//! Gemma 4 mixture of experts on lb-lfm2's expert interfaces ([`Route`], [`ExpertStore`],
//! [`PlacementStats`], the per-layer decode counters in `stage`).
//!
//! Router (llama.cpp `gemma4.cpp` + `build_moe_ffn`, `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX`,
//! `norm_w = true`, no selection bias):
//! ```text
//! r      = rms(x_mid) / sqrt(n_embd) * router_scale     (x_mid = the residual after attention,
//!                                                         NOT the experts' own pre-norm)
//! p      = softmax(W_r r)            over all experts
//! sel    = top-k of p                (ties: the lower expert id first)
//! w_e    = p_e / max(sum_sel p, 2^-14)
//! moe(x) = sum_e  w_e * down_scale[e] * down_e( gelu(gate_e x) * up_e x ),  x = rms(x_mid, pre_ffw_norm_2)
//! ```
//! GELU is the tanh form (`ggml_gelu`). The outputs are summed in ascending expert id, the
//! engine's determinism rule (lb-lfm2 `moe_ffn_with`): llama.cpp sums in top-k order, so the
//! two differ only in float rounding.
//!
//! Every expert runs on the CPU here. A GPU-resident expert would need a GEGLU variant of
//! lb-q4-gpu's SwiGLU kernels (the device kernels hard-code `silu`); until that exists the
//! store's residents are counted but the forward refuses to run with any (see
//! [`check_store`]).

use crate::numerics::matmul;
use crate::weights::{Moe, Weights};
pub use lb_lfm2::moe::{ExpertStore, Home, MissPolicy, Placement, PlacementStats, Route, Scraper};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::sync::atomic::Ordering;

/// `ggml_gelu_f32` (tanh approximation). llama.cpp's CPU backend reads it from an f16 table
/// (`GGML_GELU_FP16`), its CUDA backend computes this formula; we compute the formula.
#[inline]
pub fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const GELU_COEF_A: f32 = 0.044715;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x)).tanh())
}

/// The router's input for one token: `rms(x) / sqrt(n_embd) * scale` (no norm weight).
pub fn router_input(x: &[f32], scale: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let r = crate::model::rms_scale(x, eps);
    let c = 1.0 / (n as f32).sqrt();
    x.iter()
        .zip(scale)
        .map(|(v, s)| ((v * r) * c) * s)
        .collect()
}

/// Softmax router, top-k, weights renormalised over the selected (module doc).
pub fn route_softmax(router: &[f32], r: &[f32], n_expert: usize, k: usize) -> Route {
    let n = r.len();
    debug_assert_eq!(router.len(), n_expert * n);
    let logits: Vec<f32> = (0..n_expert)
        .map(|e| {
            router[e * n..(e + 1) * n]
                .iter()
                .zip(r)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect();
    let probs = softmax(&logits);
    let mut order: Vec<usize> = (0..n_expert).collect();
    // stable: equal probabilities keep the lower id first
    order.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let experts: Vec<usize> = order[..k].to_vec();
    let sum: f32 = experts
        .iter()
        .map(|&e| probs[e])
        .sum::<f32>()
        .max(6.103_515_6e-5);
    let weights = experts.iter().map(|&e| probs[e] / sum).collect();
    Route { experts, weights }
}

/// `ggml_soft_max` on one row: max-subtracted, sum in f64.
pub fn softmax(x: &[f32]) -> Vec<f32> {
    let mx = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let ex: Vec<f32> = x.iter().map(|v| (v - mx).exp()).collect();
    let sum: f64 = ex.iter().map(|&v| v as f64).sum();
    let inv = (1.0 / sum) as f32;
    ex.into_iter().map(|v| v * inv).collect()
}

/// One GEGLU expert over `m` rows: `down(gelu(gate x) * up x)` (without the down scale).
pub fn expert_ffn(w: &Weights, moe: &Moe, e: usize, x: &[f32], m: usize) -> Vec<f32> {
    let f = moe.gate.rows;
    let n_embd = moe.down.rows;
    let mut g = vec![0f32; m * f];
    let mut u = vec![0f32; m * f];
    matmul(w.mat(&moe.gate.expert(e)), x, m, &mut g);
    matmul(w.mat(&moe.up.expert(e)), x, m, &mut u);
    for (gi, ui) in g.iter_mut().zip(&u) {
        *gi = gelu(*gi) * ui;
    }
    let mut out = vec![0f32; m * n_embd];
    matmul(w.mat(&moe.down.expert(e)), &g, m, &mut out);
    out
}

/// Refuse a store with device residents: gemma's GEGLU experts have no device kernel yet.
pub fn check_store(store: &ExpertStore) -> Result<(), String> {
    if store.n_resident() > 0 {
        return Err("lb-gemma4 has no device kernel for GEGLU experts (lb-q4-gpu hard-codes SwiGLU); residents cannot run".into());
    }
    Ok(())
}

/// MoE FFN for `m` tokens. `x` is the experts' input (`rms(x_mid, pre_ffw_norm_2)`), `r` the
/// router input per token ([`router_input`]); both `[m, n_embd]`. Returns `[m, n_embd]` and the
/// routes. Counters go to `stats` exactly as lb-lfm2 counts them (`cpu_calls`, and per layer
/// for decode tokens), so the hit-rate tooling reads gemma runs unchanged.
#[allow(clippy::too_many_arguments)]
pub fn moe_ffn(
    w: &Weights,
    moe: &Moe,
    x: &[f32],
    r: &[f32],
    m: usize,
    store: &ExpertStore,
    layer: usize,
    stats: &PlacementStats,
) -> (Vec<f32>, Vec<Route>) {
    let cfg = &w.cfg;
    let n = cfg.n_embd;
    let routes: Vec<Route> = (0..m)
        .map(|t| {
            route_softmax(
                &moe.router,
                &r[t * n..(t + 1) * n],
                cfg.n_expert,
                cfg.n_expert_used,
            )
        })
        .collect();
    // expert -> (token, weight), ascending expert id
    let mut by_expert: BTreeMap<usize, Vec<(usize, f32)>> = BTreeMap::new();
    for (t, rt) in routes.iter().enumerate() {
        for (&e, &wt) in rt.experts.iter().zip(&rt.weights) {
            by_expert.entry(e).or_default().push((t, wt));
        }
    }
    let experts: Vec<(usize, Vec<(usize, f32)>)> = by_expert.into_iter().collect();
    let resident = experts
        .iter()
        .filter(|(e, _)| store.is_resident(layer, *e))
        .count();
    stats.gpu_hits.fetch_add(resident as u64, Ordering::Relaxed);
    stats
        .cpu_calls
        .fetch_add((experts.len() - resident) as u64, Ordering::Relaxed);
    if m == 1 {
        stats
            .stage
            .layer(layer, resident, 0, experts.len() - resident);
    }
    let t0 = std::time::Instant::now();
    let parts: Vec<(usize, Vec<(usize, f32)>, Vec<f32>)> = experts
        .into_par_iter()
        .map(|(e, toks)| {
            let mut xs = Vec::with_capacity(toks.len() * n);
            for (t, _) in &toks {
                xs.extend_from_slice(&x[t * n..(t + 1) * n]);
            }
            let y = expert_ffn(w, moe, e, &xs, toks.len());
            (e, toks, y)
        })
        .collect();
    stats
        .cpu_ns
        .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let mut out = vec![0f32; m * n];
    for (e, toks, y) in parts {
        let s = moe.down_scale[e];
        for (i, (t, wt)) in toks.iter().enumerate() {
            let o = &mut out[t * n..(t + 1) * n];
            for (oj, yj) in o.iter_mut().zip(&y[i * n..(i + 1) * n]) {
                *oj += (yj * s) * wt;
            }
        }
    }
    (out, routes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_selects_top_k_by_softmax_and_renormalises() {
        // 3 experts over a 2-dim input; logits = [1, 3, 2]
        let router = vec![1.0, 0.0, 3.0, 0.0, 2.0, 0.0];
        let rt = route_softmax(&router, &[1.0, 0.0], 3, 2);
        assert_eq!(rt.experts, vec![1, 2]);
        let (p1, p2) = (3f32.exp(), 2f32.exp());
        assert!((rt.weights[0] - p1 / (p1 + p2)).abs() < 1e-6);
        assert!((rt.weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn router_ties_keep_the_lower_id() {
        let router = vec![1.0, 1.0, 1.0];
        let rt = route_softmax(&router, &[1.0], 3, 2);
        assert_eq!(rt.experts, vec![0, 1]);
    }

    #[test]
    fn gelu_matches_the_tanh_form() {
        assert_eq!(gelu(0.0), 0.0);
        assert!((gelu(1.0) - 0.841_192).abs() < 1e-5);
        assert!((gelu(-1.0) + 0.158_808).abs() < 1e-5);
    }

    #[test]
    fn router_input_is_unit_rms_over_sqrt_n_times_scale() {
        let x = [3.0f32, 4.0];
        let r = router_input(&x, &[2.0, 2.0], 0.0);
        // rms = sqrt(12.5); r = x / rms / sqrt(2) * 2
        let rms = 12.5f32.sqrt();
        assert!((r[0] - 3.0 / rms / 2f32.sqrt() * 2.0).abs() < 1e-6);
    }
}

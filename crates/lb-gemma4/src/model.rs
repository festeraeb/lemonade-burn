//! Gemma 4 decoder over GGUF weights, on the CPU (f32 activations, quantized weights through
//! `lb_gguf::matvec_f32`). Mirrors llama.cpp e14a779 `src/models/gemma4.cpp`.
//!
//! Per layer (`x` the residual stream; `rms(v, w)` = RMS norm then times `w`):
//! ```text
//! h        = rms(x, attn_norm)
//! q        = rope(rms_head(Wq h, q_norm))         per head
//! k        = rope(rms_head(Wk h, k_norm))
//! v        = rms_head(Wv h)  (no weight; Wv = Wk on layers without attn_v)
//! a        = softmax(q k^T * 1.0  + mask) v        (scale 1.0; mask causal, plus the window
//!                                                    on sliding layers)
//! x_mid    = rms(Wo a, post_attention_norm) + x
//! d        = rms(down(gelu(gate y) * up y), post_ffw_norm_1),  y = rms(x_mid, ffn_norm)
//! d       += rms(moe(x_mid), post_ffw_norm_2)      (MoE layers; moe.rs)
//! x        = (rms(d, post_ffw_norm) + x_mid) * layer_output_scale
//! ```
//! Input: `token_embd[t] * sqrt(n_embd)`. Output: `softcap(rms(x, output_norm) @ output^T)`,
//! `softcap(z) = c * tanh(z / c)`, `c = final_logit_softcapping`.
//! RoPE is NEOX (pairs `j`, `j + n_rot/2`); sliding layers use base `freq_base_swa`, global
//! layers `freq_base` with `theta_j / rope_freqs[j]` (factors of 1e30 switch a pair off).

use crate::config::Config;
use crate::moe::{
    check_store, moe_ffn, router_input, ExpertStore, Placement, PlacementStats, Scraper,
};
use crate::weights::{MatRef, Weights};
use anyhow::Result;
use lb_gguf::Gguf;
pub use lb_lfm2::model::{CapturedLayer, Timings};
use rayon::prelude::*;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub struct Gemma4 {
    pub w: Weights,
    pub placement: Placement,
    /// lb-lfm2's expert store: residency, the miss policy and the live knobs.
    pub store: ExpertStore,
    pub stats: Arc<PlacementStats>,
    pub scraper: Arc<Scraper>,
    pub timings: Arc<Timings>,
    capture_on: AtomicBool,
    capture: Mutex<Vec<CapturedLayer>>,
    trace: Mutex<Option<Vec<Traced>>>,
}

/// One named intermediate of the LAST row of a forward pass (parity diagnosis against
/// llama.cpp's eval callback, whose names these follow: `layer` = the `-N` suffix).
#[derive(Clone, Debug)]
pub struct Traced {
    pub name: &'static str,
    pub layer: i32,
    pub data: Vec<f32>,
}

/// KV cache: per layer, `[pos][kv_heads * head_dim]` for K and V.
pub struct Cache {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub len: usize,
}

/// `1 / sqrt(mean(x^2) + eps)`, the sum in f64 (`ggml_compute_forward_rms_norm_f32`).
pub fn rms_scale(x: &[f32], eps: f32) -> f32 {
    let sum: f64 = x.iter().map(|&v| (v * v) as f64).sum();
    let mean = (sum / x.len() as f64) as f32;
    1.0 / (mean + eps).sqrt()
}

/// RMS-normalise each `n`-wide row of `x` in place, then multiply by `w` when given.
pub fn rms_rows(x: &mut [f32], n: usize, w: Option<&[f32]>, eps: f32) {
    for row in x.chunks_exact_mut(n) {
        let r = rms_scale(row, eps);
        match w {
            Some(w) => row.iter_mut().zip(w).for_each(|(v, wi)| *v = (*v * r) * wi),
            None => row.iter_mut().for_each(|v| *v *= r),
        }
    }
}

fn rms_copy(x: &[f32], n: usize, w: &[f32], eps: f32) -> Vec<f32> {
    let mut y = x.to_vec();
    rms_rows(&mut y, n, Some(w), eps);
    y
}

/// NEOX RoPE on one head vector at position `pos` (`ggml_rope_cache_init` arithmetic, f32).
pub fn rope_neox(v: &mut [f32], pos: usize, n_rot: usize, base: f32, factors: Option<&[f32]>) {
    let half = n_rot / 2;
    let theta_scale = base.powf(-2.0 / n_rot as f32);
    let mut theta = pos as f32;
    for j in 0..half {
        let ff = factors.map_or(1.0, |f| f[j]);
        let t = theta / ff;
        let (s, c) = (t.sin(), t.cos());
        let (x0, x1) = (v[j], v[j + half]);
        v[j] = x0 * c - x1 * s;
        v[j + half] = x0 * s + x1 * c;
        theta *= theta_scale;
    }
}

impl Gemma4 {
    pub fn load(path: &Path) -> Result<Self> {
        let g = Arc::new(Gguf::open(path)?);
        let w = Weights::load(g)?;
        let (nl, ne) = (w.cfg.n_layer, w.cfg.n_expert);
        let model_id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("gemma4")
            .to_string();
        let scraper = Scraper::from_env(&model_id);
        if std::env::var_os("NAUTIBURN_CLASS_ID").is_none() {
            scraper.set_class_id("lb-gemma4".into());
        }
        Ok(Self {
            w,
            placement: Placement::all_cpu(nl, ne),
            store: ExpertStore::default(),
            stats: Arc::new(PlacementStats::default()),
            scraper: Arc::new(scraper),
            timings: Arc::new(Timings::default()),
            capture_on: AtomicBool::new(false),
            capture: Mutex::new(Vec::new()),
            trace: Mutex::new(None),
        })
    }

    pub fn cfg(&self) -> &Config {
        &self.w.cfg
    }

    /// Per-layer hidden-state capture (lb-lfm2's hook): while on, every forward pass appends one
    /// [`CapturedLayer`] per layer. `x_mid` is the residual after attention (what the router
    /// and both FFN pre-norms read), `x_out` the layer output after `layer_output_scale`.
    pub fn set_capture(&self, on: bool) {
        self.capture_on.store(on, Ordering::Relaxed);
        if !on {
            self.capture.lock().unwrap().clear();
        }
    }

    pub fn capture_enabled(&self) -> bool {
        self.capture_on.load(Ordering::Relaxed)
    }

    pub fn take_capture(&self) -> Vec<CapturedLayer> {
        std::mem::take(&mut *self.capture.lock().unwrap())
    }

    /// Record named intermediates of the last row of every following pass (`take_trace`).
    pub fn set_trace(&self, on: bool) {
        *self.trace.lock().unwrap() = if on { Some(Vec::new()) } else { None };
    }

    pub fn take_trace(&self) -> Vec<Traced> {
        self.trace
            .lock()
            .unwrap()
            .as_mut()
            .map(std::mem::take)
            .unwrap_or_default()
    }

    fn tr(&self, name: &'static str, layer: i32, x: &[f32], width: usize) {
        if let Some(t) = self.trace.lock().unwrap().as_mut() {
            t.push(Traced {
                name,
                layer,
                data: x[x.len() - width..].to_vec(),
            });
        }
    }

    /// The expert-prediction guess of `target`'s experts from the residual `x_mid` of the
    /// layer before (lb-lfm2 `stage::predict`'s `mid1` rule). Gemma's router ranks by softmax
    /// of `W_r (rms(x) / sqrt(n) * scale)`; softmax and sigmoid are both monotone in the logit,
    /// so lb-lfm2's `stage::rank` gives the same ranking with `norm = scale / sqrt(n)` and a
    /// zero bias (its `margin` is then a sigmoid-space margin, not a softmax one).
    pub fn predict(
        &self,
        target: usize,
        x_mid: &[f32],
        slots: usize,
    ) -> Option<lb_lfm2::stage::Guess> {
        let moe = self.w.layers.get(target)?.moe.as_ref()?;
        let cfg = self.cfg();
        let c = 1.0 / (cfg.n_embd as f32).sqrt();
        let norm: Vec<f32> = moe.router_scale.iter().map(|s| s * c).collect();
        let bias = vec![0f32; cfg.n_expert];
        Some(lb_lfm2::stage::rank(
            &moe.router,
            &bias,
            &norm,
            x_mid,
            cfg.rms_eps,
            cfg.n_expert,
            cfg.n_expert_used,
            slots,
            target,
        ))
    }

    pub fn new_cache(&self) -> Cache {
        let n = self.cfg().n_layer;
        Cache {
            k: vec![Vec::new(); n],
            v: vec![Vec::new(); n],
            len: 0,
        }
    }

    /// `x [m, w.cols] @ W^T -> [m, w.rows]`.
    pub fn linear(&self, x: &[f32], w: &MatRef, m: usize) -> Vec<f32> {
        assert_eq!(x.len(), m * w.cols, "linear: input width");
        let t = Instant::now();
        let mut out = vec![0f32; m * w.rows];
        crate::numerics::matmul(self.w.mat(w), x, m, &mut out);
        self.timings
            .matmul_ns
            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        out
    }

    /// GEGLU dense FFN on `m` rows.
    fn dense_ffn(
        &self,
        y: &[f32],
        gate: &MatRef,
        up: &MatRef,
        down: &MatRef,
        m: usize,
    ) -> Vec<f32> {
        let mut g = self.linear(y, gate, m);
        let u = self.linear(y, up, m);
        for (gi, ui) in g.iter_mut().zip(&u) {
            *gi = crate::moe::gelu(*gi) * ui;
        }
        self.linear(&g, down, m)
    }

    /// Attention block for `m` tokens at `pos0..`; returns `Wo a` `[m, n_embd]`.
    fn attention(
        &self,
        li: usize,
        h: &[f32],
        m: usize,
        pos0: usize,
        cache: &mut Cache,
    ) -> Vec<f32> {
        let cfg = self.cfg();
        let lw = &self.w.layers[li];
        let a = &lw.attn;
        let (nh, kvh, hd) = (cfg.n_head[li], cfg.kv_heads[li], cfg.head_dim(li));
        let (n_rot, base) = (cfg.n_rot(li), cfg.rope_base(li) as f32);
        let factors = if cfg.swa[li] {
            None
        } else {
            self.w.rope_freqs.as_deref()
        };
        let eps = cfg.rms_eps;
        let mut q = self.linear(h, &a.q, m);
        let mut k = self.linear(h, &a.k, m);
        let mut v = match &a.v {
            Some(wv) => self.linear(h, wv, m),
            None => k.clone(),
        };
        rms_rows(&mut q, hd, Some(&a.q_norm), eps);
        rms_rows(&mut k, hd, Some(&a.k_norm), eps);
        rms_rows(&mut v, hd, None, eps);
        for t in 0..m {
            for hq in q[t * nh * hd..(t + 1) * nh * hd].chunks_exact_mut(hd) {
                rope_neox(hq, pos0 + t, n_rot, base, factors);
            }
            for hk in k[t * kvh * hd..(t + 1) * kvh * hd].chunks_exact_mut(hd) {
                rope_neox(hk, pos0 + t, n_rot, base, factors);
            }
        }
        self.tr("Qcur_pos", li as i32, &q, nh * hd);
        self.tr("Kcur_pos", li as i32, &k, kvh * hd);
        // llama.cpp: K and V live in an f16 cache, and flash attention takes Q as f16
        crate::numerics::kv_store(&mut q);
        crate::numerics::kv_store(&mut k);
        crate::numerics::kv_store(&mut v);
        let kc = &mut cache.k[li];
        let vc = &mut cache.v[li];
        kc.truncate(pos0 * kvh * hd);
        vc.truncate(pos0 * kvh * hd);
        kc.extend_from_slice(&k);
        vc.extend_from_slice(&v);
        let (kc, vc) = (&cache.k[li], &cache.v[li]);
        let group = nh / kvh;
        let total = pos0 + m;
        // one (token, head) per task
        let mut out = vec![0f32; m * nh * hd];
        out.par_chunks_mut(hd).enumerate().for_each(|(idx, o)| {
            let (t, hh) = (idx / nh, idx % nh);
            let g = hh / group;
            let qpos = pos0 + t;
            let qv = &q[(t * nh + hh) * hd..(t * nh + hh + 1) * hd];
            let mut scores: Vec<f32> = Vec::with_capacity(total);
            let mut keys: Vec<usize> = Vec::with_capacity(total);
            for j in 0..total {
                if !cfg.visible(li, qpos, j) {
                    continue;
                }
                let kv = &kc[(j * kvh + g) * hd..(j * kvh + g + 1) * hd];
                scores.push(qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f32>());
                keys.push(j);
            }
            let p = crate::moe::softmax(&scores);
            for (pj, &j) in p.iter().zip(&keys) {
                let vv = &vc[(j * kvh + g) * hd..(j * kvh + g + 1) * hd];
                for (oi, vi) in o.iter_mut().zip(vv) {
                    *oi += pj * vi;
                }
            }
        });
        self.tr("kqv_out", li as i32, &out, nh * hd);
        self.linear(&out, &a.o, m)
    }

    /// Run `tokens` at positions `cache.len ..`. Returns logits `[m, vocab]` when `all_logits`,
    /// else `[1, vocab]` for the last position.
    pub fn forward(&self, tokens: &[u32], cache: &mut Cache, all_logits: bool) -> Result<Vec<f32>> {
        check_store(&self.store).map_err(anyhow::Error::msg)?;
        let mm0 = self.timings.matmul_ns.load(Ordering::Relaxed);
        let t_fwd = Instant::now();
        let out = self.forward_inner(tokens, cache, all_logits);
        let dt = t_fwd.elapsed().as_nanos() as u64;
        let t = &self.timings;
        t.forward_ns.fetch_add(dt, Ordering::Relaxed);
        t.forwards.fetch_add(1, Ordering::Relaxed);
        if tokens.len() == 1 {
            t.decode_forward_ns.fetch_add(dt, Ordering::Relaxed);
            t.decode_matmul_ns.fetch_add(
                t.matmul_ns.load(Ordering::Relaxed).wrapping_sub(mm0),
                Ordering::Relaxed,
            );
            t.decodes.fetch_add(1, Ordering::Relaxed);
        }
        Ok(out)
    }

    fn forward_inner(&self, tokens: &[u32], cache: &mut Cache, all_logits: bool) -> Vec<f32> {
        let cfg = self.cfg();
        let (m, e, eps) = (tokens.len(), cfg.n_embd, cfg.rms_eps);
        let pos0 = cache.len;
        let capturing = self.capture_enabled();
        let scale = (e as f32).sqrt();
        let mut x = vec![0f32; m * e];
        for (i, &t) in tokens.iter().enumerate() {
            let row = &mut x[i * e..(i + 1) * e];
            self.w.embed_row(t, row);
            row.iter_mut().for_each(|v| *v *= scale);
        }
        self.tr("inp_scaled", -1, &x, e);
        for (li, lw) in self.w.layers.iter().enumerate() {
            let h = rms_copy(&x, e, &lw.attn_norm, eps);
            self.tr("attn_norm", li as i32, &h, e);
            let mut a = self.attention(li, &h, m, pos0, cache);
            rms_rows(&mut a, e, Some(&lw.attn.post_norm), eps);
            // x_mid = attn_post_norm + x
            let x_mid: Vec<f32> = a.iter().zip(&x).map(|(p, q)| p + q).collect();
            self.tr("attn_out", li as i32, &x_mid, e);

            let y = rms_copy(&x_mid, e, &lw.dense.norm, eps);
            let mut d = self.dense_ffn(&y, &lw.dense.gate, &lw.dense.up, &lw.dense.down, m);
            if let Some(pn) = &lw.dense.post_norm {
                rms_rows(&mut d, e, Some(pn), eps);
            }
            self.tr("ffn_mlp", li as i32, &d, e);
            let mut routes = Vec::new();
            if let Some(moe) = &lw.moe {
                let xin = rms_copy(&x_mid, e, &moe.pre_norm, eps);
                let mut r = Vec::with_capacity(m * e);
                for t in 0..m {
                    r.extend(router_input(
                        &x_mid[t * e..(t + 1) * e],
                        &moe.router_scale,
                        eps,
                    ));
                }
                let t0 = Instant::now();
                let (mut mo, rt) = moe_ffn(&self.w, moe, &xin, &r, m, &self.store, li, &self.stats);
                self.timings
                    .matmul_ns
                    .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                self.scraper.record(li, pos0, &rt);
                if let Some(last) = rt.last() {
                    self.tr(
                        "ffn_moe_weights_norm",
                        li as i32,
                        &last.weights,
                        last.weights.len(),
                    );
                    let ids: Vec<f32> = last.experts.iter().map(|&e| e as f32).collect();
                    self.tr("ffn_moe_topk", li as i32, &ids, ids.len());
                }
                rms_rows(&mut mo, e, Some(&moe.post_norm), eps);
                self.tr("ffn_moe", li as i32, &mo, e);
                d.iter_mut().zip(&mo).for_each(|(a, b)| *a += b);
                routes = rt;
            }
            rms_rows(&mut d, e, Some(&lw.post_ffw_norm), eps);
            let s = lw.out_scale;
            for ((xi, di), mi) in x.iter_mut().zip(&d).zip(&x_mid) {
                *xi = (di + mi) * s;
            }
            self.tr("l_out", li as i32, &x, e);
            if capturing {
                self.capture.lock().unwrap().push(CapturedLayer {
                    layer: li,
                    pos0,
                    m,
                    x_mid,
                    x_out: x.clone(),
                    routes,
                });
            }
        }
        cache.len = pos0 + m;
        let rows = if all_logits { m } else { 1 };
        let mut xf = x[(m - rows) * e..].to_vec();
        rms_rows(&mut xf, e, Some(&self.w.final_norm), eps);
        self.tr("result_norm", -1, &xf, e);
        let mut logits = self.linear(&xf, &self.w.output, rows);
        let c = cfg.final_softcap;
        if c > 0.0 {
            logits.iter_mut().for_each(|z| *z = (*z / c).tanh() * c);
        }
        logits
    }

    /// Greedy index of a logits row.
    pub fn argmax(row: &[f32]) -> u32 {
        let mut best = 0usize;
        for (j, &x) in row.iter().enumerate() {
            if x > row[best] {
                best = j;
            }
        }
        best as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_rotates_pairs_half_apart_and_factors_switch_pairs_off() {
        let mut v = vec![1.0, 0.0, 0.0, 0.0];
        rope_neox(&mut v, 1, 4, 10000.0, None);
        // pair (0, 2): theta = 1
        assert!((v[0] - 1f32.cos()).abs() < 1e-6 && (v[2] - 1f32.sin()).abs() < 1e-6);
        let mut w = vec![1.0, 1.0, 0.0, 0.0];
        rope_neox(&mut w, 7, 4, 10000.0, Some(&[1e30, 1e30]));
        assert!(w
            .iter()
            .zip([1.0, 1.0, 0.0, 0.0])
            .all(|(a, b)| (a - b).abs() < 1e-20));
    }

    #[test]
    fn rms_rows_normalises_each_row() {
        let mut x = vec![3.0, 4.0, 1.0, 1.0];
        rms_rows(&mut x, 2, Some(&[1.0, 2.0]), 0.0);
        let r = 1.0 / 12.5f32.sqrt();
        assert!((x[0] - 3.0 * r).abs() < 1e-6 && (x[1] - 8.0 * r).abs() < 1e-6);
        assert!((x[2] - 1.0).abs() < 1e-6 && (x[3] - 2.0).abs() < 1e-6);
    }
}

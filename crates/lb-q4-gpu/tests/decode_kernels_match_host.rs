//! The device-decode glue kernels (`decode.rs`) against host references written out longhand.
//!
//! Each reference is the host operation the kernel replaces (`Lfm2::rms_norm`, `rope_tables` +
//! `apply_rope`, the cached attention, `route_sigmoid`, the expert sum in `moe_ffn`). Sums run in
//! a different order on the device, so floats are compared to a tolerance; the router's
//! *selection* is compared exactly, ties included, because a flipped expert is not rounding.
//!
//! These open a device, so they run only when one is named (`LB_WGPU_DEVICE` or
//! `LB_CUDA_DEVICE`); without it each test prints why and passes, so a plain `cargo test` on a
//! host whose cards are busy never touches them.
mod common;
use common::Rng;
use lb_gguf::types::quantize_q4_0;
use lb_gguf::{GgmlType, QuantMat};
use lb_q4_gpu::decode::rope_table;
use lb_q4_gpu::{Gpu, GpuExpert, GpuExpertLayer, TokenParams};

fn gpu() -> Option<Gpu> {
    if std::env::var_os("LB_WGPU_DEVICE").is_none() && std::env::var_os("LB_CUDA_DEVICE").is_none() {
        eprintln!("skipped: no device named (set LB_WGPU_DEVICE or LB_CUDA_DEVICE)");
        return None;
    }
    Some(Gpu::open().unwrap_or_else(|e| panic!("no GPU: {e}")))
}

fn read(gpu: &Gpu, h: &lb_q4_gpu::cubecl_server::Handle, n: usize) -> Vec<f32> {
    gpu.read_all(vec![(h.clone(), n)]).expect("read")[0].clone()
}

fn close(got: &[f32], want: &[f32], tol_rel: f32, tol_abs: f32, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let mut worst = 0f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        assert!(d <= tol_abs + tol_rel * w.abs(), "{label} [{i}]: gpu {g} host {w} (|d| {d})");
        worst = worst.max(d);
    }
    eprintln!("{label}: max |gpu-host| = {worst:e}");
}

/// `Lfm2::rms_norm` for one row: `x * (1 / sqrt(mean(x^2) + eps)) * w`.
fn host_rms(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len() as f64;
    let r = 1.0 / ((ms as f32) + eps).sqrt();
    x.iter().zip(w).map(|(v, w)| v * r * w).collect()
}

#[test]
fn add_rms_norm_matches_host() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(11);
    let eps = 1e-5f32;
    for &n in &[2048usize, 100, 1] {
        let x: Vec<f32> = (0..n).map(|_| rng.unit() * 4.0).collect();
        let y: Vec<f32> = (0..n).map(|_| rng.unit()).collect();
        let w: Vec<f32> = (0..n).map(|_| 0.5 + rng.unit()).collect();
        let (xh, yh, wh) = (gpu.upload_f32(&x), gpu.upload_f32(&y), gpu.upload_f32(&w));
        // first norm of a token: no residual add, x untouched
        let h = gpu.add_rms_norm(&xh, None, &wh, n, eps);
        close(&read(&gpu, &h, n), &host_rms(&x, &w, eps), 2e-6, 2e-6, &format!("rms n={n}"));
        close(&read(&gpu, &xh, n), &x, 0.0, 0.0, &format!("x untouched n={n}"));
        // fused residual: x <- x + y, then the norm of the sum
        let h = gpu.add_rms_norm(&xh, Some(&yh), &wh, n, eps);
        let sum: Vec<f32> = x.iter().zip(&y).map(|(a, b)| a + b).collect();
        close(&read(&gpu, &xh, n), &sum, 0.0, 0.0, &format!("x+y n={n}"));
        close(&read(&gpu, &h, n), &host_rms(&sum, &w, eps), 2e-6, 2e-6, &format!("add+rms n={n}"));
    }
}

/// `rms_norm` per head, then `apply_rope` with the f64 tables: `x * cos + rot(x) * sin`.
fn host_norm_rope(x: &[f32], norm: &[f32], table: &[f32], pos: usize, hd: usize, eps: f32) -> Vec<f32> {
    let n = host_rms(x, norm, eps);
    let (c, s) = (&table[pos * 2 * hd..pos * 2 * hd + hd], &table[pos * 2 * hd + hd..(pos + 1) * 2 * hd]);
    (0..hd).map(|j| {
        let rot = if j < hd / 2 { -n[j + hd / 2] } else { n[j - hd / 2] };
        n[j] * c[j] + rot * s[j]
    }).collect()
}

#[test]
fn qk_norm_rope_and_kv_append_match_host() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(5);
    let (nh, kvh, hd, cap, eps) = (8usize, 2usize, 64usize, 300usize, 1e-5f32);
    let table = rope_table(1_000_000.0, hd, cap);
    let rope = gpu.upload_f32(&table);
    let kc0: Vec<f32> = (0..kvh * cap * hd).map(|_| rng.unit()).collect();
    let vc0: Vec<f32> = (0..kvh * cap * hd).map(|_| rng.unit()).collect();
    let (kc, vc) = (gpu.upload_f32(&kc0), gpu.upload_f32(&vc0));
    let qn: Vec<f32> = (0..hd).map(|_| 1.0 + rng.unit() * 0.3).collect();
    let kn: Vec<f32> = (0..hd).map(|_| 1.0 + rng.unit() * 0.3).collect();
    let (qnh, knh) = (gpu.upload_f32(&qn), gpu.upload_f32(&kn));
    let mut want_k = kc0.clone();
    let mut want_v = vc0.clone();
    let mut tp = TokenParams::new(&gpu, 0, 1, cap);
    for &pos in &[0usize, 1, 37, 255, 299] {
        tp.set(&gpu, pos, pos + 1, cap);
        let q: Vec<f32> = (0..nh * hd).map(|_| rng.unit() * 2.0).collect();
        let k: Vec<f32> = (0..kvh * hd).map(|_| rng.unit() * 2.0).collect();
        let v: Vec<f32> = (0..kvh * hd).map(|_| rng.unit() * 2.0).collect();
        let (qh, kh, vh) = (gpu.upload_f32(&q), gpu.upload_f32(&k), gpu.upload_f32(&v));
        let qo = gpu.qk_norm_rope(&qh, &kh, &vh, &qnh, &knh, &rope, cap, &kc, &vc, &tp, (nh, kvh, hd), eps);
        let want_q: Vec<f32> = (0..nh).flat_map(|h| host_norm_rope(&q[h * hd..(h + 1) * hd], &qn, &table, pos, hd, eps)).collect();
        close(&read(&gpu, &qo, nh * hd), &want_q, 3e-6, 3e-6, &format!("q pos={pos}"));
        for h in 0..kvh {
            let row = (h * cap + pos) * hd;
            want_k[row..row + hd].copy_from_slice(&host_norm_rope(&k[h * hd..(h + 1) * hd], &kn, &table, pos, hd, eps));
            want_v[row..row + hd].copy_from_slice(&v[h * hd..(h + 1) * hd]);
        }
        // the whole cache: the new row is right AND no other row moved
        close(&read(&gpu, &kc, kvh * cap * hd), &want_k, 3e-6, 3e-6, &format!("k cache pos={pos}"));
        close(&read(&gpu, &vc, kvh * cap * hd), &want_v, 0.0, 0.0, &format!("v cache pos={pos}"));
    }
}

/// Plain two-pass softmax attention in f64 for query head `h` over `total` positions.
fn host_attn(q: &[f32], kc: &[f32], vc: &[f32], (nh, kvh, hd, total, cap): (usize, usize, usize, usize, usize)) -> Vec<f32> {
    let group = nh / kvh;
    let scale = (hd as f64).powf(-0.5);
    let mut out = vec![0f32; nh * hd];
    for h in 0..nh {
        let kh = h / group;
        let scores: Vec<f64> = (0..total)
            .map(|p| (0..hd).map(|d| q[h * hd + d] as f64 * kc[(kh * cap + p) * hd + d] as f64).sum::<f64>() * scale)
            .collect();
        let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
        let z: f64 = e.iter().sum();
        for d in 0..hd {
            out[h * hd + d] = ((0..total).map(|p| e[p] * vc[(kh * cap + p) * hd + d] as f64).sum::<f64>() / z) as f32;
        }
    }
    out
}

#[test]
fn attention_decode_matches_host_over_1_to_257_positions_with_gqa() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(9);
    // the 24B's attention layers: 32 query heads over 8 KV heads, hd 64; and a 2:1 grouping
    for &(nh, kvh) in &[(32usize, 8usize), (4, 2), (8, 8)] {
        let hd = 64usize;
        let cap = 512usize;
        let kc: Vec<f32> = (0..kvh * cap * hd).map(|_| rng.unit() * 2.0).collect();
        let vc: Vec<f32> = (0..kvh * cap * hd).map(|_| rng.unit()).collect();
        let (kch, vch) = (gpu.upload_f32(&kc), gpu.upload_f32(&vc));
        let mut tp = TokenParams::new(&gpu, 0, 1, cap);
        for &total in &[1usize, 2, 3, 63, 64, 65, 128, 200, 256, 257] {
            tp.set(&gpu, total - 1, total, cap);
            // large q so the softmax is peaked and the online rescale really runs
            let q: Vec<f32> = (0..nh * hd).map(|_| rng.unit() * 3.0).collect();
            let qh = gpu.upload_f32(&q);
            let got = read(&gpu, &gpu.attn_decode(&qh, &kch, &vch, &tp, (nh, kvh, hd)), nh * hd);
            close(&got, &host_attn(&q, &kc, &vc, (nh, kvh, hd, total, cap)), 1e-5, 2e-6, &format!("attn nh={nh} kvh={kvh} total={total}"));
        }
    }
}

/// `lb_lfm2::moe::route_sigmoid`, copied verbatim (that crate pulls in Burn; this one must not).
fn route_sigmoid(router: &[f32], bias: &[f32], x: &[f32], n_expert: usize, k: usize) -> (Vec<usize>, Vec<f32>) {
    let n_embd = x.len();
    let probs: Vec<f32> = (0..n_expert)
        .map(|e| {
            let logit: f32 = router[e * n_embd..(e + 1) * n_embd].iter().zip(x).map(|(a, b)| a * b).sum();
            1.0 / (1.0 + (-logit).exp())
        })
        .collect();
    let mut order: Vec<usize> = (0..n_expert).collect();
    order.sort_by(|&a, &b| (probs[b] + bias[b]).partial_cmp(&(probs[a] + bias[a])).unwrap_or(std::cmp::Ordering::Equal));
    let experts: Vec<usize> = order[..k].to_vec();
    let mut weights: Vec<f32> = experts.iter().map(|&e| probs[e]).collect();
    let sum: f32 = weights.iter().sum::<f32>().max(6.103515625e-5);
    for w in &mut weights {
        *w /= sum;
    }
    (experts, weights)
}

/// Run the router kernel into slot `base` of a `n_slots`-slot buffer pre-filled with a marker,
/// return (experts, weights) for that slot and check every other slot kept its marker.
fn device_route(gpu: &Gpu, router: &[f32], bias: &[f32], x: &[f32], n_expert: usize, k: usize, base: usize, n_slots: usize, seq: bool) -> (Vec<usize>, Vec<f32>) {
    let n_embd = x.len();
    let routes = gpu.upload_u32(vec![0xDEAD_BEEF; 2 * n_slots]);
    let weights = gpu.upload_f32(&vec![-7.0; n_slots]);
    gpu.router_topk(&gpu.upload_f32(x), &gpu.upload_f32(router), &gpu.upload_f32(bias), &routes, &weights, (n_expert, k, n_embd, base, n_slots), seq);
    let r = gpu.read_u32(&routes, 2 * n_slots).unwrap();
    let w = read(gpu, &weights, n_slots);
    for s in (0..n_slots).filter(|s| *s < base || *s >= base + k) {
        assert_eq!((r[2 * s], r[2 * s + 1], w[s]), (0xDEAD_BEEF, 0xDEAD_BEEF, -7.0), "slot {s} outside the layer's was written");
    }
    for i in 0..k {
        assert_eq!(r[2 * (base + i) + 1], 0, "token index of slot {i}");
    }
    ((0..k).map(|i| r[2 * (base + i)] as usize).collect(), w[base..base + k].to_vec())
}

#[test]
fn router_topk_matches_route_sigmoid_including_ties_bias_and_the_floor() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(21);
    let (n_expert, k, n_embd) = (64usize, 4usize, 2048usize);
    for seq in [false, true] {
        // 1. random routers: same selection, weights to rounding
        for trial in 0..20 {
            let router: Vec<f32> = (0..n_expert * n_embd).map(|_| rng.unit() * 0.05).collect();
            let bias: Vec<f32> = (0..n_expert).map(|_| rng.unit() * 0.02).collect();
            let x: Vec<f32> = (0..n_embd).map(|_| rng.unit()).collect();
            let (we, ww) = route_sigmoid(&router, &bias, &x, n_expert, k);
            let (ge, gw) = device_route(&gpu, &router, &bias, &x, n_expert, k, 3 * (trial % 5), 3 * 5 + k, seq);
            assert_eq!(ge, we, "seq={seq} trial {trial}: selection");
            close(&gw, &ww, 2e-5, 1e-6, &format!("seq={seq} trial {trial} weights"));
        }
        // 2. exact ties: experts 9, 3, 40, 7 have identical router rows and bias, so identical
        // scores; the top-4 cut falls inside the tie and the lower ids must win, in id order
        let mut router: Vec<f32> = (0..n_expert * n_embd).map(|_| rng.unit() * 0.001).collect();
        let x: Vec<f32> = (0..n_embd).map(|_| 0.5 + rng.unit().abs()).collect();
        let hot: Vec<f32> = (0..n_embd).map(|_| 0.01).collect();
        for e in [40usize, 9, 3, 7, 50] {
            router[e * n_embd..(e + 1) * n_embd].copy_from_slice(&hot);
        }
        let mut bias = vec![0f32; n_expert];
        bias[50] = 0.5; // forced to rank 0 by the bias, not the probability
        let (we, ww) = route_sigmoid(&router, &bias, &x, n_expert, k);
        assert_eq!(we, vec![50, 3, 7, 9], "the reference itself: bias first, then the tie by id");
        let (ge, gw) = device_route(&gpu, &router, &bias, &x, n_expert, k, 0, k, seq);
        assert_eq!(ge, we, "seq={seq}: tie broken by lower expert id");
        close(&gw, &ww, 2e-6, 1e-7, &format!("seq={seq} tie weights"));
        // equal probabilities, so the three tied weights must be bit-equal on the device too
        assert!(gw[1] == gw[2] && gw[2] == gw[3], "seq={seq}: tied experts got different weights {gw:?}");
        // 3. the normalisation floor: every probability tiny, so sum < 2^-14 and w = p / 2^-14
        let router: Vec<f32> = (0..n_expert * n_embd).map(|_| -0.02 - rng.unit().abs() * 0.001).collect();
        let x: Vec<f32> = (0..n_embd).map(|_| 0.5).collect();
        let bias = vec![0f32; n_expert];
        let (we, ww) = route_sigmoid(&router, &bias, &x, n_expert, k);
        assert!(ww.iter().sum::<f32>() < 0.5, "the floor did not engage: {ww:?}");
        let (ge, gw) = device_route(&gpu, &router, &bias, &x, n_expert, k, 1, 1 + k, seq);
        assert_eq!(ge, we, "seq={seq}: floor case selection");
        close(&gw, &ww, 2e-4, 0.0, &format!("seq={seq} floor weights"));
    }
}

#[test]
fn weighted_sum_adds_in_ascending_expert_id() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(3);
    let (k, n, n_slots, base) = (4usize, 2048usize, 12usize, 4usize);
    let outs: Vec<f32> = (0..k * n).map(|_| rng.unit() * 10.0).collect();
    let mut routes = vec![0u32; 2 * n_slots];
    let experts = [41u32, 2, 63, 17];
    for (i, e) in experts.iter().enumerate() {
        routes[2 * (base + i)] = *e;
    }
    let mut weights = vec![0f32; n_slots];
    for i in 0..k {
        weights[base + i] = 0.1 + rng.unit().abs();
    }
    let y = gpu.weighted_sum(&gpu.upload_f32(&outs), &gpu.upload_f32(&weights), &gpu.upload_u32(routes.clone()), (k, n, base, n_slots));
    // the host's order: experts ascending (2, 17, 41, 63) = slots 1, 3, 0, 2, from zero
    let order = [1usize, 3, 0, 2];
    let want: Vec<f32> = (0..n).map(|i| order.iter().fold(0f32, |acc, &s| acc + weights[base + s] * outs[s * n + i])).collect();
    close(&read(&gpu, &y, n), &want, 1e-6, 1e-6, "weighted sum");
}

#[test]
fn conv_decode_wrapper_matches_host_over_a_sequence() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(17);
    let (e, l) = (256usize, 3usize);
    let taps: Vec<f32> = (0..l * e).map(|_| rng.unit()).collect();
    let tap_h = gpu.upload_f32(&taps);
    let state = gpu.upload_f32(&vec![0f32; (l - 1) * e]);
    let mut host_state = vec![vec![0f32; e]; l - 1];
    for t in 0..5 {
        let bcx: Vec<f32> = (0..3 * e).map(|_| rng.unit()).collect();
        let y = gpu.conv_decode(&gpu.upload_f32(&bcx), &tap_h, &state, e, l);
        let bx: Vec<f32> = (0..e).map(|i| bcx[i] * bcx[2 * e + i]).collect();
        let want: Vec<f32> = (0..e)
            .map(|i| {
                let mut acc = 0f32;
                for (kk, row) in host_state.iter().enumerate() {
                    acc += taps[kk * e + i] * row[i];
                }
                acc += taps[(l - 1) * e + i] * bx[i];
                bcx[e + i] * acc
            })
            .collect();
        host_state.remove(0);
        host_state.push(bx);
        close(&read(&gpu, &y, e), &want, 1e-6, 1e-6, &format!("conv token {t}"));
    }
}

fn q4(rng: &mut Rng, rows: usize, cols: usize) -> Vec<u8> {
    let wf: Vec<f32> = (0..rows * cols).map(|_| rng.unit()).collect();
    wf.chunks_exact(cols).flat_map(|r| quantize_q4_0(r)).collect()
}

/// The stacked layer upload, its per-expert views and the device-routed launch must all be the
/// very same arithmetic as the per-expert path: compared on f32 bits.
#[test]
fn stacked_layer_views_and_device_routes_are_bit_identical_to_per_expert() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(77);
    // 64 x 128 experts: 256 blocks a matrix, so every view lands on a 256-byte boundary
    let (n_exp, n_ff, n_embd, k) = (8usize, 64usize, 128usize, 3usize);
    let (g, u, d) = (q4(&mut rng, n_exp * n_ff, n_embd), q4(&mut rng, n_exp * n_ff, n_embd), q4(&mut rng, n_exp * n_embd, n_ff));
    let layer = GpuExpertLayer::upload_stacked(
        &gpu,
        QuantMat::new(GgmlType::Q4_0, n_exp * n_ff, n_embd, &g),
        QuantMat::new(GgmlType::Q4_0, n_exp * n_ff, n_embd, &u),
        QuantMat::new(GgmlType::Q4_0, n_exp * n_embd, n_ff, &d),
        n_exp,
    )
    .expect("stacked upload");
    let x: Vec<f32> = (0..n_embd).map(|_| rng.unit() * 2.0).collect();
    let xh = gpu.upload_f32(&x);
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    let (gb, ub, db) = (n_ff * n_embd / 32 * 18, n_ff * n_embd / 32 * 18, n_embd * n_ff / 32 * 18);
    let mut per_expert = Vec::new();
    for e in 0..n_exp {
        let own = GpuExpert::upload(
            &gpu,
            QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, &g[e * gb..(e + 1) * gb]),
            QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, &u[e * ub..(e + 1) * ub]),
            QuantMat::new(GgmlType::Q4_0, n_embd, n_ff, &d[e * db..(e + 1) * db]),
        )
        .unwrap();
        let want = own.ffn(&gpu, &x, 1).unwrap();
        let view = layer.expert(&gpu, e).expect("view");
        assert_eq!(bits(&view.ffn(&gpu, &x, 1).unwrap()), bits(&want), "expert {e}: view of the stacked layer");
        per_expert.push(want);
    }
    assert!(layer.expert(&gpu, n_exp).is_err(), "an expert past the layer must be refused");
    // device routes: the router kernel picks top-k into slot 2 of a 9-slot buffer; the layer
    // reads them there, and the outputs equal the per-expert outputs of the chosen experts
    let router: Vec<f32> = (0..n_exp * n_embd).map(|_| rng.unit() * 0.1).collect();
    let bias = vec![0f32; n_exp];
    let (n_slots, base) = (9usize, 2usize);
    let routes = gpu.upload_u32(vec![0; 2 * n_slots]);
    let weights = gpu.upload_f32(&vec![0.0; n_slots]);
    gpu.router_topk(&xh, &gpu.upload_f32(&router), &gpu.upload_f32(&bias), &routes, &weights, (n_exp, k, n_embd, base, n_slots), false);
    let (out, n) = layer.launch_device(&gpu, &xh, 1, &routes, n_slots, base, k);
    let got = read(&gpu, &out, n);
    let r = gpu.read_u32(&routes, 2 * n_slots).unwrap();
    for s in 0..k {
        let e = r[2 * (base + s)] as usize;
        assert_eq!(bits(&got[s * n_embd..(s + 1) * n_embd]), bits(&per_expert[e]), "slot {s} (expert {e}) on device routes");
    }
    // and the host-routes launch on the same routes agrees with the device-routes one
    let host_routes: Vec<u32> = r[2 * base..2 * (base + k)].to_vec();
    let (out2, n2) = layer.launch(&gpu, &xh, 1, &host_routes).unwrap();
    assert_eq!(bits(&read(&gpu, &out2, n2)), bits(&got), "host-routes and device-routes launches");
}

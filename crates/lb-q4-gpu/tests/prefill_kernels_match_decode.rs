//! The batched prefill kernels (`prefill.rs`) against their decode twins run one token at a time.
//!
//! The decode kernels are already checked against host references (`decode_kernels_match_host.rs`),
//! and each prefill kernel is the same arithmetic per row, so the strongest check is: a chunk of `m`
//! rows through the prefill kernel equals `m` decode steps, including the state each leaves behind
//! (the KV cache rows, the conv state). Router selection is compared exactly.
//!
//! These open a device, so they run only when one is named (`LB_WGPU_DEVICE` or `LB_CUDA_DEVICE`).
mod common;
use common::Rng;
use lb_q4_gpu::decode::rope_table;
use lb_q4_gpu::{Gpu, TokenParams};

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
        assert!(d <= tol_abs + tol_rel * w.abs(), "{label} [{i}]: rows {g} per-token {w} (|d| {d})");
        worst = worst.max(d);
    }
    eprintln!("{label}: max |rows - per-token| = {worst:e}");
}

fn vecr(rng: &mut Rng, n: usize, s: f32) -> Vec<f32> {
    (0..n).map(|_| rng.unit() * s).collect()
}

#[test]
fn rms_norm_rows_equals_one_row_at_a_time() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(21);
    let (m, n, eps) = (7usize, 2048usize, 1e-5f32);
    let x = vecr(&mut rng, m * n, 3.0);
    let y = vecr(&mut rng, m * n, 1.0);
    let w: Vec<f32> = (0..n).map(|_| 0.5 + rng.unit()).collect();
    let wh = gpu.upload_f32(&w);
    let xr = gpu.upload_f32(&x);
    let hr = gpu.rms_norm_rows(&xr, Some(&gpu.upload_f32(&y)), &wh, m, n, eps);
    let (got_h, got_x) = (read(&gpu, &hr, m * n), read(&gpu, &xr, m * n));
    let mut want_h = Vec::new();
    let mut want_x = Vec::new();
    for t in 0..m {
        let xt = gpu.upload_f32(&x[t * n..(t + 1) * n]);
        let h = gpu.add_rms_norm(&xt, Some(&gpu.upload_f32(&y[t * n..(t + 1) * n])), &wh, n, eps);
        want_h.extend(read(&gpu, &h, n));
        want_x.extend(read(&gpu, &xt, n));
    }
    close(&got_x, &want_x, 0.0, 0.0, "residual add");
    close(&got_h, &want_h, 1e-6, 1e-6, "rms rows");
    // without the add, x is untouched
    let x2 = gpu.upload_f32(&x);
    let _ = gpu.rms_norm_rows(&x2, None, &wh, m, n, eps);
    close(&read(&gpu, &x2, m * n), &x, 0.0, 0.0, "x untouched without y");
}

/// A chunk of `m` rows at `pos0` through `qk_norm_rope_rows` + `attn_prefill`, against `m` decode
/// steps through `qk_norm_rope` + `attn_decode` on a cache that already holds `pos0` positions.
#[test]
fn qk_rope_and_causal_attention_equal_decode_steps_with_gqa_and_a_prior_chunk() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(9);
    let (nh, kvh, hd, cap, eps) = (8usize, 2usize, 64usize, 256usize, 1e-5f32);
    let table = rope_table(1_000_000.0, hd, cap);
    let rope = gpu.upload_f32(&table);
    let qn: Vec<f32> = (0..hd).map(|_| 1.0 + rng.unit() * 0.3).collect();
    let kn: Vec<f32> = (0..hd).map(|_| 1.0 + rng.unit() * 0.3).collect();
    let (qnh, knh) = (gpu.upload_f32(&qn), gpu.upload_f32(&kn));
    for &(pos0, m) in &[(0usize, 1usize), (0, 37), (40, 23), (100, 130)] {
        // a cache holding pos0 earlier positions (random), the rest zero
        let mut kc0 = vec![0f32; kvh * cap * hd];
        let mut vc0 = vec![0f32; kvh * cap * hd];
        for h in 0..kvh {
            for p in 0..pos0 {
                for j in 0..hd {
                    kc0[(h * cap + p) * hd + j] = rng.unit();
                    vc0[(h * cap + p) * hd + j] = rng.unit();
                }
            }
        }
        let q = vecr(&mut rng, m * nh * hd, 2.0);
        let k = vecr(&mut rng, m * kvh * hd, 2.0);
        let v = vecr(&mut rng, m * kvh * hd, 2.0);
        // rows path
        let (kc, vc) = (gpu.upload_f32(&kc0), gpu.upload_f32(&vc0));
        let (qh, kh, vh) = (gpu.upload_f32(&q), gpu.upload_f32(&k), gpu.upload_f32(&v));
        let qo = gpu.qk_norm_rope_rows(&qh, &kh, &vh, &qnh, &knh, &rope, cap, &kc, &vc, (m, pos0, cap), (nh, kvh, hd), eps);
        let att = gpu.attn_prefill(&qo, &kc, &vc, (m, pos0, cap), (nh, kvh, hd));
        let (got_q, got_att, got_k, got_v) = (read(&gpu, &qo, m * nh * hd), read(&gpu, &att, m * nh * hd), read(&gpu, &kc, kvh * cap * hd), read(&gpu, &vc, kvh * cap * hd));
        // per-token path on its own copy of the cache
        let (kd, vd) = (gpu.upload_f32(&kc0), gpu.upload_f32(&vc0));
        let mut tp = TokenParams::new(&gpu, 0, 1, cap);
        let (mut want_q, mut want_att) = (Vec::new(), Vec::new());
        for t in 0..m {
            let pos = pos0 + t;
            tp.set(&gpu, pos, pos + 1, cap);
            let qt = gpu.upload_f32(&q[t * nh * hd..(t + 1) * nh * hd]);
            let kt = gpu.upload_f32(&k[t * kvh * hd..(t + 1) * kvh * hd]);
            let vt = gpu.upload_f32(&v[t * kvh * hd..(t + 1) * kvh * hd]);
            let qo1 = gpu.qk_norm_rope(&qt, &kt, &vt, &qnh, &knh, &rope, cap, &kd, &vd, &tp, (nh, kvh, hd), eps);
            let a1 = gpu.attn_decode(&qo1, &kd, &vd, &tp, (nh, kvh, hd));
            want_q.extend(read(&gpu, &qo1, nh * hd));
            want_att.extend(read(&gpu, &a1, nh * hd));
        }
        let label = format!("pos0={pos0} m={m}");
        close(&got_q, &want_q, 0.0, 0.0, &format!("q {label}"));
        close(&got_k, &read(&gpu, &kd, kvh * cap * hd), 0.0, 0.0, &format!("k cache {label}"));
        close(&got_v, &read(&gpu, &vd, kvh * cap * hd), 0.0, 0.0, &format!("v cache {label}"));
        close(&got_att, &want_att, 0.0, 0.0, &format!("attention {label}"));
    }
}

#[test]
fn conv_prefill_equals_decode_steps_and_leaves_the_same_state() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(17);
    for &(e, l, m) in &[(256usize, 3usize, 1usize), (256, 3, 9), (2048, 3, 64), (128, 4, 5)] {
        let taps = vecr(&mut rng, l * e, 1.0);
        let tap_h = gpu.upload_f32(&taps);
        let s0 = vecr(&mut rng, (l - 1) * e, 1.0);
        let bcx = vecr(&mut rng, m * 3 * e, 1.0);
        let state = gpu.upload_f32(&s0);
        let y = gpu.conv_prefill(&gpu.upload_f32(&bcx), &tap_h, &state, m, e, l);
        let (got_y, got_s) = (read(&gpu, &y, m * e), read(&gpu, &state, (l - 1) * e));
        let sd = gpu.upload_f32(&s0);
        let mut want_y = Vec::new();
        for t in 0..m {
            let yt = gpu.conv_decode(&gpu.upload_f32(&bcx[t * 3 * e..(t + 1) * 3 * e]), &tap_h, &sd, e, l);
            want_y.extend(read(&gpu, &yt, e));
        }
        let label = format!("e={e} l={l} m={m}");
        close(&got_y, &want_y, 0.0, 0.0, &format!("conv y {label}"));
        close(&got_s, &read(&gpu, &sd, (l - 1) * e), 0.0, 0.0, &format!("conv state {label}"));
    }
}

#[test]
fn router_rows_equal_per_token_router_exactly_and_tag_the_token() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(33);
    let (m, n_exp, k, cols) = (19usize, 64usize, 4usize, 2048usize);
    let router = vecr(&mut rng, n_exp * cols, 0.05);
    let mut bias = vecr(&mut rng, n_exp, 0.01);
    bias[5] = bias[6]; // a tie candidate
    let h = vecr(&mut rng, m * cols, 1.0);
    let (rh, bh) = (gpu.upload_f32(&router), gpu.upload_f32(&bias));
    for seq in [true, false] {
        if !seq && !gpu.use_plane {
            continue;
        }
        let routes = gpu.upload_u32(vec![0xDEAD_BEEF; 2 * m * k]);
        let weights = gpu.upload_f32(&vec![-7.0; m * k]);
        gpu.router_topk_rows(&gpu.upload_f32(&h), &rh, &bh, &routes, &weights, (m, n_exp, k, cols), seq);
        let (gr, gw) = (gpu.read_u32(&routes, 2 * m * k).unwrap(), read(&gpu, &weights, m * k));
        for t in 0..m {
            let r1 = gpu.upload_u32(vec![0; 2 * k]);
            let w1 = gpu.upload_f32(&vec![0.0; k]);
            gpu.router_topk(&gpu.upload_f32(&h[t * cols..(t + 1) * cols]), &rh, &bh, &r1, &w1, (n_exp, k, cols, 0, k), seq);
            let (er, ew) = (gpu.read_u32(&r1, 2 * k).unwrap(), read(&gpu, &w1, k));
            for i in 0..k {
                assert_eq!(gr[2 * (t * k + i)], er[2 * i], "seq={seq} token {t} rank {i}: expert");
                assert_eq!(gr[2 * (t * k + i) + 1], t as u32, "seq={seq} token {t} rank {i}: token tag");
            }
            close(&gw[t * k..(t + 1) * k], &ew, 0.0, 0.0, &format!("seq={seq} token {t} weights"));
        }
    }
}

#[test]
fn weighted_sum_rows_equals_per_token_sum() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(3);
    let (m, k, n) = (11usize, 4usize, 2048usize);
    let outs = vecr(&mut rng, m * k * n, 10.0);
    let mut routes = vec![0u32; 2 * m * k];
    for t in 0..m {
        let mut used = Vec::new();
        for i in 0..k {
            let mut e = (rng.next() % 64) as u32;
            while used.contains(&e) {
                e = (e + 1) % 64;
            }
            used.push(e);
            routes[2 * (t * k + i)] = e;
            routes[2 * (t * k + i) + 1] = t as u32;
        }
    }
    let weights: Vec<f32> = (0..m * k).map(|_| 0.1 + rng.unit().abs()).collect();
    let y = gpu.weighted_sum_rows(&gpu.upload_f32(&outs), &gpu.upload_f32(&weights), &gpu.upload_u32(routes.clone()), (m, k, n));
    let got = read(&gpu, &y, m * n);
    let mut want = Vec::new();
    for t in 0..m {
        let yt = gpu.weighted_sum(
            &gpu.upload_f32(&outs[t * k * n..(t + 1) * k * n]),
            &gpu.upload_f32(&weights[t * k..(t + 1) * k]),
            &gpu.upload_u32(routes[2 * t * k..2 * (t + 1) * k].to_vec()),
            (k, n, 0, k),
        );
        want.extend(read(&gpu, &yt, n));
    }
    close(&got, &want, 0.0, 0.0, "weighted sum rows");
}

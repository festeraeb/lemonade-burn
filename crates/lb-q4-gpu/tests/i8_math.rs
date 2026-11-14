//! K1: the staged-int8 math (`Q4Math::I8`, `LB_DD_I8=1`).
//!
//! 1. `word_math_is_exact_on_the_extremes` (host only): the packed-multiply trick of the K1 notes in
//!    lib.rs, emulated with the same 32-bit wrapping words, against the exact integer sum. It
//!    covers the worst cases of the fold bound (all nibbles 15 against all quants +-127, with
//!    each sign mix) and 400k random blocks, and checks the bound itself.
//! 2. `device_matvec_matches_cpu_i8_scalar` (card): the kernel against `lb_gguf::kernels_i8`'s
//!    scalar result (near exact: only the f32 sum order differs) and against the f32 kernel
//!    (< 1 % of the output norm), on the model's shapes.
//! 3. `routed_expert_layer_stays_close_to_f32` (card): the routed gate/up/down launches, which
//!    stage a different input row per slot.
mod common;
use common::Rng;
use lb_gguf::kernels_i8::{dot_q4_0_i8_scalar, quantize_act_q8};
use lb_gguf::types::quantize_q4_0;
use lb_gguf::{GgmlType, QuantMat};
use lb_q4_gpu::{Gpu, GpuExpertLayer, GpuQ4Mat, Q4Math};

/// The kernel's per-block word math on the host: `n[v]` is the nibble of value v (0..15), `q[v]`
/// the quant (-127..127), v in 0..32. Mirrors `q4_rows_acc_i8` step by step; returns
/// `sum_v n[v] * q[v]` (the `- 8 * sumq` offset is applied by the caller).
fn packed_block(n: &[u32; 32], q: &[i32; 32]) -> i32 {
    // weight word k: nibble j holds value 4j + k (j in 0..8)
    let mut w = [0u32; 4];
    for k in 0..4 {
        for j in 0..8 {
            w[k] |= n[4 * j + k] << (4 * j);
        }
    }
    // staged word [j][k] = q[4j+k+16] + (q[4j+k] << 16), as the stager builds it
    let xw = |j: usize, k: usize| (q[4 * j + k + 16].wrapping_add(q[4 * j + k] << 16)) as u32;
    let mut tot = 0i32;
    for h in 0..2 {
        let mut s = 0u32;
        let (mut stray, mut wanted) = (0i64, 0i64);
        for jj in 0..2 {
            let j = 2 * h + jj;
            for k in 0..4 {
                let wm = (w[k] >> (4 * j)) & 0x000f_000f;
                s = s.wrapping_add(wm.wrapping_mul(xw(j, k)));
                // the bound being proved, tracked in exact arithmetic
                let (n1, n2) = (n[4 * j + k] as i64, n[4 * j + k + 16] as i64);
                stray += n1 * q[4 * j + k + 16] as i64;
                wanted += n1 * q[4 * j + k] as i64 + n2 * q[4 * j + k + 16] as i64;
            }
        }
        assert!(stray.abs() <= 15 * 127 * 8 && stray.abs() < 1 << 15, "stray {stray}");
        assert!(wanted.abs() <= 2 * 15 * 127 * 8 && wanted.abs() < 1 << 15, "wanted {wanted}");
        let got = (s.wrapping_add(0x8000) as i32) >> 16;
        assert_eq!(got as i64, wanted, "fold {h}: packed {got} vs exact {wanted}");
        tot += got;
    }
    tot
}

fn exact_block(n: &[u32; 32], q: &[i32; 32]) -> i32 {
    (0..32).map(|v| n[v] as i32 * q[v]).sum()
}

#[test]
fn word_math_is_exact_on_the_extremes() {
    // worst cases: every nibble 15 (or 0, or alternating) against quants at +-127 in every sign mix
    for nib in [[15u32; 32], [0u32; 32], std::array::from_fn(|i| if i % 2 == 0 { 15 } else { 0 }), std::array::from_fn(|i| if i < 16 { 15 } else { 1 })] {
        for qsel in 0..16u32 {
            let q: [i32; 32] = std::array::from_fn(|v| {
                // sign pattern chosen per (quarter of the block) by the bits of qsel
                if (qsel >> (v / 8)) & 1 == 1 { -127 } else { 127 }
            });
            assert_eq!(packed_block(&nib, &q), exact_block(&nib, &q));
        }
    }
    // the two halves of a pair at opposite extremes (stray large, wanted cancelling) and vice versa
    let n = [15u32; 32];
    let q: [i32; 32] = std::array::from_fn(|v| if v < 16 { 127 } else { -127 });
    assert_eq!(packed_block(&n, &q), exact_block(&n, &q));
    let q: [i32; 32] = std::array::from_fn(|v| if v < 16 { -127 } else { 127 });
    assert_eq!(packed_block(&n, &q), exact_block(&n, &q));
    // random blocks, with a bias towards the extremes
    let mut rng = Rng(7);
    for _ in 0..400_000 {
        let n: [u32; 32] = std::array::from_fn(|_| match rng.next() % 4 { 0 => 15, 1 => 0, _ => (rng.next() % 16) as u32 });
        let q: [i32; 32] = std::array::from_fn(|_| match rng.next() % 4 { 0 => 127, 1 => -127, _ => (rng.next() % 255) as i32 - 127 });
        assert_eq!(packed_block(&n, &q), exact_block(&n, &q));
    }
    // and one fold longer than the kernel's would NOT be safe: 16 words put |wanted| up to 60960
    assert!(2 * 15 * 127 * 16 >= 1 << 15, "the fold interval is 8 words because 16 would overflow 16 bits");
}

fn gpu() -> Option<Gpu> {
    if std::env::var_os("LB_WGPU_DEVICE").is_none() && std::env::var_os("LB_CUDA_DEVICE").is_none() {
        eprintln!("skipped: no device named (set LB_WGPU_DEVICE or LB_CUDA_DEVICE)");
        return None;
    }
    let g = Gpu::open().unwrap_or_else(|e| panic!("no GPU: {e}"));
    if !g.use_plane {
        eprintln!("skipped: the rows kernels need plane ops and this device has none");
        return None;
    }
    Some(g)
}

fn q4(rng: &mut Rng, rows: usize, cols: usize) -> Vec<u8> {
    let wf: Vec<f32> = (0..rows * cols).map(|_| rng.unit()).collect();
    wf.chunks_exact(cols).flat_map(|r| quantize_q4_0(r)).collect()
}

fn rel_err(got: &[f32], want: &[f32]) -> f64 {
    let num: f64 = got.iter().zip(want).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
    let den: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

#[test]
fn device_matvec_matches_cpu_i8_scalar() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(91);
    for &(rows, cols) in &[(1536usize, 2048usize), (2048, 1536), (6144, 2048), (512, 2048), (2048, 4096), (11776, 2048), (2048, 2048), (36, 64), (8, 32)] {
        let w = q4(&mut rng, rows, cols);
        let mat = GpuQ4Mat::upload(&gpu, QuantMat::new(GgmlType::Q4_0, rows, cols, &w)).unwrap();
        // gate input: uniform activations (the spec's < 1 % of norm); `outlier` input (every 97th value
        // 20x) is the harsh case: blocks holding one are quantised 20x coarser, reported and held to
        // the docs' dp4a bound (3e-2), not to 1 %
        let plain: Vec<f32> = (0..cols).map(|_| rng.unit()).collect();
        let outlier: Vec<f32> = (0..cols).map(|i| rng.unit() * if i % 97 == 0 { 20.0 } else { 1.0 }).collect();
        gpu.set_q4_math(Q4Math::F32);
        let want_o = mat.matvec(&gpu, &outlier, 1).unwrap();
        gpu.set_q4_math(Q4Math::I8);
        let got_o = mat.matvec(&gpu, &outlier, 1).unwrap();
        let e_o = rel_err(&got_o, &want_o);
        eprintln!("{rows}x{cols}: outlier input i8 vs f32 rel {e_o:.2e}");
        assert!(e_o < 3e-2, "{rows}x{cols}: outlier input {e_o:e}");
        let x = plain;
        gpu.set_q4_math(Q4Math::F32);
        let f32out = mat.matvec(&gpu, &x, 1).unwrap();
        gpu.set_q4_math(Q4Math::I8);
        let got = mat.matvec(&gpu, &x, 1).unwrap();
        gpu.set_q4_math(Q4Math::F32);
        let act = quantize_act_q8(&x);
        let rb = cols / 32 * 18;
        let cpu: Vec<f32> = w.chunks_exact(rb).map(|row| dot_q4_0_i8_scalar(row, &act)).collect();
        let scale: f64 = cpu.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt() / (rows as f64).sqrt();
        let worst = got.iter().zip(&cpu).map(|(a, b)| ((a - b) as f64).abs()).fold(0f64, f64::max);
        let (e_cpu, e_f32) = (rel_err(&got, &cpu), rel_err(&got, &f32out));
        eprintln!("{rows}x{cols}: vs cpu i8 scalar rel {e_cpu:.2e} (max abs {worst:.2e}, rms out {scale:.2e}); vs f32 kernel rel {e_f32:.2e}");
        assert!(e_cpu < 1e-5, "{rows}x{cols}: device vs CPU int8 scalar {e_cpu:e}");
        assert!(e_f32 < 1e-2, "{rows}x{cols}: device int8 vs f32 kernel {e_f32:e}");
    }
}

#[test]
fn routed_expert_layer_stays_close_to_f32() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(5);
    let (n_ff, n_embd, n_experts) = (1536usize, 2048usize, 8usize);
    let raw: Vec<_> = (0..n_experts).map(|_| (q4(&mut rng, n_ff, n_embd), q4(&mut rng, n_ff, n_embd), q4(&mut rng, n_embd, n_ff))).collect();
    let layer = GpuExpertLayer::upload(&gpu, raw.iter().map(|(g, u, d)| (QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, g), QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, u), QuantMat::new(GgmlType::Q4_0, n_embd, n_ff, d)))).unwrap();
    for (nt, routes) in [(1usize, vec![0u32, 0, 3, 0, 5, 0, 7, 0]), (1, vec![2, 0, 2, 0, 6, 0, 1, 0])] {
        let x: Vec<f32> = (0..nt * n_embd).map(|_| rng.unit() * 3.0).collect();
        let xh = gpu.upload_f32(&x);
        let mut outs = Vec::new();
        for m in [Q4Math::F32, Q4Math::I8] {
            gpu.set_q4_math(m);
            let (o, n) = layer.launch(&gpu, &xh, nt, &routes).unwrap();
            outs.push(gpu.read_all(vec![(o, n)]).unwrap()[0].clone());
        }
        gpu.set_q4_math(Q4Math::F32);
        for slot in 0..routes.len() / 2 {
            let (a, b) = (&outs[1][slot * n_embd..(slot + 1) * n_embd], &outs[0][slot * n_embd..(slot + 1) * n_embd]);
            let e = rel_err(a, b);
            eprintln!("routed slot {slot}: i8 vs f32 rel {e:.2e}");
            assert!(e < 1e-2, "routed slot {slot}: {e:e}");
        }
    }
}

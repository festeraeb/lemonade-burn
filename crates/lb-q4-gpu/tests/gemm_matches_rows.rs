//! The tiled Q4_0 GEMM (`gemm.rs`) against the rows kernels it replaces in prefill.
//!
//! Same weights, same inputs: dense `launch_gemm` vs `launch`, fused gate/up/SiLU vs
//! `launch_gate_up_silu`, and the grouped expert path (`launch_grouped` over `group_slots`) vs the
//! stacked layer's routed `launch_device`. The summation order inside a Q4_0 block differs, so the
//! comparison is to f32 rounding, scaled by the size of the dot (`cols`).
//!
//! Opens a device only when one is named (`LB_WGPU_DEVICE` or `LB_CUDA_DEVICE`).
mod common;
use common::Rng;
use lb_gguf::types::quantize_q4_0;
use lb_gguf::{GgmlType, QuantMat};
use lb_q4_gpu::{Gpu, GpuExpertLayer, GpuQ4Mat};

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

fn q4(rng: &mut Rng, rows: usize, cols: usize) -> Vec<u8> {
    let wf: Vec<f32> = (0..rows * cols).map(|_| rng.unit()).collect();
    wf.chunks_exact(cols).flat_map(|r| quantize_q4_0(r)).collect()
}

/// f32 tiles: rounding only. Tensor cores (`Gpu::tensor_cores`): activations and dequantized
/// weights are f16, so the bound is relative to the output's scale: max |d| <= 4e-3 * max |want|.
fn close_on(gpu: &Gpu, got: &[f32], want: &[f32], cols: usize, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let scale = want.iter().fold(0f32, |a, w| a.max(w.abs()));
    let mut worst = 0f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = if gpu.tensor_cores || gpu.fp16_pipe { 4e-3 * scale.max(1e-3) } else { 2e-5 * (1.0 + w.abs()) + 2e-6 * cols as f32 };
        assert!((g - w).abs() <= tol, "{label} [{i}]: gemm {g} rows {w} (tensor cores: {})", gpu.tensor_cores);
        worst = worst.max((g - w).abs());
    }
    eprintln!("{label} ({}): max |gemm - rows| = {worst:e} of max |out| {scale:e}", if gpu.tensor_cores { "tensor cores" } else if gpu.fp16_pipe { "fp16 pipe" } else { "f32 tiles" });
}

#[test]
fn dense_gemm_and_fused_gate_up_match_the_rows_kernels() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(41);
    // rows not a multiple of 64, m below, at and past one tile
    for &(rows, cols, m) in &[(64usize, 128usize, 1usize), (200, 256, 37), (2048, 2048, 64), (6144, 2048, 130), (96, 4096, 513)] {
        let wb = q4(&mut rng, rows, cols);
        let w = GpuQ4Mat::upload(&gpu, QuantMat::new(GgmlType::Q4_0, rows, cols, &wb)).unwrap();
        let x: Vec<f32> = (0..m * cols).map(|_| rng.unit() * 2.0).collect();
        let xh = gpu.upload_f32(&x);
        let want = read(&gpu, &w.launch(&gpu, &xh, m), m * rows);
        let got = read(&gpu, &w.launch_gemm(&gpu, &xh, m), m * rows);
        close_on(&gpu, &got, &want, cols, &format!("dense {rows}x{cols} m={m}"));
        let ub = q4(&mut rng, rows, cols);
        let u = GpuQ4Mat::upload(&gpu, QuantMat::new(GgmlType::Q4_0, rows, cols, &ub)).unwrap();
        let want = read(&gpu, &GpuQ4Mat::launch_gate_up_silu(&gpu, &w, &u, &xh, m), m * rows);
        let got = read(&gpu, &GpuQ4Mat::launch_gemm_gate_up(&gpu, &w, &u, &xh, m), m * rows);
        close_on(&gpu, &got, &want, cols, &format!("gate/up {rows}x{cols} m={m}"));
    }
}

#[test]
fn grouped_experts_match_the_routed_stacked_layer() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(5);
    // 64 x 128 experts (views land on 256-byte boundaries), 150 tokens x top-4 = 600 slots, so
    // popular experts span several tiles
    let (n_exp, n_ff, n_embd, k, m) = (8usize, 64usize, 128usize, 4usize, 150usize);
    let (g, u, d) = (q4(&mut rng, n_exp * n_ff, n_embd), q4(&mut rng, n_exp * n_ff, n_embd), q4(&mut rng, n_exp * n_embd, n_ff));
    let layer = GpuExpertLayer::upload_stacked(
        &gpu,
        QuantMat::new(GgmlType::Q4_0, n_exp * n_ff, n_embd, &g),
        QuantMat::new(GgmlType::Q4_0, n_exp * n_ff, n_embd, &u),
        QuantMat::new(GgmlType::Q4_0, n_exp * n_embd, n_ff, &d),
        n_exp,
    )
    .expect("stacked upload");
    let x: Vec<f32> = (0..m * n_embd).map(|_| rng.unit() * 2.0).collect();
    let xh = gpu.upload_f32(&x);
    // routes as the router writes them: token t's k distinct experts at slots t k .., skewed so
    // expert 0 and 1 are hot
    let mut routes = Vec::with_capacity(2 * m * k);
    for t in 0..m {
        let mut used: Vec<u32> = Vec::new();
        for _ in 0..k {
            let mut e = ((rng.next() % 16) as u32).min(n_exp as u32 - 1) % (n_exp as u32);
            if rng.next() % 3 == 0 {
                e = (rng.next() % 2) as u32;
            }
            while used.contains(&e) {
                e = (e + 1) % n_exp as u32;
            }
            used.push(e);
            routes.extend_from_slice(&[e, t as u32]);
        }
    }
    let slots = m * k;
    let rh = gpu.upload_u32(routes.clone());
    let (want_h, n) = layer.launch_device(&gpu, &xh, m, &rh, slots, 0, slots);
    let want = read(&gpu, &want_h, n);
    let (perm, tiles) = lb_q4_gpu::gemm::group_slots(&routes, n_exp);
    let n_tiles = tiles.len() / 3;
    assert!(n_tiles > n_exp / 2, "several tiles: {n_tiles}");
    let (got_h, n2) = layer.launch_grouped(&gpu, &xh, m, &rh, slots, &gpu.upload_u32(perm), &gpu.upload_u32(tiles), n_tiles);
    assert_eq!(n, n2);
    close_on(&gpu, &read(&gpu, &got_h, n2), &want, n_ff.max(n_embd), "grouped experts (gate/up/silu -> down) vs routed");
}

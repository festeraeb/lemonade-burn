//! D2: the R-rows-per-plane kernels (`LB_Q4_ROWS` 4 / 8, optionally `LB_Q4_XSHARED`) against the
//! one-row kernels they replace, on f32 bits. Each row is the same blocks in the same order per
//! lane; only the plane reduce is batched, so the outputs should be identical, not merely close.
//! Where a backend's vector `plane_sum` rounds differently from the scalar one this would show
//! as a bit difference, and the test reports it with the worst |diff| before failing.
//!
//! Switches the kernels live (`Gpu::set_q4_rows`), so both arms run on the same upload.
//! Runs only when a device is named (`LB_WGPU_DEVICE` / `LB_CUDA_DEVICE`).
mod common;
use common::Rng;
use lb_gguf::types::quantize_q4_0;
use lb_gguf::{GgmlType, QuantMat};
use lb_q4_gpu::{Gpu, GpuExpert, GpuExpertLayer, GpuQ4Mat};

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

fn same(got: &[f32], want: &[f32], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let diff = got.iter().zip(want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    let worst = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert_eq!(diff, 0, "{label}: {diff} of {} outputs differ in bits (max |diff| {worst:e})", got.len());
}

/// Every configuration the knobs allow: (rows per plane, x in shared memory).
const ARMS: [(usize, bool); 4] = [(4, false), (8, false), (4, true), (8, true)];

#[test]
fn matvec_rows_equal_one_row_on_the_model_shapes() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(41);
    // the 24B's dense and expert shapes, plus rows not a multiple of 8 (falls back to one row
    // for R = 8) and a wide matrix past the shared-x limit
    for &(rows, cols) in &[(1536usize, 2048usize), (2048, 1536), (6144, 2048), (512, 2048), (2048, 11776), (36, 64), (4, 32)] {
        let w = q4(&mut rng, rows, cols);
        let mat = GpuQ4Mat::upload(&gpu, QuantMat::new(GgmlType::Q4_0, rows, cols, &w)).unwrap();
        for m in [1usize, 3] {
            let x: Vec<f32> = (0..m * cols).map(|_| rng.unit() * 2.0).collect();
            gpu.set_q4_rows(1);
            gpu.set_q4_xshared(false);
            let want = mat.matvec(&gpu, &x, m).unwrap();
            for (r, xs) in ARMS {
                gpu.set_q4_rows(r);
                gpu.set_q4_xshared(xs);
                same(&mat.matvec(&gpu, &x, m).unwrap(), &want, &format!("matvec {rows}x{cols} m={m} R={r} xshared={xs}"));
            }
        }
    }
    gpu.set_q4_rows(1);
    gpu.set_q4_xshared(false);
}

#[test]
fn expert_ffn_rows_equal_one_row() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(43);
    let (n_ff, n_embd) = (1536usize, 2048usize);
    let (g, u, d) = (q4(&mut rng, n_ff, n_embd), q4(&mut rng, n_ff, n_embd), q4(&mut rng, n_embd, n_ff));
    let ex = GpuExpert::upload(&gpu, QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, &g), QuantMat::new(GgmlType::Q4_0, n_ff, n_embd, &u), QuantMat::new(GgmlType::Q4_0, n_embd, n_ff, &d)).unwrap();
    for m in [1usize, 2] {
        let x: Vec<f32> = (0..m * n_embd).map(|_| rng.unit()).collect();
        gpu.set_q4_rows(1);
        gpu.set_q4_xshared(false);
        let want = ex.ffn(&gpu, &x, m).unwrap();
        for (r, xs) in ARMS {
            gpu.set_q4_rows(r);
            gpu.set_q4_xshared(xs);
            same(&ex.ffn(&gpu, &x, m).unwrap(), &want, &format!("expert m={m} R={r} xshared={xs}"));
        }
    }
    gpu.set_q4_rows(1);
    gpu.set_q4_xshared(false);
}

#[test]
fn routed_layer_rows_equal_one_row() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(47);
    let (n_exp, n_ff, n_embd) = (6usize, 256usize, 512usize);
    let (g, u, d) = (q4(&mut rng, n_exp * n_ff, n_embd), q4(&mut rng, n_exp * n_ff, n_embd), q4(&mut rng, n_exp * n_embd, n_ff));
    let layer = GpuExpertLayer::upload_stacked(
        &gpu,
        QuantMat::new(GgmlType::Q4_0, n_exp * n_ff, n_embd, &g),
        QuantMat::new(GgmlType::Q4_0, n_exp * n_ff, n_embd, &u),
        QuantMat::new(GgmlType::Q4_0, n_exp * n_embd, n_ff, &d),
        n_exp,
    )
    .unwrap();
    // two tokens, four slots (an expert repeated), the last expert in the buffer included
    let n_tokens = 2usize;
    let routes = [5u32, 0, 1, 1, 5, 1, 0, 0];
    let x: Vec<f32> = (0..n_tokens * n_embd).map(|_| rng.unit()).collect();
    let xh = gpu.upload_f32(&x);
    let run = |gpu: &Gpu| {
        let (o, n) = layer.launch(gpu, &xh, n_tokens, &routes).unwrap();
        gpu.read_all(vec![(o, n)]).unwrap().remove(0)
    };
    gpu.set_q4_rows(1);
    gpu.set_q4_xshared(false);
    let want = run(&gpu);
    // decode shape: one token, device routes at a slot base, x in shared memory allowed
    let x1h = gpu.upload_f32(&x[..n_embd]);
    let rh = gpu.upload_u32(vec![9, 9, 9, 9, 3, 0, 5, 0, 0, 0]);
    let run1 = |gpu: &Gpu| {
        let (o, n) = layer.launch_device(gpu, &x1h, 1, &rh, 5, 2, 3);
        gpu.read_all(vec![(o, n)]).unwrap().remove(0)
    };
    let want1 = run1(&gpu);
    for (r, xs) in ARMS {
        gpu.set_q4_rows(r);
        gpu.set_q4_xshared(xs);
        same(&run(&gpu), &want, &format!("routed layer, 2 tokens, R={r} xshared={xs}"));
        same(&run1(&gpu), &want1, &format!("routed layer, device routes, R={r} xshared={xs}"));
    }
    gpu.set_q4_rows(1);
    gpu.set_q4_xshared(false);
}

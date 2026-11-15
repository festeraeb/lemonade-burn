//! Q5_K device kernels against the CPU reference (`lb_gguf`, bit-identical to llama.cpp on the real file):
//! dense matvec, and the routed expert layer (Q5_K gate/up + SiLU, Q6_K down) against a host composition of the same maths.
mod common;
use common::{assert_close, Rng};
use lb_gguf::{matvec_f32, GgmlType, QuantMat};
use lb_q4_gpu::{q5_k_matvec_gpu, Gpu, GpuKExpertLayer};

fn random_rows(rng: &mut Rng, ty: GgmlType, rows: usize, cols: usize) -> Vec<u8> {
    let rb = ty.row_bytes(cols).unwrap();
    let bs = ty.block().unwrap().1;
    let mut v: Vec<u8> = (0..rows * rb).map(|_| rng.byte()).collect();
    for blk in v.chunks_exact_mut(bs) {
        // keep d (and for Q5_K dmin) sane f16 values
        let d = half::f16::from_f32(0.001 + 0.02 * (rng.unit() * 0.5 + 0.5)).to_le_bytes();
        if ty == GgmlType::Q5K {
            blk[0..2].copy_from_slice(&d);
            blk[2..4].copy_from_slice(&half::f16::from_f32(0.001 + 0.02 * (rng.unit() * 0.5 + 0.5)).to_le_bytes());
        } else {
            blk[bs - 2..].copy_from_slice(&d);
        }
    }
    v
}

fn check_dense(rows: usize, cols: usize, m: usize, seed: u64) {
    let mut rng = Rng(seed);
    let bytes = random_rows(&mut rng, GgmlType::Q5K, rows, cols);
    let w = QuantMat::new(GgmlType::Q5K, rows, cols, &bytes);
    let x: Vec<f32> = (0..m * cols).map(|_| rng.unit() * 4.0).collect();
    let mut want = vec![0f32; m * rows];
    matvec_f32(w, &x, m, &mut want);
    let got = q5_k_matvec_gpu(w, &x, m).unwrap_or_else(|e| panic!("GPU call failed: {e}"));
    assert_close(&got, &want, cols, rows, &format!("q5_k ({rows},{cols},{m})"));
}

#[test]
fn q5_k_one_superblock_rows() {
    check_dense(8, 256, 1, 1);
}
#[test]
fn q5_k_odd_rows_three_tokens() {
    check_dense(37, 512, 3, 2);
}
#[test]
fn q5_k_qwen_attn_gate_shape() {
    check_dense(4096, 2048, 1, 3);
}

#[test]
fn q5_k_real_rows_if_present() {
    let path = std::path::Path::new("/mnt/raid0/models/Qwen3.6-35B-A3B-Q5_K_M.gguf");
    if !path.exists() {
        eprintln!("skip: no model file");
        return;
    }
    let g = lb_gguf::Gguf::open(path).unwrap();
    for name in ["blk.0.attn_gate.weight", "token_embd.weight", "blk.20.ffn_gate_exps.weight"] {
        let t = g.tensor(name).unwrap();
        assert_eq!(t.ty, GgmlType::Q5K);
        let cols = t.dims[0];
        let rows = 3000usize;
        let bytes = &g.tensor_bytes(t)[..rows * t.ty.row_bytes(cols).unwrap()];
        let w = QuantMat::new(GgmlType::Q5K, rows, cols, bytes);
        let mut rng = Rng(9);
        let x: Vec<f32> = (0..2 * cols).map(|_| rng.unit()).collect();
        let mut want = vec![0f32; 2 * rows];
        matvec_f32(w, &x, 2, &mut want);
        let got = q5_k_matvec_gpu(w, &x, 2).unwrap();
        assert_close(&got, &want, cols, rows, &format!("q5_k real {name}"));
    }
}

fn silu(g: f32) -> f32 {
    g / (1.0 + (-g).exp())
}

/// Host composition of one routed expert FFN from dequantised rows (f64 accumulation).
fn host_expert(gate: QuantMat<'_>, up: QuantMat<'_>, down: QuantMat<'_>, ff: usize, embd: usize, e: usize, x: &[f32]) -> Vec<f32> {
    let dot = |m: &QuantMat<'_>, r: usize, v: &[f32]| -> f32 {
        let mut b = vec![0f32; m.cols];
        m.dequant_row(r, &mut b);
        b.iter().zip(v).map(|(a, c)| *a as f64 * *c as f64).sum::<f64>() as f32
    };
    let h: Vec<f32> = (0..ff).map(|r| silu(dot(&gate, e * ff + r, x)) * dot(&up, e * ff + r, x)).collect();
    (0..embd).map(|r| dot(&down, e * embd + r, &h)).collect()
}

fn check_experts(n_expert: usize, ff: usize, embd: usize, tokens: usize, routes: &[(u32, u32)], seed: u64, down_ty: GgmlType) {
    let mut rng = Rng(seed);
    let gb = random_rows(&mut rng, GgmlType::Q5K, n_expert * ff, embd);
    let ub = random_rows(&mut rng, GgmlType::Q5K, n_expert * ff, embd);
    let db = random_rows(&mut rng, down_ty, n_expert * embd, ff);
    let (gate, up, down) = (QuantMat::new(GgmlType::Q5K, n_expert * ff, embd, &gb), QuantMat::new(GgmlType::Q5K, n_expert * ff, embd, &ub), QuantMat::new(down_ty, n_expert * embd, ff, &db));
    let x: Vec<f32> = (0..tokens * embd).map(|_| rng.unit()).collect();
    let gpu = Gpu::open().unwrap();
    let layer = GpuKExpertLayer::upload(&gpu, n_expert, gate, up, down).unwrap();
    let got = layer.forward(&gpu, &x, routes).unwrap();
    let mut want = Vec::new();
    for &(e, t) in routes {
        want.extend(host_expert(gate, up, down, ff, embd, e as usize, &x[t as usize * embd..(t as usize + 1) * embd]));
    }
    assert_scaled(&got, &want, &format!("routed experts ({n_expert},{ff},{embd})"));
}

/// Two chained reductions (gate/up then down) with SiLU in between: compare to the largest output magnitude, not elementwise.
fn assert_scaled(got: &[f32], want: &[f32], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let worst = got.iter().zip(want).fold(0f32, |m, (g, w)| m.max((g - w).abs()));
    eprintln!("{label}: max |gpu-cpu| = {worst:e} on outputs up to {scale:e} (ratio {:e})", worst / scale);
    assert!(worst <= 2e-5 * scale, "{label}: worst {worst} vs scale {scale}");
}

#[test]
fn routed_experts_small() {
    check_experts(4, 256, 256, 2, &[(0, 0), (3, 0), (2, 1), (1, 1), (3, 1)], 11, GgmlType::Q6K);
    check_experts(4, 256, 256, 2, &[(0, 0), (3, 0), (2, 1), (1, 1), (3, 1)], 13, GgmlType::Q5K);
}

#[test]
fn routed_experts_qwen_shape() {
    // 6 experts of the real shape (ff 512, embd 2048); routes include the first and last expert
    check_experts(6, 512, 2048, 1, &[(0, 0), (5, 0), (2, 0), (3, 0)], 12, GgmlType::Q6K);
    check_experts(6, 512, 2048, 1, &[(0, 0), (5, 0), (2, 0), (3, 0)], 14, GgmlType::Q5K);
}

#[test]
fn routed_experts_real_layers_if_present() {
    for il in [20usize, 21, 4, 7, 39] {
        real_layer(il);
    }
}

fn real_layer(il: usize) {
    let path = std::path::Path::new("/mnt/raid0/models/Qwen3.6-35B-A3B-Q5_K_M.gguf");
    if !path.exists() {
        eprintln!("skip: no model file");
        return;
    }
    let g = lb_gguf::Gguf::open(path).unwrap();
    let (tg, tu, td) = (g.tensor(&format!("blk.{il}.ffn_gate_exps.weight")).unwrap(), g.tensor(&format!("blk.{il}.ffn_up_exps.weight")).unwrap(), g.tensor(&format!("blk.{il}.ffn_down_exps.weight")).unwrap());
    // first 64 experts of the 256: one wgpu buffer is limited to 256 MiB (the CUDA runtime has no such limit)
    let (embd, ff, ne) = (tg.dims[0], tg.dims[1], 64usize);
    // four experts' worth of rows, picked by routes: keep the whole layer view, the kernel indexes by expert
    let (gate, up, down) = (
        QuantMat::new(GgmlType::Q5K, ne * ff, embd, &g.tensor_bytes(tg)[..ne * ff * GgmlType::Q5K.row_bytes(embd).unwrap()]),
        QuantMat::new(GgmlType::Q5K, ne * ff, embd, &g.tensor_bytes(tu)[..ne * ff * GgmlType::Q5K.row_bytes(embd).unwrap()]),
        QuantMat::new(td.ty, ne * embd, ff, &g.tensor_bytes(td)[..ne * embd * td.ty.row_bytes(ff).unwrap()]),
    );
    let mut rng = Rng(5);
    let x: Vec<f32> = (0..embd).map(|_| rng.unit()).collect();
    let routes = [(0u32, 0u32), (63, 0), (17, 0), (30, 0), (45, 0), (52, 0), (3, 0), (9, 0)];
    let gpu = Gpu::open().unwrap();
    let layer = GpuKExpertLayer::upload(&gpu, ne, gate, up, down).unwrap();
    let got = layer.forward(&gpu, &x, &routes).unwrap();
    let mut want = Vec::new();
    for &(e, _) in &routes {
        want.extend(host_expert(gate, up, down, ff, embd, e as usize, &x));
    }
    assert_scaled(&got, &want, &format!("routed experts, real layer {il} (down {}), 8 routed of the first 64", td.ty.name()));
}

/// The Q6_K and Q8_0 kernels (written for LFM2.5) on the real tensors of this file.
#[test]
fn q6_k_and_q8_0_real_qwen_rows_if_present() {
    let path = std::path::Path::new("/mnt/raid0/models/Qwen3.6-35B-A3B-Q5_K_M.gguf");
    if !path.exists() {
        eprintln!("skip: no model file");
        return;
    }
    let g = lb_gguf::Gguf::open(path).unwrap();
    for (name, ty) in [("output.weight", GgmlType::Q6K), ("blk.3.attn_q.weight", GgmlType::Q6K), ("blk.4.ffn_down_exps.weight", GgmlType::Q6K), ("blk.0.attn_qkv.weight", GgmlType::Q8_0), ("blk.0.ffn_gate_shexp.weight", GgmlType::Q8_0), ("blk.0.ssm_out.weight", GgmlType::Q8_0)] {
        let t = g.tensor(name).unwrap();
        assert_eq!(t.ty, ty, "{name}");
        let cols = t.dims[0];
        let rows = 3000usize.min(t.dims[1]);
        let bytes = &g.tensor_bytes(t)[..rows * t.ty.row_bytes(cols).unwrap()];
        let w = QuantMat::new(ty, rows, cols, bytes);
        let mut rng = Rng(21);
        let x: Vec<f32> = (0..2 * cols).map(|_| rng.unit()).collect();
        let mut want = vec![0f32; 2 * rows];
        matvec_f32(w, &x, 2, &mut want);
        let got = if ty == GgmlType::Q6K { lb_q4_gpu::q6_k_matvec_gpu(w, &x, 2) } else { lb_q4_gpu::q8_0_matvec_gpu(w, &x, 2) }.unwrap();
        assert_close(&got, &want, cols, rows, &format!("{} real {name}", ty.name()));
    }
}

//! Forge seed: the GPU Q6_K matvec must match `lb_gguf::matvec_f32` (whose Q6_K dequant
//! is checked against gguf-py on the real file in lb-gguf's tests). Ignored until the
//! kernel exists; the forge check runs `--include-ignored` on c3 (LB_WGPU_DEVICE=discrete:1).
mod common;
use common::{assert_close, Rng};
use lb_gguf::{matvec_f32, GgmlType, QuantMat};
use lb_q4_gpu::q6_k_matvec_gpu;

/// Any 210-byte pattern is a valid Q6_K super-block; keep `d` a sane f16 (0.001..0.03).
fn random_q6k_rows(rng: &mut Rng, rows: usize, cols: usize) -> Vec<u8> {
    let rb = GgmlType::Q6K.row_bytes(cols).unwrap();
    let mut v: Vec<u8> = (0..rows * rb).map(|_| rng.byte()).collect();
    for blk in v.chunks_exact_mut(210) {
        let d = half::f16::from_f32(0.001 + 0.03 * (rng.unit() * 0.5 + 0.5));
        blk[208..210].copy_from_slice(&d.to_le_bytes());
    }
    v
}

fn check(rows: usize, cols: usize, m: usize, seed: u64) {
    let mut rng = Rng(seed);
    let bytes = random_q6k_rows(&mut rng, rows, cols);
    let w = QuantMat::new(GgmlType::Q6K, rows, cols, &bytes);
    let x: Vec<f32> = (0..m * cols).map(|_| rng.unit() * 4.0).collect();
    let mut want = vec![0f32; m * rows];
    matvec_f32(w, &x, m, &mut want);
    let got = q6_k_matvec_gpu(w, &x, m).unwrap_or_else(|e| panic!("GPU call failed: {e}"));
    assert_close(&got, &want, cols, rows, &format!("q6_k ({rows},{cols},{m})"));
}

#[test]
#[ignore]
fn q6_k_one_superblock_rows() {
    check(8, 256, 1, 1);
}

#[test]
#[ignore]
fn q6_k_odd_rows_three_tokens() {
    check(37, 512, 3, 2);
}

#[test]
#[ignore]
fn q6_k_lfm2_head_slice() {
    // 4096 rows of the 128000 x 2048 tied head
    check(4096, 2048, 1, 3);
}

#[test]
#[ignore]
fn q6_k_real_embedding_rows_if_present() {
    let path = std::path::Path::new("/mnt/t440/raid0/models/lfm25/LFM2.5-24B-A2B-QAD-Q4_0.gguf");
    if !path.exists() {
        eprintln!("skip: no model file");
        return;
    }
    let g = lb_gguf::Gguf::open(path).unwrap();
    let t = g.tensor("token_embd.weight").unwrap();
    assert_eq!(t.ty, GgmlType::Q6K);
    let cols = t.dims[0];
    let rows = 2048usize; // first rows of the embedding
    let bytes = &g.tensor_bytes(t)[..rows * t.ty.row_bytes(cols).unwrap()];
    let w = QuantMat::new(GgmlType::Q6K, rows, cols, bytes);
    let mut rng = Rng(9);
    let x: Vec<f32> = (0..2 * cols).map(|_| rng.unit()).collect();
    let mut want = vec![0f32; 2 * rows];
    matvec_f32(w, &x, 2, &mut want);
    let got = q6_k_matvec_gpu(w, &x, 2).unwrap();
    assert_close(&got, &want, cols, rows, "q6_k real embedding rows");
}

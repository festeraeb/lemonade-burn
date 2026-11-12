//! Forge seed: the GPU Q8_0 matvec must match `lb_gguf::matvec_f32`. Ignored until the
//! kernel exists; the forge check runs `--include-ignored` on c3 (LB_WGPU_DEVICE=discrete:1).
mod common;
use common::{assert_close, Rng};
use lb_gguf::types::quantize_q8_0;
use lb_gguf::{matvec_f32, GgmlType, QuantMat};
use lb_q4_gpu::q8_0_matvec_gpu;

fn check(rows: usize, cols: usize, m: usize, seed: u64) {
    let mut rng = Rng(seed);
    let wf: Vec<f32> = (0..rows * cols).map(|_| rng.unit()).collect();
    let bytes: Vec<u8> = wf.chunks_exact(cols).flat_map(|r| quantize_q8_0(r)).collect();
    let w = QuantMat::new(GgmlType::Q8_0, rows, cols, &bytes);
    let x: Vec<f32> = (0..m * cols).map(|_| rng.unit() * 4.0).collect();
    let mut want = vec![0f32; m * rows];
    matvec_f32(w, &x, m, &mut want);
    let got = q8_0_matvec_gpu(w, &x, m).unwrap_or_else(|e| panic!("GPU call failed: {e}"));
    assert_close(&got, &want, cols, rows, &format!("q8_0 ({rows},{cols},{m})"));
}

#[test]
#[ignore]
fn q8_0_one_block_rows() {
    check(8, 32, 1, 1);
}

#[test]
#[ignore]
fn q8_0_odd_rows_three_tokens() {
    check(37, 96, 3, 2);
}

#[test]
#[ignore]
fn q8_0_lfm2_hidden_square() {
    check(2048, 2048, 2, 3);
}

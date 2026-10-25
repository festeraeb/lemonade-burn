//! The GPU ternary GEMM must give exactly the integers of the scalar CPU reference.
use lb_ternary::{quantize_acts, ternary_gemm_i32, Kernel, PackedTernary};
use lb_ternary_gpu::ternary_gemm_i32_gpu;

/// xorshift64*: fixed-seed generator, no rand crate.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn ternary(&mut self) -> i8 {
        (self.next() % 3) as i8 - 1
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn check(rows: usize, cols: usize, m: usize, seed: u64) {
    let mut rng = Rng(seed);
    let values: Vec<i8> = (0..rows * cols).map(|_| rng.ternary()).collect();
    let w = PackedTernary::from_ternary(rows, cols, &values, 1.0);
    let acts: Vec<f32> = (0..m * cols).map(|_| rng.unit() * 4.0).collect();
    let x = quantize_acts(&acts, m, cols);
    let want = ternary_gemm_i32(w.view(), &x, Kernel::Scalar);
    let got = ternary_gemm_i32_gpu(w.view(), &x).unwrap_or_else(|e| panic!("GPU call failed: {e}"));
    assert_eq!(got.len(), want.len(), "length for ({rows}, {cols}, {m})");
    for (i, (g, e)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g, e, "shape ({rows}, {cols}, {m}): element {i} (token {}, row {})", i / rows, i % rows);
    }
}

#[test]
fn small_one_block() {
    check(64, 128, 1, 1);
}

#[test]
fn padded_cols_three_tokens() {
    check(130, 300, 3, 2);
}

#[test]
fn bitnet_square() {
    check(2560, 2560, 1, 3);
}

#[test]
fn bitnet_ffn_four_tokens() {
    check(6912, 2560, 4, 4);
}

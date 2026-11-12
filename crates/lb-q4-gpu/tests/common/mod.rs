//! Shared helpers for the GPU-vs-CPU kernel tests.
#![allow(dead_code)]
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    pub fn byte(&mut self) -> u8 {
        (self.next() >> 56) as u8
    }
}

/// Assert `got` matches `want` to f32 summation-order tolerance.
pub fn assert_close(got: &[f32], want: &[f32], cols: usize, rows: usize, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let mut worst = 0f32;
    for (i, (g, e)) in got.iter().zip(want).enumerate() {
        let tol = 1e-4 * (1.0 + e.abs()) + 1e-5 * cols as f32;
        assert!((g - e).abs() <= tol, "{label} elem {i} (token {}, row {}): gpu {g} cpu {e}", i / rows, i % rows);
        worst = worst.max((g - e).abs());
    }
    eprintln!("{label}: max |gpu-cpu| = {worst:e}");
}

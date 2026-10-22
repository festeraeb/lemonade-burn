//! Golden tests: every kernel against a naive reference built from unpacked values.
//!
//! * Integer agreement: `ternary_gemm_i32` must equal `sum_k t[r,k] * q[t,k]` exactly.
//! * Float agreement after scaling: `ternary_gemm` vs an f64 evaluation of
//!   `sum_k (q/s_a) * (t/s_w)`; documented tolerance is relative 1e-5 of
//!   `sum_k |q/s_a| * |t/s_w|` (the magnitude bound of the sum), which covers one f32
//!   rounding of the exact integer result and of the scale product.

use lb_ternary::pack::ternarize_absmean;
use lb_ternary::{quantize_acts, ternary_gemm, ternary_gemm_i32, Kernel, PackedTernary};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f32 {
        (self.next() >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
    }
}

fn case(rows: usize, cols: usize, m: usize, seed: u64) {
    let mut rng = Rng(seed);
    // Gaussian-ish weights -> absmean ternarization (realistic ~1/3 zeros)
    let wf: Vec<f32> = (0..rows * cols).map(|_| (rng.f() + rng.f() + rng.f()) * 0.02).collect();
    let (t, scale) = ternarize_absmean(&wf);
    let packed = PackedTernary::from_ternary(rows, cols, &t, scale);
    // activations with a few outliers, like real hidden states
    let x: Vec<f32> = (0..m * cols)
        .map(|i| if i % 97 == 3 { rng.f() * 20.0 } else { rng.f() })
        .collect();
    let qa = quantize_acts(&x, m, cols);

    let mut ref_i = vec![0i64; m * rows];
    let mut ref_f = vec![0f64; m * rows];
    let mut ref_abs = vec![0f64; m * rows];
    for tk in 0..m {
        for r in 0..rows {
            let mut acc = 0i64;
            let mut af = 0f64;
            let mut aa = 0f64;
            for k in 0..cols {
                let w = t[r * cols + k] as i64;
                let q = qa.row(tk)[k] as i64;
                acc += w * q;
                let term = (q as f64 / qa.scale[tk] as f64) * (w as f64 / scale as f64);
                af += term;
                aa += term.abs();
            }
            ref_i[tk * rows + r] = acc;
            ref_f[tk * rows + r] = af;
            ref_abs[tk * rows + r] = aa;
        }
    }
    for kernel in Kernel::available() {
        let got = ternary_gemm_i32(packed.view(), &qa, kernel);
        for (i, (&g, &e)) in got.iter().zip(&ref_i).enumerate() {
            assert_eq!(g as i64, e, "{kernel:?} {rows}x{cols} m={m} idx {i}: int mismatch");
        }
        let mut out = vec![0f32; m * rows];
        ternary_gemm(packed.view(), &qa, kernel, &mut out);
        for i in 0..m * rows {
            let tol = 1e-5 * ref_abs[i].max(1e-6);
            assert!(
                (out[i] as f64 - ref_f[i]).abs() <= tol,
                "{kernel:?} {rows}x{cols} m={m} idx {i}: {} vs {} (tol {tol})",
                out[i],
                ref_f[i]
            );
        }
    }
}

#[test]
fn kernels_report() {
    eprintln!("available kernels: {:?}", Kernel::available());
}

#[test]
fn gemv_shapes() {
    for (rows, cols) in [(1, 128), (7, 128), (33, 256), (17, 384), (64, 640), (40, 2560), (9, 6912)] {
        case(rows, cols, 1, rows as u64 * 31 + cols as u64);
    }
}

#[test]
fn gemm_token_tiles() {
    for m in 1..=9 {
        case(23, 384, m, 1000 + m as u64);
    }
    case(48, 2560, 5, 77);
}

#[test]
fn padded_columns() {
    // logical cols not a multiple of 128
    for cols in [1, 100, 129, 300] {
        case(5, cols, 3, cols as u64);
    }
}

#[test]
fn extreme_values() {
    // all +1 weights against all +127 activations: max-magnitude accumulation
    let rows = 4;
    let cols = 6912;
    let t = vec![1i8; rows * cols];
    let p = PackedTernary::from_ternary(rows, cols, &t, 1.0);
    for sign in [1.0f32, -1.0] {
        let x = vec![sign; cols];
        let qa = quantize_acts(&x, 1, cols);
        for kernel in Kernel::available() {
            let got = ternary_gemm_i32(p.view(), &qa, kernel);
            assert!(got.iter().all(|&g| g == sign as i32 * 127 * cols as i32), "{kernel:?}");
        }
    }
}

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
    // even block count (no odd tail) with several token tiles and an odd row count
    case(37, 2560, 13, 78);
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

/// Fused multi-matrix GEMM: bit-identical to the definition
/// `y = (exact i32 dot) as f32 * (1 / (s_a * s_w))` for each matrix, for decode (m=1) and
/// prefill shapes, under several thread counts (chunking must not change results).
#[test]
fn multi_matches_separate_bit_for_bit() {
    use lb_ternary::ternary_gemm_multi;
    let cols = 640;
    let shapes = [(96usize, 0.7f32), (24, 1.3), (200, 2.1)];
    let mut rng = Rng(4242);
    let mats: Vec<PackedTernary> = shapes
        .iter()
        .map(|&(rows, sc)| {
            let t: Vec<i8> = (0..rows * cols).map(|_| (rng.next() % 3) as i8 - 1).collect();
            PackedTernary::from_ternary(rows, cols, &t, sc)
        })
        .collect();
    for m in [1usize, 2, 5, 64] {
        let x: Vec<f32> = (0..m * cols).map(|i| if i % 89 == 1 { rng.f() * 30.0 } else { rng.f() }).collect();
        let qa = quantize_acts(&x, m, cols);
        for kernel in Kernel::available() {
            let expect: Vec<Vec<f32>> = mats
                .iter()
                .map(|p| {
                    let ints = ternary_gemm_i32(p.view(), &qa, kernel);
                    ints.iter()
                        .enumerate()
                        .map(|(i, &d)| d as f32 * (1.0 / (qa.scale[i / p.rows] * p.scale)))
                        .collect()
                })
                .collect();
            for threads in [1usize, 3, 8] {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
                pool.install(|| {
                    let views: Vec<_> = mats.iter().map(|p| p.view()).collect();
                    let mut bufs: Vec<Vec<f32>> = mats.iter().map(|p| vec![0f32; m * p.rows]).collect();
                    {
                        let mut outs: Vec<&mut [f32]> = bufs.iter_mut().map(|b| b.as_mut_slice()).collect();
                        ternary_gemm_multi(&views, &qa, kernel, &mut outs);
                    }
                    for (i, (b, e)) in bufs.iter().zip(&expect).enumerate() {
                        assert!(
                            b.iter().zip(e).all(|(x, y)| x.to_bits() == y.to_bits()),
                            "{kernel:?} m={m} threads={threads} matrix {i}"
                        );
                        let mut single = vec![0f32; m * mats[i].rows];
                        ternary_gemm(mats[i].view(), &qa, kernel, &mut single);
                        assert!(single.iter().zip(e).all(|(x, y)| x.to_bits() == y.to_bits()), "single {kernel:?} m={m}");
                    }
                });
            }
        }
    }
}

/// Decode on the spinning pool: bit-identical to `ternary_gemm_multi` (rayon) for several
/// matrices, pool sizes and kernels.
#[test]
fn pool_gemv_matches_rayon_bit_for_bit() {
    use lb_ternary::{ternary_gemm_multi, ternary_gemv_pool, SharedBytes, SharedTernary, SpinPool};
    use std::sync::Arc;
    let cols = 384 + 128;
    let mut rng = Rng(777);
    let mats: Vec<SharedTernary> = [(300usize, 0.9f32), (64, 1.7), (129, 0.4)]
        .iter()
        .map(|&(rows, sc)| {
            let t: Vec<i8> = (0..rows * cols).map(|_| (rng.next() % 3) as i8 - 1).collect();
            let p = PackedTernary::from_ternary(rows, cols, &t, sc);
            let data: SharedBytes = Arc::new(p.data);
            SharedTernary { data, rows, cols, scale: sc }
        })
        .collect();
    let x: Vec<f32> = (0..cols).map(|i| if i % 41 == 0 { rng.f() * 25.0 } else { rng.f() }).collect();
    let qa = Arc::new(quantize_acts(&x, 1, cols));
    for kernel in Kernel::available() {
        let views: Vec<_> = mats.iter().map(|m| m.view()).collect();
        let mut bufs: Vec<Vec<f32>> = mats.iter().map(|m| vec![0f32; m.rows]).collect();
        {
            let mut outs: Vec<&mut [f32]> = bufs.iter_mut().map(|b| b.as_mut_slice()).collect();
            ternary_gemm_multi(&views, &qa, kernel, &mut outs);
        }
        for threads in [1usize, 2, 7] {
            let pool = SpinPool::new(threads, std::time::Duration::from_micros(100));
            for _ in 0..3 {
                let got = ternary_gemv_pool(&pool, &mats, qa.clone(), kernel);
                for (g, e) in got.iter().zip(&bufs) {
                    assert!(g.iter().zip(e).all(|(a, b)| a.to_bits() == b.to_bits()), "{kernel:?} threads {threads}");
                }
            }
        }
    }
}

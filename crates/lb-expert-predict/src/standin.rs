//! Expert stand-in scoring: how well expert `j` reproduces expert `i` on the hidden states
//! the router actually sent to `i`.
//!
//! For every sample `x` routed to `i` (the router input, `rms(x_mid, ffn_norm)`), both
//! experts run on `x` and `j` is scored by the cosine between `y_j(x)` and `y_i(x)` and by
//! the relative L2 error `|y_j - y_i| / |y_i|`, each averaged over the samples. Cosine says
//! whether `j` pushes the residual stream the same way; the relative error also charges a
//! wrong magnitude (a stand-in with cos 0.95 at twice the norm still has rel_err >= 1).
//!
//! Experts are evaluated from dequantized f32 copies with `gemm` (offline analysis: the
//! numbers are for ranking, not for serving).

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64, *y as f64);
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    if aa == 0.0 || bb == 0.0 {
        return 0.0;
    }
    ab / (aa.sqrt() * bb.sqrt())
}

/// `|pred - reference| / |reference|` (infinite when the reference is all zeros and pred is not).
pub fn rel_l2(pred: &[f32], reference: &[f32]) -> f64 {
    let (mut d, mut r) = (0f64, 0f64);
    for (p, q) in pred.iter().zip(reference) {
        let (p, q) = (*p as f64, *q as f64);
        d += (p - q) * (p - q);
        r += q * q;
    }
    if r == 0.0 {
        return if d == 0.0 { 0.0 } else { f64::INFINITY };
    }
    (d / r).sqrt()
}

/// Running means of cosine and relative error over samples.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct PairAcc {
    pub n: u64,
    pub cos_sum: f64,
    pub rel_sum: f64,
}

impl PairAcc {
    pub fn add(&mut self, pred: &[f32], reference: &[f32]) {
        self.n += 1;
        self.cos_sum += cosine(pred, reference);
        self.rel_sum += rel_l2(pred, reference);
    }
    pub fn cos(&self) -> f64 {
        if self.n == 0 { f64::NAN } else { self.cos_sum / self.n as f64 }
    }
    pub fn rel_err(&self) -> f64 {
        if self.n == 0 { f64::NAN } else { self.rel_sum / self.n as f64 }
    }
}

/// `Y[m, n] = X[m, k] * W[n, k]^T` (both row-major), multithreaded through `gemm`.
pub fn matmul_xt(x: &[f32], m: usize, k: usize, w: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(x.len(), m * k);
    assert_eq!(w.len(), n * k);
    let mut y = vec![0f32; m * n];
    if m == 0 {
        return y;
    }
    // SAFETY: the pointers cover m*k, n*k and m*n f32 with the strides given below.
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            y.as_mut_ptr(),
            1,
            n as isize,
            false,
            x.as_ptr(),
            1,
            k as isize,
            w.as_ptr(),
            k as isize,
            1,
            0.0f32,
            1.0f32,
            false,
            false,
            false,
            gemm::Parallelism::Rayon(rayon::current_num_threads()),
        );
    }
    y
}

/// One SwiGLU expert in f32: `down(silu(gate x) * up x)`.
pub struct DenseExpert {
    pub n_embd: usize,
    pub n_ff: usize,
    /// `[n_ff][n_embd]`
    pub gate: Vec<f32>,
    /// `[n_ff][n_embd]`
    pub up: Vec<f32>,
    /// `[n_embd][n_ff]`
    pub down: Vec<f32>,
}

impl DenseExpert {
    /// `x` is `m` rows of `n_embd`; returns `m` rows of `n_embd`.
    pub fn forward(&self, x: &[f32], m: usize) -> Vec<f32> {
        let g = matmul_xt(x, m, self.n_embd, &self.gate, self.n_ff);
        let u = matmul_xt(x, m, self.n_embd, &self.up, self.n_ff);
        let a: Vec<f32> = g.par_iter().zip(&u).map(|(g, u)| g / (1.0 + (-g).exp()) * u).collect();
        matmul_xt(&a, m, self.n_ff, &self.down, self.n_embd)
    }
}

/// Score every expert `j` against every expert `i`.
///
/// * `outputs[j]`: expert `j` run on all `n_rows` sample rows (`n_rows * n_embd`).
/// * `samples[i]`: the rows (indices into those) that the router sent to expert `i`.
///
/// Returns `table[i][j]` (the diagonal is the trivial cos 1 / err 0).
pub fn score_table(outputs: &[Vec<f32>], samples: &[Vec<usize>], n_embd: usize) -> Vec<Vec<PairAcc>> {
    let n = outputs.len();
    samples
        .par_iter()
        .enumerate()
        .map(|(i, rows)| {
            let mut line = vec![PairAcc::default(); n];
            for &r in rows {
                let reference = &outputs[i][r * n_embd..(r + 1) * n_embd];
                for (j, acc) in line.iter_mut().enumerate() {
                    acc.add(&outputs[j][r * n_embd..(r + 1) * n_embd], reference);
                }
            }
            line
        })
        .collect()
}

/// The two best stand-ins for `i` by mean cosine (excluding `i` itself), best first.
pub fn best_two(line: &[PairAcc], i: usize) -> (usize, usize) {
    let mut order: Vec<usize> = (0..line.len()).filter(|&j| j != i).collect();
    // NaN (no samples) sorts last, so the order stays total
    let key = |j: usize| if line[j].cos().is_nan() { f64::NEG_INFINITY } else { line[j].cos() };
    order.sort_by(|&a, &b| key(b).total_cmp(&key(a)).then(a.cmp(&b)));
    (order[0], order[1])
}

/// `(w1 * y1 + w2 * y2) / (w1 + w2)`: the "almost expert" made of two neighbours, weighted by
/// what the router gives each on this token.
pub fn blend2(y1: &[f32], w1: f32, y2: &[f32], w2: f32) -> Vec<f32> {
    let s = (w1 + w2).max(1e-12);
    y1.iter().zip(y2).map(|(a, b)| (w1 * a + w2 * b) / s).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_and_rel_err_on_known_vectors() {
        let a = [1.0, 2.0, 3.0];
        let b2 = [2.0, 4.0, 6.0];
        assert!((cosine(&a, &b2) - 1.0).abs() < 1e-12);
        assert!((rel_l2(&b2, &a) - 1.0).abs() < 1e-12); // twice the norm: err 1
        assert!((cosine(&[1.0, 0.0], &[0.0, 3.0])).abs() < 1e-12);
        assert!((cosine(&[1.0, 0.0], &[-2.0, 0.0]) + 1.0).abs() < 1e-12);
        assert_eq!(rel_l2(&a, &a), 0.0);
        assert_eq!(cosine(&[0.0, 0.0], &a), 0.0);
        assert!(rel_l2(&a, &[0.0, 0.0, 0.0]).is_infinite());
    }

    #[test]
    fn gemm_matches_naive() {
        let (m, k, n) = (5, 7, 3);
        let x: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.37).sin()).collect();
        let w: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.11).cos()).collect();
        let y = matmul_xt(&x, m, k, &w, n);
        for r in 0..m {
            for c in 0..n {
                let want: f32 = (0..k).map(|t| x[r * k + t] * w[c * k + t]).sum();
                assert!((y[r * n + c] - want).abs() < 1e-4, "{r},{c}");
            }
        }
    }

    fn expert(seed: f32, scale: f32) -> DenseExpert {
        let (e, f) = (4, 6);
        let v = |n: usize, s: f32| -> Vec<f32> { (0..n).map(|i| ((i as f32 + s) * 0.7).sin() * 0.5).collect() };
        let mut down = v(e * f, seed + 2.0);
        for d in &mut down {
            *d *= scale;
        }
        DenseExpert { n_embd: e, n_ff: f, gate: v(f * e, seed), up: v(f * e, seed + 1.0), down }
    }

    #[test]
    fn expert_forward_matches_naive_swiglu() {
        let ex = expert(0.3, 1.0);
        let x = [0.5f32, -1.0, 2.0, 0.25];
        let y = ex.forward(&x, 1);
        let mut want = vec![0f32; 4];
        let mut a = vec![0f32; 6];
        for r in 0..6 {
            let g: f32 = (0..4).map(|c| ex.gate[r * 4 + c] * x[c]).sum();
            let u: f32 = (0..4).map(|c| ex.up[r * 4 + c] * x[c]).sum();
            a[r] = g / (1.0 + (-g).exp()) * u;
        }
        for (o, w) in want.iter_mut().enumerate() {
            *w = (0..6).map(|r| ex.down[o * 6 + r] * a[r]).sum();
        }
        for (p, q) in y.iter().zip(&want) {
            assert!((p - q).abs() < 1e-5);
        }
    }

    #[test]
    fn stand_in_ranks_a_scaled_copy_first() {
        // expert 1 = expert 0 with its down projection scaled by 1.5 (same direction),
        // expert 2 unrelated: on expert 0's samples, 1 must score cos 1 / err 0.5.
        let experts = [expert(0.3, 1.0), expert(0.3, 1.5), expert(9.1, 1.0)];
        let rows: Vec<f32> = (0..3 * 4).map(|i| ((i as f32) * 1.3).cos()).collect();
        let outputs: Vec<Vec<f32>> = experts.iter().map(|ex| ex.forward(&rows, 3)).collect();
        let samples = vec![vec![0, 1, 2], vec![0, 2], vec![1]];
        let t = score_table(&outputs, &samples, 4);
        assert!((t[0][0].cos() - 1.0).abs() < 1e-9 && t[0][0].rel_err() == 0.0);
        assert!((t[0][1].cos() - 1.0).abs() < 1e-6);
        assert!((t[0][1].rel_err() - 0.5).abs() < 1e-5);
        assert!(t[0][2].cos() < t[0][1].cos());
        assert_eq!(t[0][1].n, 3);
        assert_eq!(t[1][0].n, 2);
        // scoring 0 against 1's output: y0 = y1 / 1.5, err = 1/3
        assert!((t[1][0].rel_err() - 1.0 / 3.0).abs() < 1e-5);
        assert_eq!(best_two(&t[0], 0), (1, 2));
        let b = blend2(&[1.0, 0.0], 3.0, &[0.0, 1.0], 1.0);
        assert!((b[0] - 0.75).abs() < 1e-7 && (b[1] - 0.25).abs() < 1e-7);
    }
}

//! Opt-in int8 LM head: per-row symmetric int8 weights with one f32 scale per row.
//! The exact bf16 head ([`crate::bf16`]) stays the default; this halves the head's weight
//! bytes (2 -> 1 per weight) at the cost of exact logits.

use rayon::prelude::*;

use crate::bf16::bf16_to_f32;

pub struct Q8Head {
    pub rows: usize,
    pub cols: usize,
    /// Row-major `[rows, cols]`.
    pub q: Vec<i8>,
    pub scale: Vec<f32>,
}

impl Q8Head {
    /// Quantize row-major bf16 weights: `scale = amax / 127` (1.0 for an all-zero row),
    /// `q = round_ties_even(w / scale)` clamped to `[-127, 127]`.
    pub fn from_bf16(w: &[u16], rows: usize, cols: usize) -> Q8Head {
        assert_eq!(w.len(), rows * cols);
        let mut q = vec![0i8; rows * cols];
        let mut scale = vec![0f32; rows];
        q.par_chunks_mut(cols).zip(scale.par_iter_mut()).enumerate().for_each(|(r, (qr, s))| {
            let row = &w[r * cols..(r + 1) * cols];
            let amax = row.iter().fold(0f32, |m, &b| m.max(bf16_to_f32(b).abs()));
            *s = if amax == 0.0 { 1.0 } else { amax / 127.0 };
            for (d, &b) in qr.iter_mut().zip(row) {
                *d = (bf16_to_f32(b) / *s).round_ties_even().clamp(-127.0, 127.0) as i8;
            }
        });
        Q8Head { rows, cols, q, scale }
    }

    /// `out[r] = scale[r] * sum_k q[r, k] * x[k]`.
    pub fn gemv(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(out.len(), self.rows);
        let (cols, chunk) = (self.cols, 256);
        out.par_chunks_mut(chunk).enumerate().for_each(|(ci, dst)| {
            for (i, o) in dst.iter_mut().enumerate() {
                let r = ci * chunk + i;
                let row = &self.q[r * cols..(r + 1) * cols];
                let mut acc = [0f32; 8];
                let n8 = cols / 8 * 8;
                for k in (0..n8).step_by(8) {
                    for j in 0..8 {
                        acc[j] += row[k + j] as f32 * x[k + j];
                    }
                }
                let mut sum: f32 = acc.iter().sum();
                for k in n8..cols {
                    sum += row[k] as f32 * x[k];
                }
                *o = self.scale[r] * sum;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bf16::bf16_gemv;
    use crate::Kernel;

    fn rng(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
        }
    }

    fn bf16_matrix(rnd: &mut impl FnMut() -> f32, n: usize, amp: f32) -> Vec<u16> {
        (0..n).map(|_| ((rnd() * amp).to_bits() >> 16) as u16).collect()
    }

    #[test]
    fn q8_matches_dequantized_reference() {
        let (rows, cols) = (300, 2560);
        let mut rnd = rng(0x9e3779b97f4a7c15);
        let w = bf16_matrix(&mut rnd, rows * cols, 0.05);
        let x: Vec<f32> = (0..cols).map(|_| rnd()).collect();
        let h = Q8Head::from_bf16(&w, rows, cols);
        let mut out = vec![0f32; rows];
        h.gemv(&x, &mut out);
        for r in 0..rows {
            let terms = (0..cols).map(|k| h.scale[r] as f64 * h.q[r * cols + k] as f64 * x[k] as f64);
            let (refv, mag) = terms.fold((0f64, 0f64), |(s, m), t| (s + t, m + t.abs()));
            assert!((out[r] as f64 - refv).abs() <= 1e-4 * mag, "row {r}: {} vs {refv}", out[r]);
        }
    }

    #[test]
    fn q8_argmax_agrees_with_bf16() {
        let (rows, cols) = (2000, 512);
        let mut rnd = rng(42);
        let w = bf16_matrix(&mut rnd, rows * cols, 0.05);
        let h = Q8Head::from_bf16(&w, rows, cols);
        let argmax = |v: &[f32]| v.iter().enumerate().fold((0, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0;
        let (mut a, mut b) = (vec![0f32; rows], vec![0f32; rows]);
        let mut agree = 0;
        for _ in 0..200 {
            let x: Vec<f32> = (0..cols).map(|_| rnd()).collect();
            h.gemv(&x, &mut a);
            bf16_gemv(&w, rows, cols, &x, &mut b, Kernel::Scalar);
            agree += (argmax(&a) == argmax(&b)) as usize;
        }
        eprintln!("q8 head: argmax agrees with bf16 on {agree}/200");
        assert!(agree >= 190, "argmax agreed on only {agree}/200");
    }

    #[test]
    fn zero_row_is_safe() {
        let (rows, cols) = (3, 40);
        let mut rnd = rng(7);
        let mut w = bf16_matrix(&mut rnd, rows * cols, 0.05);
        w[cols..2 * cols].iter_mut().for_each(|b| *b = 0);
        let h = Q8Head::from_bf16(&w, rows, cols);
        assert_eq!(h.scale[1], 1.0);
        let x: Vec<f32> = (0..cols).map(|_| rnd()).collect();
        let mut out = vec![1f32; rows];
        h.gemv(&x, &mut out);
        assert_eq!(out[1], 0.0);
    }
}

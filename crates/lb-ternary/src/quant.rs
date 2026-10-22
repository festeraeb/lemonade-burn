//! Per-token absmax int8 activation quantization, reproducing
//! `transformers.integrations.bitnet.ActQuant`:
//!
//! ```text
//! x = activation.float()
//! scale = 127 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
//! q = (x * scale).round().clamp(-128, 127)    # round-half-to-even
//! ```
//!
//! Because `|x * scale| <= 127` the value -128 never occurs, which the SIMD kernels rely
//! on (no `maddubs` saturation, see `gemm`).

use crate::pack::padded_cols;

/// `m` quantized activation rows of logical width `k`, each padded with zeros to
/// `padded_cols(k)`.
#[derive(Clone, Debug, Default)]
pub struct QuantActs {
    pub m: usize,
    pub k: usize,
    pub kp: usize,
    pub q: Vec<i8>,
    /// Per-row `127 / max|x|`.
    pub scale: Vec<f32>,
    /// Per-row sum of `q` (used to undo the `+1` code offset).
    pub sum: Vec<i32>,
}

impl QuantActs {
    pub fn row(&self, i: usize) -> &[i8] {
        &self.q[i * self.kp..(i + 1) * self.kp]
    }
}

/// Quantize `m` rows of `x` (row-major, width `k`).
pub fn quantize_acts(x: &[f32], m: usize, k: usize) -> QuantActs {
    let mut out = QuantActs::default();
    quantize_acts_into(x, m, k, &mut out);
    out
}

/// Like [`quantize_acts`] but reuses `out`'s buffers.
pub fn quantize_acts_into(x: &[f32], m: usize, k: usize, out: &mut QuantActs) {
    assert_eq!(x.len(), m * k);
    let kp = padded_cols(k);
    out.m = m;
    out.k = k;
    out.kp = kp;
    out.q.clear();
    out.q.resize(m * kp, 0);
    out.scale.clear();
    out.sum.clear();
    for i in 0..m {
        let row = &x[i * k..(i + 1) * k];
        let (s, sum) = quantize_row(row, &mut out.q[i * kp..i * kp + k]);
        out.scale.push(s);
        out.sum.push(sum);
    }
}

/// Quantize one row; returns `(scale, sum(q))`.
pub fn quantize_row(x: &[f32], q: &mut [i8]) -> (f32, i32) {
    let mut amax = 0.0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }
    let scale = 127.0f32 / amax.max(1e-5);
    let mut sum = 0i32;
    for (o, &v) in q.iter_mut().zip(x) {
        let r = (v * scale).round_ties_even().clamp(-128.0, 127.0) as i8;
        *o = r;
        sum += r as i32;
    }
    (scale, sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantizes_like_actquant() {
        let x = [1.0f32, -0.5, 0.25, 0.0, -1.0];
        let qa = quantize_acts(&x, 1, 5);
        assert_eq!(qa.scale[0], 127.0);
        // 63.5 -> 64 (half-even), 31.75 -> 32
        assert_eq!(&qa.row(0)[..5], &[127, -64, 32, 0, -127]);
        assert!(qa.row(0)[5..].iter().all(|&v| v == 0));
        assert_eq!(qa.sum[0], 127 - 64 + 32 - 127);
    }

    #[test]
    fn zero_row_is_safe() {
        let qa = quantize_acts(&[0.0; 4], 1, 4);
        assert_eq!(qa.scale[0], 127.0 / 1e-5);
        assert!(qa.q.iter().all(|&v| v == 0));
    }
}

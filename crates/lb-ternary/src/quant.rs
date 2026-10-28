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
use rayon::prelude::*;

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

/// Like [`quantize_acts`] but reuses `out`'s buffers. Rows are quantized in parallel
/// (rayon) when there are several; each row's result does not depend on that.
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
    if m == 0 {
        return;
    }
    if m == 1 || m * k < PAR_MIN_ELEMS {
        for i in 0..m {
            let row = &x[i * k..(i + 1) * k];
            let (s, sum) = quantize_row(row, &mut out.q[i * kp..i * kp + k]);
            out.scale.push(s);
            out.sum.push(sum);
        }
        return;
    }
    let mut res: Vec<(f32, i32)> = Vec::with_capacity(m);
    out.q
        .par_chunks_mut(kp)
        .zip(x.par_chunks(k))
        .map(|(q, row)| quantize_row(row, &mut q[..k]))
        .collect_into_vec(&mut res);
    out.scale.extend(res.iter().map(|r| r.0));
    out.sum.extend(res.iter().map(|r| r.1));
}

/// Below this many activations a multi-row quantization stays on the calling thread.
const PAR_MIN_ELEMS: usize = 16 * 1024;

/// Quantize one row; returns `(scale, sum(q))`. Uses AVX2 when the CPU has it; the result
/// is bit-identical to [`quantize_row_scalar`] (tested).
pub fn quantize_row(x: &[f32], q: &mut [i8]) -> (f32, i32) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            return unsafe { x86::quantize_row_avx2(x, q) };
        }
    }
    quantize_row_scalar(x, q)
}

#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    use std::sync::OnceLock;
    static AVX2: OnceLock<bool> = OnceLock::new();
    *AVX2.get_or_init(|| is_x86_feature_detected!("avx2"))
}

/// Scalar reference quantizer (the `ActQuant` definition, element by element).
pub fn quantize_row_scalar(x: &[f32], q: &mut [i8]) -> (f32, i32) {
    let mut amax = 0.0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }
    let scale = 127.0f32 / amax.max(1e-5);
    let mut sum = 0i32;
    for (o, &v) in q.iter_mut().zip(x) {
        let r = quant_one(v, scale);
        *o = r;
        sum += r as i32;
    }
    (scale, sum)
}

#[inline]
fn quant_one(v: f32, scale: f32) -> i8 {
    (v * scale).round_ties_even().clamp(-128.0, 127.0) as i8
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::quant_one;
    use std::arch::x86_64::*;

    /// 8 floats -> 8 clamped, rounded i32 (NaN -> 0).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn quant8(s: __m256, lo: __m256, hi: __m256, p: *const f32) -> __m256i {
        let y = _mm256_mul_ps(_mm256_loadu_ps(p), s);
        let r = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(y);
        let c = _mm256_min_ps(_mm256_max_ps(r, lo), hi);
        let c = _mm256_and_ps(c, _mm256_cmp_ps::<_CMP_ORD_Q>(y, y));
        _mm256_cvtps_epi32(c)
    }

    /// AVX2 version of `quantize_row_scalar`, same results for every input:
    /// * `max_ps(v, acc)` returns `acc` when `v` is NaN, like `f32::max` ignoring NaN;
    /// * `round_ps(NEAREST_INT)` is round-half-to-even, like `round_ties_even`;
    /// * NaN products become 0, like `NaN as i8`; +-inf clamp to 127 / -128.
    #[target_feature(enable = "avx2")]
    pub unsafe fn quantize_row_avx2(x: &[f32], q: &mut [i8]) -> (f32, i32) {
        let n = x.len();
        assert!(q.len() >= n);
        let xp = x.as_ptr();
        let qp = q.as_mut_ptr();

        let absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
        let mut mx = _mm256_setzero_ps();
        let n8 = n / 8 * 8;
        let mut i = 0;
        while i < n8 {
            let v = _mm256_and_ps(_mm256_loadu_ps(xp.add(i)), absmask);
            mx = _mm256_max_ps(v, mx);
            i += 8;
        }
        let mut lanes = [0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), mx);
        let mut amax = 0.0f32;
        for &l in &lanes {
            amax = amax.max(l);
        }
        for &v in &x[n8..] {
            amax = amax.max(v.abs());
        }
        let scale = 127.0f32 / amax.max(1e-5);

        let s = _mm256_set1_ps(scale);
        let lo = _mm256_set1_ps(-128.0);
        let hi = _mm256_set1_ps(127.0);
        let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        let mut acc = _mm256_setzero_si256();
        let n32 = n / 32 * 32;
        let mut i = 0;
        while i < n32 {
            let a = quant8(s, lo, hi, xp.add(i));
            let b = quant8(s, lo, hi, xp.add(i + 8));
            let c = quant8(s, lo, hi, xp.add(i + 16));
            let d = quant8(s, lo, hi, xp.add(i + 24));
            acc = _mm256_add_epi32(acc, _mm256_add_epi32(_mm256_add_epi32(a, b), _mm256_add_epi32(c, d)));
            let packed = _mm256_packs_epi16(_mm256_packs_epi32(a, b), _mm256_packs_epi32(c, d));
            _mm256_storeu_si256(qp.add(i) as *mut __m256i, _mm256_permutevar8x32_epi32(packed, perm));
            i += 32;
        }
        let mut sums = [0i32; 8];
        _mm256_storeu_si256(sums.as_mut_ptr() as *mut __m256i, acc);
        let mut sum: i32 = sums.iter().sum();
        for j in n32..n {
            let r = quant_one(x[j], scale);
            q[j] = r;
            sum += r as i32;
        }
        (scale, sum)
    }
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

    fn rnd_vec(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|i| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u = (s >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0;
                if i % 53 == 7 { u * 40.0 } else { u }
            })
            .collect()
    }

    /// The dispatched (SIMD) quantizer must match the scalar definition bit for bit,
    /// including ties, tails, zeros, tiny/denormal values, infinities and NaN.
    #[test]
    fn simd_matches_scalar() {
        let mut cases: Vec<Vec<f32>> = Vec::new();
        for (seed, n) in [(1u64, 2560usize), (2, 6912), (3, 31), (4, 33), (5, 7), (6, 1), (7, 130), (8, 0)] {
            cases.push(rnd_vec(seed, n));
        }
        // exact .5 ties after scaling: amax 127 -> scale 1, values k + 0.5
        let mut ties: Vec<f32> = (0..96).map(|i| (i as f32 - 48.0) + 0.5).collect();
        ties.push(127.0);
        cases.push(ties);
        let mut odd = rnd_vec(9, 70);
        odd[3] = f32::NAN;
        odd[40] = f32::NAN;
        odd[5] = 1e-42; // denormal
        odd[6] = -0.0;
        cases.push(odd);
        let mut inf = rnd_vec(10, 64);
        inf[10] = f32::INFINITY;
        inf[50] = f32::NEG_INFINITY;
        cases.push(inf);
        cases.push(vec![0.0; 40]);
        cases.push(vec![1e-30; 40]);
        for x in &cases {
            let mut a = vec![0i8; x.len()];
            let mut b = vec![0i8; x.len()];
            let ra = quantize_row(x, &mut a);
            let rb = quantize_row_scalar(x, &mut b);
            assert_eq!(ra.0.to_bits(), rb.0.to_bits(), "scale, n={}", x.len());
            assert_eq!(ra.1, rb.1, "sum, n={}", x.len());
            assert_eq!(a, b, "codes, n={}", x.len());
        }
    }

    /// Parallel multi-row quantization equals row-by-row scalar quantization.
    #[test]
    fn parallel_rows_match() {
        for (m, k) in [(64usize, 2560usize), (5, 6912), (3, 100), (17, 1000)] {
            let x = rnd_vec(m as u64 * 31 + k as u64, m * k);
            let qa = quantize_acts(&x, m, k);
            for i in 0..m {
                let mut q = vec![0i8; k];
                let (s, sum) = quantize_row_scalar(&x[i * k..(i + 1) * k], &mut q);
                assert_eq!(qa.scale[i].to_bits(), s.to_bits());
                assert_eq!(qa.sum[i], sum);
                assert_eq!(&qa.row(i)[..k], &q[..]);
                assert!(qa.row(i)[k..].iter().all(|&v| v == 0));
            }
        }
    }
}

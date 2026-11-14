//! CPU expert kernel with 8-bit activations: `out[r] = sum_c W[r, c] * x[c]`, `W` in Q4_0.
//!
//! The reference kernels in [`crate::kernels`] dequantize every weight to f32. Here the
//! activation row is quantized once per call to signed 8-bit values in blocks of 32 (one f32
//! scale per block), so a Q4_0 block times an activation block is an integer dot product of
//! 32 small numbers, followed by one multiply with the two block scales:
//!
//! ```text
//! out[r] = sum over blocks b of  d_w[r, b] * d_x[b] * isum[r, b]
//! isum[r, b] = sum over j in 0..32 of (nibble[r, b, j] - 8) * q[b * 32 + j]      (exact, i32)
//! ```
//!
//! `isum` is bounded by 32 * 8 * 127 = 32512, so it is exact in i32 (and in i16).
//! The scalar functions here define the arithmetic; the SIMD paths must give the same `isum`
//! for every block and a final value within float rounding of the scalar one.

use crate::kernels::QuantMat;
use crate::types::GgmlType;
use half::f16;
use rayon::prelude::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// An activation row quantized to i8 in blocks of 32.
#[derive(Clone, Debug)]
pub struct ActQ8 {
    /// `q[c]` in -127..=127; value `c` is `scale[c / 32] * q[c]`.
    pub q: Vec<i8>,
    /// One scale per block of 32 values (0.0 for an all-zero block).
    pub scale: Vec<f32>,
}

/// Quantize `x` (a whole number of 32-value blocks) to [`ActQ8`]: per block,
/// `scale = max|x| / 127` and `q = round(x / scale)` (ties away from zero), clamped to +-127.
pub fn quantize_act_q8(x: &[f32]) -> ActQ8 {
    assert_eq!(x.len() % 32, 0, "activation length must be a whole number of 32-value blocks");
    let mut q = vec![0i8; x.len()];
    let mut scale = vec![0f32; x.len() / 32];
    for (b, blk) in x.chunks_exact(32).enumerate() {
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        if amax == 0.0 || !amax.is_finite() {
            continue;
        }
        let s = amax / 127.0;
        let inv = 127.0 / amax;
        scale[b] = s;
        for (j, &v) in blk.iter().enumerate() {
            q[b * 32 + j] = (v * inv).round().clamp(-127.0, 127.0) as i8;
        }
    }
    ActQ8 { q, scale }
}

/// The exact integer dot of one Q4_0 block's 16 nibble bytes with 32 activation values.
#[inline]
pub fn isum_q4_0_block(nibbles: &[u8], q: &[i8]) -> i32 {
    debug_assert_eq!(nibbles.len(), 16);
    debug_assert_eq!(q.len(), 32);
    let mut s = 0i32;
    for j in 0..16 {
        let byte = nibbles[j];
        s += ((byte & 0x0f) as i32 - 8) * q[j] as i32;
        s += ((byte >> 4) as i32 - 8) * q[j + 16] as i32;
    }
    s
}

/// Scalar reference: dot of one Q4_0 row (`cols / 32` blocks of 18 bytes) with `a`.
pub fn dot_q4_0_i8_scalar(row: &[u8], a: &ActQ8) -> f32 {
    debug_assert_eq!(row.len() / 18 * 32, a.q.len());
    let mut acc = 0f32;
    for (b, blk) in row.chunks_exact(18).enumerate() {
        let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let isum = isum_q4_0_block(&blk[2..18], &a.q[b * 32..b * 32 + 32]);
        acc += d * a.scale[b] * isum as f32;
    }
    acc
}

/// Which kernel [`dot_q4_0_i8`] uses on this CPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdLevel {
    Scalar,
    Avx2,
    Avx512,
}

/// The best level this CPU supports (AVX-512 needs `avx512f` and `avx512bw`).
pub fn simd_level() -> SimdLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            return SimdLevel::Avx512;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return SimdLevel::Avx2;
        }
    }
    SimdLevel::Scalar
}

/// Dot of one Q4_0 row with `a`, on the fastest kernel this CPU has.
#[inline]
pub fn dot_q4_0_i8(row: &[u8], a: &ActQ8) -> f32 {
    dot_q4_0_i8_at(simd_level(), row, a)
}

/// Dot of one Q4_0 row with `a` on a named kernel. Panics if the CPU lacks it.
pub fn dot_q4_0_i8_at(level: SimdLevel, row: &[u8], a: &ActQ8) -> f32 {
    match level {
        SimdLevel::Scalar => dot_q4_0_i8_scalar(row, a),
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2 => {
            assert!(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"));
            // SAFETY: the features were checked on the line above.
            unsafe { dot_q4_0_i8_avx2(row, a) }
        }
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx512 => {
            assert!(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw"));
            // SAFETY: the features were checked on the line above.
            unsafe { dot_q4_0_i8_avx512(row, a) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        _ => dot_q4_0_i8_scalar(row, a),
    }
}

/// One 32-value block (`b`) of `row` accumulated into `acc`: adds `d * a.scale[b] * isum`
/// spread over the 8 lanes, where `isum` is the exact integer dot of [`isum_q4_0_block`].
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_block_avx2(acc: __m256, row: &[u8], a: &ActQ8, b: usize) -> __m256 {
    let blk = &row[b * 18..b * 18 + 18];
    let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
    let nib = _mm_loadu_si128(blk.as_ptr().add(2).cast::<__m128i>());
    // Byte j holds value j in its low nibble and value j + 16 in its high nibble:
    // concatenate low nibbles then high nibbles into 32 bytes in value order 0..32.
    let lo = _mm_and_si128(nib, _mm_set1_epi8(0x0f));
    let hi = _mm_and_si128(_mm_srli_epi16(nib, 4), _mm_set1_epi8(0x0f));
    let nib_u = _mm256_set_m128i(hi, lo);
    let q = _mm256_loadu_si256(a.q.as_ptr().add(b * 32).cast::<__m256i>());
    // sum(nibble * q) over byte pairs (max 15 * 127 * 2 = 3810, fits i16), then pair-sum to 8 i32 lanes.
    let prod = _mm256_maddubs_epi16(nib_u, q);
    let sum = _mm256_madd_epi16(prod, _mm256_set1_epi16(1));
    // Subtract the -8 offset term: 8 * sum(q), computed the same way.
    let q8 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi8(8), q), _mm256_set1_epi16(1));
    let isum = _mm256_sub_epi32(sum, q8);
    let s = _mm256_set1_ps(d * a.scale[b]);
    _mm256_fmadd_ps(_mm256_cvtepi32_ps(isum), s, acc)
}

/// AVX2 kernel for one Q4_0 row. Same result contract as [`dot_q4_0_i8_scalar`].
///
/// # Safety
/// The CPU must support `avx2` and `fma`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q4_0_i8_avx2(row: &[u8], a: &ActQ8) -> f32 {
    debug_assert_eq!(row.len() / 18 * 32, a.q.len());
    let nb = row.len() / 18;
    // Four independent accumulators: the per-block fmadd chain is the latency
    // bottleneck, so spread consecutive blocks over four of them.
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut b = 0;
    while b + 4 <= nb {
        acc0 = unsafe { dot_block_avx2(acc0, row, a, b) };
        acc1 = unsafe { dot_block_avx2(acc1, row, a, b + 1) };
        acc2 = unsafe { dot_block_avx2(acc2, row, a, b + 2) };
        acc3 = unsafe { dot_block_avx2(acc3, row, a, b + 3) };
        b += 4;
    }
    let mut acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    while b < nb {
        acc = unsafe { dot_block_avx2(acc, row, a, b) };
        b += 1;
    }
    let mut lanes = [0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    lanes.iter().sum()
}

/// AVX-512 (F + BW) kernel for one Q4_0 row. Same result contract as [`dot_q4_0_i8_scalar`].
///
/// # Safety
/// The CPU must support `avx512f` and `avx512bw`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn dot_q4_0_i8_avx512(row: &[u8], a: &ActQ8) -> f32 {
    debug_assert_eq!(row.len() / 18 * 32, a.q.len());
    let nb = row.len() / 18;
    // Two 256-bit accumulators: in each 512-bit isum step lanes 0..8 are block b
    // and lanes 8..16 are block b + 1, so the float stage runs at full 256-bit
    // rate (no 512-bit cvt/fmadd, and each block's scale is a plain broadcast).
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut b = 0;
    // Two blocks (64 values) per step in one __m512i.
    while b + 2 <= nb {
        let blk0 = &row[b * 18..b * 18 + 18];
        let blk1 = &row[(b + 1) * 18..(b + 1) * 18 + 18];
        let d0 = f16::from_le_bytes([blk0[0], blk0[1]]).to_f32();
        let d1 = f16::from_le_bytes([blk1[0], blk1[1]]).to_f32();
        let n0 = _mm_loadu_si128(blk0.as_ptr().add(2).cast::<__m128i>());
        let n1 = _mm_loadu_si128(blk1.as_ptr().add(2).cast::<__m128i>());
        let v0 = _mm256_set_m128i(
            _mm_and_si128(_mm_srli_epi16(n0, 4), _mm_set1_epi8(0x0f)),
            _mm_and_si128(n0, _mm_set1_epi8(0x0f)),
        );
        let v1 = _mm256_set_m128i(
            _mm_and_si128(_mm_srli_epi16(n1, 4), _mm_set1_epi8(0x0f)),
            _mm_and_si128(n1, _mm_set1_epi8(0x0f)),
        );
        let nib = _mm512_inserti64x4(_mm512_castsi256_si512(v0), v1, 1);
        let q0 = _mm256_loadu_si256(a.q.as_ptr().add(b * 32).cast::<__m256i>());
        let q1 = _mm256_loadu_si256(a.q.as_ptr().add(b * 32 + 32).cast::<__m256i>());
        let q = _mm512_inserti64x4(_mm512_castsi256_si512(q0), q1, 1);
        let prod = _mm512_maddubs_epi16(nib, q);
        let sum = _mm512_madd_epi16(prod, _mm512_set1_epi16(1));
        let q8 = _mm512_madd_epi16(_mm512_maddubs_epi16(_mm512_set1_epi8(8), q), _mm512_set1_epi16(1));
        let isum = _mm512_sub_epi32(sum, q8);
        acc0 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm512_castsi512_si256(isum)),
            _mm256_set1_ps(d0 * a.scale[b]),
            acc0,
        );
        acc1 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm512_extracti64x4_epi64(isum, 1)),
            _mm256_set1_ps(d1 * a.scale[b + 1]),
            acc1,
        );
        b += 2;
    }
    let mut acc = _mm256_add_ps(acc0, acc1);
    // Odd last block on the 256-bit path.
    while b < nb {
        acc = unsafe { dot_block_avx2(acc, row, a, b) };
        b += 1;
    }
    let mut lanes = [0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    lanes.iter().sum()
}

/// `out[r] = W[r, :] . x` for one activation row, Q4_0 weights, rows split across rayon threads.
/// The activation is quantized once. `min_rows_per_task` bounds the split (0 = 64).
pub fn matvec_q4_0_i8(w: QuantMat<'_>, x: &[f32], out: &mut [f32], min_rows_per_task: usize) {
    assert_eq!(w.ty, GgmlType::Q4_0, "matvec_q4_0_i8 takes Q4_0 weights");
    assert_eq!(x.len(), w.cols);
    assert_eq!(out.len(), w.rows);
    let a = quantize_act_q8(x);
    let level = simd_level();
    let rb = w.row_bytes();
    let chunk = if min_rows_per_task == 0 { 64 } else { min_rows_per_task };
    out.par_chunks_mut(chunk).enumerate().for_each(|(ci, o)| {
        let r0 = ci * chunk;
        for (i, v) in o.iter_mut().enumerate() {
            let r = r0 + i;
            *v = dot_q4_0_i8_at(level, &w.bytes[r * rb..(r + 1) * rb], &a);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::quantize_q4_0;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    #[test]
    fn act_quantization_round_trips_within_half_a_step() {
        let mut s = 3u64;
        let x: Vec<f32> = (0..96).map(|_| lcg(&mut s) * 3.0).collect();
        let a = quantize_act_q8(&x);
        for (c, &v) in x.iter().enumerate() {
            let back = a.scale[c / 32] * a.q[c] as f32;
            assert!((back - v).abs() <= a.scale[c / 32] * 0.5 + 1e-6, "c{c}: {back} vs {v}");
        }
        let z = quantize_act_q8(&[0f32; 32]);
        assert!(z.q.iter().all(|&q| q == 0) && z.scale[0] == 0.0);
    }

    #[test]
    fn scalar_i8_dot_is_close_to_the_f32_reference() {
        let (rows, cols) = (41, 2048);
        let mut s = 7u64;
        let wf: Vec<f32> = (0..rows * cols).map(|_| lcg(&mut s)).collect();
        let x: Vec<f32> = (0..cols).map(|_| lcg(&mut s)).collect();
        let bytes: Vec<u8> = wf.chunks_exact(cols).flat_map(quantize_q4_0).collect();
        let qm = QuantMat::new(GgmlType::Q4_0, rows, cols, &bytes);
        let a = quantize_act_q8(&x);
        let mut buf = vec![0f32; cols];
        let (mut err2, mut ref2) = (0f64, 0f64);
        for r in 0..rows {
            qm.dequant_row(r, &mut buf);
            let want: f32 = buf.iter().zip(&x).map(|(p, q)| p * q).sum();
            let got = dot_q4_0_i8_scalar(qm.row(r), &a);
            err2 += ((got - want) as f64).powi(2);
            ref2 += (want as f64).powi(2);
        }
        let rel = (err2 / ref2).sqrt();
        assert!(rel < 0.01, "relative error {rel}");
    }

    #[test]
    fn matvec_matches_the_scalar_dot() {
        let (rows, cols) = (300, 512);
        let mut s = 11u64;
        let wf: Vec<f32> = (0..rows * cols).map(|_| lcg(&mut s)).collect();
        let x: Vec<f32> = (0..cols).map(|_| lcg(&mut s)).collect();
        let bytes: Vec<u8> = wf.chunks_exact(cols).flat_map(quantize_q4_0).collect();
        let qm = QuantMat::new(GgmlType::Q4_0, rows, cols, &bytes);
        let a = quantize_act_q8(&x);
        let mut out = vec![0f32; rows];
        matvec_q4_0_i8(qm, &x, &mut out, 0);
        for r in 0..rows {
            let want = dot_q4_0_i8_scalar(qm.row(r), &a);
            assert!((out[r] - want).abs() <= 1e-4 * want.abs().max(1.0), "r{r}: {} vs {want}", out[r]);
        }
    }
}

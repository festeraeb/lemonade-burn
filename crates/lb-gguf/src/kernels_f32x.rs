//! Exact CPU expert kernel: f32 activations, Q4_0 weights dequantized in SIMD registers.
//!
//! The same arithmetic as [`crate::kernels::dot_q4_0`]: every weight is `d * (nibble - 8)` as an
//! f32 (the nibble minus 8 is exact), multiplied with the f32 activation, no activation rounding.
//! Only the summation order differs (lanes, fused multiply-add), so results agree with the scalar
//! reference to float rounding, unlike [`crate::kernels_i8`] which rounds the activations to 8 bits.
//!
//! Layout of a block (18 bytes): f16 `d`, then 16 bytes; byte `j` holds value `j` in the low nibble
//! and value `j + 16` in the high nibble.

use crate::kernels::QuantMat;
use crate::kernels_i8::{simd_level, SimdLevel};
use crate::types::GgmlType;
use half::f16;
use rayon::prelude::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Scalar version (the reference's arithmetic, plain loop).
pub fn dot_q4_0_f32x_scalar(row: &[u8], x: &[f32]) -> f32 {
    crate::kernels::dot_q4_0(row, x)
}

/// Dot of one Q4_0 row with `x` on the fastest kernel this CPU has.
#[inline]
pub fn dot_q4_0_f32x(row: &[u8], x: &[f32]) -> f32 {
    dot_q4_0_f32x_at(simd_level(), row, x)
}

/// Dot of one Q4_0 row with `x` on a named kernel. Panics if the CPU lacks it.
pub fn dot_q4_0_f32x_at(level: SimdLevel, row: &[u8], x: &[f32]) -> f32 {
    assert_eq!(row.len() / 18 * 32, x.len());
    match level {
        SimdLevel::Scalar => dot_q4_0_f32x_scalar(row, x),
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2 => {
            assert!(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"));
            // SAFETY: the features were checked on the line above.
            unsafe { dot_avx2(row, x) }
        }
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx512 => {
            assert!(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw"));
            // SAFETY: the features were checked on the line above.
            unsafe { dot_avx512(row, x) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        _ => dot_q4_0_f32x_scalar(row, x),
    }
}

/// One block into the 8-lane accumulator: `acc += d * (dequantized nibbles * x)` per lane.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn block_avx2(acc: __m256, blk: *const u8, xp: *const f32) -> __m256 {
    let d = f16::from_le_bytes([*blk, *blk.add(1)]).to_f32();
    let bytes = _mm_loadu_si128(blk.add(2) as *const __m128i);
    let m = _mm_set1_epi8(0x0f);
    let lo = _mm_and_si128(bytes, m);
    let hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), m);
    let eight = _mm256_set1_epi32(8);
    let cv = |v: __m128i| _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(v), eight));
    let mut s = _mm256_mul_ps(cv(lo), _mm256_loadu_ps(xp));
    s = _mm256_fmadd_ps(cv(_mm_srli_si128(lo, 8)), _mm256_loadu_ps(xp.add(8)), s);
    s = _mm256_fmadd_ps(cv(hi), _mm256_loadu_ps(xp.add(16)), s);
    s = _mm256_fmadd_ps(cv(_mm_srli_si128(hi, 8)), _mm256_loadu_ps(xp.add(24)), s);
    _mm256_fmadd_ps(_mm256_set1_ps(d), s, acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(row: &[u8], x: &[f32]) -> f32 {
    let nb = row.len() / 18;
    let (mut a0, mut a1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let mut b = 0;
    while b + 2 <= nb {
        a0 = block_avx2(a0, row.as_ptr().add(b * 18), x.as_ptr().add(b * 32));
        a1 = block_avx2(a1, row.as_ptr().add(b * 18 + 18), x.as_ptr().add(b * 32 + 32));
        b += 2;
    }
    if b < nb {
        a0 = block_avx2(a0, row.as_ptr().add(b * 18), x.as_ptr().add(b * 32));
    }
    let mut lanes = [0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), _mm256_add_ps(a0, a1));
    lanes.iter().sum()
}

/// One block into the 16-lane accumulator: low nibbles against `x[0..16]`, high against `x[16..32]`.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx2,fma")]
unsafe fn block_avx512(acc: __m512, blk: *const u8, xp: *const f32) -> __m512 {
    let d = f16::from_le_bytes([*blk, *blk.add(1)]).to_f32();
    let bytes = _mm_loadu_si128(blk.add(2) as *const __m128i);
    let m = _mm_set1_epi8(0x0f);
    let lo = _mm_and_si128(bytes, m);
    let hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), m);
    let eight = _mm512_set1_epi32(8);
    let cv = |v: __m128i| _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(v), eight));
    let mut s = _mm512_mul_ps(cv(lo), _mm512_loadu_ps(xp));
    s = _mm512_fmadd_ps(cv(hi), _mm512_loadu_ps(xp.add(16)), s);
    _mm512_fmadd_ps(_mm512_set1_ps(d), s, acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx2,fma")]
unsafe fn dot_avx512(row: &[u8], x: &[f32]) -> f32 {
    let nb = row.len() / 18;
    let (mut a0, mut a1) = (_mm512_setzero_ps(), _mm512_setzero_ps());
    let mut b = 0;
    while b + 2 <= nb {
        a0 = block_avx512(a0, row.as_ptr().add(b * 18), x.as_ptr().add(b * 32));
        a1 = block_avx512(a1, row.as_ptr().add(b * 18 + 18), x.as_ptr().add(b * 32 + 32));
        b += 2;
    }
    if b < nb {
        a0 = block_avx512(a0, row.as_ptr().add(b * 18), x.as_ptr().add(b * 32));
    }
    _mm512_reduce_add_ps(_mm512_add_ps(a0, a1))
}

/// `out[r] = W[r, :] . x` for one activation row, Q4_0 weights, rows split across rayon threads.
/// `min_rows_per_task` bounds the split (0 = 64). Any other weight type falls back to the reference.
pub fn matvec_q4_0_f32x(w: QuantMat<'_>, x: &[f32], out: &mut [f32], min_rows_per_task: usize) {
    assert_eq!(x.len(), w.cols);
    assert_eq!(out.len(), w.rows);
    if w.ty != GgmlType::Q4_0 {
        crate::kernels::matvec_f32(w, x, 1, out);
        return;
    }
    let level = simd_level();
    let rb = w.row_bytes();
    let chunk = if min_rows_per_task == 0 { 64 } else { min_rows_per_task };
    out.par_chunks_mut(chunk).enumerate().for_each(|(ci, o)| {
        let r0 = ci * chunk;
        for (i, v) in o.iter_mut().enumerate() {
            let r = r0 + i;
            *v = dot_q4_0_f32x_at(level, &w.bytes[r * rb..(r + 1) * rb], x);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::dot_q4_0;
    use crate::types::quantize_q4_0;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn levels() -> Vec<SimdLevel> {
        let mut v = vec![SimdLevel::Scalar];
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                v.push(SimdLevel::Avx2);
            }
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                v.push(SimdLevel::Avx512);
            }
        }
        v
    }

    #[test]
    fn every_level_agrees_with_the_scalar_reference() {
        // odd block counts exercise the tail; 2048 is the real expert width
        for &cols in &[32usize, 96, 2048, 1568] {
            let rows = 37;
            let mut s = 5u64 + cols as u64;
            let wf: Vec<f32> = (0..rows * cols).map(|_| lcg(&mut s) * 2.0).collect();
            let x: Vec<f32> = (0..cols).map(|_| lcg(&mut s) * 3.0).collect();
            let bytes: Vec<u8> = wf.chunks_exact(cols).flat_map(quantize_q4_0).collect();
            let rb = cols / 32 * 18;
            for lv in levels() {
                let (mut err2, mut ref2, mut worst) = (0f64, 0f64, 0f64);
                for r in 0..rows {
                    let row = &bytes[r * rb..(r + 1) * rb];
                    let want = dot_q4_0(row, &x);
                    let got = dot_q4_0_f32x_at(lv, row, &x);
                    let e = (got - want).abs() as f64;
                    worst = worst.max(e / (want.abs() as f64).max(1.0));
                    err2 += e * e;
                    ref2 += (want as f64).powi(2);
                }
                assert!(worst < 2e-5, "{lv:?} cols {cols}: worst relative-to-max(1,|v|) {worst}");
                assert!((err2 / ref2).sqrt() < 1e-5, "{lv:?} cols {cols}");
            }
        }
    }

    #[test]
    fn matvec_matches_the_reference_dot() {
        let (rows, cols) = (300, 512);
        let mut s = 21u64;
        let wf: Vec<f32> = (0..rows * cols).map(|_| lcg(&mut s)).collect();
        let x: Vec<f32> = (0..cols).map(|_| lcg(&mut s)).collect();
        let bytes: Vec<u8> = wf.chunks_exact(cols).flat_map(quantize_q4_0).collect();
        let qm = QuantMat::new(GgmlType::Q4_0, rows, cols, &bytes);
        let mut out = vec![0f32; rows];
        matvec_q4_0_f32x(qm, &x, &mut out, 0);
        for r in 0..rows {
            let want = dot_q4_0(qm.row(r), &x);
            assert!((out[r] - want).abs() <= 2e-5 * want.abs().max(1.0), "r{r}: {} vs {want}", out[r]);
        }
    }
}

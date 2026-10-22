//! f32 activations x bf16 weights GEMV (`y = W x`, W row-major `[rows, cols]` as raw
//! bf16 bits). Used for the tied, unquantized LM head (`lm_head = embed_tokens`), which at
//! 128256 x 2560 is larger than all ternary layers together.

use crate::gemm::Kernel;
use rayon::prelude::*;

#[inline]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// `out[r] = sum_k x[k] * W[r, k]`.
pub fn bf16_gemv(w: &[u16], rows: usize, cols: usize, x: &[f32], out: &mut [f32], kernel: Kernel) {
    assert_eq!(w.len(), rows * cols);
    assert_eq!(x.len(), cols);
    assert_eq!(out.len(), rows);
    let chunk = 256;
    out.par_chunks_mut(chunk).enumerate().for_each(|(ci, dst)| {
        for (i, o) in dst.iter_mut().enumerate() {
            let r = ci * chunk + i;
            *o = dot(kernel, &w[r * cols..(r + 1) * cols], x);
        }
    });
}

#[inline]
fn dot(kernel: Kernel, w: &[u16], x: &[f32]) -> f32 {
    match kernel {
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx512 | Kernel::Avx512Vnni => unsafe { x86::dot_avx512(w, x) },
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx2 if std::arch::is_x86_feature_detected!("fma") => unsafe { x86::dot_avx2(w, x) },
        _ => dot_scalar(w, x),
    }
}

pub fn dot_scalar(w: &[u16], x: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let n8 = w.len() / 8 * 8;
    for i in (0..n8).step_by(8) {
        for l in 0..8 {
            acc[l] += bf16_to_f32(w[i + l]) * x[i + l];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for i in n8..w.len() {
        s += bf16_to_f32(w[i]) * x[i];
    }
    s
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_avx2(w: &[u16], x: &[f32]) -> f32 {
        let n = w.len();
        let n32 = n / 32 * 32;
        let wp = w.as_ptr();
        let xp = x.as_ptr();
        let mut a = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i < n32 {
            for u in 0..4 {
                let wi = _mm256_cvtepu16_epi32(_mm_loadu_si128(wp.add(i + u * 8) as *const __m128i));
                let wf = _mm256_castsi256_ps(_mm256_slli_epi32(wi, 16));
                a[u] = _mm256_fmadd_ps(wf, _mm256_loadu_ps(xp.add(i + u * 8)), a[u]);
            }
            i += 32;
        }
        let s = _mm256_add_ps(_mm256_add_ps(a[0], a[1]), _mm256_add_ps(a[2], a[3]));
        let mut buf = [0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), s);
        let mut r: f32 = buf.iter().sum();
        for j in n32..n {
            r += super::bf16_to_f32(w[j]) * x[j];
        }
        r
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn dot_avx512(w: &[u16], x: &[f32]) -> f32 {
        let n = w.len();
        let n64 = n / 64 * 64;
        let wp = w.as_ptr();
        let xp = x.as_ptr();
        let mut a = [_mm512_setzero_ps(); 4];
        let mut i = 0;
        while i < n64 {
            for u in 0..4 {
                let wi = _mm512_cvtepu16_epi32(_mm256_loadu_si256(wp.add(i + u * 16) as *const __m256i));
                let wf = _mm512_castsi512_ps(_mm512_slli_epi32(wi, 16));
                a[u] = _mm512_fmadd_ps(wf, _mm512_loadu_ps(xp.add(i + u * 16)), a[u]);
            }
            i += 64;
        }
        let s = _mm512_add_ps(_mm512_add_ps(a[0], a[1]), _mm512_add_ps(a[2], a[3]));
        let mut r = _mm512_reduce_add_ps(s);
        for j in n64..n {
            r += super::bf16_to_f32(w[j]) * x[j];
        }
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_kernels_agree() {
        let rows = 37;
        let cols = 2560 + 24;
        let mut s = 12345u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
        };
        let wf: Vec<f32> = (0..rows * cols).map(|_| rnd() * 0.05).collect();
        let w: Vec<u16> = wf.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        let x: Vec<f32> = (0..cols).map(|_| rnd()).collect();
        let refv: Vec<f64> = (0..rows)
            .map(|r| (0..cols).map(|k| bf16_to_f32(w[r * cols + k]) as f64 * x[k] as f64).sum())
            .collect();
        for k in Kernel::available() {
            let mut out = vec![0f32; rows];
            bf16_gemv(&w, rows, cols, &x, &mut out, k);
            for r in 0..rows {
                assert!((out[r] as f64 - refv[r]).abs() < 1e-4, "{k:?} row {r}");
            }
        }
    }
}

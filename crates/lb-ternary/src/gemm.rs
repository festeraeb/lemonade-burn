//! int8 x ternary GEMV / GEMM.
//!
//! For activation row `x_q` (int8, scale `s_a`) and packed weight row with codes
//! `c = w + 1` (scale `s_w`):
//!
//! ```text
//! acc  = sum_k c[k] * x_q[k]           (u8 x i8 products, exact in i32)
//! dot  = acc - sum_k x_q[k]            (= sum_k w[k] * x_q[k], exact)
//! y    = dot / (s_a * s_w)             (f32)
//! ```
//!
//! The integer `dot` is exact and identical across all kernels (tested). Only the final
//! f32 scaling differs from the reference `F.linear(x_q / s_a, t / s_w)` (float summation
//! order), see the crate tests for the documented tolerance.
//!
//! SIMD notes: codes are in 0..=2 and |x_q| <= 127, so `maddubs` pair sums are <= 508
//! and four of them <= 2032; nothing saturates in i16.

use crate::pack::{padded_cols, TernaryView, BLOCK, BLOCK_BYTES};
use crate::quant::QuantActs;
use rayon::prelude::*;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    Scalar,
    Avx2,
    /// AVX-512F + BW (`vpmaddubsw` on 512-bit vectors).
    Avx512,
    /// AVX-512 VNNI (`vpdpbusd`).
    Avx512Vnni,
}

impl Kernel {
    pub fn name(self) -> &'static str {
        match self {
            Kernel::Scalar => "scalar",
            Kernel::Avx2 => "avx2",
            Kernel::Avx512 => "avx512bw",
            Kernel::Avx512Vnni => "avx512vnni",
        }
    }

    pub fn parse(s: &str) -> Option<Kernel> {
        Some(match s {
            "scalar" => Kernel::Scalar,
            "avx2" => Kernel::Avx2,
            "avx512" | "avx512bw" => Kernel::Avx512,
            "avx512vnni" | "vnni" => Kernel::Avx512Vnni,
            _ => return None,
        })
    }

    /// Kernels this CPU can run, slowest first.
    pub fn available() -> Vec<Kernel> {
        let mut v = vec![Kernel::Scalar];
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                v.push(Kernel::Avx2);
            }
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                v.push(Kernel::Avx512);
                if is_x86_feature_detected!("avx512vnni") {
                    v.push(Kernel::Avx512Vnni);
                }
            }
        }
        v
    }

    pub fn is_available(self) -> bool {
        Self::available().contains(&self)
    }

    /// Best available kernel, overridable with `LB_KERNEL=scalar|avx2|avx512|avx512vnni`
    /// (an unavailable override falls back to the best kernel with a warning).
    pub fn detect() -> Kernel {
        static K: OnceLock<Kernel> = OnceLock::new();
        *K.get_or_init(|| {
            let best = *Self::available().last().unwrap();
            match std::env::var("LB_KERNEL").ok().as_deref().map(Kernel::parse) {
                Some(Some(k)) if k.is_available() => k,
                Some(_) => {
                    eprintln!("lb-ternary: LB_KERNEL not usable here, using {}", best.name());
                    best
                }
                None => best,
            }
        })
    }
}

/// Raw integer dot products: `out[t * rows + r] = sum_k w[r,k] * x_q[t,k]` (exact).
pub fn ternary_gemm_i32(w: TernaryView<'_>, x: &QuantActs, kernel: Kernel) -> Vec<i32> {
    check(&w, x);
    let (m, n) = (x.m, w.rows);
    let mut tmp = vec![0i32; n * m]; // [row][token]
    let chunk = chunk_rows(n);
    tmp.par_chunks_mut(chunk * m).enumerate().for_each(|(ci, dst)| {
        let r0 = ci * chunk;
        let rows = dst.len() / m;
        for r in 0..rows {
            dot_row(kernel, w.row(r0 + r), x, &mut dst[r * m..(r + 1) * m]);
        }
    });
    let mut out = vec![0i32; m * n];
    for r in 0..n {
        for t in 0..m {
            out[t * n + r] = tmp[r * m + t];
        }
    }
    out
}

/// `out[t * rows + r] = y` (row-major `[m, rows]`, i.e. `x @ W^T`).
pub fn ternary_gemm(w: TernaryView<'_>, x: &QuantActs, kernel: Kernel, out: &mut [f32]) {
    ternary_gemm_multi(&[w], x, kernel, &mut [out]);
}

/// Several matrices that share the same input `x` (e.g. q/k/v, or gate/up) in one
/// parallel region: `outs[i][t * ws[i].rows + r] = (x @ ws[i]^T)[t, r]`.
///
/// Every output element is computed exactly as [`ternary_gemm`] computes it (same integer
/// dot, same `1 / (s_a * s_w)` scaling), so results are bit-identical to separate calls;
/// fusing only saves the fork/join and lets the row chunks of all matrices balance
/// across threads together.
pub fn ternary_gemm_multi(ws: &[TernaryView<'_>], x: &QuantActs, kernel: Kernel, outs: &mut [&mut [f32]]) {
    assert_eq!(ws.len(), outs.len());
    let m = x.m;
    for (w, o) in ws.iter().zip(outs.iter()) {
        check(w, x);
        assert_eq!(o.len(), m * w.rows);
    }
    let total_rows: usize = ws.iter().map(|w| w.rows).sum();
    let chunk = chunk_rows(total_rows);
    if m == 1 {
        // decode: write straight into the outputs, one task per row chunk
        let mut tasks: Vec<(TernaryView<'_>, usize, f32, &mut [f32])> = Vec::new();
        for (w, o) in ws.iter().zip(outs.iter_mut()) {
            let inv = 1.0 / (x.scale[0] * w.scale);
            for (ci, dst) in o.chunks_mut(chunk).enumerate() {
                tasks.push((*w, ci * chunk, inv, dst));
            }
        }
        tasks.into_par_iter().for_each(|(w, r0, inv, dst)| {
            let mut acc = [0i32; 1];
            for (r, o) in dst.iter_mut().enumerate() {
                dot_row(kernel, w.row(r0 + r), x, &mut acc);
                *o = acc[0] as f32 * inv;
            }
        });
        return;
    }
    // prefill: each task computes a row chunk for all tokens into `tmp` ([row][token]),
    // then a second parallel pass transposes into the `[token][row]` outputs.
    let mut tmps: Vec<Vec<f32>> = ws.iter().map(|w| vec![0f32; w.rows * m]).collect();
    let invs: Vec<Vec<f32>> = ws.iter().map(|w| x.scale.iter().map(|&sa| 1.0 / (sa * w.scale)).collect()).collect();
    {
        let mut tasks: Vec<(TernaryView<'_>, usize, &[f32], &mut [f32])> = Vec::new();
        for ((w, inv), tmp) in ws.iter().zip(&invs).zip(tmps.iter_mut()) {
            for (ci, dst) in tmp.chunks_mut(chunk * m).enumerate() {
                tasks.push((*w, ci * chunk, inv.as_slice(), dst));
            }
        }
        // AVX-512: register-tiled kernel over 2 rows x 4 tokens on re-laid-out activations
        let paired = matches!(kernel, Kernel::Avx512 | Kernel::Avx512Vnni).then(|| PairedActs::new(x));
        tasks.into_par_iter().for_each(|(w, r0, inv, dst)| {
            let rows = dst.len() / m;
            if let Some(p) = &paired {
                tile_rows(w, r0, rows, x, p, inv, dst);
                return;
            }
            let mut acc = vec![0i32; m];
            for r in 0..rows {
                dot_row(kernel, w.row(r0 + r), x, &mut acc);
                for t in 0..m {
                    dst[r * m + t] = acc[t] as f32 * inv[t];
                }
            }
        });
    }
    let mut rows_out: Vec<(&[f32], usize, &mut [f32])> = Vec::new();
    for (tmp, o) in tmps.iter().zip(outs.iter_mut()) {
        let n = o.len() / m;
        for (t, dst) in o.chunks_mut(n).enumerate() {
            rows_out.push((tmp.as_slice(), t, dst));
        }
    }
    rows_out.into_par_iter().for_each(|(tmp, t, dst)| {
        for (r, d) in dst.iter_mut().enumerate() {
            *d = tmp[r * m + t];
        }
    });
}

fn check(w: &TernaryView<'_>, x: &QuantActs) {
    assert_eq!(w.cols, x.k, "inner dimension mismatch");
    assert_eq!(padded_cols(w.cols), x.kp);
}

/// Rows per parallel task: about `LB_TASKS_PER_THREAD` (default 8) tasks per thread.
/// Measured on c4 (20 threads): 1 per thread was ~10% slower in decode, 2..16 within noise.
fn chunk_rows(n: usize) -> usize {
    static PER: OnceLock<usize> = OnceLock::new();
    let per = *PER.get_or_init(|| std::env::var("LB_TASKS_PER_THREAD").ok().and_then(|v| v.parse().ok()).unwrap_or(8));
    let threads = rayon::current_num_threads().max(1);
    n.div_ceil(threads * per).clamp(8, 1 << 20)
}

/// Activations re-laid out for the AVX-512 tile kernel so that one 64-byte load matches
/// one `decode512` lane group: for token `t` and block pair `bb` (blocks `2bb`, `2bb+1`),
/// 64-byte group `k` (0..4) holds `x[t, 2bb*128 + 32k ..][..32]` then
/// `x[t, (2bb+1)*128 + 32k ..][..32]`. An odd last block stays in `QuantActs::q`.
struct PairedActs {
    data: Vec<i8>,
    /// Bytes per token (`pairs * 256`).
    stride: usize,
    pairs: usize,
}

impl PairedActs {
    fn new(x: &QuantActs) -> Self {
        let pairs = x.kp / BLOCK / 2;
        let stride = pairs * 2 * BLOCK;
        let mut data = vec![0i8; x.m * stride];
        for t in 0..x.m {
            let src = x.row(t);
            let dst = &mut data[t * stride..(t + 1) * stride];
            for bb in 0..pairs {
                for k in 0..4 {
                    let d = &mut dst[bb * 256 + k * 64..bb * 256 + k * 64 + 64];
                    let b0 = 2 * bb * BLOCK + k * 32;
                    d[..32].copy_from_slice(&src[b0..b0 + 32]);
                    d[32..].copy_from_slice(&src[b0 + BLOCK..b0 + BLOCK + 32]);
                }
            }
        }
        Self { data, stride, pairs }
    }
}

/// Prefill rows `r0 .. r0 + rows` of `w` for all tokens into `dst` (`[row][token]`) with
/// the 2-row x 4-token AVX-512BW tile. Same integer dot and scaling as `dot_row`.
fn tile_rows(w: TernaryView<'_>, r0: usize, rows: usize, x: &QuantActs, p: &PairedActs, inv: &[f32], dst: &mut [f32]) {
    let m = x.m;
    // token tile outer: its activations (4 x row width) stay in L1 while the task's
    // weight rows stream from L2, instead of re-streaming all tokens per row pair
    let mut t = 0;
    while t < m {
        let tt = (m - t).min(4);
        let mut r = 0;
        while r < rows {
            let rr = (rows - r).min(2);
            let rws = [w.row(r0 + r), w.row(r0 + r + rr - 1)];
            let mut acc = [[0i32; 4]; 2];
            tile_dispatch(rr, tt, &rws, x, p, t, &mut acc);
            for i in 0..rr {
                for j in 0..tt {
                    let d = acc[i][j] - x.sum[t + j];
                    dst[(r + i) * m + t + j] = d as f32 * inv[t + j];
                }
            }
            r += rr;
        }
        t += tt;
    }
}

#[allow(unused_variables)]
fn tile_dispatch(rr: usize, tt: usize, rws: &[&[u8]; 2], x: &QuantActs, p: &PairedActs, t0: usize, acc: &mut [[i32; 4]; 2]) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use x86::tile_avx512bw as k;
        match (rr, tt) {
            (2, 4) => k::<2, 4>(rws, x, p, t0, acc),
            (2, 3) => k::<2, 3>(rws, x, p, t0, acc),
            (2, 2) => k::<2, 2>(rws, x, p, t0, acc),
            (2, 1) => k::<2, 1>(rws, x, p, t0, acc),
            (1, 4) => k::<1, 4>(rws, x, p, t0, acc),
            (1, 3) => k::<1, 3>(rws, x, p, t0, acc),
            (1, 2) => k::<1, 2>(rws, x, p, t0, acc),
            _ => k::<1, 1>(rws, x, p, t0, acc),
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    unreachable!("tile kernel is x86-only");
}

/// Exact `sum_k w[k] * x_q[t,k]` for every token `t` of `x`, for one packed row.
#[inline]
fn dot_row(kernel: Kernel, row: &[u8], x: &QuantActs, out: &mut [i32]) {
    let m = x.m;
    let mut t = 0;
    while t < m {
        let left = m - t;
        let xs = &x.q[t * x.kp..];
        match left {
            1 => run::<1>(kernel, row, xs, x.kp, &mut out[t..t + 1]),
            2 => run::<2>(kernel, row, xs, x.kp, &mut out[t..t + 2]),
            3 => run::<3>(kernel, row, xs, x.kp, &mut out[t..t + 3]),
            _ => run::<4>(kernel, row, xs, x.kp, &mut out[t..t + 4]),
        }
        t += left.min(4);
    }
    for (o, s) in out.iter_mut().zip(&x.sum) {
        *o -= *s;
    }
}

#[inline]
fn run<const T: usize>(kernel: Kernel, row: &[u8], x: &[i8], stride: usize, out: &mut [i32]) {
    let mut acc = [0i32; T];
    match kernel {
        Kernel::Scalar => codes_dot_scalar::<T>(row, x, stride, &mut acc),
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx2 => unsafe { x86::codes_dot_avx2::<T>(row, x, stride, &mut acc) },
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx512 => unsafe { x86::codes_dot_avx512bw::<T>(row, x, stride, &mut acc) },
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx512Vnni => unsafe { x86::codes_dot_avx512vnni::<T>(row, x, stride, &mut acc) },
        #[cfg(not(target_arch = "x86_64"))]
        _ => codes_dot_scalar::<T>(row, x, stride, &mut acc),
    }
    out[..T].copy_from_slice(&acc);
}

/// Scalar reference: `acc[t] = sum_k code[k] * x[t*stride + k]` over the padded row.
pub fn codes_dot_scalar<const T: usize>(row: &[u8], x: &[i8], stride: usize, acc: &mut [i32; T]) {
    let nb = row.len() / BLOCK_BYTES;
    for b in 0..nb {
        let blk = &row[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        for k in 0..4 {
            for (j, &byte) in blk.iter().enumerate() {
                let c = ((byte >> (2 * k)) & 3) as i32;
                let idx = b * BLOCK + k * 32 + j;
                for (t, a) in acc.iter_mut().enumerate() {
                    *a += c * x[t * stride + idx] as i32;
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::*;
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum256(v: __m256i) -> i32 {
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256(v, 1);
        let s = _mm_add_epi32(lo, hi);
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
        _mm_cvtsi128_si32(s)
    }

    /// One 128-weight block, AVX2: returns 8 x i32 partial sums.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn block_avx2(c: &[__m256i; 4], xb: *const i8) -> __m256i {
        let s0 = _mm256_maddubs_epi16(c[0], _mm256_loadu_si256(xb as *const __m256i));
        let s1 = _mm256_maddubs_epi16(c[1], _mm256_loadu_si256(xb.add(32) as *const __m256i));
        let s2 = _mm256_maddubs_epi16(c[2], _mm256_loadu_si256(xb.add(64) as *const __m256i));
        let s3 = _mm256_maddubs_epi16(c[3], _mm256_loadu_si256(xb.add(96) as *const __m256i));
        let s = _mm256_add_epi16(_mm256_add_epi16(s0, s1), _mm256_add_epi16(s2, s3));
        _mm256_madd_epi16(s, _mm256_set1_epi16(1))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn decode256(p: *const u8) -> [__m256i; 4] {
        let mask = _mm256_set1_epi8(3);
        let v = _mm256_loadu_si256(p as *const __m256i);
        [
            _mm256_and_si256(v, mask),
            _mm256_and_si256(_mm256_srli_epi16(v, 2), mask),
            _mm256_and_si256(_mm256_srli_epi16(v, 4), mask),
            _mm256_and_si256(_mm256_srli_epi16(v, 6), mask),
        ]
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn codes_dot_avx2<const T: usize>(
        row: &[u8],
        x: &[i8],
        stride: usize,
        acc: &mut [i32; T],
    ) {
        let nb = row.len() / BLOCK_BYTES;
        assert!(x.len() >= (T - 1) * stride + nb * BLOCK);
        let mut a = [_mm256_setzero_si256(); T];
        let rp = row.as_ptr();
        let xp = x.as_ptr();
        for b in 0..nb {
            let c = decode256(rp.add(b * BLOCK_BYTES));
            for t in 0..T {
                a[t] = _mm256_add_epi32(a[t], block_avx2(&c, xp.add(t * stride + b * BLOCK)));
            }
        }
        for t in 0..T {
            acc[t] = hsum256(a[t]);
        }
    }

    /// Codes for two consecutive blocks: lane half 0 = block b, half 1 = block b+1.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn decode512(p: *const u8) -> [__m512i; 4] {
        let mask = _mm512_set1_epi8(3);
        let v = _mm512_loadu_si512(p as *const __m512i);
        [
            _mm512_and_si512(v, mask),
            _mm512_and_si512(_mm512_srli_epi16(v, 2), mask),
            _mm512_and_si512(_mm512_srli_epi16(v, 4), mask),
            _mm512_and_si512(_mm512_srli_epi16(v, 6), mask),
        ]
    }

    /// Activations matching `decode512` lane k: x[b*128 + k*32 ..] | x[(b+1)*128 + k*32 ..].
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn xpair(xb: *const i8, k: usize) -> __m512i {
        let lo = _mm256_loadu_si256(xb.add(k * 32) as *const __m256i);
        let hi = _mm256_loadu_si256(xb.add(BLOCK + k * 32) as *const __m256i);
        _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1)
    }

    #[target_feature(enable = "avx512f,avx512bw,avx2")]
    pub unsafe fn codes_dot_avx512bw<const T: usize>(
        row: &[u8],
        x: &[i8],
        stride: usize,
        acc: &mut [i32; T],
    ) {
        let nb = row.len() / BLOCK_BYTES;
        assert!(x.len() >= (T - 1) * stride + nb * BLOCK);
        let ones = _mm512_set1_epi16(1);
        let mut a = [_mm512_setzero_si512(); T];
        let rp = row.as_ptr();
        let xp = x.as_ptr();
        let pairs = nb / 2;
        for bb in 0..pairs {
            let c = decode512(rp.add(bb * 2 * BLOCK_BYTES));
            for t in 0..T {
                let xb = xp.add(t * stride + bb * 2 * BLOCK);
                let s0 = _mm512_maddubs_epi16(c[0], xpair(xb, 0));
                let s1 = _mm512_maddubs_epi16(c[1], xpair(xb, 1));
                let s2 = _mm512_maddubs_epi16(c[2], xpair(xb, 2));
                let s3 = _mm512_maddubs_epi16(c[3], xpair(xb, 3));
                let s = _mm512_add_epi16(_mm512_add_epi16(s0, s1), _mm512_add_epi16(s2, s3));
                a[t] = _mm512_add_epi32(a[t], _mm512_madd_epi16(s, ones));
            }
        }
        let mut tail = [_mm256_setzero_si256(); T];
        if nb % 2 == 1 {
            let b = nb - 1;
            let c = decode256(rp.add(b * BLOCK_BYTES));
            for t in 0..T {
                tail[t] = block_avx2(&c, xp.add(t * stride + b * BLOCK));
            }
        }
        for t in 0..T {
            acc[t] = _mm512_reduce_add_epi32(a[t]) + hsum256(tail[t]);
        }
    }

    /// Codes of lane group `S/2` (S = 0, 2, 4, 6) from 64 raw packed bytes.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn codes<const S: u32>(raw: __m512i) -> __m512i {
        _mm512_and_si512(_mm512_srli_epi16::<S>(raw), _mm512_set1_epi8(3))
    }

    /// `acc[r][t] = sum_k code[r][k] * x[t0 + t, k]` (code dot, before the `-sum(x)`
    /// correction) for `R` rows x `T` tokens. Codes are decoded once per row and each
    /// activation load serves all `R` rows; `maddubs` pair sums are accumulated in i16 over
    /// up to 8 block pairs (bound below) before widening to i32.
    #[target_feature(enable = "avx512f,avx512bw,avx2")]
    pub unsafe fn tile_avx512bw<const R: usize, const T: usize>(
        rws: &[&[u8]; 2],
        x: &QuantActs,
        p: &PairedActs,
        t0: usize,
        acc: &mut [[i32; 4]; 2],
    ) {
        let pairs = p.pairs;
        let nb = x.kp / BLOCK;
        for r in 0..R {
            assert_eq!(rws[r].len(), nb * BLOCK_BYTES);
        }
        assert!(t0 + T <= x.m && p.data.len() >= (t0 + T) * p.stride);
        let ones = _mm512_set1_epi16(1);
        let xp = p.data.as_ptr().add(t0 * p.stride);
        let mut a = [[_mm512_setzero_si512(); T]; R];
        // i16 partial sums are widened to i32 every PAIRS_PER_WIDEN block pairs: a pair
        // adds at most 4 * 2 * 3 * 127 = 3048 per lane (even for the never-emitted code
        // 3), so 8 pairs stay below 32767 and one `madd` serves 8 pairs instead of one.
        const PAIRS_PER_WIDEN: usize = 8;
        let mut bb0 = 0;
        while bb0 < pairs {
            let bb1 = (bb0 + PAIRS_PER_WIDEN).min(pairs);
            let mut s = [[_mm512_setzero_si512(); T]; R];
            for bb in bb0..bb1 {
                let mut raw = [_mm512_setzero_si512(); R];
                for r in 0..R {
                    raw[r] = _mm512_loadu_si512(rws[r].as_ptr().add(bb * 2 * BLOCK_BYTES) as *const __m512i);
                }
                let xb = xp.add(bb * 256);
                macro_rules! group {
                    ($k:expr, $shift:expr) => {
                        let mut c = [_mm512_setzero_si512(); R];
                        for r in 0..R {
                            c[r] = codes::<$shift>(raw[r]);
                        }
                        for t in 0..T {
                            let xv = _mm512_loadu_si512(xb.add(t * p.stride + $k * 64) as *const __m512i);
                            for r in 0..R {
                                s[r][t] = _mm512_add_epi16(s[r][t], _mm512_maddubs_epi16(c[r], xv));
                            }
                        }
                    };
                }
                group!(0, 0);
                group!(1, 2);
                group!(2, 4);
                group!(3, 6);
            }
            for r in 0..R {
                for t in 0..T {
                    a[r][t] = _mm512_add_epi32(a[r][t], _mm512_madd_epi16(s[r][t], ones));
                }
            }
            bb0 = bb1;
        }
        for r in 0..R {
            for t in 0..T {
                let mut v = _mm512_reduce_add_epi32(a[r][t]);
                if nb % 2 == 1 {
                    let b = nb - 1;
                    let c = decode256(rws[r].as_ptr().add(b * BLOCK_BYTES));
                    v += hsum256(block_avx2(&c, x.q.as_ptr().add((t0 + t) * x.kp + b * BLOCK)));
                }
                acc[r][t] = v;
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx2")]
    pub unsafe fn codes_dot_avx512vnni<const T: usize>(
        row: &[u8],
        x: &[i8],
        stride: usize,
        acc: &mut [i32; T],
    ) {
        let nb = row.len() / BLOCK_BYTES;
        assert!(x.len() >= (T - 1) * stride + nb * BLOCK);
        let mut a = [_mm512_setzero_si512(); T];
        let rp = row.as_ptr();
        let xp = x.as_ptr();
        let pairs = nb / 2;
        for bb in 0..pairs {
            let c = decode512(rp.add(bb * 2 * BLOCK_BYTES));
            for t in 0..T {
                let xb = xp.add(t * stride + bb * 2 * BLOCK);
                let mut s = a[t];
                s = _mm512_dpbusd_epi32(s, c[0], xpair(xb, 0));
                s = _mm512_dpbusd_epi32(s, c[1], xpair(xb, 1));
                s = _mm512_dpbusd_epi32(s, c[2], xpair(xb, 2));
                s = _mm512_dpbusd_epi32(s, c[3], xpair(xb, 3));
                a[t] = s;
            }
        }
        let mut tail = [_mm256_setzero_si256(); T];
        if nb % 2 == 1 {
            let b = nb - 1;
            let c = decode256(rp.add(b * BLOCK_BYTES));
            for t in 0..T {
                tail[t] = block_avx2(&c, xp.add(t * stride + b * BLOCK));
            }
        }
        for t in 0..T {
            acc[t] = _mm512_reduce_add_epi32(a[t]) + hsum256(tail[t]);
        }
    }
}

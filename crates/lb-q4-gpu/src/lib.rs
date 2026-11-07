//! Q4_0 matvec on the GPU (CubeCL, wgpu runtime): `out[t, r] = sum_c W[r, c] * x[t, c]`
//! with `W` in GGUF Q4_0 blocks (32 values: f16 scale `d`, 16 nibble bytes; low nibble =
//! value j, high nibble = value j + 16, each minus 8). Same numbers as
//! `lb_gguf::matvec_f32` up to f32 summation order.
//!
//! Device layout (repacked once at upload, see [`repack_q4_0`]): GGUF blocks are 18 bytes
//! and never 4-byte aligned, so the matrix is split into two aligned arrays.
//!   * `qs`: 4 u32 words per block, read as one `Vector<u32, 4>` (16-byte load). Nibble `j`
//!     (bits 4j..4j+3) of word `k` holds value `4j + k` of the block, so
//!     `(qs >> 4j) & 15` is exactly the quants of x's 4-wide vector number `j`.
//!   * `sc`: the f16 scales, two per u32, decoded in the kernel.
//!
//! One plane (warp/wave/subgroup) computes one output element: its lanes take consecutive
//! blocks (coalesced 16-byte loads) and `plane_sum` reduces. A cube holds several planes, so
//! it produces several output rows; the plane size is read at run time (`PLANE_DIM`), so the
//! same kernel is right for wave32 and wave64. Devices without subgroup ops fall back to one
//! output per cube with a shared-memory tree.

pub mod decode;
pub mod store;
pub use decode::TokenParams;
pub use store::{pick_cube_device, Gpu, GpuConv, GpuExpert, GpuExpertLayer, GpuQ4Mat, GpuQ6KMat};
#[cfg(feature = "cuda")]
pub mod pinned;
#[cfg(feature = "cuda")]
pub mod registered;
pub use cubecl::server as cubecl_server;

use cubecl::prelude::*;
use lb_gguf::QuantMat;

type V4F = Vector<f32, Const<4>>;
type V4U = Vector<u32, Const<4>>;

/// f16 bits -> f32, exactly: normal numbers are rebuilt bit for bit (exponent rebased
/// 15 -> 127, mantissa widened 10 -> 23 bits); subnormals are `man * 2^-24`.
#[cube]
fn f16_to_f32(bits: u32) -> f32 {
    let sign = (bits >> 15u32) & 1u32;
    let exp = (bits >> 10u32) & 31u32;
    let man = bits & 1023u32;
    let mut v = 0.0f32;
    if exp == 0u32 {
        v = man as f32 * 5.9604645e-8f32;
        if sign == 1u32 {
            v = -v;
        }
    } else {
        let b = (sign << 31u32) | ((exp + 112u32) << 23u32) | (man << 13u32);
        v = f32::reinterpret(b);
    }
    v
}

/// This lane's partial dot of one row (`bpr` blocks starting at global block `gb0`) with the
/// activation row starting at `x4_0` (in 4-wide vectors): blocks `lane, lane + width, ...`.
#[cube]
fn q4_row_partial(qs: &[V4U], sc: &[u32], x: &[V4F], gb0: usize, x4_0: usize, bpr: usize, lane: usize, width: usize) -> f32 {
    let mut acc = V4F::new(0.0f32);
    let mut b = lane;
    while b < bpr {
        let gb = gb0 + b;
        let q = qs[gb];
        let d = f16_to_f32((sc[gb / 2usize] >> (((gb % 2usize) * 16usize) as u32)) & 65535u32);
        let xb = x4_0 + b * 8usize;
        let mut s = V4F::new(0.0f32);
        #[unroll]
        for j in 0..8u32 {
            let nib = (q >> V4U::new(4u32 * j)) & V4U::new(15u32);
            let f = V4F::cast_from(nib) - V4F::new(8.0f32);
            s += f * x[xb + j as usize];
        }
        acc += V4F::new(d) * s;
        b += width;
    }
    acc.extract(0usize) + acc.extract(1usize) + acc.extract(2usize) + acc.extract(3usize)
}

/// Sum a per-lane partial across the lanes working on one output; valid in every lane.
/// Plane path: `plane_sum`. Fallback: the whole cube is one output, shared-memory tree.
#[cube]
fn lane_sum(sm: &mut Shared<[f32]>, v: f32, #[comptime] use_plane: bool) -> f32 {
    let mut r = v;
    if use_plane {
        r = plane_sum(v);
    } else {
        sm[UNIT_POS as usize] = v;
        sync_cube();
        let mut s = CUBE_DIM_X / 2u32;
        while s > 0u32 {
            if UNIT_POS < s {
                sm[UNIT_POS as usize] += sm[(UNIT_POS + s) as usize];
            }
            sync_cube();
            s = s / 2u32;
        }
        r = sm[0usize];
        sync_cube();
    }
    r
}

/// `dims` = [rows, cols]. One plane (or, in the fallback, one cube) per output element
/// `o = t * rows + r`, grid-strided so the result is right for any launched cube count.
#[cube(launch)]
fn q4_0_matvec_kernel(qs: &[V4U], sc: &[u32], x: &[V4F], dims: &[u32], out: &mut [f32], #[comptime] use_plane: bool) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let bpr = cols / 32usize;
    let mut sm = Shared::<[f32]>::new_slice(256usize);
    let mut width = CUBE_DIM_X as usize;
    let mut plane_id = 0usize;
    let mut lane = UNIT_POS as usize;
    if use_plane {
        width = PLANE_DIM as usize;
        plane_id = (UNIT_POS / PLANE_DIM) as usize;
        lane = UNIT_POS_PLANE as usize;
    }
    let ppc = CUBE_DIM_X as usize / width;
    let step = CUBE_COUNT as usize * ppc;
    let mut o = CUBE_POS as usize * ppc + plane_id;
    while o < out.len() {
        let t = o / rows;
        let r = o % rows;
        let part = q4_row_partial(qs, sc, x, r * bpr, t * cols / 4usize, bpr, lane, width);
        let total = lane_sum(&mut sm, part, use_plane);
        if lane == 0usize {
            out[o] = total;
        }
        o += step;
    }
}

/// Fused SwiGLU front half: `h[t, r] = silu(x_t . gate_r) * (x_t . up_r)`. Gate and up have
/// the same shape (`dims`), so one dispatch reads both and the `g`/`u` buffers never exist.
#[cube(launch)]
fn q4_0_gate_up_silu_kernel(
    gq: &[V4U],
    gs: &[u32],
    uq: &[V4U],
    us: &[u32],
    x: &[V4F],
    dims: &[u32],
    h: &mut [f32],
    #[comptime] use_plane: bool,
) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let bpr = cols / 32usize;
    let mut sm = Shared::<[f32]>::new_slice(256usize);
    let mut width = CUBE_DIM_X as usize;
    let mut plane_id = 0usize;
    let mut lane = UNIT_POS as usize;
    if use_plane {
        width = PLANE_DIM as usize;
        plane_id = (UNIT_POS / PLANE_DIM) as usize;
        lane = UNIT_POS_PLANE as usize;
    }
    let ppc = CUBE_DIM_X as usize / width;
    let step = CUBE_COUNT as usize * ppc;
    let mut o = CUBE_POS as usize * ppc + plane_id;
    while o < h.len() {
        let t = o / rows;
        let r = o % rows;
        let pg = q4_row_partial(gq, gs, x, r * bpr, t * cols / 4usize, bpr, lane, width);
        let pu = q4_row_partial(uq, us, x, r * bpr, t * cols / 4usize, bpr, lane, width);
        let g = lane_sum(&mut sm, pg, use_plane);
        let u = lane_sum(&mut sm, pu, use_plane);
        if lane == 0usize {
            h[o] = g / (1.0f32 + Exp::exp(-g)) * u;
        }
        o += step;
    }
}

/// This lane's partial dot of one Q6_K row (`gpr` groups of 16 from global group `gg0`).
///
/// Mirrors [`q4_row_partial`]: byte `j` of word `k` of a group holds value `4j + k`, so a
/// vector-wide `(w >> 8j) & 255` lands exactly on x's 4-wide vector `j`. The group scale is
/// already `d * sc[g]` (see [`repack_q6_k`]), so there is no f16 decode in the inner loop.
#[cube]
fn q6_row_partial(qv: &[V4U], sd: &[f32], x: &[V4F], gg0: usize, x4_0: usize, gpr: usize, lane: usize, width: usize) -> f32 {
    let mut acc = V4F::new(0.0f32);
    let mut g = lane;
    while g < gpr {
        let gg = gg0 + g;
        let w = qv[gg];
        let s = sd[gg];
        let xb = x4_0 + g * 4usize;
        let mut part = V4F::new(0.0f32);
        #[unroll]
        for j in 0..4u32 {
            let q = (w >> V4U::new(8u32 * j)) & V4U::new(255u32);
            let f = V4F::cast_from(q) - V4F::new(32.0f32);
            part += f * x[xb + j as usize];
        }
        acc += V4F::new(s) * part;
        g += width;
    }
    acc.extract(0usize) + acc.extract(1usize) + acc.extract(2usize) + acc.extract(3usize)
}

/// Q6_K matvec. `dims` = [rows, cols]; one plane (or cube, in the fallback) per output.
#[cube(launch)]
fn q6_k_matvec_kernel(qv: &[V4U], sd: &[f32], x: &[V4F], dims: &[u32], out: &mut [f32], #[comptime] use_plane: bool) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let gpr = cols / 16usize;
    let mut sm = Shared::<[f32]>::new_slice(256usize);
    let mut width = CUBE_DIM_X as usize;
    let mut plane_id = 0usize;
    let mut lane = UNIT_POS as usize;
    if use_plane {
        width = PLANE_DIM as usize;
        plane_id = (UNIT_POS / PLANE_DIM) as usize;
        lane = UNIT_POS_PLANE as usize;
    }
    let ppc = CUBE_DIM_X as usize / width;
    let step = CUBE_COUNT as usize * ppc;
    let mut o = CUBE_POS as usize * ppc + plane_id;
    while o < out.len() {
        let t = o / rows;
        let r = o % rows;
        let part = q6_row_partial(qv, sd, x, r * gpr, t * cols / 4usize, gpr, lane, width);
        let total = lane_sum(&mut sm, part, use_plane);
        if lane == 0usize {
            out[o] = total;
        }
        o += step;
    }
}

/// The whole middle of an LFM2 short-conv block, for one token, on the device.
///
/// `in_proj` and `out_proj` are both resident matmuls, but everything between them was Burn
/// CPU tensor ops, so each conv layer paid two host syncs -- and 30 of the model's 40 layers
/// are conv layers, which is 60 of a token's ~145 syncs. Measured, a sync costs ~0.24 ms more
/// than it needs to (see `dense_bench`'s chain row), and the arithmetic here is 5 flops per
/// channel: it is pure latency.
///
/// Per channel `i` of `e`, with `bcx = in_proj(h)` laid out `[b | c | x]`:
/// ```text
/// bx   = b[i] * x[i]
/// acc  = sum_k conv[k][i] * z[k]      where z = [state rows ..., bx]
/// y[i] = c[i] * acc
/// state <- z[1..]                      (the l-1 most recent rows)
/// ```
/// `state` is read and rewritten in place, so it stays on the device across tokens and never
/// costs a transfer. Unit `i` touches only indices `i, e+i, ...`, so there is no cross-unit
/// dependency and no barrier is needed; the state shift ascends, so each row is read before
/// it is overwritten.
///
/// One token only (`m = 1`). Prefill keeps the CPU path: it is a different, cheaper problem
/// (one pass over the whole prompt) and is not the bottleneck.
#[cube(launch)]
fn conv_decode_kernel(bcx: &[f32], conv: &[f32], state: &mut [f32], y: &mut [f32], dims: &[u32]) {
    let e = dims[0] as usize;
    let l = dims[1] as usize;
    let mut i = ABSOLUTE_POS as usize;
    let stride = (CUBE_COUNT as usize) * (CUBE_DIM_X as usize);
    while i < e {
        let bx = bcx[i] * bcx[2usize * e + i];
        let c = bcx[e + i];
        let mut acc = 0.0f32;
        let mut k = 0usize;
        while k < l - 1usize {
            acc += conv[k * e + i] * state[k * e + i];
            k += 1usize;
        }
        acc += conv[(l - 1usize) * e + i] * bx;
        y[i] = c * acc;
        // drop the oldest row, append bx
        let mut k = 0usize;
        while k + 2usize < l {
            state[k * e + i] = state[(k + 1usize) * e + i];
            k += 1usize;
        }
        state[(l - 2usize) * e + i] = bx;
        i += stride;
    }
}

/// Repack GGUF Q6_K super-blocks into the device layout.
///
/// A super-block is 256 values in 210 bytes: `ql[128]` low nibbles, `qh[64]` two high bits
/// each, `scales[16]` i8, then the f16 `d`. Three things make that unusable as it stands:
/// it is not 4-byte aligned, the 6 bits of a value are split across two arrays at indices
/// that depend on which quarter of a half it is in, and the scale lookup in the reference
/// reads as if it needed the same case analysis.
///
/// It does not: for value `v` the scale index works out to exactly `v / 16`, so the 256
/// values are 16 contiguous groups of 16, each with one scale. So the repack writes
///   * `qv`: 4 u32 words per group — one `Vector<u32, 4>`, 16 values, the 6-bit quants
///     widened to a byte each and left unsigned (0..63; the kernel subtracts 32)
///   * `sd`: one f32 per group, already `d * scales[g]`, so `d` never reaches the kernel
///
/// 320 bytes per super-block against the file's 210 (1.52x: the tied head goes 215 MB to
/// 327 MB on the device), in exchange for an inner loop with no bit surgery in it.
pub fn repack_q6_k(bytes: &[u8], n_super: usize) -> (Vec<u32>, Vec<f32>) {
    use rayon::prelude::*;
    let mut qv = vec![0u32; n_super * 64];
    let mut sd = vec![0f32; n_super * 16];
    qv.par_chunks_mut(64).zip(sd.par_chunks_mut(16)).enumerate().for_each(|(b, (qv, sd))| {
        let blk = &bytes[b * 210..(b + 1) * 210];
        let d = half::f16::from_le_bytes([blk[208], blk[209]]).to_f32();
        for (g, s) in sd.iter_mut().enumerate() {
            *s = d * (blk[192 + g] as i8) as f32;
        }
        for v in 0..256usize {
            let (h, r) = (v / 128, v % 128);
            let (sub, l) = (r / 32, r % 32);
            let ql = &blk[64 * h..];
            let qh = &blk[128 + 32 * h..];
            let q = match sub {
                0 => (ql[l] & 0x0f) | ((qh[l] & 3) << 4),
                1 => (ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4),
                2 => (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4),
                _ => (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4),
            } as u32;
            let (g, u) = (v / 16, v % 16);
            qv[g * 4 + u % 4] |= q << (8 * (u / 4) as u32);
        }
    });
    (qv, sd)
}

/// Default cube size (units) when the device is not asked: see `Gpu::open`.
pub const THREADS: usize = 64;

/// Blocks per repack work unit. Even, so an `sc` word (two blocks) never straddles two units.
/// 2048 blocks = 36 KB in, 32 KB out: with the SIMD inner loop below a unit is ~10 us of work,
/// so a 1536x2048 matrix (98,304 blocks) is 48 units, enough to spread and not so many that
/// rayon's per-task cost shows (measured on c2, `examples/repack_bench.rs`).
const REPACK_CHUNK: usize = 2048;

/// Repack GGUF Q4_0 rows into the device layout (`qs` words, packed f16 `sc` words).
/// `bytes` is the unchanged GGUF data: `n_blocks` blocks of 18 bytes.
///
/// Parallel over blocks, because this is on the critical path twice. It runs for every
/// resident expert at load (2066 of them on a 12 GB budget, 23.5 ms each single-threaded
/// was 49 s of a 147 s load), and an on-demand upload of a non-resident expert has to pay
/// it before the copy, where it was over half the cost of serving the miss.
pub fn repack_q4_0(bytes: &[u8], n_blocks: usize) -> (Vec<u32>, Vec<u32>) {
    let mut qs = vec![0u32; n_blocks * 4];
    let mut sc = vec![0u32; n_blocks.div_ceil(2).max(1)];
    repack_q4_0_into(bytes, n_blocks, &mut qs, &mut sc);
    (qs, sc)
}

/// `repack_q4_0` into caller-owned words: `qs` is `n_blocks * 4` long, `sc` is
/// `ceil(n_blocks / 2).max(1)`. **Every word is assigned**, none is accumulated into, so the
/// slices need no zeroing first and can be page-locked memory the device DMAs from directly
/// (`Gpu::upload_q4_0`): that is the whole point of the `_into` form (see `pinned.rs`).
///
/// The quant words come from [`pack_block`] (one block = one 4x4 byte transpose and two
/// nibble packs); on x86-64 with SSSE3 that is one `pshufb` and a dozen vector ops per block
/// ([`repack_quants_ssse3`]), the scalar form is the same arithmetic in `u32`s and is what the
/// unit test checks both against.
pub fn repack_q4_0_into(bytes: &[u8], n_blocks: usize, qs: &mut [u32], sc: &mut [u32]) {
    use rayon::prelude::*;
    assert_eq!(qs.len(), n_blocks * 4, "qs words");
    assert_eq!(sc.len(), n_blocks.div_ceil(2).max(1), "sc words");
    assert!(bytes.len() >= n_blocks * 18, "{} bytes for {n_blocks} Q4_0 blocks", bytes.len());
    if n_blocks == 0 {
        sc[0] = 0;
        return;
    }
    #[cfg(target_arch = "x86_64")]
    let simd = std::is_x86_feature_detected!("ssse3") && std::env::var_os("LB_REPACK_SCALAR").is_none();
    #[cfg(not(target_arch = "x86_64"))]
    let simd = false;
    qs.par_chunks_mut(4 * REPACK_CHUNK)
        .zip(sc.par_chunks_mut(REPACK_CHUNK / 2))
        .enumerate()
        .for_each(|(ci, (qs, sc))| {
            let b0 = ci * REPACK_CHUNK;
            let nb = qs.len() / 4;
            let src = &bytes[b0 * 18..(b0 + nb) * 18];
            repack_scales(src, nb, sc);
            #[cfg(target_arch = "x86_64")]
            if simd {
                // SAFETY: `ssse3` was detected at run time just above.
                unsafe { repack_quants_ssse3(src, nb, qs) };
                return;
            }
            repack_quants_scalar(src, nb, qs);
        });
}

/// The packed f16 scales of `nb` blocks at `src`: two per word, the odd block in the high
/// half; a missing last block is 0.
fn repack_scales(src: &[u8], nb: usize, sc: &mut [u32]) {
    let scale = |b: usize| u16::from_le_bytes([src[b * 18], src[b * 18 + 1]]) as u32;
    for (j, w) in sc.iter_mut().enumerate() {
        let lo = if 2 * j < nb { scale(2 * j) } else { 0 };
        let hi = if 2 * j + 1 < nb { scale(2 * j + 1) } else { 0 };
        *w = lo | (hi << 16);
    }
}

/// Four nibbles at bits 0, 8, 16, 24 of `x` (the rest zero) -> bits 0, 4, 8, 12.
#[inline(always)]
fn nibbles16(x: u32) -> u32 {
    let x = (x | (x >> 4)) & 0x00ff_00ff;
    (x | (x >> 8)) & 0x0000_ffff
}

/// The 16 nibble bytes of one block, as four little-endian words (`a[j]` = bytes 4j..4j+4,
/// so byte k of `a[j]` is value 4j+k) -> the block's 4 device words. Word k gathers byte k
/// of every `a[j]` (a 4x4 byte transpose), then packs those bytes' low nibbles into its low
/// half and their high nibbles into its high half: value 4j+k at nibble j, value 4j+k+16
/// at nibble j+4, which is the layout the kernels read (module doc).
#[inline(always)]
fn pack_block(a: [u32; 4]) -> [u32; 4] {
    let mut w = [0u32; 4];
    for (k, w) in w.iter_mut().enumerate() {
        let col = |j: usize| (a[j] >> (8 * k)) & 0xff;
        let t = col(0) | (col(1) << 8) | (col(2) << 16) | (col(3) << 24);
        *w = nibbles16(t & 0x0f0f_0f0f) | (nibbles16((t >> 4) & 0x0f0f_0f0f) << 16);
    }
    w
}

fn repack_quants_scalar(src: &[u8], nb: usize, qs: &mut [u32]) {
    for (b, out) in qs.chunks_exact_mut(4).enumerate().take(nb) {
        let q = &src[b * 18 + 2..b * 18 + 18];
        let a = [
            u32::from_le_bytes([q[0], q[1], q[2], q[3]]),
            u32::from_le_bytes([q[4], q[5], q[6], q[7]]),
            u32::from_le_bytes([q[8], q[9], q[10], q[11]]),
            u32::from_le_bytes([q[12], q[13], q[14], q[15]]),
        ];
        out.copy_from_slice(&pack_block(a));
    }
}

/// [`pack_block`] on 128-bit vectors: the 16 nibble bytes are one unaligned load, the 4x4
/// byte transpose is one `pshufb`, and the two nibble packs run on all four words at once.
///
/// # Safety
/// The caller has checked that SSSE3 is available. `src` holds `nb` whole 18-byte blocks and
/// `qs` at least `4 * nb` words (the slice bounds are asserted; the loads never pass `src`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn repack_quants_ssse3(src: &[u8], nb: usize, qs: &mut [u32]) {
    use std::arch::x86_64::*;
    assert!(src.len() >= nb * 18 && qs.len() >= nb * 4);
    // lane k <- bytes k, 4+k, 8+k, 12+k (byte k of each a[j], low to high)
    let shuf = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    let m_nib = _mm_set1_epi32(0x0f0f_0f0f);
    let m_pair = _mm_set1_epi32(0x00ff_00ff);
    let m_half = _mm_set1_epi32(0x0000_ffff);
    for b in 0..nb {
        // SAFETY: bytes b*18+2 .. b*18+18 are inside `src` (asserted above); qs[4b..4b+4] inside `qs`.
        unsafe {
            let v = _mm_loadu_si128(src.as_ptr().add(b * 18 + 2) as *const __m128i);
            let t = _mm_shuffle_epi8(v, shuf);
            let lo = _mm_and_si128(t, m_nib);
            let hi = _mm_and_si128(_mm_srli_epi32(t, 4), m_nib);
            let lo = _mm_and_si128(_mm_or_si128(lo, _mm_srli_epi32(lo, 4)), m_pair);
            let lo = _mm_and_si128(_mm_or_si128(lo, _mm_srli_epi32(lo, 8)), m_half);
            let hi = _mm_and_si128(_mm_or_si128(hi, _mm_srli_epi32(hi, 4)), m_pair);
            let hi = _mm_and_si128(_mm_or_si128(hi, _mm_srli_epi32(hi, 8)), m_half);
            let w = _mm_or_si128(lo, _mm_slli_epi32(hi, 16));
            _mm_storeu_si128(qs.as_mut_ptr().add(b * 4) as *mut __m128i, w);
        }
    }
}

/// [`pack_block`] on the device: `a0..a3` are the 16 nibble bytes of one block as
/// little-endian words, the result is the block's device word `k`.
#[cube]
fn pack_word(a0: u32, a1: u32, a2: u32, a3: u32, k: u32) -> u32 {
    let s = 8u32 * k;
    let t = ((a0 >> s) & 0xffu32) | (((a1 >> s) & 0xffu32) << 8u32) | (((a2 >> s) & 0xffu32) << 16u32) | (((a3 >> s) & 0xffu32) << 24u32);
    let lo = t & 0x0f0f_0f0fu32;
    let lo = (lo | (lo >> 4u32)) & 0x00ff_00ffu32;
    let lo = (lo | (lo >> 8u32)) & 0x0000_ffffu32;
    let hi = (t >> 4u32) & 0x0f0f_0f0fu32;
    let hi = (hi | (hi >> 4u32)) & 0x00ff_00ffu32;
    let hi = (hi | (hi >> 8u32)) & 0x0000_ffffu32;
    lo | (hi << 16u32)
}

/// Units per cube for [`q4_0_repack_kernel`].
pub const REPACK_CUBE: usize = 256;

/// Repack raw GGUF Q4_0 blocks that are already on the device. `raw` is the file's bytes
/// as u32 words: two 18-byte blocks are exactly 9 words, so an even block count is 4-byte
/// aligned and one unit handles one pair. Block `2p` has its f16 scale in the low half of
/// word `9p` and its nibble bytes straddling words `9p..9p+5` by a half word; block `2p+1`
/// has its scale in the high half of word `9p+4` and its nibble bytes exactly in words
/// `9p+5..9p+9`. `n[0]` is the pair count. Output is what [`repack_q4_0_into`] writes.
///
/// This is how a miss is served with no host work at all: the file mapping is page-locked
/// once (`Gpu::register_mapping`), the driver DMAs the expert's raw bytes straight out of
/// the page cache, and the device does the repack (measured 2026-11-03 on c2, `docs/WORK_LOG.md`).
#[cube(launch)]
fn q4_0_repack_kernel(raw: &[u32], qs: &mut [u32], sc: &mut [u32], n: &[u32]) {
    let p = CUBE_POS * CUBE_DIM_X as usize + UNIT_POS_X as usize;
    if p < n[0] as usize {
        let w0 = raw[9 * p];
        let w1 = raw[9 * p + 1];
        let w2 = raw[9 * p + 2];
        let w3 = raw[9 * p + 3];
        let w4 = raw[9 * p + 4];
        let w5 = raw[9 * p + 5];
        let w6 = raw[9 * p + 6];
        let w7 = raw[9 * p + 7];
        let w8 = raw[9 * p + 8];
        let a0 = (w0 >> 16u32) | (w1 << 16u32);
        let a1 = (w1 >> 16u32) | (w2 << 16u32);
        let a2 = (w2 >> 16u32) | (w3 << 16u32);
        let a3 = (w3 >> 16u32) | (w4 << 16u32);
        sc[p] = (w0 & 0xffffu32) | (w4 & 0xffff_0000u32);
        #[unroll]
        for k in 0..4u32 {
            qs[8 * p + k as usize] = pack_word(a0, a1, a2, a3, k);
            qs[8 * p + 4 + k as usize] = pack_word(w5, w6, w7, w8, k);
        }
    }
}

#[cfg(test)]
mod repack_tests {
    use super::*;

    /// The accumulate-into-zeros form `repack_q4_0` had before `_into` existed, kept as the
    /// reference the assigning form must match word for word.
    fn reference(bytes: &[u8], n_blocks: usize) -> (Vec<u32>, Vec<u32>) {
        let mut qs = vec![0u32; n_blocks * 4];
        let mut sc = vec![0u32; n_blocks.div_ceil(2).max(1)];
        for b in 0..n_blocks {
            let blk = &bytes[b * 18..b * 18 + 18];
            sc[b / 2] |= (u16::from_le_bytes([blk[0], blk[1]]) as u32) << (16 * (b % 2));
            let w = &mut qs[b * 4..b * 4 + 4];
            for i in 0..16usize {
                let byte = blk[2 + i] as u32;
                w[i % 4] |= ((byte & 15) << (4 * (i / 4))) | ((byte >> 4) << (4 * (i / 4) + 16));
            }
        }
        (qs, sc)
    }

    #[test]
    fn assigning_repack_matches_reference() {
        // odd and even counts, under, at and past one REPACK_CHUNK and its sc half-chunk
        for &n in &[1usize, 2, 3, 31, 1023, 1024, 1025, 2047, 2049, 4096 + 7] {
            let mut x = 0x9E37_79B9u32;
            let bytes: Vec<u8> = (0..n * 18)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 17;
                    x ^= x << 5;
                    (x >> 24) as u8
                })
                .collect();
            let (rq, rs) = reference(&bytes, n);
            let (q, s) = repack_q4_0(&bytes, n);
            assert_eq!(q, rq, "qs differ at n_blocks={n}");
            assert_eq!(s, rs, "sc differ at n_blocks={n}");
            // and into a slice that starts dirty, since pinned slots are reused
            let mut dq = vec![0xFFFF_FFFFu32; n * 4];
            let mut ds = vec![0xFFFF_FFFFu32; n.div_ceil(2).max(1)];
            repack_q4_0_into(&bytes, n, &mut dq, &mut ds);
            assert_eq!(dq, rq, "qs into dirty slice differ at n_blocks={n}");
            assert_eq!(ds, rs, "sc into dirty slice differ at n_blocks={n}");
            // the scalar inner loop, whatever the host supports (the SIMD one is what
            // `repack_q4_0` took above on an SSSE3 host)
            let mut sq = vec![0xFFFF_FFFFu32; n * 4];
            repack_quants_scalar(&bytes, n, &mut sq);
            assert_eq!(sq, rq, "scalar qs differ at n_blocks={n}");
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("ssse3") {
                let mut vq = vec![0xFFFF_FFFFu32; n * 4];
                unsafe { repack_quants_ssse3(&bytes, n, &mut vq) };
                assert_eq!(vq, rq, "ssse3 qs differ at n_blocks={n}");
            }
        }
    }
}

/// Cube count for `n` outputs given `per_cube` outputs per cube: wgpu caps each dispatch
/// dimension at 65535, so the cubes are laid out in two dimensions and the kernel reads the
/// linear `CUBE_POS`.
pub fn cube_count(n: usize, per_cube: usize) -> CubeCount {
    let cubes = n.div_ceil(per_cube.max(1)).max(1);
    let x = cubes.min(65535);
    let y = cubes.div_ceil(x).max(1);
    CubeCount::Static(x as u32, y as u32, 1)
}

/// `dims` for the kernels: [rows, cols].
pub fn matvec_dims(rows: usize, cols: usize) -> [u32; 2] {
    [rows as u32, cols as u32]
}

/// `x`: `m` rows of `w.cols` f32. Returns `m * w.rows` values, token-major like the CPU kernel.
pub fn q4_0_matvec_gpu(w: QuantMat<'_>, x: &[f32], m: usize) -> Result<Vec<f32>, String> {
    if w.ty != lb_gguf::GgmlType::Q4_0 {
        return Err(format!("expected Q4_0, got {}", w.ty.name()));
    }
    if x.len() != m * w.cols {
        return Err(format!("x has {} values, expected {} x {}", x.len(), m, w.cols));
    }
    if m * w.rows == 0 {
        return Ok(Vec::new());
    }
    let gpu = Gpu::open()?;
    GpuQ4Mat::upload(&gpu, w)?.matvec(&gpu, x, m)
}

// ---- Seeds for forge lanes (bodies to be written; tests in tests/q6k_matches_scalar.rs and
// tests/q8_0_matches_scalar.rs are `#[ignore]` until then: run with `--include-ignored`
// on a GPU host, never on t440). ----

/// Q8_0 matvec on the GPU: blocks of 32 values, f16 scale `d` then 32 signed bytes
/// (`value = d * q`), row bytes `cols / 32 * 34`. Same contract as [`q4_0_matvec_gpu`].
pub fn q8_0_matvec_gpu(w: QuantMat<'_>, x: &[f32], m: usize) -> Result<Vec<f32>, String> {
    if w.ty != lb_gguf::GgmlType::Q8_0 {
        return Err(format!("expected Q8_0, got {}", w.ty.name()));
    }
    let _ = (x, m);
    Err("q8_0_matvec_gpu: not implemented yet (forge seed)".to_string())
}

/// Q6_K matvec on the GPU: super-blocks of 256 values, 210 bytes (`ql[128]`, `qh[64]`,
/// `scales[16]` i8, `d` f16); see `lb_gguf::types::dequant_q6_k_block` for the exact
/// bit layout. Row bytes `cols / 256 * 210`. Same contract as [`q4_0_matvec_gpu`].
pub fn q6_k_matvec_gpu(w: QuantMat<'_>, x: &[f32], m: usize) -> Result<Vec<f32>, String> {
    if w.ty != lb_gguf::GgmlType::Q6K {
        return Err(format!("expected Q6_K, got {}", w.ty.name()));
    }
    if x.len() != m * w.cols {
        return Err(format!("x has {} values, expected {} x {}", x.len(), m, w.cols));
    }
    if m * w.rows == 0 {
        return Ok(Vec::new());
    }
    let gpu = Gpu::open()?;
    GpuQ6KMat::upload(&gpu, w)?.matvec(&gpu, x, m)
}

/// One dispatch for a whole MoE layer's gate+up+silu, over experts concatenated into one buffer.
///
/// The per-expert kernel costs a dispatch per expert, and a dispatch on a loaded device measures
/// **0.108 ms** -- so a top-4 layer pays 8 of them, 38 layers pay 304, and that is 42% of a
/// token's matmul budget. Here all 64 experts of a layer live in one `gq`/`uq` buffer and the
/// routing comes in as data, so the layer is one dispatch whatever the top-k.
///
/// `routes` holds two u32 per slot: `[expert_id, token_index]`. A slot is one (expert, token)
/// pair the router chose, so `h` is `[slots, rows]` and output `o` maps to slot `o / rows`,
/// row `o % rows`. Expert `e`'s rows start at block `e * rows * bpr`. `dims` = `[rows, cols,
/// base]`: the launch's slots start at slot `base` of `routes`, so a device-resident decode can
/// keep every layer's routes in one buffer and read them all back with the logits.
///
/// Concatenation is only sound because `q4_row_partial` reads the f16 scale at `sc[gb / 2]` from
/// the GLOBAL block index: each expert's block count must therefore be even, or an expert
/// boundary would land mid-u32 and shift every scale after it. `GpuExpertLayer::upload` asserts it.
#[cube(launch)]
fn q4_0_routed_gate_up_silu_kernel(
    gq: &[V4U],
    gs: &[u32],
    uq: &[V4U],
    us: &[u32],
    x: &[V4F],
    routes: &[u32],
    dims: &[u32],
    h: &mut [f32],
    #[comptime] use_plane: bool,
) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let bpr = cols / 32usize;
    let bpe = rows * bpr;
    // first slot of this launch in `routes` (a per-token buffer holds every layer's slots)
    let rbase = dims[2] as usize;
    let mut sm = Shared::<[f32]>::new_slice(256usize);
    let mut width = CUBE_DIM_X as usize;
    let mut plane_id = 0usize;
    let mut lane = UNIT_POS as usize;
    if use_plane {
        width = PLANE_DIM as usize;
        plane_id = (UNIT_POS / PLANE_DIM) as usize;
        lane = UNIT_POS_PLANE as usize;
    }
    let ppc = CUBE_DIM_X as usize / width;
    let step = CUBE_COUNT as usize * ppc;
    let mut o = CUBE_POS as usize * ppc + plane_id;
    while o < h.len() {
        let slot = o / rows;
        let r = o % rows;
        let e = routes[2usize * (rbase + slot)] as usize;
        let tok = routes[2usize * (rbase + slot) + 1usize] as usize;
        let gb0 = e * bpe + r * bpr;
        let x4 = tok * cols / 4usize;
        let pg = q4_row_partial(gq, gs, x, gb0, x4, bpr, lane, width);
        let pu = q4_row_partial(uq, us, x, gb0, x4, bpr, lane, width);
        let g = lane_sum(&mut sm, pg, use_plane);
        let u = lane_sum(&mut sm, pu, use_plane);
        if lane == 0usize {
            h[o] = g / (1.0f32 + Exp::exp(-g)) * u;
        }
        o += step;
    }
}

/// One dispatch for a whole MoE layer's `down` projection, experts concatenated as above.
///
/// Unlike the gate/up pass, the input is the previous kernel's own output, so slot `s` reads
/// `h[s * cols ..]` -- the slot index, not a token index. `routes` is the same buffer.
#[cube(launch)]
fn q4_0_routed_matvec_kernel(
    qs: &[V4U],
    sc: &[u32],
    h: &[V4F],
    routes: &[u32],
    dims: &[u32],
    out: &mut [f32],
    #[comptime] use_plane: bool,
) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let bpr = cols / 32usize;
    let bpe = rows * bpr;
    // first slot of this launch in `routes` (a per-token buffer holds every layer's slots)
    let rbase = dims[2] as usize;
    let mut sm = Shared::<[f32]>::new_slice(256usize);
    let mut width = CUBE_DIM_X as usize;
    let mut plane_id = 0usize;
    let mut lane = UNIT_POS as usize;
    if use_plane {
        width = PLANE_DIM as usize;
        plane_id = (UNIT_POS / PLANE_DIM) as usize;
        lane = UNIT_POS_PLANE as usize;
    }
    let ppc = CUBE_DIM_X as usize / width;
    let step = CUBE_COUNT as usize * ppc;
    let mut o = CUBE_POS as usize * ppc + plane_id;
    while o < out.len() {
        let slot = o / rows;
        let r = o % rows;
        let e = routes[2usize * (rbase + slot)] as usize;
        let p = q4_row_partial(qs, sc, h, e * bpe + r * bpr, slot * cols / 4usize, bpr, lane, width);
        let v = lane_sum(&mut sm, p, use_plane);
        if lane == 0usize {
            out[o] = v;
        }
        o += step;
    }
}

// ---- D2: R rows per plane (docs/DEVICE_DECODE_PLAN.md) ----
//
// The kernels above give each plane one output row: two 16-byte loads a lane (`qs` and its
// scale word) and then the FMAs, so few bytes are in flight per SM and the P100 streams at ~7%
// of HBM2. Here a plane takes `r` consecutive rows (4 or 8, `LB_Q4_ROWS`): per block it issues
// the `r` quant loads, the `r` scale loads and the 8 activation loads **before** any FMA, keeps
// `r` partials, and reduces them together at the end (`plane_sum` on a 4-wide vector, one per 4
// rows) instead of once a row. With `xshared` the activation row (m = 1) is staged in shared
// memory once per cube, since every row reads all of it.
//
// Per row the arithmetic is `q4_row_partial`'s exactly (same blocks per lane, same order), so
// the outputs are the same numbers; only the reduction is batched.

/// `r` row partials of one plane's lane: rows `0..r` start at global blocks `gb0 + i * bpr`, the
/// activation row at vector `x4_0` (in `xs` when `xshared`). Written into `acc[0..r]`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn q4_rows_acc(
    qs: &[V4U],
    sc: &[u32],
    x: &[V4F],
    xs: &Shared<[V4F]>,
    gb0: usize,
    bpr: usize,
    x4_0: usize,
    lane: usize,
    width: usize,
    acc: &mut Array<f32>,
    #[comptime] r: usize,
    #[comptime] xshared: bool,
) {
    let mut accv = Array::<V4F>::new(r);
    #[unroll]
    for i in 0..r {
        accv[i] = V4F::new(0.0f32);
    }
    let mut b = lane;
    while b < bpr {
        // every load of this block first: r quant vectors, r scales, 8 activation vectors
        let mut q = Array::<V4U>::new(r);
        let mut d = Array::<u32>::new(r);
        #[unroll]
        for i in 0..r {
            let gb = gb0 + i * bpr + b;
            q[i] = qs[gb];
            d[i] = (sc[gb / 2usize] >> (((gb % 2usize) * 16usize) as u32)) & 65535u32;
        }
        let mut xv = Array::<V4F>::new(8usize);
        let xb = x4_0 + b * 8usize;
        #[unroll]
        for j in 0..8usize {
            if xshared {
                xv[j] = xs[xb + j];
            } else {
                xv[j] = x[xb + j];
            }
        }
        #[unroll]
        for i in 0..r {
            let mut s = V4F::new(0.0f32);
            #[unroll]
            for j in 0..8u32 {
                let nib = (q[i] >> V4U::new(4u32 * j)) & V4U::new(15u32);
                let f = V4F::cast_from(nib) - V4F::new(8.0f32);
                s += f * xv[j as usize];
            }
            accv[i] += V4F::new(f16_to_f32(d[i])) * s;
        }
        b += width;
    }
    #[unroll]
    for i in 0..r {
        let a = accv[i];
        acc[i] = a.extract(0usize) + a.extract(1usize) + a.extract(2usize) + a.extract(3usize);
    }
}

/// Reduce `acc[c*4 .. c*4+4]` across the plane in one `plane_sum`; the 4 totals, valid in every lane.
#[cube]
fn plane_sum4(acc: &Array<f32>, c: usize) -> V4F {
    let mut v = V4F::new(0.0f32);
    v.insert(0usize, acc[4usize * c]);
    v.insert(1usize, acc[4usize * c + 1usize]);
    v.insert(2usize, acc[4usize * c + 2usize]);
    v.insert(3usize, acc[4usize * c + 3usize]);
    plane_sum(v)
}

/// Stage `n4` activation vectors in shared memory (whole cube; ends on a barrier).
#[cube]
fn stage_x(x: &[V4F], xs: &mut Shared<[V4F]>, n4: usize) {
    let mut i = UNIT_POS as usize;
    while i < n4 {
        xs[i] = x[i];
        i += CUBE_DIM_X as usize;
    }
    sync_cube();
}

/// Q4_0 matvec, `r` rows per plane (needs plane ops; `rows % r == 0`, `r % 4 == 0`).
/// Plain: `dims` = [rows, cols], output `o = t * rows + row`. `routed`: `dims` = [rows, cols,
/// base], output `o = slot * rows + row`, expert `routes[2 (base + slot)]`, input row `slot`
/// (the routed `down` of an expert layer). `xshared` only with one input row.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn q4_0_matvec_rows_kernel(
    qs: &[V4U],
    sc: &[u32],
    x: &[V4F],
    routes: &[u32],
    dims: &[u32],
    out: &mut [f32],
    #[comptime] r: usize,
    #[comptime] xshared: bool,
    #[comptime] xlen: usize,
    #[comptime] routed: bool,
) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let bpr = cols / 32usize;
    let mut xs = Shared::<[V4F]>::new_slice(xlen);
    if xshared {
        stage_x(x, &mut xs, cols / 4usize);
    }
    let width = PLANE_DIM as usize;
    let plane_id = (UNIT_POS / PLANE_DIM) as usize;
    let lane = UNIT_POS_PLANE as usize;
    let ppc = CUBE_DIM_X as usize / width;
    let step = CUBE_COUNT as usize * ppc;
    let groups = out.len() / r;
    let mut g = CUBE_POS as usize * ppc + plane_id;
    let mut acc = Array::<f32>::new(r);
    while g < groups {
        let o0 = g * r;
        let t = o0 / rows;
        let r0 = o0 % rows;
        let mut gb0 = r0 * bpr;
        if routed {
            let e = routes[2usize * (dims[2] as usize + t)] as usize;
            gb0 += e * rows * bpr;
        }
        q4_rows_acc(qs, sc, x, &xs, gb0, bpr, t * cols / 4usize, lane, width, &mut acc, r, xshared);
        #[unroll]
        for c in 0..r / 4 {
            let v = plane_sum4(&acc, c);
            if lane == 0usize {
                #[unroll]
                for k in 0..4usize {
                    out[o0 + 4 * c + k] = v.extract(k);
                }
            }
        }
        g += step;
    }
}

/// `h = silu(x . gate) * (x . up)`, `r` rows per plane (gate's `r` rows, then up's).
/// Plain: `dims` = [rows, cols], `o = t * rows + row`. `routed`: `dims` = [rows, cols, base],
/// `o = slot * rows + row`, expert and token from `routes[2 (base + slot) ..]`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn q4_0_gate_up_silu_rows_kernel(
    gq: &[V4U],
    gs: &[u32],
    uq: &[V4U],
    us: &[u32],
    x: &[V4F],
    routes: &[u32],
    dims: &[u32],
    h: &mut [f32],
    #[comptime] r: usize,
    #[comptime] xshared: bool,
    #[comptime] xlen: usize,
    #[comptime] routed: bool,
) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let bpr = cols / 32usize;
    let mut xs = Shared::<[V4F]>::new_slice(xlen);
    if xshared {
        stage_x(x, &mut xs, cols / 4usize);
    }
    let width = PLANE_DIM as usize;
    let plane_id = (UNIT_POS / PLANE_DIM) as usize;
    let lane = UNIT_POS_PLANE as usize;
    let ppc = CUBE_DIM_X as usize / width;
    let step = CUBE_COUNT as usize * ppc;
    let groups = h.len() / r;
    let mut g = CUBE_POS as usize * ppc + plane_id;
    let mut ag = Array::<f32>::new(r);
    let mut au = Array::<f32>::new(r);
    while g < groups {
        let o0 = g * r;
        let t = o0 / rows;
        let r0 = o0 % rows;
        let mut gb0 = r0 * bpr;
        let mut tok = t;
        if routed {
            let base = dims[2] as usize;
            gb0 += routes[2usize * (base + t)] as usize * rows * bpr;
            tok = routes[2usize * (base + t) + 1usize] as usize;
        }
        q4_rows_acc(gq, gs, x, &xs, gb0, bpr, tok * cols / 4usize, lane, width, &mut ag, r, xshared);
        q4_rows_acc(uq, us, x, &xs, gb0, bpr, tok * cols / 4usize, lane, width, &mut au, r, xshared);
        #[unroll]
        for c in 0..r / 4 {
            let vg = plane_sum4(&ag, c);
            let vu = plane_sum4(&au, c);
            if lane == 0usize {
                #[unroll]
                for k in 0..4usize {
                    let gv = vg.extract(k);
                    h[o0 + 4 * c + k] = gv / (1.0f32 + Exp::exp(-gv)) * vu.extract(k);
                }
            }
        }
        g += step;
    }
}

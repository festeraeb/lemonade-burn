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

pub mod store;
pub use store::{Gpu, GpuExpert, GpuExpertStack, GpuQ4Mat};
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

/// Batched SwiGLU front half over a whole layer's experts. `jobs` holds two words per job:
/// [expert slot, activation row]; job `j` produces `h[j * rows .. (j + 1) * rows]`. The
/// weights are the fused GGUF tensor `[n_expert * rows, cols]` (all experts back to back),
/// so expert `e`'s row `r` starts at global block `(e * rows + r) * bpr`. `dims` = [rows, cols].
///
/// One dispatch for every routed expert replaces two dispatches per expert: a per-expert
/// launch streams only ~1.5 MB, too little in flight to fill the memory system, and wgpu
/// puts a barrier between consecutive passes so they never overlap.
#[cube(launch)]
fn q4_0_stack_gate_up_silu_kernel(
    gq: &[V4U],
    gs: &[u32],
    uq: &[V4U],
    us: &[u32],
    x: &[V4F],
    jobs: &[u32],
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
        let j = o / rows;
        let r = o % rows;
        let e = jobs[2usize * j] as usize;
        let xr = jobs[2usize * j + 1usize] as usize;
        let gb0 = (e * rows + r) * bpr;
        let pg = q4_row_partial(gq, gs, x, gb0, xr * cols / 4usize, bpr, lane, width);
        let pu = q4_row_partial(uq, us, x, gb0, xr * cols / 4usize, bpr, lane, width);
        let g = lane_sum(&mut sm, pg, use_plane);
        let u = lane_sum(&mut sm, pu, use_plane);
        if lane == 0usize {
            h[o] = g / (1.0f32 + Exp::exp(-g)) * u;
        }
        o += step;
    }
}

/// Batched down projection: job `j` multiplies its own row `j` of `x` (the `h` the front
/// half wrote) by expert `jobs[2j]`'s matrix, `out[j * rows + r]`. `dims` = [rows, cols].
#[cube(launch)]
fn q4_0_stack_matvec_kernel(qs: &[V4U], sc: &[u32], x: &[V4F], jobs: &[u32], dims: &[u32], out: &mut [f32], #[comptime] use_plane: bool) {
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
        let j = o / rows;
        let r = o % rows;
        let e = jobs[2usize * j] as usize;
        let part = q4_row_partial(qs, sc, x, (e * rows + r) * bpr, j * cols / 4usize, bpr, lane, width);
        let total = lane_sum(&mut sm, part, use_plane);
        if lane == 0usize {
            out[o] = total;
        }
        o += step;
    }
}

/// Default cube size (units) when the device is not asked: see `Gpu::open`.
pub const THREADS: usize = 64;

/// Repack GGUF Q4_0 rows into the device layout (`qs` words, packed f16 `sc` words).
/// `bytes` is the unchanged GGUF data: `n_blocks` blocks of 18 bytes.
pub fn repack_q4_0(bytes: &[u8], n_blocks: usize) -> (Vec<u32>, Vec<u32>) {
    let mut qs = vec![0u32; n_blocks * 4];
    let mut sc = vec![0u32; n_blocks.div_ceil(2).max(1)];
    for b in 0..n_blocks {
        let blk = &bytes[b * 18..b * 18 + 18];
        let d = u16::from_le_bytes([blk[0], blk[1]]) as u32;
        sc[b / 2] |= d << (16 * (b % 2));
        let q = &blk[2..];
        for v in 0..32usize {
            let nib = if v < 16 { q[v] & 15 } else { q[v - 16] >> 4 } as u32;
            let (j, k) = (v / 4, v % 4);
            qs[b * 4 + k] |= nib << (4 * j);
        }
    }
    (qs, sc)
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
    let _ = (x, m);
    Err("q6_k_matvec_gpu: not implemented yet (forge seed)".to_string())
}

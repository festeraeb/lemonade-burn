//! Tiled GEMM for prefill on the K-quant and Q8_0 matrices of the `qwen35moe` family at file density: `out[t, r] = sum_c W[r, c] x[t, c]`
//! for `m` token rows, so each weight tile is read once per 64 tokens instead of once per token. Same tile as the Q4_0 `gemm.rs` (a cube owns
//! 64 tokens x 64 weight rows, 256 units each a 4 x 4 sub-tile, one 32-column slab staged in shared memory a step) but the weight slab is
//! dequantised to f32 values in the staging step (`d * sc * q - dmin * mn` for Q5_K, `d * sc * (q - 32)` for Q6_K, `d * q` for Q8_0), so the
//! inner loop is plain FMAs and the three types share it. Sums agree with the matvec kernels to f32 rounding (order inside a slab differs).
//!
//! Grouped (MoE) mode: the routed (token, expert) slots of a chunk are sorted by expert on the host (`gemm::group_slots`) and cut into tiles of at
//! most 64 slots of one expert; a tile reads its input rows through `perm` (the slot's token for gate/up, the slot itself for down), and writes
//! output row `perm[start + i]`. With `gate_up` a tile carries two matrices and writes `silu(x . gate) * (x . up)`.

use crate::kquant::{k4_min_pub, k4_scale_pub};
use crate::f16_to_f32;
use cubecl::prelude::*;

const LD: usize = 33;
pub const QT_Q5K: u32 = 0;
pub const QT_Q6K: u32 = 1;
pub const QT_Q8: u32 = 2;
/// Q8_0 in the decode (`GpuQ8Mat`) layout: quants repacked 4 bytes apart (`repack_q8_0`), f16 scales in pairs. Lets one device copy serve decode and prefill.
pub const QT_Q8R: u32 = 3;

/// Stage the weight slab (rows `r0..r0+64`, columns `32 kb ..`) as f32 values: unit `u` takes row `u / 4` and columns `8 (u % 4) ..+8`.
/// `a0..a2` are the matrix's arrays (Q5_K: all-in-one words in `a0`; Q6_K: `ql+qh` words, scales, f16 `d`; Q8_0: quants, f16 `d`);
/// `eb` = first block (in the type's block unit) of this expert/matrix.
#[cube]
#[allow(clippy::too_many_arguments)]
fn stage_k(a0: &[u32], a1: &[u32], a2: &[u32], ws: &mut Shared<[f32]>, eb: usize, cols: usize, r0: usize, rows: usize, kb: usize, #[comptime] qt: u32) {
    let u = UNIT_POS as usize;
    let rr = u / 4usize;
    let p = u % 4usize;
    let r = r0 + rr;
    if r < rows {
        if comptime!(qt == 0u32) {
            let bpr = cols / 256usize;
            let blk = eb + r * bpr + kb / 8usize;
            let g = kb % 8usize;
            let base = blk * 44usize;
            let w0 = a0[base];
            let d = f16_to_f32(w0 & 65535u32);
            let dmin = f16_to_f32((w0 >> 16u32) & 65535u32);
            let s0 = a0[base + 1usize];
            let s1 = a0[base + 2usize];
            let s2 = a0[base + 3usize];
            let sc = d * (k4_scale_pub(s0, s1, s2, g as u32) as f32);
            let mn = dmin * (k4_min_pub(s0, s1, s2, g as u32) as f32);
            let shift = ((g % 2usize) * 4usize) as u32;
            #[unroll]
            for wi in 0..2usize {
                let qw = a0[base + 12usize + 8usize * (g / 2usize) + 2usize * p + wi];
                let hw = a0[base + 4usize + 2usize * p + wi];
                #[unroll]
                for i in 0..4u32 {
                    let nib = (qw >> (8u32 * i + shift)) & 15u32;
                    let hb = (hw >> (8u32 * i + g as u32)) & 1u32;
                    ws[rr * LD + 8usize * p + 4usize * wi + i as usize] = sc * ((nib + hb * 16u32) as f32) - mn;
                }
            }
        } else if comptime!(qt == 1u32) {
            let bpr = cols / 256usize;
            let blk = eb + r * bpr + kb / 8usize;
            let g = kb % 8usize;
            let h = g / 4usize;
            let sub = g % 4usize;
            let d = f16_to_f32((a2[blk / 2usize] >> (((blk % 2usize) * 16usize) as u32)) & 65535u32);
            let sidx = 8usize * h + 2usize * sub + p / 2usize;
            let sw = a1[blk * 4usize + sidx / 4usize];
            let sb = (sw >> (((sidx % 4usize) * 8usize) as u32)) & 255u32;
            let sf = (sb as f32) - 256.0f32 * ((sb >> 7u32) as f32);
            let s = d * sf;
            let nsh = (4usize * (sub / 2usize)) as u32;
            let qsh = (2usize * sub) as u32;
            #[unroll]
            for wi in 0..2usize {
                let lw = a0[blk * 48usize + 16usize * h + 8usize * (sub % 2usize) + 2usize * p + wi];
                let hw = a0[blk * 48usize + 32usize + 8usize * h + 2usize * p + wi];
                #[unroll]
                for i in 0..4u32 {
                    let nib = (lw >> (8u32 * i + nsh)) & 15u32;
                    let hb = (hw >> (8u32 * i + qsh)) & 3u32;
                    ws[rr * LD + 8usize * p + 4usize * wi + i as usize] = s * (((nib + hb * 16u32) as f32) - 32.0f32);
                }
            }
        } else {
            let bpr = cols / 32usize;
            let blk = eb + r * bpr + kb;
            let d = f16_to_f32((a1[blk / 2usize] >> (((blk % 2usize) * 16usize) as u32)) & 65535u32);
            if comptime!(qt == 3u32) {
                // repacked layout: word `4 g + k` of a block holds bytes `16 g + 4 b + k`, b = 0..3; this unit's columns are `8 p ..+8`
                let g16 = p / 2usize;
                let bsh = (16usize * (p % 2usize)) as u32;
                #[unroll]
                for k in 0..4usize {
                    let qw = a0[blk * 8usize + 4usize * g16 + k];
                    #[unroll]
                    for hb in 0..2u32 {
                        let by = (qw >> (bsh + 8u32 * hb)) & 255u32;
                        let sv = (by as f32) - 256.0f32 * ((by >> 7u32) as f32);
                        ws[rr * LD + 8usize * p + 4usize * (hb as usize) + k] = d * sv;
                    }
                }
            } else {
                #[unroll]
                for wi in 0..2usize {
                    let qw = a0[blk * 8usize + 2usize * p + wi];
                    #[unroll]
                    for i in 0..4u32 {
                        let by = (qw >> (8u32 * i)) & 255u32;
                        let sv = (by as f32) - 256.0f32 * ((by >> 7u32) as f32);
                        ws[rr * LD + 8usize * p + 4usize * wi + i as usize] = d * sv;
                    }
                }
            }
        }
    } else {
        #[unroll]
        for j in 0..8usize {
            ws[rr * LD + 8usize * p + j] = 0.0f32;
        }
    }
}

/// `dims` = `[rows, cols, m]`. Grid `(ceil(rows / 64), y)`: dense `y = ceil(m / 64)`, grouped `y` = number of tiles. `CubeDim` 256.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn kq_gemm_kernel(
    a0: &[u32], a1: &[u32], a2: &[u32],
    b0: &[u32], b1: &[u32], b2: &[u32],
    x: &[f32], routes: &[u32], perm: &[u32], tiles: &[u32], dims: &[u32], out: &mut [f32],
    #[comptime] qt: u32, #[comptime] grouped: bool, #[comptime] gate_up: bool, #[comptime] src_tok: bool,
) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let m = dims[2] as usize;
    let r0 = CUBE_POS_X as usize * 64usize;
    let ty = (UNIT_POS / 16u32) as usize;
    let tx = (UNIT_POS % 16u32) as usize;
    let mut bpr = cols / 256usize;
    if comptime!(qt >= 2u32) {
        bpr = cols / 32usize;
    }
    let mut t0 = 0usize;
    let mut count = 64usize;
    let mut eb = 0usize;
    if grouped {
        let tile = CUBE_POS_Y as usize;
        let e = tiles[3usize * tile] as usize;
        t0 = tiles[3usize * tile + 1usize] as usize;
        count = tiles[3usize * tile + 2usize] as usize;
        eb = e * rows * bpr;
    } else {
        t0 = CUBE_POS_Y as usize * 64usize;
        if t0 + 64usize > m {
            count = m - t0;
        }
    }
    let mut xs = Shared::<[f32]>::new_slice(64usize * LD);
    let mut ws = Shared::<[f32]>::new_slice(64usize * LD);
    let mut ws2 = Shared::<[f32]>::new_slice(64usize * LD);
    let xr = UNIT_POS as usize / 4usize;
    let xpart = UNIT_POS as usize % 4usize;
    let mut src = 0usize;
    let valid = xr < count;
    if valid {
        if grouped {
            let s = perm[t0 + xr] as usize;
            if src_tok {
                src = routes[2usize * s + 1usize] as usize;
            } else {
                src = s;
            }
        } else {
            src = t0 + xr;
        }
    }
    let mut acc = Array::<f32>::new(16usize);
    let mut acc2 = Array::<f32>::new(16usize);
    #[unroll]
    for i in 0..16usize {
        acc[i] = 0.0f32;
        acc2[i] = 0.0f32;
    }
    let nslab = cols / 32usize;
    let mut kb = 0usize;
    while kb < nslab {
        let c0 = kb * 32usize + xpart * 8usize;
        #[unroll]
        for j in 0..8usize {
            let mut v = 0.0f32;
            if valid {
                v = x[src * cols + c0 + j];
            }
            xs[xr * LD + xpart * 8usize + j] = v;
        }
        stage_k(a0, a1, a2, &mut ws, eb, cols, r0, rows, kb, qt);
        if gate_up {
            stage_k(b0, b1, b2, &mut ws2, eb, cols, r0, rows, kb, qt);
        }
        sync_cube();
        for kk in 0..32usize {
            let mut a = Array::<f32>::new(4usize);
            #[unroll]
            for i in 0..4usize {
                a[i] = xs[(ty * 4usize + i) * LD + kk];
            }
            #[unroll]
            for j in 0..4usize {
                let b = ws[(tx * 4usize + j) * LD + kk];
                #[unroll]
                for i in 0..4usize {
                    acc[i * 4usize + j] += a[i] * b;
                }
                if gate_up {
                    let b2 = ws2[(tx * 4usize + j) * LD + kk];
                    #[unroll]
                    for i in 0..4usize {
                        acc2[i * 4usize + j] += a[i] * b2;
                    }
                }
            }
        }
        sync_cube();
        kb += 1usize;
    }
    #[unroll]
    for i in 0..4usize {
        let ti = ty * 4usize + i;
        if ti < count {
            let mut dst = t0 + ti;
            if grouped {
                dst = perm[t0 + ti] as usize;
            }
            #[unroll]
            for j in 0..4usize {
                let r = r0 + tx * 4usize + j;
                if r < rows {
                    let mut v = acc[i * 4usize + j];
                    if gate_up {
                        v = v / (1.0f32 + Exp::exp(-v)) * acc2[i * 4usize + j];
                    }
                    out[dst * rows + r] = v;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------
// The 4-wide variant (the structure of `gemm::q4_0_gemm_v4_kernel`): 128 units, a unit owns 8 token rows x 4 weight rows, shared memory holds
// column vectors (`xs[kk * 16 + g]` = column kk of token rows 4g.., `ws[kk * 16 + t]` = value of weight rows 4t.. at column kk), so a column
// costs three shared loads for 32 multiply-adds.
// ---------------------------------------------------------------------------------------------------------------------------

type V4F = Vector<f32, Const<4>>;

/// Stage a 64-row x 32-column weight slab as column vectors: unit `u` takes weight rows `4 wg ..+4` (`wg = u / 8`) and columns `4 p ..+4` (`p = u % 8`),
/// one 32-bit word of each row.
#[cube]
#[allow(clippy::too_many_arguments)]
fn stage_k4(a0: &[u32], a1: &[u32], a2: &[u32], ws: &mut Shared<[V4F]>, eb: usize, cols: usize, r0: usize, rows: usize, kb: usize, wg: usize, p: usize, #[comptime] qt: u32) {
    let mut c0 = V4F::new(0.0f32);
    let mut c1 = V4F::new(0.0f32);
    let mut c2 = V4F::new(0.0f32);
    let mut c3 = V4F::new(0.0f32);
    #[unroll]
    for i in 0..4usize {
        let r = r0 + wg * 4usize + i;
        if r < rows {
            let mut v0 = 0.0f32;
            let mut v1 = 0.0f32;
            let mut v2 = 0.0f32;
            let mut v3 = 0.0f32;
            if comptime!(qt == 0u32) {
                let bpr = cols / 256usize;
                let blk = eb + r * bpr + kb / 8usize;
                let g = kb % 8usize;
                let base = blk * 44usize;
                let w0 = a0[base];
                let d = f16_to_f32(w0 & 65535u32);
                let dmin = f16_to_f32((w0 >> 16u32) & 65535u32);
                let s0 = a0[base + 1usize];
                let s1 = a0[base + 2usize];
                let s2 = a0[base + 3usize];
                let sc = d * (k4_scale_pub(s0, s1, s2, g as u32) as f32);
                let mn = dmin * (k4_min_pub(s0, s1, s2, g as u32) as f32);
                let shift = ((g % 2usize) * 4usize) as u32;
                let qw = a0[base + 12usize + 8usize * (g / 2usize) + p];
                let hw = a0[base + 4usize + p];
                let gg = g as u32;
                v0 = sc * ((((qw >> shift) & 15u32) + ((hw >> gg) & 1u32) * 16u32) as f32) - mn;
                v1 = sc * ((((qw >> (8u32 + shift)) & 15u32) + ((hw >> (8u32 + gg)) & 1u32) * 16u32) as f32) - mn;
                v2 = sc * ((((qw >> (16u32 + shift)) & 15u32) + ((hw >> (16u32 + gg)) & 1u32) * 16u32) as f32) - mn;
                v3 = sc * ((((qw >> (24u32 + shift)) & 15u32) + ((hw >> (24u32 + gg)) & 1u32) * 16u32) as f32) - mn;
            } else if comptime!(qt == 1u32) {
                let bpr = cols / 256usize;
                let blk = eb + r * bpr + kb / 8usize;
                let g = kb % 8usize;
                let h = g / 4usize;
                let sub = g % 4usize;
                let d = f16_to_f32((a2[blk / 2usize] >> (((blk % 2usize) * 16usize) as u32)) & 65535u32);
                let sidx = 8usize * h + 2usize * sub + p / 4usize;
                let sw = a1[blk * 4usize + sidx / 4usize];
                let sb = (sw >> (((sidx % 4usize) * 8usize) as u32)) & 255u32;
                let s = d * ((sb as f32) - 256.0f32 * ((sb >> 7u32) as f32));
                let nsh = (4usize * (sub / 2usize)) as u32;
                let qsh = (2usize * sub) as u32;
                let lw = a0[blk * 48usize + 16usize * h + 8usize * (sub % 2usize) + p];
                let hw = a0[blk * 48usize + 32usize + 8usize * h + p];
                v0 = s * ((((lw >> nsh) & 15u32) + ((hw >> qsh) & 3u32) * 16u32) as f32 - 32.0f32);
                v1 = s * ((((lw >> (8u32 + nsh)) & 15u32) + ((hw >> (8u32 + qsh)) & 3u32) * 16u32) as f32 - 32.0f32);
                v2 = s * ((((lw >> (16u32 + nsh)) & 15u32) + ((hw >> (16u32 + qsh)) & 3u32) * 16u32) as f32 - 32.0f32);
                v3 = s * ((((lw >> (24u32 + nsh)) & 15u32) + ((hw >> (24u32 + qsh)) & 3u32) * 16u32) as f32 - 32.0f32);
            } else {
                let bpr = cols / 32usize;
                let blk = eb + r * bpr + kb;
                let d = f16_to_f32((a1[blk / 2usize] >> (((blk % 2usize) * 16usize) as u32)) & 65535u32);
                let qw = a0[blk * 8usize + p];
                let b0 = qw & 255u32;
                let b1 = (qw >> 8u32) & 255u32;
                let b2 = (qw >> 16u32) & 255u32;
                let b3 = (qw >> 24u32) & 255u32;
                v0 = d * ((b0 as f32) - 256.0f32 * ((b0 >> 7u32) as f32));
                v1 = d * ((b1 as f32) - 256.0f32 * ((b1 >> 7u32) as f32));
                v2 = d * ((b2 as f32) - 256.0f32 * ((b2 >> 7u32) as f32));
                v3 = d * ((b3 as f32) - 256.0f32 * ((b3 >> 7u32) as f32));
            }
            c0.insert(i, v0);
            c1.insert(i, v1);
            c2.insert(i, v2);
            c3.insert(i, v3);
        }
    }
    ws[(4usize * p) * 16usize + wg] = c0;
    ws[(4usize * p + 1usize) * 16usize + wg] = c1;
    ws[(4usize * p + 2usize) * 16usize + wg] = c2;
    ws[(4usize * p + 3usize) * 16usize + wg] = c3;
}

/// Same contract as [`kq_gemm_kernel`] but `x` is read as 4-vectors and the cube has 128 units.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn kq_gemm_v4_kernel(
    a0: &[u32], a1: &[u32], a2: &[u32],
    b0: &[u32], b1: &[u32], b2: &[u32],
    x: &[V4F], routes: &[u32], perm: &[u32], tiles: &[u32], dims: &[u32], out: &mut [f32],
    #[comptime] qt: u32, #[comptime] grouped: bool, #[comptime] gate_up: bool, #[comptime] src_tok: bool,
) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let m = dims[2] as usize;
    let r0 = CUBE_POS_X as usize * 64usize;
    let u = UNIT_POS as usize;
    let mut bpr = cols / 256usize;
    if comptime!(qt == 2u32) {
        bpr = cols / 32usize;
    }
    let mut t0 = 0usize;
    let mut count = 64usize;
    let mut eb = 0usize;
    if grouped {
        let tile = CUBE_POS_Y as usize;
        let e = tiles[3usize * tile] as usize;
        t0 = tiles[3usize * tile + 1usize] as usize;
        count = tiles[3usize * tile + 2usize] as usize;
        eb = e * rows * bpr;
    } else {
        t0 = CUBE_POS_Y as usize * 64usize;
        if t0 + 64usize > m {
            count = m - t0;
        }
    }
    let mut xs = Shared::<[V4F]>::new_slice(32usize * 16usize);
    let mut ws = Shared::<[V4F]>::new_slice(32usize * 16usize);
    let mut ws2 = Shared::<[V4F]>::new_slice(32usize * 16usize);
    let xg = u / 8usize;
    let xq = u % 8usize;
    let mut srcs = Array::<usize>::new(4usize);
    let mut oks = Array::<bool>::new(4usize);
    #[unroll]
    for i in 0..4usize {
        let ti = xg * 4usize + i;
        let ok = ti < count;
        let mut sidx = 0usize;
        if ok {
            if grouped {
                let sl = perm[t0 + ti] as usize;
                if src_tok {
                    sidx = routes[2usize * sl + 1usize] as usize;
                } else {
                    sidx = sl;
                }
            } else {
                sidx = t0 + ti;
            }
        }
        srcs[i] = sidx;
        oks[i] = ok;
    }
    let wg = u / 8usize;
    let wp = u % 8usize;
    let ty = u / 16usize;
    let tx = u % 16usize;
    let mut acc = Array::<V4F>::new(8usize);
    let mut acc2 = Array::<V4F>::new(8usize);
    #[unroll]
    for i in 0..8usize {
        acc[i] = V4F::new(0.0f32);
        acc2[i] = V4F::new(0.0f32);
    }
    let nslab = cols / 32usize;
    let mut kb = 0usize;
    while kb < nslab {
        let mut rv = Array::<V4F>::new(4usize);
        #[unroll]
        for i in 0..4usize {
            let mut v = V4F::new(0.0f32);
            if oks[i] {
                v = x[(srcs[i] * cols + kb * 32usize) / 4usize + xq];
            }
            rv[i] = v;
        }
        #[unroll]
        for c in 0..4usize {
            let mut col = V4F::new(rv[0usize].extract(c));
            col.insert(1usize, rv[1usize].extract(c));
            col.insert(2usize, rv[2usize].extract(c));
            col.insert(3usize, rv[3usize].extract(c));
            xs[(xq * 4usize + c) * 16usize + xg] = col;
        }
        stage_k4(a0, a1, a2, &mut ws, eb, cols, r0, rows, kb, wg, wp, qt);
        if gate_up {
            stage_k4(b0, b1, b2, &mut ws2, eb, cols, r0, rows, kb, wg, wp, qt);
        }
        sync_cube();
        for kk in 0..32usize {
            let xa = xs[kk * 16usize + ty * 2usize];
            let xb = xs[kk * 16usize + ty * 2usize + 1usize];
            let b = ws[kk * 16usize + tx];
            #[unroll]
            for i in 0..4usize {
                acc[i] = fma(V4F::new(xa.extract(i)), b, acc[i]);
                acc[4usize + i] = fma(V4F::new(xb.extract(i)), b, acc[4usize + i]);
            }
            if gate_up {
                let bb = ws2[kk * 16usize + tx];
                #[unroll]
                for i in 0..4usize {
                    acc2[i] = fma(V4F::new(xa.extract(i)), bb, acc2[i]);
                    acc2[4usize + i] = fma(V4F::new(xb.extract(i)), bb, acc2[4usize + i]);
                }
            }
        }
        sync_cube();
        kb += 1usize;
    }
    #[unroll]
    for i in 0..8usize {
        let ti = ty * 8usize + i;
        if ti < count {
            let mut dst = t0 + ti;
            if grouped {
                dst = perm[t0 + ti] as usize;
            }
            #[unroll]
            for j in 0..4usize {
                let r = r0 + tx * 4usize + j;
                if r < rows {
                    let mut v = acc[i].extract(j);
                    if gate_up {
                        v = v / (1.0f32 + Exp::exp(-v)) * acc2[i].extract(j);
                    }
                    out[dst * rows + r] = v;
                }
            }
        }
    }
}

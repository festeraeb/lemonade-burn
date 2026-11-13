//! Tiled Q4_0 GEMM for prefill: `out[t, r] = sum_c W[r, c] x[t, c]` over many rows of `x`,
//! reading each weight tile once per 64 rows instead of once per row.
//!
//! The rows kernels (`q4_0_matvec_rows_kernel`) give every output row its own plane, so a chunk of
//! `m` prompt rows reads every weight `m` times. For decode (`m = 1`) that is the whole job; for a
//! 512-row prefill chunk through LFM2.5's experts it is ~11 GB of weight reads per MoE layer (each
//! routed (token, expert) pair re-reads 5.3 MB), so prefill was bandwidth-bound at a few hundred
//! tok/s. Here a cube owns a 64 x 64 output tile (64 input rows x 64 weight rows), stages one
//! Q4_0 block (32 columns) of both into shared memory, and each of its 256 units accumulates a
//! 4 x 4 sub-tile from shared memory.
//!
//! Numbers: per Q4_0 block the unit sums `(q - 8) * x` over the block's 32 columns, then adds
//! `d * partial`, the grouping ggml's reference uses, so results agree with the rows kernels to
//! f32 rounding (not bit for bit: the summation order inside a block differs).
//!
//! **Grouped (MoE) mode.** The routed (token, expert) slots of a chunk are sorted by expert on the
//! host into `perm` and cut into tiles of at most 64 slots of one expert (`tiles`: `[expert, start,
//! count]` each). Tile `y` reads its input rows through `perm` (the token of a slot from `routes`
//! for gate/up, the slot itself for down) and writes output row `perm[start + i]`, so the outputs
//! stay in slot order and `weighted_sum_rows_kernel` is unchanged. With `gate_up` the tile carries
//! two weight matrices and writes `silu(x.gate) * (x.up)`.

use crate::f16_to_f32;
use cubecl::prelude::*;

/// Output tile edge (rows of x and rows of W per cube).
pub const TILE: usize = 64;
/// Shared-memory row stride (TILE + 1 would conflict on the 32-column rows; 33 is the block + 1).
const LD: usize = 33;

/// Stage one Q4_0 block column (`kb`) of 64 weight rows: `ws[rr * LD + c] = q - 8` (unscaled) and
/// `ds[rr] = d`. Unit `u` takes row `u / 4`, word `u % 4` (word `k` holds values `4j + k`).
#[cube]
#[allow(clippy::too_many_arguments)]
fn stage_w(qs: &[u32], sc: &[u32], ws: &mut Shared<[f32]>, ds: &mut Shared<[f32]>, base_block: usize, bpr: usize, r0: usize, rows: usize, kb: usize) {
    let u = UNIT_POS as usize;
    let rr = u / 4usize;
    let word = u % 4usize;
    let r = r0 + rr;
    if r < rows {
        let gb = base_block + r * bpr + kb;
        let w = qs[gb * 4usize + word];
        #[unroll]
        for j in 0..8u32 {
            ws[rr * LD + 4usize * j as usize + word] = f32::cast_from((w >> (4u32 * j)) & 15u32) - 8.0f32;
        }
        if word == 0usize {
            ds[rr] = f16_to_f32((sc[gb / 2usize] >> (((gb % 2usize) * 16usize) as u32)) & 65535u32);
        }
    } else {
        #[unroll]
        for j in 0..8u32 {
            ws[rr * LD + 4usize * j as usize + word] = 0.0f32;
        }
        if word == 0usize {
            ds[rr] = 0.0f32;
        }
    }
}

/// `dims` = `[rows, cols, m, base_ff]`: weight rows, columns, input rows (dense) and, grouped, the
/// per-expert block stride is `rows * cols / 32`. Grid `(ceil(rows / 64), y)`: dense `y` =
/// `ceil(m / 64)`; grouped `y` = number of tiles. `CubeDim` 256.
///
/// `src_tok`: grouped input row of slot `s` is `routes[2 s + 1]` (gate/up reads the token's
/// activation); otherwise it is `s` (down reads the slot's hidden row).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn q4_0_gemm_kernel(
    gq: &[u32],
    gs: &[u32],
    uq: &[u32],
    us: &[u32],
    x: &[f32],
    routes: &[u32],
    perm: &[u32],
    tiles: &[u32],
    dims: &[u32],
    out: &mut [f32],
    #[comptime] grouped: bool,
    #[comptime] gate_up: bool,
    #[comptime] src_tok: bool,
) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let m = dims[2] as usize;
    let bpr = cols / 32usize;
    let r0 = CUBE_POS_X as usize * 64usize;
    let ty = (UNIT_POS / 16u32) as usize;
    let tx = (UNIT_POS % 16u32) as usize;

    // which input rows this tile covers, and where its outputs go
    let mut t0 = 0usize;
    let mut count = 64usize;
    let mut base_block = 0usize;
    if grouped {
        let tile = CUBE_POS_Y as usize;
        let e = tiles[3usize * tile] as usize;
        t0 = tiles[3usize * tile + 1usize] as usize;
        count = tiles[3usize * tile + 2usize] as usize;
        base_block = e * rows * bpr;
    } else {
        t0 = CUBE_POS_Y as usize * 64usize;
        if t0 + 64usize > m {
            count = m - t0;
        }
    }

    let mut xs = Shared::<[f32]>::new_slice(64usize * LD);
    let mut ws = Shared::<[f32]>::new_slice(64usize * LD);
    let mut ds = Shared::<[f32]>::new_slice(64usize);
    let mut ws2 = Shared::<[f32]>::new_slice(64usize * LD);
    let mut ds2 = Shared::<[f32]>::new_slice(64usize);

    // the source row of each of this tile's 64 input rows (4 per unit when staging x)
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
    let mut part = Array::<f32>::new(16usize);
    let mut part2 = Array::<f32>::new(16usize);
    #[unroll]
    for i in 0..16usize {
        acc[i] = 0.0f32;
        acc2[i] = 0.0f32;
    }

    let mut kb = 0usize;
    while kb < bpr {
        // stage x: row xr, columns xpart*8 .. +8 of this block
        let c0 = kb * 32usize + xpart * 8usize;
        #[unroll]
        for j in 0..8usize {
            let mut v = 0.0f32;
            if valid {
                v = x[src * cols + c0 + j];
            }
            xs[xr * LD + xpart * 8usize + j] = v;
        }
        stage_w(gq, gs, &mut ws, &mut ds, base_block, bpr, r0, rows, kb);
        if gate_up {
            stage_w(uq, us, &mut ws2, &mut ds2, base_block, bpr, r0, rows, kb);
        }
        sync_cube();
        #[unroll]
        for i in 0..16usize {
            part[i] = 0.0f32;
            part2[i] = 0.0f32;
        }
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
                    part[i * 4usize + j] += a[i] * b;
                }
                if gate_up {
                    let b2 = ws2[(tx * 4usize + j) * LD + kk];
                    #[unroll]
                    for i in 0..4usize {
                        part2[i * 4usize + j] += a[i] * b2;
                    }
                }
            }
        }
        #[unroll]
        for j in 0..4usize {
            let d = ds[tx * 4usize + j];
            #[unroll]
            for i in 0..4usize {
                acc[i * 4usize + j] += d * part[i * 4usize + j];
            }
            if gate_up {
                let d2 = ds2[tx * 4usize + j];
                #[unroll]
                for i in 0..4usize {
                    acc2[i * 4usize + j] += d2 * part2[i * 4usize + j];
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

/// f16 shared-memory row stride for the tensor-core kernel: a multiple of 8 halves (wmma needs a
/// 16-byte `ldm`), padded past 32 so the 64 rows do not all start on one bank.
const LDH: usize = 40;
/// f32 staging row stride for the accumulators (a multiple of 4 floats).
const LDS: usize = 68;

/// The tensor-core form of [`q4_0_gemm_kernel`] (same grid, tiling, grouping and outputs), for
/// cards with f16 cooperative matrix ops (`cmma` 16x16x16, f32 accumulate: NVIDIA sm_70+; our
/// fleet's Turing cards). The 64 x 64 tile is 8 warps of 16 x 32: warp `w` owns rows
/// `16 (w / 2)` and columns `32 (w % 2) .. + 32` as two accumulators (four with `gate_up`).
///
/// Numbers differ from the f32 kernel in two ways, both by design: activations are rounded to
/// f16 before the multiply, and a weight is dequantized to f16 (`(q - 8) * d`) instead of scaling
/// a per-block f32 partial. Accumulation is f32 throughout. This is the arithmetic llama.cpp's
/// prefill uses on these cards (dequantize to f16, tensor-core GEMM); its cost is checked by the
/// model-level KLD gate, not assumed (`docs/PREFILL_SPLIT.md`).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn q4_0_gemm_tc_kernel(
    gq: &[u32],
    gs: &[u32],
    uq: &[u32],
    us: &[u32],
    x: &[f32],
    routes: &[u32],
    perm: &[u32],
    tiles: &[u32],
    dims: &[u32],
    out: &mut [f32],
    #[comptime] grouped: bool,
    #[comptime] gate_up: bool,
    #[comptime] src_tok: bool,
) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let m = dims[2] as usize;
    let bpr = cols / 32usize;
    let r0 = CUBE_POS_X as usize * 64usize;
    let u = UNIT_POS as usize;
    let warp = (UNIT_POS / PLANE_DIM) as usize;
    let wm = warp / 2usize;
    let wn = warp % 2usize;

    let mut t0 = 0usize;
    let mut count = 64usize;
    let mut base_block = 0usize;
    if grouped {
        let tile = CUBE_POS_Y as usize;
        let e = tiles[3usize * tile] as usize;
        t0 = tiles[3usize * tile + 1usize] as usize;
        count = tiles[3usize * tile + 2usize] as usize;
        base_block = e * rows * bpr;
    } else {
        t0 = CUBE_POS_Y as usize * 64usize;
        if t0 + 64usize > m {
            count = m - t0;
        }
    }

    let mut xs = Shared::<[half::f16]>::new_aligned_slice(64usize * LDH, 32usize);
    let mut ws = Shared::<[half::f16]>::new_aligned_slice(64usize * LDH, 32usize);
    let mut ws2 = Shared::<[half::f16]>::new_aligned_slice(64usize * LDH, 32usize);
    let mut st = Shared::<[f32]>::new_aligned_slice(64usize * LDS, 32usize);

    let xr = u / 4usize;
    let part = u % 4usize;
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
    let wr = r0 + xr;

    let c0 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0f32);
    let c1 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0f32);
    let c2 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0f32);
    let c3 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0f32);

    let mut kb = 0usize;
    while kb < bpr {
        // x: row xr, columns part*8 .. +8 of this block, rounded to f16
        let xc = kb * 32usize + part * 8usize;
        #[unroll]
        for j in 0..8usize {
            let mut v = 0.0f32;
            if valid {
                v = x[src * cols + xc + j];
            }
            xs[xr * LDH + part * 8usize + j] = half::f16::cast_from(v);
        }
        // W: row xr of the tile, word `part` of its block, dequantized (q - 8) * d
        if wr < rows {
            let gb = base_block + wr * bpr + kb;
            let w = gq[gb * 4usize + part];
            let d = f16_to_f32((gs[gb / 2usize] >> (((gb % 2usize) * 16usize) as u32)) & 65535u32);
            #[unroll]
            for j in 0..8u32 {
                ws[xr * LDH + 4usize * j as usize + part] = half::f16::cast_from((f32::cast_from((w >> (4u32 * j)) & 15u32) - 8.0f32) * d);
            }
            if gate_up {
                let w2 = uq[gb * 4usize + part];
                let d2 = f16_to_f32((us[gb / 2usize] >> (((gb % 2usize) * 16usize) as u32)) & 65535u32);
                #[unroll]
                for j in 0..8u32 {
                    ws2[xr * LDH + 4usize * j as usize + part] = half::f16::cast_from((f32::cast_from((w2 >> (4u32 * j)) & 15u32) - 8.0f32) * d2);
                }
            }
        } else {
            #[unroll]
            for j in 0..8u32 {
                ws[xr * LDH + 4usize * j as usize + part] = half::f16::cast_from(0.0f32);
                if gate_up {
                    ws2[xr * LDH + 4usize * j as usize + part] = half::f16::cast_from(0.0f32);
                }
            }
        }
        sync_cube();
        #[unroll]
        for ks in 0..2usize {
            let ao = wm * 16usize * LDH + ks * 16usize;
            let bo0 = wn * 32usize * LDH + ks * 16usize;
            let bo1 = (wn * 32usize + 16usize) * LDH + ks * 16usize;
            let a = cmma::Matrix::<half::f16>::from_slice(cmma::MatrixIdent::A, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor, &xs[ao..64usize * LDH], LDH as u32);
            let b0 = cmma::Matrix::<half::f16>::from_slice(cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor, &ws[bo0..64usize * LDH], LDH as u32);
            let b1 = cmma::Matrix::<half::f16>::from_slice(cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor, &ws[bo1..64usize * LDH], LDH as u32);
            cmma::execute(&a, &b0, &c0, &c0);
            cmma::execute(&a, &b1, &c1, &c1);
            if gate_up {
                let b2 = cmma::Matrix::<half::f16>::from_slice(cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor, &ws2[bo0..64usize * LDH], LDH as u32);
                let b3 = cmma::Matrix::<half::f16>::from_slice(cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor, &ws2[bo1..64usize * LDH], LDH as u32);
                cmma::execute(&a, &b2, &c2, &c2);
                cmma::execute(&a, &b3, &c3, &c3);
            }
        }
        sync_cube();
        kb += 1usize;
    }

    // accumulators -> shared staging -> each unit's 16 outputs (row u / 4, columns 16 (u % 4) ..)
    let so = wm * 16usize * LDS + wn * 32usize;
    cmma::store(&mut st[so..64usize * LDS], &c0, LDS as u32, cmma::MatrixLayout::RowMajor);
    cmma::store(&mut st[so + 16usize..64usize * LDS], &c1, LDS as u32, cmma::MatrixLayout::RowMajor);
    sync_cube();
    let mut g = Array::<f32>::new(16usize);
    #[unroll]
    for j in 0..16usize {
        g[j] = st[xr * LDS + part * 16usize + j];
    }
    if gate_up {
        sync_cube();
        cmma::store(&mut st[so..64usize * LDS], &c2, LDS as u32, cmma::MatrixLayout::RowMajor);
        cmma::store(&mut st[so + 16usize..64usize * LDS], &c3, LDS as u32, cmma::MatrixLayout::RowMajor);
        sync_cube();
        #[unroll]
        for j in 0..16usize {
            let gv = g[j];
            g[j] = gv / (1.0f32 + Exp::exp(-gv)) * st[xr * LDS + part * 16usize + j];
        }
    }
    if valid {
        let mut dst = t0 + xr;
        if grouped {
            dst = perm[t0 + xr] as usize;
        }
        #[unroll]
        for j in 0..16usize {
            let r = r0 + part * 16usize + j;
            if r < rows {
                out[dst * rows + r] = g[j];
            }
        }
    }
}

/// Sort a chunk's routed slots by expert: `perm` (slot ids, expert-major, ascending slot within an
/// expert) and the tiles (`[expert, start, count]`, at most `TILE` slots each). `routes` is the
/// `[expert, token]` pair list of the chunk.
pub fn group_slots(routes: &[u32], n_expert: usize) -> (Vec<u32>, Vec<u32>) {
    let slots = routes.len() / 2;
    let mut counts = vec![0usize; n_expert + 1];
    for s in 0..slots {
        counts[routes[2 * s] as usize + 1] += 1;
    }
    for e in 0..n_expert {
        counts[e + 1] += counts[e];
    }
    let starts = counts.clone();
    let mut fill = counts;
    let mut perm = vec![0u32; slots];
    for s in 0..slots {
        let e = routes[2 * s] as usize;
        perm[fill[e]] = s as u32;
        fill[e] += 1;
    }
    let mut tiles = Vec::new();
    for e in 0..n_expert {
        let (a, b) = (starts[e], starts[e + 1]);
        let mut s = a;
        while s < b {
            let c = TILE.min(b - s);
            tiles.extend_from_slice(&[e as u32, s as u32, c as u32]);
            s += c;
        }
    }
    (perm, tiles)
}

#[cfg(test)]
mod tests {
    use super::group_slots;

    #[test]
    fn slots_group_by_expert_in_tiles_of_at_most_64() {
        // 150 slots: expert 3 gets 130 (3 tiles: 64, 64, 2), expert 0 gets 20, expert 7 none
        let mut routes = Vec::new();
        for s in 0..150u32 {
            routes.extend_from_slice(&[if s % 15 < 2 { 0 } else { 3 }, s / 4]);
        }
        let (perm, tiles) = group_slots(&routes, 8);
        assert_eq!(perm.len(), 150);
        let mut seen = perm.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..150).collect::<Vec<u32>>(), "a permutation of the slots");
        let t: Vec<[u32; 3]> = tiles.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
        assert_eq!(t, vec![[0, 0, 20], [3, 20, 64], [3, 84, 64], [3, 148, 2]]);
        for [e, s, c] in t {
            for i in s..s + c {
                assert_eq!(routes[2 * perm[i as usize] as usize], e, "tile of expert {e} holds only its slots");
            }
        }
    }
}

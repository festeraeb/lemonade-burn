//! Q5_K device kernels (the `qwen35moe` family: Q5_K attention/SSM-gate/embedding/expert gate+up, with Q6_K expert down and
//! Q8_0 shared experts, whose kernels already exist in `lib.rs`).
//!
//! Layout of a GGUF Q5_K super-block (256 values, 176 bytes): `d` f16, `dmin` f16, `scales[12]` (eight 6-bit scales and eight
//! 6-bit mins), `qh[32]` (fifth bits), `qs[128]` (low nibbles). Value `32 g + l` (g in 0..8, l in 0..32) is
//! `d * sc[g] * q - dmin * mn[g]` with `q = nibble + 16 * bit`, the nibble from `qs[32 (g/2) + l]` (low for even g, high for odd)
//! and the bit `(qh[l] >> g) & 1`. The CPU reference is `lb_gguf::types::dequant_q5_k_block`, bit-identical to llama.cpp's
//! `dequantize_row_q5_K` on every Q5_K row of Qwen3.6-35B-A3B (`scripts/qwen/dequant_check.sh`).
//!
//! Device layout (same idea as Q6_K, [`crate::repack_q6_k`]): the 5-bit quants are widened to one byte each and the block's
//! two f32 factors are folded in at repack, so the kernel has no bit surgery and never sees `d`:
//!   * `qv`: 16-value half groups, one `Vector<u32, 4>` each (value `u` of the half group in word `u % 4`, byte `u / 4`);
//!   * `sd`, `mn`: per half group, `d * sc[g]` and `dmin * mn[g]` (both halves of a 32-group carry the same pair).
//! A half group's partial is `sd * sum(q x) - mn * sum(x)`: the per-value arithmetic of the reference, summed in a different
//! order (hence the f32-summation tolerance in the tests).
//!
//! Routed (MoE) variants: all experts of a layer concatenated, routes as `[expert, token]` u32 pairs, exactly as the Q4_0 routed
//! kernels in `lib.rs`.

use crate::store::Gpu;
use crate::qwen_ops::sigmoid_f;
use crate::{cube_count, f16_to_f32, lane_sum, q6_row_partial, V4F, V4U};
use cubecl::prelude::*;
use lb_gguf::{GgmlType, QuantMat};

type Handle = cubecl::server::Handle;

/// Repack Q5_K super-blocks into the device layout. `bytes` is the GGUF data, `n_super` super-blocks of 176 bytes.
pub fn repack_q5_k(bytes: &[u8], n_super: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    use rayon::prelude::*;
    let mut qv = vec![0u32; n_super * 64];
    let mut sd = vec![0f32; n_super * 16];
    let mut mn = vec![0f32; n_super * 16];
    qv.par_chunks_mut(64).zip(sd.par_chunks_mut(16)).zip(mn.par_chunks_mut(16)).enumerate().for_each(|(b, ((qv, sd), mn))| {
        let blk = &bytes[b * 176..(b + 1) * 176];
        let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let qs = &blk[48..176];
        for g in 0..8usize {
            let (sc, m) = if g < 4 {
                (scales[g] & 63, scales[g + 4] & 63)
            } else {
                ((scales[g + 4] & 0x0f) | ((scales[g - 4] >> 6) << 4), (scales[g + 4] >> 4) | ((scales[g] >> 6) << 4))
            };
            let (s, mm) = (d * sc as f32, dmin * m as f32);
            sd[2 * g] = s;
            sd[2 * g + 1] = s;
            mn[2 * g] = mm;
            mn[2 * g + 1] = mm;
            for l in 0..32usize {
                let q = if g % 2 == 0 { qs[32 * (g / 2) + l] & 0x0f } else { qs[32 * (g / 2) + l] >> 4 } as u32
                    + if (qh[l] >> g) & 1 != 0 { 16 } else { 0 };
                let (half, u) = (l / 16, l % 16);
                qv[(2 * g + half) * 4 + u % 4] |= q << (8 * (u / 4) as u32);
            }
        }
    });
    (qv, sd, mn)
}

/// Lane partial of one row: half groups `lane, lane + width, ...` (16 values each).
#[cube]
fn q5_row_partial(qv: &[V4U], sd: &[f32], mn: &[f32], x: &[V4F], hg0: usize, x4_0: usize, hpr: usize, lane: usize, width: usize) -> f32 {
    let mut acc = V4F::new(0.0f32);
    let mut g = lane;
    while g < hpr {
        let gg = hg0 + g;
        let w = qv[gg];
        let xb = x4_0 + g * 4usize;
        let mut part = V4F::new(0.0f32);
        let mut xs = V4F::new(0.0f32);
        #[unroll]
        for j in 0..4u32 {
            let q = (w >> V4U::new(8u32 * j)) & V4U::new(255u32);
            let xv = x[xb + j as usize];
            part += V4F::cast_from(q) * xv;
            xs += xv;
        }
        acc += V4F::new(sd[gg]) * part - V4F::new(mn[gg]) * xs;
        g += width;
    }
    acc.extract(0usize) + acc.extract(1usize) + acc.extract(2usize) + acc.extract(3usize)
}

/// Q5_K matvec, `dims` = [rows, cols]; one plane (or cube, in the fallback) per output, grid-strided.
#[cube(launch)]
fn q5_k_matvec_kernel(qv: &[V4U], sd: &[f32], mn: &[f32], x: &[V4F], dims: &[u32], out: &mut [f32], #[comptime] use_plane: bool) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let hpr = cols / 16usize;
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
        let part = q5_row_partial(qv, sd, mn, x, r * hpr, t * cols / 4usize, hpr, lane, width);
        let total = lane_sum(&mut sm, part, use_plane);
        if lane == 0usize {
            out[o] = total;
        }
        o += step;
    }
}

/// One routed MoE layer's gate+up+silu with Q5_K experts concatenated: `h[slot, r] = silu(gate_e r . x) * (up_e r . x)`.
/// `routes` = `[expert, token]` pairs from slot `dims[2]`; `dims` = [rows, cols, base].
#[cube(launch)]
fn q5_k_routed_gate_up_silu_kernel(
    gq: &[V4U], gs: &[f32], gm: &[f32],
    uq: &[V4U], us: &[f32], um: &[f32],
    x: &[V4F], routes: &[u32], dims: &[u32], h: &mut [f32], #[comptime] use_plane: bool,
) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let hpr = cols / 16usize;
    let hpe = rows * hpr;
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
        let hg0 = e * hpe + r * hpr;
        let x4 = tok * cols / 4usize;
        let pg = q5_row_partial(gq, gs, gm, x, hg0, x4, hpr, lane, width);
        let pu = q5_row_partial(uq, us, um, x, hg0, x4, hpr, lane, width);
        let g = lane_sum(&mut sm, pg, use_plane);
        let u = lane_sum(&mut sm, pu, use_plane);
        if lane == 0usize {
            h[o] = g / (1.0f32 + Exp::exp(-g)) * u;
        }
        o += step;
    }
}

/// Routed `down` with Q6_K experts concatenated; slot `s` reads `h[s * cols ..]`.
#[cube(launch)]
fn q6_k_routed_matvec_kernel(qv: &[V4U], sd: &[f32], h: &[V4F], routes: &[u32], dims: &[u32], out: &mut [f32], #[comptime] use_plane: bool) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let gpr = cols / 16usize;
    let gpe = rows * gpr;
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
        let p = q6_row_partial(qv, sd, h, e * gpe + r * gpr, slot * cols / 4usize, gpr, lane, width);
        let v = lane_sum(&mut sm, p, use_plane);
        if lane == 0usize {
            out[o] = v;
        }
        o += step;
    }
}

/// Routed `down` with Q5_K experts (20 of Qwen3.6's 40 layers; the other 20 use Q6_K): slot `s` reads `h[s * cols ..]`.
#[cube(launch)]
fn q5_k_routed_matvec_kernel(qv: &[V4U], sd: &[f32], mn: &[f32], h: &[V4F], routes: &[u32], dims: &[u32], out: &mut [f32], #[comptime] use_plane: bool) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let hpr = cols / 16usize;
    let hpe = rows * hpr;
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
        let p = q5_row_partial(qv, sd, mn, h, e * hpe + r * hpr, slot * cols / 4usize, hpr, lane, width);
        let v = lane_sum(&mut sm, p, use_plane);
        if lane == 0usize {
            out[o] = v;
        }
        o += step;
    }
}

/// Byte `i` (0..12) of the packed Q4_K/Q5_K scale block held in three words.
#[cube]
fn sbyte(s0: u32, s1: u32, s2: u32, i: u32) -> u32 {
    let mut w = s0;
    if i >= 8u32 {
        w = s2;
    } else if i >= 4u32 {
        w = s1;
    }
    (w >> ((i % 4u32) * 8u32)) & 255u32
}

#[cube]
fn k4_scale(s0: u32, s1: u32, s2: u32, g: u32) -> u32 {
    let mut r = sbyte(s0, s1, s2, g) & 63u32;
    if g >= 4u32 {
        r = (sbyte(s0, s1, s2, g + 4u32) & 15u32) | ((sbyte(s0, s1, s2, g - 4u32) >> 6u32) << 4u32);
    }
    r
}

#[cube]
fn k4_min(s0: u32, s1: u32, s2: u32, g: u32) -> u32 {
    let mut r = sbyte(s0, s1, s2, g + 4u32) & 63u32;
    if g >= 4u32 {
        r = (sbyte(s0, s1, s2, g + 4u32) >> 4u32) | ((sbyte(s0, s1, s2, g) >> 6u32) << 4u32);
    }
    r
}

/// Permute an activation row inside every 16 values (a 4x4 transpose): `y[16a + 4j + i] = x[16a + 4i + j]`. The native Q5_K kernel
/// reads four file words at once (lane i = word i, byte j = value 4i + j), so its x vectors must hold values `4i + j` across `i`.
#[cube(launch)]
fn perm16_kernel(x: &[f32], y: &mut [f32]) {
    let o = ABSOLUTE_POS as usize;
    if o < y.len() {
        let a = o / 16usize;
        let r = o % 16usize;
        let j = r / 4usize;
        let i = r % 4usize;
        y[o] = x[a * 16usize + 4usize * i + j];
    }
}

/// Byte `k` of each of the four words of `w` as an exact float: `(byte | 0x4B000000)` is `2^23 + byte`, minus `2^23` (`off` = the
/// constant to subtract, `8388608.0` plus any offset of the quant, e.g. 32 for Q6_K, exactly representable). Two integer ops and one add
/// where an int-to-float convert costs a quarter-rate slot (Pascal).
#[cube]
fn u2f_bytes(w: V4U, k: u32, off: f32) -> V4F {
    let b = ((w >> V4U::new(8u32 * k)) & V4U::new(255u32)) | V4U::new(1258291200u32);
    V4F::reinterpret(b) - V4F::new(off)
}

/// Lane partial of one Q5_K row at file density. The nibble/high-bit assembly is done on whole words (4 values an instruction), then
/// each value costs one extract, one add and one FMA; same products as the scalar decode, summed in a different order.
#[cube]
fn q5n_partial(w: &[V4U], x: &[V4F], blk0: usize, xrow4: usize, bpr: usize, lane: usize, width: usize) -> f32 {
    let items = bpr * 4usize;
    let mut acc = 0.0f32;
    let mut it = lane;
    while it < items {
        let b = it / 4usize;
        let j = it % 4usize;
        let base = (blk0 + b) * 11usize;
        let hdr = w[base];
        let w0 = hdr.extract(0usize);
        let d = f16_to_f32(w0 & 65535u32);
        let dmin = f16_to_f32((w0 >> 16u32) & 65535u32);
        let s0 = hdr.extract(1usize);
        let s1 = hdr.extract(2usize);
        let s2 = hdr.extract(3usize);
        let g1 = (2usize * j) as u32;
        let g2 = g1 + 1u32;
        let sc1 = d * (k4_scale(s0, s1, s2, g1) as f32);
        let sc2 = d * (k4_scale(s0, s1, s2, g2) as f32);
        let mn1 = dmin * (k4_min(s0, s1, s2, g1) as f32);
        let mn2 = dmin * (k4_min(s0, s1, s2, g2) as f32);
        let xb = xrow4 + b * 64usize + 16usize * j;
        let mut p1 = V4F::new(0.0f32);
        let mut p2 = V4F::new(0.0f32);
        let mut x1 = V4F::new(0.0f32);
        let mut x2 = V4F::new(0.0f32);
        let m4 = V4U::new(252645135u32);
        let m16 = V4U::new(269488144u32);
        #[unroll]
        for a in 0..2usize {
            let hv = w[base + 1usize + a];
            let qv = w[base + 3usize + 2usize * j + a];
            let w1 = (qv & m4) | (((hv >> V4U::new(g1)) << V4U::new(4u32)) & m16);
            let w2 = ((qv >> V4U::new(4u32)) & m4) | (((hv >> V4U::new(g2)) << V4U::new(4u32)) & m16);
            #[unroll]
            for k in 0..4u32 {
                let xv1 = x[xb + 4usize * a + k as usize];
                let xv2 = x[xb + 8usize + 4usize * a + k as usize];
                p1 += u2f_bytes(w1, k, 8388608.0f32) * xv1;
                p2 += u2f_bytes(w2, k, 8388608.0f32) * xv2;
                x1 += xv1;
                x2 += xv2;
            }
        }
        let sp1 = p1.extract(0usize) + p1.extract(1usize) + p1.extract(2usize) + p1.extract(3usize);
        let sp2 = p2.extract(0usize) + p2.extract(1usize) + p2.extract(2usize) + p2.extract(3usize);
        let sx1 = x1.extract(0usize) + x1.extract(1usize) + x1.extract(2usize) + x1.extract(3usize);
        let sx2 = x2.extract(0usize) + x2.extract(1usize) + x2.extract(2usize) + x2.extract(3usize);
        acc += sc1 * sp1 - mn1 * sx1 + sc2 * sp2 - mn2 * sx2;
        it += width;
    }
    acc
}


// ---- row-batched K-quant matvec: a plane takes `r` consecutive rows and loads every activation vector once for all of them
// (the one-row kernels re-read 256 bytes of activations for every 80 bytes of Q5_K weights: the loads, not the arithmetic, set the time).

/// `acc[i] += ` this lane's partial of row `i` (`i < r`; rows `blk0 + i * bpr` blocks apart), Q5_K file density; rows `i >= split` read
/// their blocks from `w2` (the up matrix next to the gate matrix: the same activation vectors serve both).
#[cube]
fn q5n_rows2(w: &[V4U], w2: &[V4U], x: &[V4F], blk0: usize, xrow4: usize, bpr: usize, lane: usize, width: usize, acc: &mut Array<f32>, #[comptime] r: usize, #[comptime] split: usize) {
    let items = bpr * 4usize;
    let mut it = lane;
    while it < items {
        let b = it / 4usize;
        let j = it % 4usize;
        let g1 = (2usize * j) as u32;
        let g2 = g1 + 1u32;
        let xb = xrow4 + b * 64usize + 16usize * j;
        // all loads first: 16 activation vectors once, then each row's header, qh and qs
        let mut xa = Array::<V4F>::new(16usize);
        let mut x1 = V4F::new(0.0f32);
        let mut x2 = V4F::new(0.0f32);
        #[unroll]
        for a in 0..2usize {
            #[unroll]
            for kk in 0..4usize {
                let v1 = x[xb + 4usize * a + kk];
                let v2 = x[xb + 8usize + 4usize * a + kk];
                xa[(a * 4usize + kk) * 2usize] = v1;
                xa[(a * 4usize + kk) * 2usize + 1usize] = v2;
                x1 += v1;
                x2 += v2;
            }
        }
        let sx1 = x1.extract(0usize) + x1.extract(1usize) + x1.extract(2usize) + x1.extract(3usize);
        let sx2 = x2.extract(0usize) + x2.extract(1usize) + x2.extract(2usize) + x2.extract(3usize);
        let m4 = V4U::new(252645135u32);
        let m16 = V4U::new(269488144u32);
        #[unroll]
        for i in 0..r {
            let ri = comptime!(if i >= split { i - split } else { i });
            let base = (blk0 + ri * bpr + b) * 11usize;
            let mut hdr = w[base];
            let mut h0 = w[base + 1usize];
            let mut h1v = w[base + 2usize];
            let mut q0 = w[base + 3usize + 2usize * j];
            let mut q1 = w[base + 4usize + 2usize * j];
            if comptime!(i >= split) {
                hdr = w2[base];
                h0 = w2[base + 1usize];
                h1v = w2[base + 2usize];
                q0 = w2[base + 3usize + 2usize * j];
                q1 = w2[base + 4usize + 2usize * j];
            }
            let w0 = hdr.extract(0usize);
            let d = f16_to_f32(w0 & 65535u32);
            let dmin = f16_to_f32((w0 >> 16u32) & 65535u32);
            let s0 = hdr.extract(1usize);
            let s1 = hdr.extract(2usize);
            let s2 = hdr.extract(3usize);
            let sc1 = d * (k4_scale(s0, s1, s2, g1) as f32);
            let sc2 = d * (k4_scale(s0, s1, s2, g2) as f32);
            let mn1 = dmin * (k4_min(s0, s1, s2, g1) as f32);
            let mn2 = dmin * (k4_min(s0, s1, s2, g2) as f32);
            let mut p1 = V4F::new(0.0f32);
            let mut p2 = V4F::new(0.0f32);
            #[unroll]
            for a in 0..2usize {
                let mut hv = h0;
                let mut qv = q0;
                if a == 1usize {
                    hv = h1v;
                    qv = q1;
                }
                let wd1 = (qv & m4) | (((hv >> V4U::new(g1)) << V4U::new(4u32)) & m16);
                let wd2 = ((qv >> V4U::new(4u32)) & m4) | (((hv >> V4U::new(g2)) << V4U::new(4u32)) & m16);
                #[unroll]
                for kk in 0..4u32 {
                    p1 += u2f_bytes(wd1, kk, 8388608.0f32) * xa[(a * 4usize + kk as usize) * 2usize];
                    p2 += u2f_bytes(wd2, kk, 8388608.0f32) * xa[(a * 4usize + kk as usize) * 2usize + 1usize];
                }
            }
            let sp1 = p1.extract(0usize) + p1.extract(1usize) + p1.extract(2usize) + p1.extract(3usize);
            let sp2 = p2.extract(0usize) + p2.extract(1usize) + p2.extract(2usize) + p2.extract(3usize);
            acc[i] += sc1 * sp1 - mn1 * sx1 + sc2 * sp2 - mn2 * sx2;
        }
        it += width;
    }
}

/// Q5_K native matvec, 4 rows a plane, activations shared across the rows; needs plane ops and `rows % 4 == 0`.
#[cube(launch)]
fn q5_k_native_rows4_kernel(w: &[V4U], x: &[V4F], dims: &[u32], out: &mut [f32], #[comptime] rr: usize) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let bpr = cols / 256usize;
    let lane = UNIT_POS_PLANE as usize;
    let plane_id = (UNIT_POS / PLANE_DIM) as usize;
    let ppc = CUBE_DIM_X as usize / 32usize;
    let step = CUBE_COUNT as usize * ppc;
    let groups = out.len() / rr;
    let mut g = CUBE_POS as usize * ppc + plane_id;
    while g < groups {
        let o0 = g * rr;
        let t = o0 / rows;
        let r0 = o0 % rows;
        let mut acc = Array::<f32>::new(rr);
        #[unroll]
        for i in 0..rr {
            acc[i] = 0.0f32;
        }
        q5n_rows(w, x, r0 * bpr, t * cols / 4usize, bpr, lane, 32usize, &mut acc, rr);
        #[unroll]
        for i in 0..rr {
            let sv = plane_sum(acc[i]);
            if lane == 0usize {
                out[o0 + i] = sv;
            }
        }
        g += step;
    }
}

/// Native-layout Q5_K matvec: the GGUF bytes as they are (176-byte blocks = 11 `Vector<u32,4>`, 0.6875 bytes a value), no repack.
///
/// The widened layout above costs 1.5 bytes a value (2.2x the file): 26 GB of Qwen3.6 Q5_K_M would be about 50 GB on the card.
/// This kernel keeps the file's density. One lane takes one 64-value chunk of a block (low nibbles = group 2j, high = group 2j+1):
/// five 16-byte loads (header, two of `qh`, two of `qs`) and sixteen activation vector loads, four values per lane-vector.
/// `x` must be permuted with `perm16_kernel`. `dims` = [rows, cols].
#[cube(launch)]
fn q5_k_native_matvec_kernel(w: &[V4U], x: &[V4F], dims: &[u32], out: &mut [f32], #[comptime] use_plane: bool) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let bpr = cols / 256usize;
    let items = bpr * 4usize;
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
        let acc = q5n_partial(w, x, r * bpr, t * cols / 4usize, bpr, lane, width);
        let total = lane_sum(&mut sm, acc, use_plane);
        if lane == 0usize {
            out[o] = total;
        }
        o += step;
    }
}

/// A Q5_K matrix resident on the device in the file's own layout.
pub struct GpuQ5KNative {
    pub rows: usize,
    pub cols: usize,
    w: Handle,
    n_words: usize,
    dims: Handle,
}

impl GpuQ5KNative {
    pub fn upload(gpu: &Gpu, w: QuantMat<'_>) -> Result<Self, String> {
        if w.ty != GgmlType::Q5K || w.cols % 256 != 0 {
            return Err("expected Q5_K with cols % 256 == 0".into());
        }
        let words: Vec<u32> = w.bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        Ok(Self { rows: w.rows, cols: w.cols, n_words: words.len(), w: gpu.upload_u32(words), dims: gpu.const_u32(&crate::matvec_dims(w.rows, w.cols)) })
    }

    pub fn bytes_on_device(&self) -> usize {
        self.n_words * 4
    }

    /// `out = x @ W^T`; permutes `x` first (one small launch). Callers sharing one activation across matrices use
    /// [`permute_x`](Self::permute_x) once and [`launch_permuted`](Self::launch_permuted).
    pub fn launch(&self, gpu: &Gpu, x: &Handle, m: usize) -> Handle {
        let xp = Self::permute_x(gpu, x, m * self.cols);
        self.launch_permuted(gpu, &xp, m)
    }

    pub fn permute_x(gpu: &Gpu, x: &Handle, n: usize) -> Handle {
        let y = gpu.scratch(n * 4);
        unsafe {
            perm16_kernel::launch(&gpu.client, cube_count(n, gpu.cube_dim), CubeDim::new_1d(gpu.cube_dim as u32), BufferArg::from_raw_parts(x.clone(), n), BufferArg::from_raw_parts(y.clone(), n));
        }
        y
    }

    pub fn launch_permuted(&self, gpu: &Gpu, xp: &Handle, m: usize) -> Handle {
        let n_out = m * self.rows;
        let out = gpu.scratch(n_out * 4);
        let rr: usize = std::env::var("LB_QW_R").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
        if gpu.use_plane && gpu.plane_size.1 == 32 && self.rows % rr == 0 && std::env::var("LB_QW_ROWS1").is_err() {
            unsafe {
                q5_k_native_rows4_kernel::launch(
                    &gpu.client,
                    cube_count(n_out / rr, gpu.per_cube()),
                    CubeDim::new_1d(gpu.cube_dim as u32),
                    BufferArg::from_raw_parts(self.w.clone(), self.n_words / 4),
                    BufferArg::from_raw_parts(xp.clone(), m * self.cols / 4),
                    BufferArg::from_raw_parts(self.dims.clone(), 2),
                    BufferArg::from_raw_parts(out.clone(), n_out),
                    rr,
                );
            }
            return out;
        }
        unsafe {
            q5_k_native_matvec_kernel::launch(
                &gpu.client,
                cube_count(n_out, gpu.per_cube()),
                CubeDim::new_1d(gpu.cube_dim as u32),
                BufferArg::from_raw_parts(self.w.clone(), self.n_words / 4),
                BufferArg::from_raw_parts(xp.clone(), m * self.cols / 4),
                BufferArg::from_raw_parts(self.dims.clone(), 2),
                BufferArg::from_raw_parts(out.clone(), n_out),
                gpu.use_plane,
            );
        }
        out
    }

    pub fn matvec(&self, gpu: &Gpu, x: &[f32], m: usize) -> Result<Vec<f32>, String> {
        let xh = gpu.upload_f32(x);
        let out = self.launch(gpu, &xh, m);
        Ok(gpu.read_all(vec![(out, m * self.rows)])?.remove(0))
    }
}

/// A Q5_K matrix resident on the device.
pub struct GpuQ5KMat {
    pub rows: usize,
    pub cols: usize,
    qv: Handle,
    sd: Handle,
    mn: Handle,
    n_qv: usize,
    n_sd: usize,
    dims: Handle,
}

impl GpuQ5KMat {
    pub fn upload(gpu: &Gpu, w: QuantMat<'_>) -> Result<Self, String> {
        if w.ty != GgmlType::Q5K {
            return Err(format!("expected Q5_K, got {}", w.ty.name()));
        }
        if w.cols % 256 != 0 {
            return Err(format!("Q5_K needs cols % 256 == 0, got {}", w.cols));
        }
        let (qv, sd, mn) = repack_q5_k(w.bytes, w.rows * (w.cols / 256));
        let dims = crate::matvec_dims(w.rows, w.cols);
        Ok(Self { rows: w.rows, cols: w.cols, n_qv: qv.len(), n_sd: sd.len(), qv: gpu.upload_u32(qv), sd: gpu.upload_f32(&sd), mn: gpu.upload_f32(&mn), dims: gpu.const_u32(&dims) })
    }

    pub fn bytes_on_device(&self) -> usize {
        (self.n_qv + 2 * self.n_sd) * 4
    }

    /// `out = x @ W^T` for `m` rows of `x` already on the device.
    pub fn launch(&self, gpu: &Gpu, x: &Handle, m: usize) -> Handle {
        let n_out = m * self.rows;
        let out = gpu.scratch(n_out * 4);
        unsafe {
            q5_k_matvec_kernel::launch(
                &gpu.client,
                cube_count(n_out, gpu.per_cube()),
                CubeDim::new_1d(gpu.cube_dim as u32),
                BufferArg::from_raw_parts(self.qv.clone(), self.n_qv / 4),
                BufferArg::from_raw_parts(self.sd.clone(), self.n_sd),
                BufferArg::from_raw_parts(self.mn.clone(), self.n_sd),
                BufferArg::from_raw_parts(x.clone(), m * self.cols / 4),
                BufferArg::from_raw_parts(self.dims.clone(), 2),
                BufferArg::from_raw_parts(out.clone(), n_out),
                gpu.use_plane,
            );
        }
        gpu.tr("q5_matvec", self.rows);
        out
    }

    pub fn matvec(&self, gpu: &Gpu, x: &[f32], m: usize) -> Result<Vec<f32>, String> {
        let xh = gpu.upload_f32(x);
        let out = self.launch(gpu, &xh, m);
        Ok(gpu.read_all(vec![(out, m * self.rows)])?.remove(0))
    }
}

/// Q5_K matvec on the GPU, same contract as `q6_k_matvec_gpu`.
pub fn q5_k_matvec_gpu(w: QuantMat<'_>, x: &[f32], m: usize) -> Result<Vec<f32>, String> {
    if x.len() != m * w.cols {
        return Err(format!("x has {} values, expected {} x {}", x.len(), m, w.cols));
    }
    if m * w.rows == 0 {
        return Ok(Vec::new());
    }
    let gpu = Gpu::open()?;
    GpuQ5KMat::upload(&gpu, w)?.matvec(&gpu, x, m)
}

/// All experts of one `qwen35moe` layer on the device: gate and up Q5_K `[n_expert][ff, embd]`, down Q6_K or Q5_K `[n_expert][embd, ff]`.
pub struct GpuKExpertLayer {
    pub n_expert: usize,
    pub embd: usize,
    pub ff: usize,
    gq: Handle, gs: Handle, gm: Handle,
    uq: Handle, us: Handle, um: Handle,
    dq: Handle, ds: Handle, dm: Option<Handle>,
    n_q5: usize, n_s5: usize, n_dq: usize, n_ds: usize,
}

impl GpuKExpertLayer {
    /// `gate`/`up`: `n_expert * ff` Q5_K rows of `embd`; `down`: `n_expert * embd` Q6_K or Q5_K rows of `ff`.
    pub fn upload(gpu: &Gpu, n_expert: usize, gate: QuantMat<'_>, up: QuantMat<'_>, down: QuantMat<'_>) -> Result<Self, String> {
        if gate.ty != GgmlType::Q5K || up.ty != GgmlType::Q5K || !matches!(down.ty, GgmlType::Q6K | GgmlType::Q5K) {
            return Err("expected Q5_K gate/up and Q6_K or Q5_K down".into());
        }
        let (ff, embd) = (gate.rows / n_expert, gate.cols);
        if gate.rows != n_expert * ff || up.rows != gate.rows || down.rows != n_expert * embd || down.cols != ff || embd % 256 != 0 || ff % 256 != 0 {
            return Err("shape mismatch".into());
        }
        let (gq, gs, gm) = repack_q5_k(gate.bytes, gate.rows * (embd / 256));
        let (uq, us, um) = repack_q5_k(up.bytes, up.rows * (embd / 256));
        let (dq, ds, dm) = if down.ty == GgmlType::Q6K {
            let (dq, ds) = crate::repack_q6_k(down.bytes, down.rows * (ff / 256));
            (dq, ds, None)
        } else {
            let (dq, ds, dm) = repack_q5_k(down.bytes, down.rows * (ff / 256));
            (dq, ds, Some(dm))
        };
        let (n_q5, n_s5, n_dq, n_ds) = (gq.len(), gs.len(), dq.len(), ds.len());
        Ok(Self {
            n_expert, embd, ff, n_q5, n_s5, n_dq, n_ds,
            gq: gpu.upload_u32(gq), gs: gpu.upload_f32(&gs), gm: gpu.upload_f32(&gm),
            uq: gpu.upload_u32(uq), us: gpu.upload_f32(&us), um: gpu.upload_f32(&um),
            dq: gpu.upload_u32(dq), ds: gpu.upload_f32(&ds), dm: dm.map(|m| gpu.upload_f32(&m)),
        })
    }

    /// Run the routed FFN for `routes` (`(expert, token)` per slot) on `x` (`tokens` rows of `embd`):
    /// returns per-slot outputs `[slots, embd]` (no router weights applied).
    pub fn forward(&self, gpu: &Gpu, x: &[f32], routes: &[(u32, u32)]) -> Result<Vec<f32>, String> {
        let slots = routes.len();
        let xh = gpu.upload_f32(x);
        let rt: Vec<u32> = routes.iter().flat_map(|&(e, t)| [e, t]).collect();
        let rh = gpu.upload_u32(rt.clone());
        let dims_gu = gpu.const_u32(&[self.ff as u32, self.embd as u32, 0]);
        let dims_dn = gpu.const_u32(&[self.embd as u32, self.ff as u32, 0]);
        let h = gpu.scratch(slots * self.ff * 4);
        let out = gpu.scratch(slots * self.embd * 4);
        unsafe {
            q5_k_routed_gate_up_silu_kernel::launch(
                &gpu.client,
                cube_count(slots * self.ff, gpu.per_cube()),
                CubeDim::new_1d(gpu.cube_dim as u32),
                BufferArg::from_raw_parts(self.gq.clone(), self.n_q5 / 4),
                BufferArg::from_raw_parts(self.gs.clone(), self.n_s5),
                BufferArg::from_raw_parts(self.gm.clone(), self.n_s5),
                BufferArg::from_raw_parts(self.uq.clone(), self.n_q5 / 4),
                BufferArg::from_raw_parts(self.us.clone(), self.n_s5),
                BufferArg::from_raw_parts(self.um.clone(), self.n_s5),
                BufferArg::from_raw_parts(xh.clone(), x.len() / 4),
                BufferArg::from_raw_parts(rh.clone(), rt.len()),
                BufferArg::from_raw_parts(dims_gu, 3),
                BufferArg::from_raw_parts(h.clone(), slots * self.ff),
                gpu.use_plane,
            );
            if let Some(dm) = &self.dm {
                q5_k_routed_matvec_kernel::launch(
                    &gpu.client,
                    cube_count(slots * self.embd, gpu.per_cube()),
                    CubeDim::new_1d(gpu.cube_dim as u32),
                    BufferArg::from_raw_parts(self.dq.clone(), self.n_dq / 4),
                    BufferArg::from_raw_parts(self.ds.clone(), self.n_ds),
                    BufferArg::from_raw_parts(dm.clone(), self.n_ds),
                    BufferArg::from_raw_parts(h.clone(), slots * self.ff / 4),
                    BufferArg::from_raw_parts(rh.clone(), rt.len()),
                    BufferArg::from_raw_parts(dims_dn, 3),
                    BufferArg::from_raw_parts(out.clone(), slots * self.embd),
                    gpu.use_plane,
                );
            } else {
                q6_k_routed_matvec_kernel::launch(
                    &gpu.client,
                    cube_count(slots * self.embd, gpu.per_cube()),
                    CubeDim::new_1d(gpu.cube_dim as u32),
                    BufferArg::from_raw_parts(self.dq.clone(), self.n_dq / 4),
                    BufferArg::from_raw_parts(self.ds.clone(), self.n_ds),
                    BufferArg::from_raw_parts(h.clone(), slots * self.ff / 4),
                    BufferArg::from_raw_parts(rh.clone(), rt.len()),
                    BufferArg::from_raw_parts(dims_dn, 3),
                    BufferArg::from_raw_parts(out.clone(), slots * self.embd),
                    gpu.use_plane,
                );
            }
        }
        Ok(gpu.read_all(vec![(out, slots * self.embd)])?.remove(0))
    }
}

// ---------------------------------------------------------------------------------------------------------------------------
// Q6_K and Q8_0 at file density. Their blocks (210 and 34 bytes) are not 16-byte aligned, so the bytes are split at upload into
// aligned arrays without changing a single stored value or the byte count: Q6_K -> the 192 bytes of `ql`+`qh` (12 vectors a block),
// the 16 i8 scales (one vector), the f16 `d` packed two to a word (210 B a block); Q8_0 -> the 32 quants (2 vectors a block) and the
// f16 `d` packed two to a word (34 B a block). Both read activations permuted by `perm16_kernel`.
// ---------------------------------------------------------------------------------------------------------------------------

/// Lane partial of one Q6_K row at file density (half blocks `lane, lane + width, ...`), word-wide nibble/2-bit assembly as in Q5_K.
#[cube]
fn q6n_partial(qq: &[V4U], sc: &[V4U], dd: &[u32], x: &[V4F], blk0: usize, xrow4: usize, bpr: usize, lane: usize, width: usize) -> f32 {
    let items = bpr * 2usize;
    let mut acc = 0.0f32;
    let mut it = lane;
    while it < items {
        let gb = blk0 + it / 2usize;
        let h = it % 2usize;
        let d = f16_to_f32((dd[gb / 2usize] >> (((gb % 2usize) * 16usize) as u32)) & 65535u32);
        let sv = sc[gb];
        let mut w0 = sv.extract(0usize);
        let mut w1 = sv.extract(1usize);
        if h == 1usize {
            w0 = sv.extract(2usize);
            w1 = sv.extract(3usize);
        }
        let xb = xrow4 + (it / 2usize) * 64usize + 32usize * h;
        let qb = gb * 12usize;
        let m4 = V4U::new(252645135u32);
        let m3 = V4U::new(50529027u32);
        let mut part = 0.0f32;
        #[unroll]
        for a in 0..2usize {
            let qla = qq[qb + 4usize * h + a];
            let qlb = qq[qb + 4usize * h + a + 2usize];
            let qhv = qq[qb + 8usize + 2usize * h + a];
            let l0 = qla & m4;
            let l1 = qlb & m4;
            let l2 = (qla >> V4U::new(4u32)) & m4;
            let l3 = (qlb >> V4U::new(4u32)) & m4;
            let v0 = l0 | ((qhv & m3) << V4U::new(4u32));
            let v1 = l1 | (((qhv >> V4U::new(2u32)) & m3) << V4U::new(4u32));
            let v2 = l2 | (((qhv >> V4U::new(4u32)) & m3) << V4U::new(4u32));
            let v3 = l3 | (((qhv >> V4U::new(6u32)) & m3) << V4U::new(4u32));
            let mut p0 = V4F::new(0.0f32);
            let mut p1 = V4F::new(0.0f32);
            let mut p2 = V4F::new(0.0f32);
            let mut p3 = V4F::new(0.0f32);
            #[unroll]
            for k in 0..4u32 {
                let xo = xb + 4usize * a + k as usize;
                // 8388608 + 32: the Q6_K offset folded into the exact subtract
                p0 += u2f_bytes(v0, k, 8388640.0f32) * x[xo];
                p1 += u2f_bytes(v1, k, 8388640.0f32) * x[xo + 8usize];
                p2 += u2f_bytes(v2, k, 8388640.0f32) * x[xo + 16usize];
                p3 += u2f_bytes(v3, k, 8388640.0f32) * x[xo + 24usize];
            }
            let s0 = ((w0 >> ((8usize * a) as u32)) & 255u32) as f32;
            let s1 = ((w0 >> ((8usize * (a + 2usize)) as u32)) & 255u32) as f32;
            let s2 = ((w1 >> ((8usize * a) as u32)) & 255u32) as f32;
            let s3 = ((w1 >> ((8usize * (a + 2usize)) as u32)) & 255u32) as f32;
            let f0 = s0 - 256.0f32 * (((s0 as u32) >> 7u32) as f32);
            let f1 = s1 - 256.0f32 * (((s1 as u32) >> 7u32) as f32);
            let f2 = s2 - 256.0f32 * (((s2 as u32) >> 7u32) as f32);
            let f3 = s3 - 256.0f32 * (((s3 as u32) >> 7u32) as f32);
            let e0 = p0.extract(0usize) + p0.extract(1usize) + p0.extract(2usize) + p0.extract(3usize);
            let e1 = p1.extract(0usize) + p1.extract(1usize) + p1.extract(2usize) + p1.extract(3usize);
            let e2 = p2.extract(0usize) + p2.extract(1usize) + p2.extract(2usize) + p2.extract(3usize);
            let e3 = p3.extract(0usize) + p3.extract(1usize) + p3.extract(2usize) + p3.extract(3usize);
            part += f0 * e0 + f1 * e1 + f2 * e2 + f3 * e3;
        }
        acc += d * part;
        it += width;
    }
    acc
}


/// One weight array: [`q5n_rows2`] with no second matrix.
#[cube]
fn q5n_rows(w: &[V4U], x: &[V4F], blk0: usize, xrow4: usize, bpr: usize, lane: usize, width: usize, acc: &mut Array<f32>, #[comptime] r: usize) {
    q5n_rows2(w, w, x, blk0, xrow4, bpr, lane, width, acc, r, 1000usize);
}

/// Q6_K analogue of [`q5n_rows`]: `acc[i] +=` the lane's partial of row `i`; the 32 activation vectors of a half block are loaded once.
#[cube]
fn q6n_rows(qq: &[V4U], sc: &[V4U], dd: &[u32], x: &[V4F], blk0: usize, xrow4: usize, bpr: usize, lane: usize, width: usize, acc: &mut Array<f32>, #[comptime] r: usize) {
    let items = bpr * 2usize;
    let mut it = lane;
    while it < items {
        let bb = it / 2usize;
        let h = it % 2usize;
        let xb = xrow4 + bb * 64usize + 32usize * h;
        let mut xa = Array::<V4F>::new(32usize);
        #[unroll]
        for a in 0..2usize {
            #[unroll]
            for kk in 0..4usize {
                #[unroll]
                for mm in 0..4usize {
                    xa[((a * 4usize + kk) * 4usize) + mm] = x[xb + 4usize * a + kk + 8usize * mm];
                }
            }
        }
        let m4 = V4U::new(252645135u32);
        let m3 = V4U::new(50529027u32);
        #[unroll]
        for i in 0..r {
            let gb = blk0 + i * bpr + bb;
            let d = f16_to_f32((dd[gb / 2usize] >> (((gb % 2usize) * 16usize) as u32)) & 65535u32);
            let sv = sc[gb];
            let mut w0 = sv.extract(0usize);
            let mut w1 = sv.extract(1usize);
            if h == 1usize {
                w0 = sv.extract(2usize);
                w1 = sv.extract(3usize);
            }
            let qb = gb * 12usize;
            let mut part = 0.0f32;
            #[unroll]
            for a in 0..2usize {
                let qla = qq[qb + 4usize * h + a];
                let qlb = qq[qb + 4usize * h + a + 2usize];
                let qhv = qq[qb + 8usize + 2usize * h + a];
                let v0 = (qla & m4) | ((qhv & m3) << V4U::new(4u32));
                let v1 = (qlb & m4) | (((qhv >> V4U::new(2u32)) & m3) << V4U::new(4u32));
                let v2 = ((qla >> V4U::new(4u32)) & m4) | (((qhv >> V4U::new(4u32)) & m3) << V4U::new(4u32));
                let v3 = ((qlb >> V4U::new(4u32)) & m4) | (((qhv >> V4U::new(6u32)) & m3) << V4U::new(4u32));
                let mut p0 = V4F::new(0.0f32);
                let mut p1 = V4F::new(0.0f32);
                let mut p2 = V4F::new(0.0f32);
                let mut p3 = V4F::new(0.0f32);
                #[unroll]
                for kk in 0..4u32 {
                    let xi = ((a * 4usize + kk as usize) * 4usize);
                    p0 += u2f_bytes(v0, kk, 8388640.0f32) * xa[xi];
                    p1 += u2f_bytes(v1, kk, 8388640.0f32) * xa[xi + 1usize];
                    p2 += u2f_bytes(v2, kk, 8388640.0f32) * xa[xi + 2usize];
                    p3 += u2f_bytes(v3, kk, 8388640.0f32) * xa[xi + 3usize];
                }
                let s0 = ((w0 >> ((8usize * a) as u32)) & 255u32) as f32;
                let s1 = ((w0 >> ((8usize * (a + 2usize)) as u32)) & 255u32) as f32;
                let s2 = ((w1 >> ((8usize * a) as u32)) & 255u32) as f32;
                let s3 = ((w1 >> ((8usize * (a + 2usize)) as u32)) & 255u32) as f32;
                let f0 = s0 - 256.0f32 * (((s0 as u32) >> 7u32) as f32);
                let f1 = s1 - 256.0f32 * (((s1 as u32) >> 7u32) as f32);
                let f2 = s2 - 256.0f32 * (((s2 as u32) >> 7u32) as f32);
                let f3 = s3 - 256.0f32 * (((s3 as u32) >> 7u32) as f32);
                let e0 = p0.extract(0usize) + p0.extract(1usize) + p0.extract(2usize) + p0.extract(3usize);
                let e1 = p1.extract(0usize) + p1.extract(1usize) + p1.extract(2usize) + p1.extract(3usize);
                let e2 = p2.extract(0usize) + p2.extract(1usize) + p2.extract(2usize) + p2.extract(3usize);
                let e3 = p3.extract(0usize) + p3.extract(1usize) + p3.extract(2usize) + p3.extract(3usize);
                part += f0 * e0 + f1 * e1 + f2 * e2 + f3 * e3;
            }
            acc[i] += d * part;
        }
        it += width;
    }
}

/// Q6_K native matvec, 4 rows a plane, activations shared across the rows; needs plane ops and `rows % 4 == 0`.
#[cube(launch)]
fn q6_k_native_rows4_kernel(qq: &[V4U], sc: &[V4U], dd: &[u32], x: &[V4F], dims: &[u32], out: &mut [f32], #[comptime] rr: usize) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let bpr = cols / 256usize;
    let lane = UNIT_POS_PLANE as usize;
    let plane_id = (UNIT_POS / PLANE_DIM) as usize;
    let ppc = CUBE_DIM_X as usize / 32usize;
    let step = CUBE_COUNT as usize * ppc;
    let groups = out.len() / rr;
    let mut g = CUBE_POS as usize * ppc + plane_id;
    while g < groups {
        let o0 = g * rr;
        let t = o0 / rows;
        let r0 = o0 % rows;
        let mut acc = Array::<f32>::new(rr);
        #[unroll]
        for i in 0..rr {
            acc[i] = 0.0f32;
        }
        q6n_rows(qq, sc, dd, x, r0 * bpr, t * cols / 4usize, bpr, lane, 32usize, &mut acc, rr);
        #[unroll]
        for i in 0..rr {
            let sv = plane_sum(acc[i]);
            if lane == 0usize {
                out[o0 + i] = sv;
            }
        }
        g += step;
    }
}

/// Q6_K lane = one 128-value half block. `dims` = [rows, cols].
#[cube(launch)]
fn q6_k_native_matvec_kernel(qq: &[V4U], sc: &[V4U], dd: &[u32], x: &[V4F], dims: &[u32], out: &mut [f32], #[comptime] use_plane: bool) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let bpr = cols / 256usize;
    let items = bpr * 2usize;
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
        let acc = q6n_partial(qq, sc, dd, x, r * bpr, t * cols / 4usize, bpr, lane, width);
        let total = lane_sum(&mut sm, acc, use_plane);
        if lane == 0usize {
            out[o] = total;
        }
        o += step;
    }
}

/// Q8_0 lane = one 32-value block.
#[cube(launch)]
fn q8_0_native_matvec_kernel(qs: &[V4U], dd: &[u32], x: &[V4F], dims: &[u32], out: &mut [f32], #[comptime] use_plane: bool) {
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
        let mut acc = 0.0f32;
        let mut b = lane;
        while b < bpr {
            let gb = r * bpr + b;
            let d = f16_to_f32((dd[gb / 2usize] >> (((gb % 2usize) * 16usize) as u32)) & 65535u32);
            let xb = (t * cols + b * 32usize) / 4usize;
            let mut p = V4F::new(0.0f32);
            #[unroll]
            for a in 0..2usize {
                let qv = qs[gb * 2usize + a];
                #[unroll]
                for k in 0..4u32 {
                    let by = (qv >> V4U::new(8u32 * k)) & V4U::new(255u32);
                    let f = V4F::cast_from(by) - V4F::new(256.0f32) * V4F::cast_from(by >> V4U::new(7u32));
                    p += f * x[xb + 4usize * a + k as usize];
                }
            }
            acc += d * (p.extract(0usize) + p.extract(1usize) + p.extract(2usize) + p.extract(3usize));
            b += width;
        }
        let total = lane_sum(&mut sm, acc, use_plane);
        if lane == 0usize {
            out[o] = total;
        }
        o += step;
    }
}

fn pack_f16_pairs(bytes: &[u8], block: usize, off: usize, n_blocks: usize) -> Vec<u32> {
    let mut v = vec![0u32; n_blocks.div_ceil(2)];
    for b in 0..n_blocks {
        let h = u16::from_le_bytes([bytes[b * block + off], bytes[b * block + off + 1]]) as u32;
        v[b / 2] |= h << (16 * (b % 2) as u32);
    }
    v
}

fn bytes_to_words(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// A Q6_K matrix on the device at file density (210 bytes a block, split into aligned arrays).
pub struct GpuQ6KNative {
    pub rows: usize,
    pub cols: usize,
    qq: Handle,
    sc: Handle,
    dd: Handle,
    n: (usize, usize, usize),
    dims: Handle,
}

impl GpuQ6KNative {
    pub fn upload(gpu: &Gpu, w: QuantMat<'_>) -> Result<Self, String> {
        if w.ty != GgmlType::Q6K || w.cols % 256 != 0 {
            return Err("expected Q6_K with cols % 256 == 0".into());
        }
        let nb = w.rows * (w.cols / 256);
        let mut qq = Vec::with_capacity(nb * 192);
        let mut sc = Vec::with_capacity(nb * 16);
        for blk in w.bytes.chunks_exact(210) {
            qq.extend_from_slice(&blk[..192]);
            sc.extend_from_slice(&blk[192..208]);
        }
        let dd = pack_f16_pairs(w.bytes, 210, 208, nb);
        let (qq, sc) = (bytes_to_words(&qq), bytes_to_words(&sc));
        let n = (qq.len(), sc.len(), dd.len());
        Ok(Self { rows: w.rows, cols: w.cols, n, qq: gpu.upload_u32(qq), sc: gpu.upload_u32(sc), dd: gpu.upload_u32(dd), dims: gpu.const_u32(&crate::matvec_dims(w.rows, w.cols)) })
    }
    pub fn bytes_on_device(&self) -> usize {
        (self.n.0 + self.n.1 + self.n.2) * 4
    }
    pub fn launch(&self, gpu: &Gpu, x: &Handle, m: usize) -> Handle {
        let xp = GpuQ5KNative::permute_x(gpu, x, m * self.cols);
        self.launch_permuted(gpu, &xp, m)
    }
    pub fn launch_permuted(&self, gpu: &Gpu, xp: &Handle, m: usize) -> Handle {
        let n_out = m * self.rows;
        let out = gpu.scratch(n_out * 4);
        let rr: usize = std::env::var("LB_QW_R").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
        if gpu.use_plane && gpu.plane_size.1 == 32 && self.rows % rr == 0 && std::env::var("LB_QW_ROWS1").is_err() {
            unsafe {
                q6_k_native_rows4_kernel::launch(
                    &gpu.client,
                    cube_count(n_out / rr, gpu.per_cube()),
                    CubeDim::new_1d(gpu.cube_dim as u32),
                    BufferArg::from_raw_parts(self.qq.clone(), self.n.0 / 4),
                    BufferArg::from_raw_parts(self.sc.clone(), self.n.1 / 4),
                    BufferArg::from_raw_parts(self.dd.clone(), self.n.2),
                    BufferArg::from_raw_parts(xp.clone(), m * self.cols / 4),
                    BufferArg::from_raw_parts(self.dims.clone(), 2),
                    BufferArg::from_raw_parts(out.clone(), n_out),
                    rr,
                );
            }
            return out;
        }
        unsafe {
            q6_k_native_matvec_kernel::launch(
                &gpu.client,
                cube_count(n_out, gpu.per_cube()),
                CubeDim::new_1d(gpu.cube_dim as u32),
                BufferArg::from_raw_parts(self.qq.clone(), self.n.0 / 4),
                BufferArg::from_raw_parts(self.sc.clone(), self.n.1 / 4),
                BufferArg::from_raw_parts(self.dd.clone(), self.n.2),
                BufferArg::from_raw_parts(xp.clone(), m * self.cols / 4),
                BufferArg::from_raw_parts(self.dims.clone(), 2),
                BufferArg::from_raw_parts(out.clone(), n_out),
                gpu.use_plane,
            );
        }
        out
    }
    pub fn matvec(&self, gpu: &Gpu, x: &[f32], m: usize) -> Result<Vec<f32>, String> {
        let xh = gpu.upload_f32(x);
        let out = self.launch(gpu, &xh, m);
        Ok(gpu.read_all(vec![(out, m * self.rows)])?.remove(0))
    }
}

/// A Q8_0 matrix on the device at file density (34 bytes a block, split into aligned arrays).
pub struct GpuQ8Native {
    pub rows: usize,
    pub cols: usize,
    qs: Handle,
    dd: Handle,
    n: (usize, usize),
    dims: Handle,
}

impl GpuQ8Native {
    pub fn upload(gpu: &Gpu, w: QuantMat<'_>) -> Result<Self, String> {
        if w.ty != GgmlType::Q8_0 || w.cols % 32 != 0 || (w.rows * (w.cols / 32)) % 1 != 0 {
            return Err("expected Q8_0".into());
        }
        let nb = w.rows * (w.cols / 32);
        let mut qs = Vec::with_capacity(nb * 32);
        for blk in w.bytes.chunks_exact(34) {
            qs.extend_from_slice(&blk[2..34]);
        }
        let dd = pack_f16_pairs(w.bytes, 34, 0, nb);
        let qs = bytes_to_words(&qs);
        let n = (qs.len(), dd.len());
        Ok(Self { rows: w.rows, cols: w.cols, n, qs: gpu.upload_u32(qs), dd: gpu.upload_u32(dd), dims: gpu.const_u32(&crate::matvec_dims(w.rows, w.cols)) })
    }
    pub fn bytes_on_device(&self) -> usize {
        (self.n.0 + self.n.1) * 4
    }
    pub fn launch(&self, gpu: &Gpu, x: &Handle, m: usize) -> Handle {
        let xp = GpuQ5KNative::permute_x(gpu, x, m * self.cols);
        self.launch_permuted(gpu, &xp, m)
    }
    pub fn launch_permuted(&self, gpu: &Gpu, xp: &Handle, m: usize) -> Handle {
        let n_out = m * self.rows;
        let out = gpu.scratch(n_out * 4);
        unsafe {
            q8_0_native_matvec_kernel::launch(
                &gpu.client,
                cube_count(n_out, gpu.per_cube()),
                CubeDim::new_1d(gpu.cube_dim as u32),
                BufferArg::from_raw_parts(self.qs.clone(), self.n.0 / 4),
                BufferArg::from_raw_parts(self.dd.clone(), self.n.1),
                BufferArg::from_raw_parts(xp.clone(), m * self.cols / 4),
                BufferArg::from_raw_parts(self.dims.clone(), 2),
                BufferArg::from_raw_parts(out.clone(), n_out),
                gpu.use_plane,
            );
        }
        out
    }
    pub fn matvec(&self, gpu: &Gpu, x: &[f32], m: usize) -> Result<Vec<f32>, String> {
        let xh = gpu.upload_f32(x);
        let out = self.launch(gpu, &xh, m);
        Ok(gpu.read_all(vec![(out, m * self.rows)])?.remove(0))
    }
}

// ---- routed (MoE) kernels at file density: all experts of a layer in one native-layout buffer, rows of expert e start at block
// `(e * rows + r) * bpr`; routes are `[expert, token]` pairs. `x` is permuted (`perm16_kernel`); the gate+up pass reads one
// permuted token row, the down pass reads the previous pass's output permuted per slot (the caller permutes it).

#[cube(launch)]
fn q5n_routed_gate_up_silu_kernel(gw: &[V4U], uw: &[V4U], x: &[V4F], routes: &[u32], dims: &[u32], h: &mut [f32], #[comptime] use_plane: bool) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let bpr = cols / 256usize;
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
        let e = routes[2usize * slot] as usize;
        let tok = routes[2usize * slot + 1usize] as usize;
        let blk0 = (e * rows + r) * bpr;
        let pg = q5n_partial(gw, x, blk0, tok * cols / 4usize, bpr, lane, width);
        let pu = q5n_partial(uw, x, blk0, tok * cols / 4usize, bpr, lane, width);
        let g = lane_sum(&mut sm, pg, use_plane);
        let u = lane_sum(&mut sm, pu, use_plane);
        if lane == 0usize {
            // written in the 16-value-permuted order the down pass reads (`perm16_kernel`: y[16a + 4j + i] = x[16a + 4i + j])
            let a16 = r / 16usize;
            let rem = r % 16usize;
            h[slot * rows + a16 * 16usize + (rem % 4usize) * 4usize + rem / 4usize] = g / (1.0f32 + Exp::exp(-g)) * u;
        }
        o += step;
    }
}

#[cube(launch)]
fn q5n_routed_down_kernel(w: &[V4U], h: &[V4F], routes: &[u32], dims: &[u32], out: &mut [f32], #[comptime] use_plane: bool) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let bpr = cols / 256usize;
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
        let e = routes[2usize * slot] as usize;
        let p = q5n_partial(w, h, (e * rows + r) * bpr, slot * cols / 4usize, bpr, lane, width);
        let v = lane_sum(&mut sm, p, use_plane);
        if lane == 0usize {
            out[o] = v;
        }
        o += step;
    }
}

#[cube(launch)]
fn q6n_routed_down_kernel(qq: &[V4U], sc: &[V4U], dd: &[u32], h: &[V4F], routes: &[u32], dims: &[u32], out: &mut [f32], #[comptime] use_plane: bool) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let bpr = cols / 256usize;
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
        let e = routes[2usize * slot] as usize;
        let p = q6n_partial(qq, sc, dd, h, (e * rows + r) * bpr, slot * cols / 4usize, bpr, lane, width);
        let v = lane_sum(&mut sm, p, use_plane);
        if lane == 0usize {
            out[o] = v;
        }
        o += step;
    }
}


/// Routed down with several output rows per plane, for short rows (`cols` 512: 8 or 4 items a row would leave most lanes idle):
/// a plane of 32 lanes takes `32 / ipr` rows, `ipr` lanes a row, and separates the rows with masked plane sums. Needs plane ops, 32 lanes.
#[cube(launch)]
fn q5n_routed_down_mr_kernel(w: &[V4U], h: &[V4F], routes: &[u32], dims: &[u32], out: &mut [f32], #[comptime] ipr: usize, #[comptime] rpp: usize) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let bpr = cols / 256usize;
    let lane = UNIT_POS_PLANE as usize;
    let plane_id = (UNIT_POS / PLANE_DIM) as usize;
    let ppc = CUBE_DIM_X as usize / 32usize;
    let step = CUBE_COUNT as usize * ppc * rpp;
    let rl = lane / ipr;
    let mut ob = (CUBE_POS as usize * ppc + plane_id) * rpp;
    while ob < out.len() {
        let o = ob + rl;
        let mut part = 0.0f32;
        if o < out.len() {
            let slot = o / rows;
            let r = o % rows;
            let e = routes[2usize * slot] as usize;
            part = q5n_partial(w, h, (e * rows + r) * bpr, slot * cols / 4usize, bpr, lane % ipr, ipr);
        }
        #[unroll]
        for rr in 0..rpp {
            let mut mv = 0.0f32;
            if rl == rr {
                mv = part;
            }
            let sdot = plane_sum(mv);
            if lane == 0usize {
                if ob + rr < out.len() {
                    out[ob + rr] = sdot;
                }
            }
        }
        ob += step;
    }
}

#[cube(launch)]
fn q6n_routed_down_mr_kernel(qq: &[V4U], sc: &[V4U], dd: &[u32], h: &[V4F], routes: &[u32], dims: &[u32], out: &mut [f32], #[comptime] ipr: usize, #[comptime] rpp: usize) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let bpr = cols / 256usize;
    let lane = UNIT_POS_PLANE as usize;
    let plane_id = (UNIT_POS / PLANE_DIM) as usize;
    let ppc = CUBE_DIM_X as usize / 32usize;
    let step = CUBE_COUNT as usize * ppc * rpp;
    let rl = lane / ipr;
    let mut ob = (CUBE_POS as usize * ppc + plane_id) * rpp;
    while ob < out.len() {
        let o = ob + rl;
        let mut part = 0.0f32;
        if o < out.len() {
            let slot = o / rows;
            let r = o % rows;
            let e = routes[2usize * slot] as usize;
            part = q6n_partial(qq, sc, dd, h, (e * rows + r) * bpr, slot * cols / 4usize, bpr, lane % ipr, ipr);
        }
        #[unroll]
        for rr in 0..rpp {
            let mut mv = 0.0f32;
            if rl == rr {
                mv = part;
            }
            let sdot = plane_sum(mv);
            if lane == 0usize {
                if ob + rr < out.len() {
                    out[ob + rr] = sdot;
                }
            }
        }
        ob += step;
    }
}


/// Lane partial of one Q8_0 row at file density (blocks `lane, lane + width, ...`; `bpr` blocks a row).
#[cube]
fn q8n_partial(qs: &[V4U], dd: &[u32], x: &[V4F], blk0: usize, xrow4: usize, bpr: usize, lane: usize, width: usize) -> f32 {
    let mut acc = 0.0f32;
    let mut b = lane;
    while b < bpr {
        let gb = blk0 + b;
        let d = f16_to_f32((dd[gb / 2usize] >> (((gb % 2usize) * 16usize) as u32)) & 65535u32);
        let xb = xrow4 + b * 8usize;
        let mut p = V4F::new(0.0f32);
        #[unroll]
        for a in 0..2usize {
            let qv = qs[gb * 2usize + a];
            #[unroll]
            for k in 0..4u32 {
                let by = (qv >> V4U::new(8u32 * k)) & V4U::new(255u32);
                let f = V4F::cast_from(by) - V4F::new(256.0f32) * V4F::cast_from(by >> V4U::new(7u32));
                p += f * x[xb + 4usize * a + k as usize];
            }
        }
        acc += d * (p.extract(0usize) + p.extract(1usize) + p.extract(2usize) + p.extract(3usize));
        b += width;
    }
    acc
}

/// Routed down with Q8_0 experts (some Tiel-Coder layers), several rows a plane like the K-quant `_mr` kernels: `ipr` = blocks a row.
#[cube(launch)]
fn q8n_routed_down_mr_kernel(qs: &[V4U], dd: &[u32], h: &[V4F], routes: &[u32], dims: &[u32], out: &mut [f32], #[comptime] ipr: usize, #[comptime] rpp: usize) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let bpr = cols / 32usize;
    let lane = UNIT_POS_PLANE as usize;
    let plane_id = (UNIT_POS / PLANE_DIM) as usize;
    let ppc = CUBE_DIM_X as usize / 32usize;
    let step = CUBE_COUNT as usize * ppc * rpp;
    let rl = lane / ipr;
    let mut ob = (CUBE_POS as usize * ppc + plane_id) * rpp;
    while ob < out.len() {
        let o = ob + rl;
        let mut part = 0.0f32;
        if o < out.len() {
            let slot = o / rows;
            let r = o % rows;
            let e = routes[2usize * slot] as usize;
            part = q8n_partial(qs, dd, h, (e * rows + r) * bpr, slot * cols / 4usize, bpr, lane % ipr, ipr);
        }
        #[unroll]
        for rr in 0..rpp {
            let mut mv = 0.0f32;
            if rl == rr {
                mv = part;
            }
            let sdot = plane_sum(mv);
            if lane == 0usize {
                if ob + rr < out.len() {
                    out[ob + rr] = sdot;
                }
            }
        }
        ob += step;
    }
}

/// Routed gate+up+SiLU with Q6_K experts (one Tiel-Coder layer); writes `h` permuted like the Q5_K kernel.
#[cube(launch)]
fn q6n_routed_gate_up_silu_kernel(gq: &[V4U], gs: &[V4U], gd: &[u32], uq: &[V4U], us: &[V4U], ud: &[u32], x: &[V4F], routes: &[u32], dims: &[u32], h: &mut [f32], #[comptime] use_plane: bool) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let bpr = cols / 256usize;
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
        let e = routes[2usize * slot] as usize;
        let tok = routes[2usize * slot + 1usize] as usize;
        let blk0 = (e * rows + r) * bpr;
        let pg = q6n_partial(gq, gs, gd, x, blk0, tok * cols / 4usize, bpr, lane, width);
        let pu = q6n_partial(uq, us, ud, x, blk0, tok * cols / 4usize, bpr, lane, width);
        let g = lane_sum(&mut sm, pg, use_plane);
        let u = lane_sum(&mut sm, pu, use_plane);
        if lane == 0usize {
            let a16 = r / 16usize;
            let rem = r % 16usize;
            h[slot * rows + a16 * 16usize + (rem % 4usize) * 4usize + rem / 4usize] = g / (1.0f32 + Exp::exp(-g)) * u;
        }
        o += step;
    }
}

/// Q6_K matrix split into aligned arrays at file density (`ql`+`qh`, scales, f16 `d`), as `GpuQ6KNative` holds it.
fn split_q6(gpu: &Gpu, bytes: &[u8], nb: usize) -> (Handle, Handle, Handle, (usize, usize, usize)) {
    let mut qq = Vec::with_capacity(nb * 192);
    let mut sc = Vec::with_capacity(nb * 16);
    for blk in bytes.chunks_exact(210) {
        qq.extend_from_slice(&blk[..192]);
        sc.extend_from_slice(&blk[192..208]);
    }
    let dd = pack_f16_pairs(bytes, 210, 208, nb);
    let (qq, sc) = (bytes_to_words(&qq), bytes_to_words(&sc));
    let n = (qq.len(), sc.len(), dd.len());
    (gpu.upload_u32(qq), gpu.upload_u32(sc), gpu.upload_u32(dd), n)
}


/// Routed gate+up+SiLU, Q5_K, 4 output rows a plane (gate and up of each: 8 row partials sharing the activation loads); writes `h` permuted.
#[cube(launch)]
fn q5n_routed_gate_up_rows4_kernel(gw: &[V4U], uw: &[V4U], x: &[V4F], routes: &[u32], dims: &[u32], h: &mut [f32]) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let bpr = cols / 256usize;
    let lane = UNIT_POS_PLANE as usize;
    let plane_id = (UNIT_POS / PLANE_DIM) as usize;
    let ppc = CUBE_DIM_X as usize / 32usize;
    let step = CUBE_COUNT as usize * ppc;
    let groups = h.len() / 4usize;
    let mut g = CUBE_POS as usize * ppc + plane_id;
    while g < groups {
        let o0 = g * 4usize;
        let slot = o0 / rows;
        let r0 = o0 % rows;
        let e = routes[2usize * slot] as usize;
        let tok = routes[2usize * slot + 1usize] as usize;
        let mut acc = Array::<f32>::new(8usize);
        #[unroll]
        for i in 0..8usize {
            acc[i] = 0.0f32;
        }
        // rows 0..4 of acc: gate rows r0..r0+4, rows 4..8: the same rows of up
        q5n_rows2(gw, uw, x, (e * rows + r0) * bpr, tok * cols / 4usize, bpr, lane, 32usize, &mut acc, 8usize, 4usize);
        #[unroll]
        for i in 0..4usize {
            let gsum = plane_sum(acc[i]);
            let usum = plane_sum(acc[i + 4usize]);
            if lane == 0usize {
                let r = r0 + i;
                let a16 = r / 16usize;
                let rem = r % 16usize;
                h[slot * rows + a16 * 16usize + (rem % 4usize) * 4usize + rem / 4usize] = gsum / (1.0f32 + Exp::exp(-gsum)) * usum;
            }
        }
        g += step;
    }
}

/// Routed down, Q5_K or Q6_K experts, `r` rows a lane group with the activation loaded once per group: a plane is `32 / ipr` groups of
/// `ipr` lanes (the items of one row), each group `r` consecutive rows; a group's lanes are summed with xor shuffles.
#[cube(launch)]
fn q5n_routed_down_rb_kernel(w: &[V4U], h: &[V4F], routes: &[u32], dims: &[u32], out: &mut [f32], #[comptime] ipr: usize, #[comptime] r: usize) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let bpr = cols / 256usize;
    let lane = UNIT_POS_PLANE as usize;
    let plane_id = (UNIT_POS / PLANE_DIM) as usize;
    let ppc = CUBE_DIM_X as usize / 32usize;
    let step = CUBE_COUNT as usize * ppc;
    let gpp = 32usize / ipr;
    let per_plane = gpp * r;
    let groups = out.len() / per_plane;
    let lg = lane / ipr;
    let li = lane % ipr;
    let mut g = CUBE_POS as usize * ppc + plane_id;
    while g < groups {
        let ob = g * per_plane + lg * r;
        let slot = ob / rows;
        let r0 = ob % rows;
        let e = routes[2usize * slot] as usize;
        let mut acc = Array::<f32>::new(r);
        #[unroll]
        for i in 0..r {
            acc[i] = 0.0f32;
        }
        q5n_rows(w, h, (e * rows + r0) * bpr, slot * cols / 4usize, bpr, li, ipr, &mut acc, r);
        #[unroll]
        for i in 0..r {
            let mut v = acc[i];
            let mut s = lane * 0usize + ipr / 2usize;
            while s > 0usize {
                v += plane_shuffle_xor(v, s as u32);
                s = s / 2usize;
            }
            if li == 0usize {
                out[ob + i] = v;
            }
        }
        g += step;
    }
}

#[cube(launch)]
fn q6n_routed_down_rb_kernel(qq: &[V4U], sc: &[V4U], dd: &[u32], h: &[V4F], routes: &[u32], dims: &[u32], out: &mut [f32], #[comptime] ipr: usize, #[comptime] r: usize) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let bpr = cols / 256usize;
    let lane = UNIT_POS_PLANE as usize;
    let plane_id = (UNIT_POS / PLANE_DIM) as usize;
    let ppc = CUBE_DIM_X as usize / 32usize;
    let step = CUBE_COUNT as usize * ppc;
    let gpp = 32usize / ipr;
    let per_plane = gpp * r;
    let groups = out.len() / per_plane;
    let lg = lane / ipr;
    let li = lane % ipr;
    let mut g = CUBE_POS as usize * ppc + plane_id;
    while g < groups {
        let ob = g * per_plane + lg * r;
        let slot = ob / rows;
        let r0 = ob % rows;
        let e = routes[2usize * slot] as usize;
        let mut acc = Array::<f32>::new(r);
        #[unroll]
        for i in 0..r {
            acc[i] = 0.0f32;
        }
        q6n_rows(qq, sc, dd, h, (e * rows + r0) * bpr, slot * cols / 4usize, bpr, li, ipr, &mut acc, r);
        #[unroll]
        for i in 0..r {
            let mut v = acc[i];
            let mut s = lane * 0usize + ipr / 2usize;
            while s > 0usize {
                v += plane_shuffle_xor(v, s as u32);
                s = s / 2usize;
            }
            if li == 0usize {
                out[ob + i] = v;
            }
        }
        g += step;
    }
}

/// Expert gate and up matrices (always the same type in the files seen): Q5_K, or Q6_K in a few layers.
pub enum NativeGateUp {
    Q5 { gw: Handle, uw: Handle, n: usize },
    Q6 { g: (Handle, Handle, Handle), u: (Handle, Handle, Handle), n: (usize, usize, usize) },
}

/// Expert down matrices at file density: Q5_K, Q6_K or Q8_0 (the files mix them by layer).
pub enum NativeDown {
    Q8 { qs: Handle, dd: Handle, n: (usize, usize) },
    Q5 { w: Handle, n: usize },
    Q6 { qq: Handle, sc: Handle, dd: Handle, n: (usize, usize, usize) },
}

/// All experts of one `qwen35moe` layer at file density, routes read from a device buffer (`[expert, token]` pairs, `slots` of them).
pub struct GpuNativeExperts {
    pub n_expert: usize,
    pub embd: usize,
    pub ff: usize,
    gu: NativeGateUp,
    down: NativeDown,
}

impl GpuNativeExperts {
    pub fn upload(gpu: &Gpu, n_expert: usize, gate: QuantMat<'_>, up: QuantMat<'_>, down: QuantMat<'_>) -> Result<Self, String> {
        if gate.ty != up.ty || !matches!(gate.ty, GgmlType::Q5K | GgmlType::Q6K) || !matches!(down.ty, GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q8_0) {
            return Err(format!("expert types gate {} up {} down {} not supported", gate.ty.name(), up.ty.name(), down.ty.name()));
        }
        let (ff, embd) = (gate.rows / n_expert, gate.cols);
        let gu = if gate.ty == GgmlType::Q5K {
            let (gw, uw) = (bytes_to_words(gate.bytes), bytes_to_words(up.bytes));
            let n = gw.len();
            NativeGateUp::Q5 { gw: gpu.upload_u32(gw), uw: gpu.upload_u32(uw), n }
        } else {
            let nb = gate.rows * (gate.cols / 256);
            let (a, b, c, n) = split_q6(gpu, gate.bytes, nb);
            let (d, e, f, _) = split_q6(gpu, up.bytes, nb);
            NativeGateUp::Q6 { g: (a, b, c), u: (d, e, f), n }
        };
        let down = match down.ty {
            GgmlType::Q5K => {
                let w = bytes_to_words(down.bytes);
                let n = w.len();
                NativeDown::Q5 { w: gpu.upload_u32(w), n }
            }
            GgmlType::Q6K => {
                let (qq, sc, dd, n) = split_q6(gpu, down.bytes, down.rows * (down.cols / 256));
                NativeDown::Q6 { qq, sc, dd, n }
            }
            _ => {
                let nb = down.rows * (down.cols / 32);
                let mut qs = Vec::with_capacity(nb * 32);
                for blk in down.bytes.chunks_exact(34) {
                    qs.extend_from_slice(&blk[2..34]);
                }
                let dd = pack_f16_pairs(down.bytes, 34, 0, nb);
                let qs = bytes_to_words(&qs);
                let n = (qs.len(), dd.len());
                NativeDown::Q8 { qs: gpu.upload_u32(qs), dd: gpu.upload_u32(dd), n }
            }
        };
        Ok(Self { n_expert, embd, ff, gu, down })
    }

    pub fn bytes_on_device(&self) -> usize {
        (match &self.gu {
            NativeGateUp::Q5 { n, .. } => 2 * n * 4,
            NativeGateUp::Q6 { n, .. } => 2 * (n.0 + n.1 + n.2) * 4,
        }) + match &self.down {
            NativeDown::Q8 { n, .. } => (n.0 + n.1) * 4,
            NativeDown::Q5 { n, .. } => n * 4,
                NativeDown::Q6 { n, .. } => (n.0 + n.1 + n.2) * 4,
            }
    }

    /// `x` (one token, `embd` values, already on the device) through the experts named by `routes` (`2 * slots` u32):
    /// returns `[slots, embd]` per-slot outputs (router weights not applied).
    pub fn launch(&self, gpu: &Gpu, x: &Handle, routes: &Handle, slots: usize) -> Handle {
        let xp = GpuQ5KNative::permute_x(gpu, x, self.embd);
        self.launch_permuted(gpu, xp, routes, slots)
    }

    /// As [`launch`](Self::launch) with the activation already in the permuted order (`qw_rms`'s / `qw_add_rms`'s second output).
    pub fn launch_permuted(&self, gpu: &Gpu, xp: Handle, routes: &Handle, slots: usize) -> Handle {
        let h = gpu.scratch(slots * self.ff * 4);
        let out = gpu.scratch(slots * self.embd * 4);
        let d_gu = gpu.const_u32(&[self.ff as u32, self.embd as u32]);
        let d_dn = gpu.const_u32(&[self.embd as u32, self.ff as u32]);
        // row-batched kernels need plane ops, a 32-lane plane, and (for the down pass) lane groups that divide the plane
        let rb = gpu.use_plane && gpu.plane_size.1 == 32 && std::env::var("LB_QW_ROWS1").is_err();
        unsafe {
            match &self.gu {
                NativeGateUp::Q5 { gw, uw, n } if rb && self.ff % 4 == 0 && std::env::var("LB_QW_NOGU").is_err() => q5n_routed_gate_up_rows4_kernel::launch(
                    &gpu.client,
                    cube_count(slots * self.ff / 4, gpu.per_cube()),
                    CubeDim::new_1d(gpu.cube_dim as u32),
                    BufferArg::from_raw_parts(gw.clone(), n / 4),
                    BufferArg::from_raw_parts(uw.clone(), n / 4),
                    BufferArg::from_raw_parts(xp, self.embd / 4),
                    BufferArg::from_raw_parts(routes.clone(), 2 * slots),
                    BufferArg::from_raw_parts(d_gu, 2),
                    BufferArg::from_raw_parts(h.clone(), slots * self.ff),
                ),
                NativeGateUp::Q5 { gw, uw, n } => q5n_routed_gate_up_silu_kernel::launch(
                    &gpu.client,
                    cube_count(slots * self.ff, gpu.per_cube()),
                    CubeDim::new_1d(gpu.cube_dim as u32),
                    BufferArg::from_raw_parts(gw.clone(), n / 4),
                    BufferArg::from_raw_parts(uw.clone(), n / 4),
                    BufferArg::from_raw_parts(xp, self.embd / 4),
                    BufferArg::from_raw_parts(routes.clone(), 2 * slots),
                    BufferArg::from_raw_parts(d_gu, 2),
                    BufferArg::from_raw_parts(h.clone(), slots * self.ff),
                    gpu.use_plane,
                ),
                NativeGateUp::Q6 { g, u, n } => q6n_routed_gate_up_silu_kernel::launch(
                    &gpu.client,
                    cube_count(slots * self.ff, gpu.per_cube()),
                    CubeDim::new_1d(gpu.cube_dim as u32),
                    BufferArg::from_raw_parts(g.0.clone(), n.0 / 4),
                    BufferArg::from_raw_parts(g.1.clone(), n.1 / 4),
                    BufferArg::from_raw_parts(g.2.clone(), n.2),
                    BufferArg::from_raw_parts(u.0.clone(), n.0 / 4),
                    BufferArg::from_raw_parts(u.1.clone(), n.1 / 4),
                    BufferArg::from_raw_parts(u.2.clone(), n.2),
                    BufferArg::from_raw_parts(xp, self.embd / 4),
                    BufferArg::from_raw_parts(routes.clone(), 2 * slots),
                    BufferArg::from_raw_parts(d_gu, 2),
                    BufferArg::from_raw_parts(h.clone(), slots * self.ff),
                    gpu.use_plane,
                ),
            }
        }
        let hp = h; // gate+up already wrote it permuted
        // multi-row planes need plane ops, a 32-lane plane, and an item count that divides it
        let mr = |ipr: usize| gpu.use_plane && gpu.plane_size.1 == 32 && ipr <= 32 && 32 % ipr == 0;
        unsafe {
            match &self.down {
                NativeDown::Q8 { qs, dd, n } => {
                    let ipr = self.ff / 32;
                    assert!(mr(ipr), "Q8_0 expert down needs plane ops and ff/32 dividing 32 (ff = {})", self.ff);
                    let rpp = 32 / ipr;
                    q8n_routed_down_mr_kernel::launch(
                        &gpu.client,
                        cube_count((slots * self.embd).div_ceil(rpp), gpu.per_cube()),
                        CubeDim::new_1d(gpu.cube_dim as u32),
                        BufferArg::from_raw_parts(qs.clone(), n.0 / 4),
                        BufferArg::from_raw_parts(dd.clone(), n.1),
                        BufferArg::from_raw_parts(hp, slots * self.ff / 4),
                        BufferArg::from_raw_parts(routes.clone(), 2 * slots),
                        BufferArg::from_raw_parts(d_dn, 2),
                        BufferArg::from_raw_parts(out.clone(), slots * self.embd),
                        ipr,
                        rpp,
                    )
                }
                NativeDown::Q5 { w, n } if rb && std::env::var("LB_QW_NODN").is_err() && 32 % (self.ff / 256 * 4) == 0 && self.embd % (32 / (self.ff / 256 * 4) * 4) == 0 => {
                    let ipr = self.ff / 256 * 4;
                    let per_plane = 32 / ipr * 4;
                    q5n_routed_down_rb_kernel::launch(
                        &gpu.client,
                        cube_count(slots * self.embd / per_plane, gpu.per_cube()),
                        CubeDim::new_1d(gpu.cube_dim as u32),
                        BufferArg::from_raw_parts(w.clone(), n / 4),
                        BufferArg::from_raw_parts(hp, slots * self.ff / 4),
                        BufferArg::from_raw_parts(routes.clone(), 2 * slots),
                        BufferArg::from_raw_parts(d_dn, 2),
                        BufferArg::from_raw_parts(out.clone(), slots * self.embd),
                        ipr,
                        4usize,
                    )
                }
                NativeDown::Q5 { w, n } if mr(self.ff / 256 * 4) => {
                    let ipr = self.ff / 256 * 4;
                    let rpp = 32 / ipr;
                    q5n_routed_down_mr_kernel::launch(
                        &gpu.client,
                        cube_count((slots * self.embd).div_ceil(rpp), gpu.per_cube()),
                        CubeDim::new_1d(gpu.cube_dim as u32),
                        BufferArg::from_raw_parts(w.clone(), n / 4),
                        BufferArg::from_raw_parts(hp, slots * self.ff / 4),
                        BufferArg::from_raw_parts(routes.clone(), 2 * slots),
                        BufferArg::from_raw_parts(d_dn, 2),
                        BufferArg::from_raw_parts(out.clone(), slots * self.embd),
                        ipr,
                        rpp,
                    )
                }
                NativeDown::Q5 { w, n } => q5n_routed_down_kernel::launch(
                    &gpu.client,
                    cube_count(slots * self.embd, gpu.per_cube()),
                    CubeDim::new_1d(gpu.cube_dim as u32),
                    BufferArg::from_raw_parts(w.clone(), n / 4),
                    BufferArg::from_raw_parts(hp, slots * self.ff / 4),
                    BufferArg::from_raw_parts(routes.clone(), 2 * slots),
                    BufferArg::from_raw_parts(d_dn, 2),
                    BufferArg::from_raw_parts(out.clone(), slots * self.embd),
                    gpu.use_plane,
                ),
                NativeDown::Q6 { qq, sc, dd, n } if rb && std::env::var("LB_QW_NODN").is_err() && 32 % (self.ff / 256 * 2) == 0 && self.embd % (32 / (self.ff / 256 * 2) * 4) == 0 => {
                    let ipr = self.ff / 256 * 2;
                    let per_plane = 32 / ipr * 4;
                    q6n_routed_down_rb_kernel::launch(
                        &gpu.client,
                        cube_count(slots * self.embd / per_plane, gpu.per_cube()),
                        CubeDim::new_1d(gpu.cube_dim as u32),
                        BufferArg::from_raw_parts(qq.clone(), n.0 / 4),
                        BufferArg::from_raw_parts(sc.clone(), n.1 / 4),
                        BufferArg::from_raw_parts(dd.clone(), n.2),
                        BufferArg::from_raw_parts(hp, slots * self.ff / 4),
                        BufferArg::from_raw_parts(routes.clone(), 2 * slots),
                        BufferArg::from_raw_parts(d_dn, 2),
                        BufferArg::from_raw_parts(out.clone(), slots * self.embd),
                        ipr,
                        4usize,
                    )
                }
                NativeDown::Q6 { qq, sc, dd, n } if mr(self.ff / 256 * 2) => {
                    let ipr = self.ff / 256 * 2;
                    let rpp = 32 / ipr;
                    q6n_routed_down_mr_kernel::launch(
                        &gpu.client,
                        cube_count((slots * self.embd).div_ceil(rpp), gpu.per_cube()),
                        CubeDim::new_1d(gpu.cube_dim as u32),
                        BufferArg::from_raw_parts(qq.clone(), n.0 / 4),
                        BufferArg::from_raw_parts(sc.clone(), n.1 / 4),
                        BufferArg::from_raw_parts(dd.clone(), n.2),
                        BufferArg::from_raw_parts(hp, slots * self.ff / 4),
                        BufferArg::from_raw_parts(routes.clone(), 2 * slots),
                        BufferArg::from_raw_parts(d_dn, 2),
                        BufferArg::from_raw_parts(out.clone(), slots * self.embd),
                        ipr,
                        rpp,
                    )
                }
                NativeDown::Q6 { qq, sc, dd, n } => q6n_routed_down_kernel::launch(
                    &gpu.client,
                    cube_count(slots * self.embd, gpu.per_cube()),
                    CubeDim::new_1d(gpu.cube_dim as u32),
                    BufferArg::from_raw_parts(qq.clone(), n.0 / 4),
                    BufferArg::from_raw_parts(sc.clone(), n.1 / 4),
                    BufferArg::from_raw_parts(dd.clone(), n.2),
                    BufferArg::from_raw_parts(hp, slots * self.ff / 4),
                    BufferArg::from_raw_parts(routes.clone(), 2 * slots),
                    BufferArg::from_raw_parts(d_dn, 2),
                    BufferArg::from_raw_parts(out.clone(), slots * self.embd),
                    gpu.use_plane,
                ),
            }
        }
        out
    }
}

// ---- the shared expert (Q8_0 matrices) fused into two launches: gate+up+SiLU, and down folded into the MoE combine.

/// `h[r] = silu(gate_r . x) * (up_r . x)` over `rows` rows of Q8_0 (file density), plane a row, written in the 16-value-permuted order.
#[cube(launch)]
fn q8n_gate_up_silu_kernel(gq: &[V4U], gd: &[u32], uq: &[V4U], ud: &[u32], x: &[V4F], dims: &[u32], h: &mut [f32]) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let bpr = cols / 32usize;
    let mut sm = Shared::<[f32]>::new_slice(256usize);
    let lane = UNIT_POS_PLANE as usize;
    let plane_id = (UNIT_POS / PLANE_DIM) as usize;
    let ppc = CUBE_DIM_X as usize / 32usize;
    let step = CUBE_COUNT as usize * ppc;
    let mut r = CUBE_POS as usize * ppc + plane_id;
    while r < rows {
        let pg = q8n_partial(gq, gd, x, r * bpr, 0usize, bpr, lane, 32usize);
        let pu = q8n_partial(uq, ud, x, r * bpr, 0usize, bpr, lane, 32usize);
        let g = plane_sum(pg);
        let u = plane_sum(pu);
        if lane == 0usize {
            let a16 = r / 16usize;
            let rem = r % 16usize;
            h[a16 * 16usize + (rem % 4usize) * 4usize + rem / 4usize] = g / (1.0f32 + Exp::exp(-g)) * u;
        }
        r += step;
    }
    let _ = &mut sm;
}

/// `out = x + sum_r w_r down_r + sigmoid(sgate) * (Wd . hs)`: the MoE combine with the shared expert's Q8_0 down matrix folded in
/// (`hs` = the shared hidden in permuted order, `cols` = its width), a plane an output.
#[cube(launch)]
fn q8n_combine_kernel(x: &[f32], down: &[f32], weights: &[f32], sgate: &[f32], dq: &[V4U], dd: &[u32], hs: &[V4F], dims: &[u32], out: &mut [f32], #[comptime] kslots: usize, #[comptime] sg_off: usize) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let bpr = cols / 32usize;
    let lane = UNIT_POS_PLANE as usize;
    let plane_id = (UNIT_POS / PLANE_DIM) as usize;
    let ppc = CUBE_DIM_X as usize / 32usize;
    let step = CUBE_COUNT as usize * ppc;
    let mut i = CUBE_POS as usize * ppc + plane_id;
    while i < rows {
        let p = q8n_partial(dq, dd, hs, i * bpr, 0usize, bpr, lane, 32usize);
        let sh = plane_sum(p);
        if lane == 0usize {
            let mut acc = 0.0f32;
            let mut r = 0usize;
            while r < kslots {
                acc += weights[r] * down[r * rows + i];
                r += 1usize;
            }
            acc += sigmoid_f(sgate[sg_off]) * sh;
            out[i] = x[i] + acc;
        }
        i += step;
    }
}

impl GpuQ8Native {
    /// Shared-expert hidden: `silu(gate x) * (up x)` in permuted order, `x` permuted. Needs plane ops and a 32-lane plane.
    pub fn gate_up_silu(gpu: &Gpu, gate: &GpuQ8Native, up: &GpuQ8Native, xp: &Handle) -> Handle {
        let h = gpu.scratch(gate.rows * 4);
        unsafe {
            q8n_gate_up_silu_kernel::launch(
                &gpu.client,
                cube_count(gate.rows, gpu.per_cube()),
                CubeDim::new_1d(gpu.cube_dim as u32),
                BufferArg::from_raw_parts(gate.qs.clone(), gate.n.0 / 4),
                BufferArg::from_raw_parts(gate.dd.clone(), gate.n.1),
                BufferArg::from_raw_parts(up.qs.clone(), up.n.0 / 4),
                BufferArg::from_raw_parts(up.dd.clone(), up.n.1),
                BufferArg::from_raw_parts(xp.clone(), gate.cols / 4),
                BufferArg::from_raw_parts(gate.dims.clone(), 2),
                BufferArg::from_raw_parts(h.clone(), gate.rows),
            );
        }
        h
    }

    /// `x + sum_r w_r down_r + sigmoid(sgate[sg_off]) * (self . hs)`; `self` is the shared down matrix (`rows` = n_embd).
    #[allow(clippy::too_many_arguments)]
    pub fn combine_shared(&self, gpu: &Gpu, x: &Handle, down: &Handle, weights: &Handle, sgate: &Handle, sg_len: usize, sg_off: usize, hs: &Handle, k: usize) -> Handle {
        let out = gpu.scratch(self.rows * 4);
        unsafe {
            q8n_combine_kernel::launch(
                &gpu.client,
                cube_count(self.rows, gpu.per_cube()),
                CubeDim::new_1d(gpu.cube_dim as u32),
                BufferArg::from_raw_parts(x.clone(), self.rows),
                BufferArg::from_raw_parts(down.clone(), k * self.rows),
                BufferArg::from_raw_parts(weights.clone(), k),
                BufferArg::from_raw_parts(sgate.clone(), sg_len),
                BufferArg::from_raw_parts(self.qs.clone(), self.n.0 / 4),
                BufferArg::from_raw_parts(self.dd.clone(), self.n.1),
                BufferArg::from_raw_parts(hs.clone(), self.cols / 4),
                BufferArg::from_raw_parts(self.dims.clone(), 2),
                BufferArg::from_raw_parts(out.clone(), self.rows),
                k,
                sg_off,
            );
        }
        out
    }
}

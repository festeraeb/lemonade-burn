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

/// Native-layout Q5_K matvec: the GGUF bytes as they are (176-byte blocks = 44 u32 words, 0.6875 bytes a value), no repack.
///
/// The widened layout above costs 1.5 bytes a value (2.2x the file): 26 GB of Qwen3.6 Q5_K_M would be about 50 GB on the card.
/// This kernel keeps the file's density (so 25 GB of weights fit across two 16 GB cards) and pays bit extraction in the kernel.
/// One lane per (block, 32-value group) work item; `dims` = [rows, cols].
#[cube(launch)]
fn q5_k_native_matvec_kernel(w: &[u32], x: &[f32], dims: &[u32], out: &mut [f32], #[comptime] use_plane: bool) {
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let bpr = cols / 256usize;
    let items = bpr * 8usize;
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
        let mut j = lane;
        while j < items {
            let b = j / 8usize;
            let g = j % 8usize;
            let base = (r * bpr + b) * 44usize;
            let w0 = w[base];
            let d = f16_to_f32(w0 & 65535u32);
            let dmin = f16_to_f32((w0 >> 16u32) & 65535u32);
            // 6-bit scale and min of group g from the 12 packed bytes (words base+1 .. base+3)
            let mut sc = 0u32;
            let mut mn = 0u32;
            if g < 4usize {
                // bytes g and g + 4
                sc = (w[base + 1usize + g / 4usize] >> (((g % 4usize) * 8usize) as u32)) & 63u32;
                mn = (w[base + 1usize + (g + 4usize) / 4usize] >> ((((g + 4usize) % 4usize) * 8usize) as u32)) & 63u32;
            } else {
                // low nibbles from byte g + 4, top two bits from byte g - 4 (scale) and byte g (min)
                let bs = (w[base + 1usize + (g + 4usize) / 4usize] >> ((((g + 4usize) % 4usize) * 8usize) as u32)) & 255u32;
                let bt = (w[base + 1usize + (g - 4usize) / 4usize] >> ((((g - 4usize) % 4usize) * 8usize) as u32)) & 255u32;
                let bm = (w[base + 1usize + g / 4usize] >> (((g % 4usize) * 8usize) as u32)) & 255u32;
                sc = (bs & 15u32) | ((bt >> 6u32) << 4u32);
                mn = (bs >> 4u32) | ((bm >> 6u32) << 4u32);
            }
            let s = d * (sc as f32);
            let m = dmin * (mn as f32);
            let hi = (g % 2usize) as u32 * 4u32;
            let gshift = g as u32;
            let qs0 = base + 12usize + 8usize * (g / 2usize);
            let qh0 = base + 4usize;
            let xb = t * cols + b * 256usize + 32usize * g;
            let mut qx = 0.0f32;
            let mut xs = 0.0f32;
            #[unroll]
            for wi in 0..8usize {
                let qw = w[qs0 + wi];
                let hw = w[qh0 + wi];
                #[unroll]
                for k in 0..4usize {
                    let sh = (k * 8usize) as u32;
                    let nib = (qw >> (sh + hi)) & 15u32;
                    let hb = (hw >> (sh + gshift)) & 1u32;
                    let q = (nib + hb * 16u32) as f32;
                    let xv = x[xb + wi * 4usize + k];
                    qx += q * xv;
                    xs += xv;
                }
            }
            acc += s * qx - m * xs;
            j += width;
        }
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

    pub fn launch(&self, gpu: &Gpu, x: &Handle, m: usize) -> Handle {
        let n_out = m * self.rows;
        let out = gpu.scratch(n_out * 4);
        unsafe {
            q5_k_native_matvec_kernel::launch(
                &gpu.client,
                cube_count(n_out, gpu.per_cube()),
                CubeDim::new_1d(gpu.cube_dim as u32),
                BufferArg::from_raw_parts(self.w.clone(), self.n_words),
                BufferArg::from_raw_parts(x.clone(), m * self.cols),
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

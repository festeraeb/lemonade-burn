//! Factorised sub-1-bit layers (`lb_littlebit`) on the device: two kernels per layer, every path in
//! the same launches.
//!
//! * `lb_latent_kernel`: `z[t, p, i] = s_lat[p, i] · Σ_k sign(V_p[i, k]) · s_in[p, k] · x[t, k]`
//! * `lb_out_kernel`:    `y[t, j]    = Σ_p s_out[p, j] · Σ_i sign(U_p[j, i]) · z[t, p, i]`
//!
//! One plane per output value; lanes take 32-column words in turn, `plane_sum` reduces. Signs are bit
//! set = −1. `z` rows are padded to whole words (`rpad`) and the padding is written as 0, so the second
//! kernel can run whole words of U (whose padding bits are 0 = +1). Needs plane ops and `in % 32 == 0`
//! (every LFM2 / Qwen width); otherwise `upload` declines and the caller keeps the CPU path.

use crate::{cube_count, Gpu};
use cubecl::prelude::*;
use cubecl::server::Handle;
use lb_littlebit::{words, Linear};

#[cube(launch)]
fn lb_latent_kernel(v: &[u32], x: &[f32], s_in: &[f32], s_lat: &[f32], dims: &[u32], z: &mut [f32]) {
    // dims = [rows_all = paths * rpad, in, words(in), rpad, r]
    let rows_all = dims[0] as usize;
    let inp = dims[1] as usize;
    let wi = dims[2] as usize;
    let rpad = dims[3] as usize;
    let r = dims[4] as usize;
    let width = PLANE_DIM as usize;
    let plane_id = (UNIT_POS / PLANE_DIM) as usize;
    let lane = UNIT_POS_PLANE as usize;
    let ppc = CUBE_DIM_X as usize / width;
    let step = CUBE_COUNT as usize * ppc;
    let mut g = CUBE_POS as usize * ppc + plane_id;
    while g < z.len() {
        let t = g / rows_all;
        let row = g % rows_all;
        let p = row / rpad;
        let i = row % rpad;
        let mut acc = 0.0f32;
        if i < r {
            let x0 = t * inp;
            let s0 = p * inp;
            let mut w = lane;
            while w < wi {
                let bits = v[row * wi + w];
                #[unroll]
                for b in 0..32u32 {
                    let k = w * 32usize + b as usize;
                    let sgn = 1.0f32 - 2.0f32 * f32::cast_from((bits >> b) & 1u32);
                    acc += sgn * s_in[s0 + k] * x[x0 + k];
                }
                w += width;
            }
        }
        let total = plane_sum(acc);
        if lane == 0usize {
            if i < r {
                z[g] = total * s_lat[row];
            } else {
                z[g] = 0.0f32;
            }
        }
        g += step;
    }
}

#[cube(launch)]
fn lb_out_kernel(u: &[u32], z: &[f32], s_out: &[f32], dims: &[u32], y: &mut [f32]) {
    // dims = [out, words(r), rpad, paths]
    let out = dims[0] as usize;
    let wr = dims[1] as usize;
    let rpad = dims[2] as usize;
    let paths = dims[3] as usize;
    let width = PLANE_DIM as usize;
    let plane_id = (UNIT_POS / PLANE_DIM) as usize;
    let lane = UNIT_POS_PLANE as usize;
    let ppc = CUBE_DIM_X as usize / width;
    let step = CUBE_COUNT as usize * ppc;
    let mut g = CUBE_POS as usize * ppc + plane_id;
    while g < y.len() {
        let t = g / out;
        let j = g % out;
        let mut acc = 0.0f32;
        for p in 0..paths {
            let z0 = (t * paths + p) * rpad;
            let mut part = 0.0f32;
            let mut w = lane;
            while w < wr {
                let bits = u[(p * out + j) * wr + w];
                #[unroll]
                for b in 0..32u32 {
                    let sgn = 1.0f32 - 2.0f32 * f32::cast_from((bits >> b) & 1u32);
                    part += sgn * z[z0 + w * 32usize + b as usize];
                }
                w += width;
            }
            acc += part * s_out[p * out + j];
        }
        let total = plane_sum(acc);
        if lane == 0usize {
            y[g] = total;
        }
        g += step;
    }
}

/// A factorised layer resident on the device.
pub struct GpuLittleBit {
    pub out: usize,
    pub inp: usize,
    paths: usize,
    rpad: usize,
    u: Handle,
    v: Handle,
    s_in: Handle,
    s_lat: Handle,
    s_out: Handle,
    dims_a: Handle,
    dims_b: Handle,
    n: [usize; 5],
}

impl GpuLittleBit {
    pub fn upload(gpu: &Gpu, l: &Linear) -> Result<Self, String> {
        if !gpu.use_plane || l.inp % 32 != 0 || l.paths.is_empty() {
            return Err(format!("littlebit on the device needs plane ops and in % 32 == 0 ([{}, {}])", l.out, l.inp));
        }
        let r = l.paths[0].r;
        if l.paths.iter().any(|p| p.r != r || p.out != l.out || p.inp != l.inp) {
            return Err("littlebit: paths of one layer must share rank and shape".into());
        }
        let (np, wi, wr) = (l.paths.len(), words(l.inp), words(r));
        let rpad = wr * 32;
        let mut v = vec![0u32; np * rpad * wi];
        let mut s_lat = vec![0f32; np * rpad];
        let mut u = Vec::with_capacity(np * l.out * wr);
        let (mut s_in, mut s_out) = (Vec::with_capacity(np * l.inp), Vec::with_capacity(np * l.out));
        for (pi, p) in l.paths.iter().enumerate() {
            v[pi * rpad * wi..pi * rpad * wi + r * wi].copy_from_slice(&p.v);
            s_lat[pi * rpad..pi * rpad + r].copy_from_slice(&p.s_lat);
            u.extend_from_slice(&p.u);
            s_in.extend_from_slice(&p.s_in);
            s_out.extend_from_slice(&p.s_out);
        }
        let n = [u.len(), v.len(), s_in.len(), s_lat.len(), s_out.len()];
        Ok(Self {
            out: l.out,
            inp: l.inp,
            paths: np,
            rpad,
            u: gpu.upload_u32(u),
            v: gpu.upload_u32(v),
            s_in: gpu.upload_f32(&s_in),
            s_lat: gpu.upload_f32(&s_lat),
            s_out: gpu.upload_f32(&s_out),
            dims_a: gpu.const_u32(&[(np * rpad) as u32, l.inp as u32, wi as u32, rpad as u32, r as u32]),
            dims_b: gpu.const_u32(&[l.out as u32, wr as u32, rpad as u32, np as u32]),
            n,
        })
    }

    pub fn bytes_on_device(&self) -> usize {
        self.n.iter().sum::<usize>() * 4
    }

    /// `y [m, out] = x [m, in] · Wᵀ` on the device.
    pub fn launch(&self, gpu: &Gpu, x: &Handle, m: usize) -> Handle {
        let nz = m * self.paths * self.rpad;
        let z = gpu.scratch(nz * 4);
        let ny = m * self.out;
        let y = gpu.scratch(ny * 4);
        let per_cube = gpu.per_cube();
        unsafe {
            lb_latent_kernel::launch(
                &gpu.client,
                cube_count(nz, per_cube),
                CubeDim::new_1d(gpu.cube_dim as u32),
                BufferArg::from_raw_parts(self.v.clone(), self.n[1]),
                BufferArg::from_raw_parts(x.clone(), m * self.inp),
                BufferArg::from_raw_parts(self.s_in.clone(), self.n[2]),
                BufferArg::from_raw_parts(self.s_lat.clone(), self.n[3]),
                BufferArg::from_raw_parts(self.dims_a.clone(), 5),
                BufferArg::from_raw_parts(z.clone(), nz),
            );
            lb_out_kernel::launch(
                &gpu.client,
                cube_count(ny, per_cube),
                CubeDim::new_1d(gpu.cube_dim as u32),
                BufferArg::from_raw_parts(self.u.clone(), self.n[0]),
                BufferArg::from_raw_parts(z.clone(), nz),
                BufferArg::from_raw_parts(self.s_out.clone(), self.n[4]),
                BufferArg::from_raw_parts(self.dims_b.clone(), 4),
                BufferArg::from_raw_parts(y.clone(), ny),
            );
        }
        y
    }

    /// Host in, host out (tests and the host-path hook).
    pub fn matvec(&self, gpu: &Gpu, x: &[f32], m: usize) -> Result<Vec<f32>, String> {
        let xh = gpu.upload_f32(x);
        let y = self.launch(gpu, &xh, m);
        Ok(gpu.read_all(vec![(y, m * self.out)])?.remove(0))
    }
}

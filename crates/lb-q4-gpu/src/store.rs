//! Resident Q4_0 matrices and expert FFNs on one wgpu device.
//!
//! `Gpu` owns the compute client for the device chosen by `LB_WGPU_DEVICE`. A `GpuQ4Mat`
//! is a Q4_0 matrix uploaded once; `GpuExpert` is one SwiGLU expert (gate, up, down)
//! whose `ffn` runs three matvec launches and one `silu(g) * u` launch with the
//! intermediates staying on the device. Launches are asynchronous: `GpuExpert::launch`
//! returns the output handle and `Gpu::read_all` fetches a batch of outputs in one sync,
//! so a layer's resident experts cost one round trip, not one per expert.

use crate::{
    cube_count, matvec_dims, q4_0_gate_up_silu_kernel, q4_0_matvec_kernel, q4_0_stack_gate_up_silu_kernel, q4_0_stack_matvec_kernel, repack_q4_0, QuantMat,
    THREADS,
};
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::server::Handle;
use lb_ternary_gpu::pick_device;
use std::collections::HashMap;
use std::sync::Mutex;

/// Most free buffers to keep per size. Decode reuses a handful of sizes, so this only caps
/// pathological growth (a long prefill with many distinct `m`).
const POOL_PER_SIZE: usize = 64;

pub struct Gpu {
    pub client: Client,
    pub name: String,
    /// Units per cube for the matvec kernels (`LB_Q4_CUBE` overrides).
    pub cube_dim: usize,
    /// Reduce with `plane_sum` (device has subgroup ops); otherwise one output per cube.
    pub use_plane: bool,
    /// Plane (warp/wave) size range the driver reports.
    pub plane_size: (u32, u32),
    /// Recycled scratch buffers, keyed by byte size.
    free: Mutex<HashMap<usize, Vec<Handle>>>,
    /// Scratch handed out since the last sync. Reusing one before the GPU has run would
    /// corrupt it, so these only return to `free` after `read_all` has synchronised.
    in_flight: Mutex<Vec<(usize, Handle)>>,
}

impl Gpu {
    /// Open the device named by `LB_WGPU_DEVICE` (see `lb_ternary_gpu::pick_device`).
    pub fn open() -> Result<Self, String> {
        let device = pick_device()?;
        let dev = cubecl::Device::Wgpu(device);
        let client = std::panic::catch_unwind(|| dev.client()).map_err(|_| "no wgpu adapter available".to_string())?;
        let props = client.properties();
        let (pmin, pmax) = (props.hardware.plane_size_min, props.hardware.plane_size_max);
        let has_ops = props.features.plane.contains(cubecl::ir::features::Plane::Ops);
        let use_plane = has_ops && pmax > 0 && std::env::var("LB_NO_PLANE").is_err();
        // Default: two planes per cube, from the plane size the driver reports (wave32 NVIDIA/RDNA:
        // 64 units, wave64 Polaris/GCN: 128). Measured flat from 32 to 256 on a P100 (see
        // the commit message), so this only needs to be a whole number of planes.
        let mut cube_dim = std::env::var("LB_Q4_CUBE").ok().and_then(|v| v.parse().ok()).unwrap_or(if use_plane { 2 * pmax as usize } else { THREADS });
        if use_plane {
            // a whole number of planes, or `PLANE_DIM`-sized groups would straddle the cube
            let p = pmax as usize;
            cube_dim = cube_dim.div_ceil(p).max(1) * p;
        }
        let cube_dim = cube_dim.clamp(1, 256).next_power_of_two().min(256);
        Ok(Self {
            client,
            name: format!("{dev:?}"),
            cube_dim,
            use_plane,
            plane_size: (pmin, pmax),
            free: Mutex::new(HashMap::new()),
            in_flight: Mutex::new(Vec::new()),
        })
    }

    /// Outputs one cube produces with the current cube size and reduction.
    pub fn per_cube(&self) -> usize {
        if self.use_plane { self.cube_dim / (self.plane_size.1 as usize).max(1) } else { 1 }
    }

    pub fn upload_f32(&self, x: &[f32]) -> Handle {
        self.client.create_from_slice(f32::as_bytes(x))
    }

    /// Wait for everything submitted so far (a timing fence).
    pub fn sync(&self) -> Result<(), String> {
        cubecl::future::block_on(self.client.sync()).map_err(|e| format!("GPU sync failed: {e:?}"))
    }

    /// A scratch buffer of `bytes`, reusing a recycled one when one is free.
    ///
    /// `GpuQ4Mat::launch` and `GpuExpert::launch` allocated a fresh buffer for every
    /// intermediate, which in decode is about four allocations per resident expert, so
    /// hundreds a token. Sizes repeat exactly, so recycling removes essentially all of them.
    pub fn scratch(&self, bytes: usize) -> Handle {
        let h = match self.free.lock().expect("scratch pool").get_mut(&bytes).and_then(Vec::pop) {
            Some(h) => h,
            None => self.client.empty(bytes),
        };
        self.in_flight.lock().expect("scratch pool").push((bytes, h.clone()));
        h
    }

    /// Return every scratch buffer handed out since the last sync. Only safe once the
    /// device has finished with them, which is why `read_all` is the one caller.
    fn recycle(&self) {
        let taken: Vec<(usize, Handle)> = std::mem::take(&mut *self.in_flight.lock().expect("scratch pool"));
        let mut free = self.free.lock().expect("scratch pool");
        for (bytes, h) in taken {
            let slot = free.entry(bytes).or_default();
            if slot.len() < POOL_PER_SIZE {
                slot.push(h);
            }
        }
    }

    /// Read several f32 outputs back in one synchronisation, then recycle this round's scratch.
    pub fn read_all(&self, handles: Vec<(Handle, usize)>) -> Result<Vec<Vec<f32>>, String> {
        let lens: Vec<usize> = handles.iter().map(|(_, n)| *n).collect();
        let hs: Vec<Handle> = handles.into_iter().map(|(h, _)| h).collect();
        let bytes = self.client.read(hs).map_err(|e| format!("GPU read failed: {e:?}"))?;
        let out: Vec<Vec<f32>> = bytes.iter().zip(lens).map(|(b, n)| f32::from_bytes(b)[..n].to_vec()).collect();
        // the read synchronised, so nothing handed out before it is still in flight
        self.recycle();
        Ok(out)
    }

    /// Buffers currently held for reuse, for tests and reporting.
    pub fn pooled(&self) -> usize {
        self.free.lock().expect("scratch pool").values().map(Vec::len).sum()
    }
}

/// A Q4_0 matrix resident on the device, repacked (see [`repack_q4_0`]).
pub struct GpuQ4Mat {
    pub rows: usize,
    pub cols: usize,
    pub row_bytes: usize,
    qs: Handle,
    sc: Handle,
    n_qs: usize,
    n_sc: usize,
    dims: Handle,
}

impl GpuQ4Mat {
    pub fn upload(gpu: &Gpu, w: QuantMat<'_>) -> Result<Self, String> {
        if w.ty != lb_gguf::GgmlType::Q4_0 {
            return Err(format!("expected Q4_0, got {}", w.ty.name()));
        }
        if w.cols % 32 != 0 {
            return Err(format!("Q4_0 needs cols % 32 == 0, got {}", w.cols));
        }
        let n_blocks = w.rows * (w.cols / 32);
        let (qs, sc) = repack_q4_0(w.bytes, n_blocks);
        let dims = matvec_dims(w.rows, w.cols);
        Ok(Self {
            rows: w.rows,
            cols: w.cols,
            row_bytes: w.row_bytes(),
            n_qs: qs.len(),
            n_sc: sc.len(),
            qs: gpu.client.create_from_slice(u32::as_bytes(&qs)),
            sc: gpu.client.create_from_slice(u32::as_bytes(&sc)),
            dims: gpu.client.create_from_slice(u32::as_bytes(&dims)),
        })
    }

    pub fn bytes_on_device(&self) -> usize {
        (self.n_qs + self.n_sc) * 4
    }

    /// Launch `out = x @ W^T` for `m` rows of `x` already on the device; returns the output handle.
    pub fn launch(&self, gpu: &Gpu, x: &Handle, m: usize) -> Handle {
        let n_out = m * self.rows;
        let out = gpu.scratch(n_out * 4);
        unsafe {
            q4_0_matvec_kernel::launch(
                &gpu.client,
                cube_count(n_out, gpu.per_cube()),
                CubeDim::new_1d(gpu.cube_dim as u32),
                BufferArg::from_raw_parts(self.qs.clone(), self.n_qs),
                BufferArg::from_raw_parts(self.sc.clone(), self.n_sc),
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

/// One SwiGLU expert resident on the device: `down(silu(gate x) * up x)`.
pub struct GpuExpert {
    pub gate: GpuQ4Mat,
    pub up: GpuQ4Mat,
    pub down: GpuQ4Mat,
}

impl GpuExpert {
    pub fn upload(gpu: &Gpu, gate: QuantMat<'_>, up: QuantMat<'_>, down: QuantMat<'_>) -> Result<Self, String> {
        Ok(Self { gate: GpuQ4Mat::upload(gpu, gate)?, up: GpuQ4Mat::upload(gpu, up)?, down: GpuQ4Mat::upload(gpu, down)? })
    }

    pub fn bytes_on_device(&self) -> usize {
        self.gate.bytes_on_device() + self.up.bytes_on_device() + self.down.bytes_on_device()
    }

    /// Launch the expert on `m` rows of `x` (device handle); returns (output handle, length).
    pub fn launch(&self, gpu: &Gpu, x: &Handle, m: usize) -> (Handle, usize) {
        let n_ff = self.gate.rows;
        assert!(self.gate.rows == self.up.rows && self.gate.cols == self.up.cols, "gate/up shape mismatch");
        let n_h = m * n_ff;
        let h = gpu.scratch(n_h * 4);
        unsafe {
            q4_0_gate_up_silu_kernel::launch(
                &gpu.client,
                cube_count(n_h, gpu.per_cube()),
                CubeDim::new_1d(gpu.cube_dim as u32),
                BufferArg::from_raw_parts(self.gate.qs.clone(), self.gate.n_qs),
                BufferArg::from_raw_parts(self.gate.sc.clone(), self.gate.n_sc),
                BufferArg::from_raw_parts(self.up.qs.clone(), self.up.n_qs),
                BufferArg::from_raw_parts(self.up.sc.clone(), self.up.n_sc),
                BufferArg::from_raw_parts(x.clone(), m * self.gate.cols),
                BufferArg::from_raw_parts(self.gate.dims.clone(), 2),
                BufferArg::from_raw_parts(h.clone(), n_h),
                gpu.use_plane,
            );
        }
        let out = self.down.launch(gpu, &h, m);
        (out, m * self.down.rows)
    }

    /// Upload `x`, run the expert, read the result (one expert, one round trip).
    pub fn ffn(&self, gpu: &Gpu, x: &[f32], m: usize) -> Result<Vec<f32>, String> {
        let xh = gpu.upload_f32(x);
        let (out, n) = self.launch(gpu, &xh, m);
        Ok(gpu.read_all(vec![(out, n)])?.remove(0))
    }
}

/// One layer's routed experts resident as three fused matrices (gate, up, down), each the
/// GGUF tensor `[n_expert, rows, cols]` repacked as a single `qs`/`sc` pair. A decode step
/// computes every routed expert in two dispatches (see [`GpuExpertStack::launch`]) instead
/// of two per expert.
pub struct GpuExpertStack {
    pub n_expert: usize,
    pub n_ff: usize,
    pub n_embd: usize,
    gate: StackMat,
    up: StackMat,
    down: StackMat,
}

/// One fused `[n_expert * rows, cols]` matrix: the repack is over the whole run, so block `b`
/// of expert `e`'s row `r` is global block `(e * rows + r) * bpr + b`. `sc` packs two scales
/// per word across expert boundaries, which is why it is repacked as one array and not per expert.
struct StackMat {
    qs: Handle,
    sc: Handle,
    n_qs: usize,
    n_sc: usize,
    dims: Handle,
}

impl StackMat {
    fn upload(gpu: &Gpu, w: QuantMat<'_>, n_expert: usize) -> Result<Self, String> {
        if w.ty != lb_gguf::GgmlType::Q4_0 {
            return Err(format!("expected Q4_0, got {}", w.ty.name()));
        }
        if w.cols % 32 != 0 {
            return Err(format!("Q4_0 needs cols % 32 == 0, got {}", w.cols));
        }
        if n_expert == 0 || w.rows % n_expert != 0 {
            return Err(format!("{} rows do not split into {} experts", w.rows, n_expert));
        }
        let (qs, sc) = repack_q4_0(w.bytes, w.rows * (w.cols / 32));
        let dims = matvec_dims(w.rows / n_expert, w.cols);
        Ok(Self {
            n_qs: qs.len(),
            n_sc: sc.len(),
            qs: gpu.client.create_from_slice(u32::as_bytes(&qs)),
            sc: gpu.client.create_from_slice(u32::as_bytes(&sc)),
            dims: gpu.client.create_from_slice(u32::as_bytes(&dims)),
        })
    }
}

impl GpuExpertStack {
    /// `gate`/`up` are `[n_expert * n_ff, n_embd]` and `down` is `[n_expert * n_embd, n_ff]`:
    /// the fused GGUF expert tensors viewed as one tall matrix each (rows of expert `e`
    /// are `e * rows .. (e + 1) * rows`), so their bytes can be passed straight through.
    pub fn upload(gpu: &Gpu, n_expert: usize, gate: QuantMat<'_>, up: QuantMat<'_>, down: QuantMat<'_>) -> Result<Self, String> {
        let (g, u, d) = (StackMat::upload(gpu, gate, n_expert)?, StackMat::upload(gpu, up, n_expert)?, StackMat::upload(gpu, down, n_expert)?);
        let (n_ff, n_embd) = (gate.rows / n_expert, gate.cols);
        if up.rows != gate.rows || up.cols != gate.cols || down.rows / n_expert != n_embd || down.cols != n_ff {
            return Err("gate/up/down shapes disagree".to_string());
        }
        Ok(Self { n_expert, n_ff, n_embd, gate: g, up: u, down: d })
    }

    pub fn bytes_on_device(&self) -> usize {
        [&self.gate, &self.up, &self.down].iter().map(|m| (m.n_qs + m.n_sc) * 4).sum()
    }

    /// Run jobs `(expert slot, row of x)` in two dispatches total; job `j` is
    /// `down_e(silu(gate_e x_r) * up_e x_r)` and lands in `out[j * n_embd ..]`. Decode with
    /// top-8 routing is eight jobs all reading row 0. Returns (output handle, length).
    pub fn launch(&self, gpu: &Gpu, x: &Handle, m: usize, jobs: &[(usize, usize)]) -> (Handle, usize) {
        assert!(jobs.iter().all(|&(e, r)| e < self.n_expert && r < m), "job out of range");
        let flat: Vec<u32> = jobs.iter().flat_map(|&(e, r)| [e as u32, r as u32]).collect();
        let jh = gpu.client.create_from_slice(u32::as_bytes(&flat));
        let n_h = jobs.len() * self.n_ff;
        let h = gpu.scratch(n_h * 4);
        unsafe {
            q4_0_stack_gate_up_silu_kernel::launch(
                &gpu.client,
                cube_count(n_h, gpu.per_cube()),
                CubeDim::new_1d(gpu.cube_dim as u32),
                BufferArg::from_raw_parts(self.gate.qs.clone(), self.gate.n_qs),
                BufferArg::from_raw_parts(self.gate.sc.clone(), self.gate.n_sc),
                BufferArg::from_raw_parts(self.up.qs.clone(), self.up.n_qs),
                BufferArg::from_raw_parts(self.up.sc.clone(), self.up.n_sc),
                BufferArg::from_raw_parts(x.clone(), m * self.n_embd),
                BufferArg::from_raw_parts(jh.clone(), flat.len()),
                BufferArg::from_raw_parts(self.gate.dims.clone(), 2),
                BufferArg::from_raw_parts(h.clone(), n_h),
                gpu.use_plane,
            );
        }
        let n_out = jobs.len() * self.n_embd;
        let out = gpu.scratch(n_out * 4);
        unsafe {
            q4_0_stack_matvec_kernel::launch(
                &gpu.client,
                cube_count(n_out, gpu.per_cube()),
                CubeDim::new_1d(gpu.cube_dim as u32),
                BufferArg::from_raw_parts(self.down.qs.clone(), self.down.n_qs),
                BufferArg::from_raw_parts(self.down.sc.clone(), self.down.n_sc),
                BufferArg::from_raw_parts(h.clone(), n_h),
                BufferArg::from_raw_parts(jh.clone(), flat.len()),
                BufferArg::from_raw_parts(self.down.dims.clone(), 2),
                BufferArg::from_raw_parts(out.clone(), n_out),
                gpu.use_plane,
            );
        }
        (out, n_out)
    }

    /// Upload `x`, run `jobs`, read the result (one round trip).
    pub fn ffn(&self, gpu: &Gpu, x: &[f32], m: usize, jobs: &[(usize, usize)]) -> Result<Vec<f32>, String> {
        let xh = gpu.upload_f32(x);
        let (out, n) = self.launch(gpu, &xh, m, jobs);
        Ok(gpu.read_all(vec![(out, n)])?.remove(0))
    }
}

//! Resident Q4_0 matrices and expert FFNs on one wgpu device.
//!
//! `Gpu` owns the compute client for the device chosen by `LB_WGPU_DEVICE`. A `GpuQ4Mat`
//! is a Q4_0 matrix uploaded once; `GpuExpert` is one SwiGLU expert (gate, up, down)
//! whose `ffn` runs three matvec launches and one `silu(g) * u` launch with the
//! intermediates staying on the device. Launches are asynchronous: `GpuExpert::launch`
//! returns the output handle and `Gpu::read_all` fetches a batch of outputs in one sync,
//! so a layer's resident experts cost one round trip, not one per expert.

use crate::{conv_decode_kernel, cube_count, matvec_dims, q4_0_gate_up_silu_kernel, q4_0_matvec_kernel, q6_k_matvec_kernel, repack_q4_0, repack_q6_k, QuantMat, THREADS};
use cubecl::bytes::Bytes;
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
        self.client.create(Bytes::from_elems(x.to_vec()))
    }

    /// Put an owned, typed buffer on the device without copying it on the host first.
    ///
    /// `Client::create_from_slice` copies the blob **twice** before any backend sees it:
    /// `slice.to_vec()`, and then `Bytes::from_bytes_vec(data.to_vec())` inside
    /// `do_create_from_slices` on a vector it already owns (cubecl-runtime 0.11.3
    /// `client.rs`). Above the malloc mmap threshold each fresh allocation also page-faults,
    /// which is why our measured host-to-device rate got *worse* with blob size. `create`
    /// takes `Bytes` as it is, and `Bytes::from_elems` erases the element type of a `Vec`
    /// without touching the data, so a repacked `Vec<u32>` reaches the device uncopied.
    pub fn upload_u32(&self, v: Vec<u32>) -> Handle {
        self.client.create(Bytes::from_elems(v))
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
            qs: gpu.upload_u32(qs),
            sc: gpu.upload_u32(sc),
            dims: gpu.upload_u32(dims.to_vec()),
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

/// One LFM2 short-conv block resident on the device: `in_proj`, the taps, the rolling state
/// and `out_proj`, so a decode step costs **one** host sync for the layer instead of two.
///
/// The state lives on the device and is updated in place by the kernel, so it never crosses
/// the bus. That also means [`Self::launch`] is **stateful**: calling it advances the layer
/// by one token, and it must be called once per token in order. `reset` clears it between
/// sequences, and prefill (`m > 1`) must not use this path at all -- it would leave the state
/// holding the wrong rows.
pub struct GpuConv {
    pub in_proj: GpuQ4Mat,
    pub out_proj: GpuQ4Mat,
    /// `[l_cache, n_embd]`, tap `k` at `k * n_embd`.
    conv: Handle,
    /// `[l_cache - 1, n_embd]`, the most recent rows of `b * x`.
    state: Handle,
    dims: Handle,
    e: usize,
    l: usize,
}

impl GpuConv {
    /// `conv` is `[l_cache][n_embd]`, already in tap order (see `lb_lfm2::weights`).
    pub fn upload(gpu: &Gpu, in_proj: QuantMat<'_>, conv: &[f32], out_proj: QuantMat<'_>) -> Result<Self, String> {
        let e = out_proj.rows;
        if conv.len() % e != 0 {
            return Err(format!("conv taps {} not a multiple of n_embd {e}", conv.len()));
        }
        let l = conv.len() / e;
        if l < 2 {
            return Err(format!("l_cache must be at least 2, got {l}"));
        }
        if in_proj.rows != 3 * e {
            return Err(format!("in_proj rows {} expected 3 x {e}", in_proj.rows));
        }
        Ok(Self {
            in_proj: GpuQ4Mat::upload(gpu, in_proj)?,
            out_proj: GpuQ4Mat::upload(gpu, out_proj)?,
            conv: gpu.upload_f32(conv),
            state: gpu.client.create(Bytes::from_elems(vec![0f32; (l - 1) * e])),
            dims: gpu.upload_u32(vec![e as u32, l as u32]),
            e,
            l,
        })
    }

    pub fn bytes_on_device(&self) -> usize {
        self.in_proj.bytes_on_device() + self.out_proj.bytes_on_device() + (self.l * 2 - 1) * self.e * 4
    }

    /// Rows the state holds: `l_cache - 1` rows of `n_embd`.
    pub fn state_rows(&self) -> (usize, usize) {
        (self.l - 1, self.e)
    }

    /// Overwrite the state with `rows` (`(l_cache - 1) * n_embd` values, oldest first).
    ///
    /// Prefill runs on the CPU, so after it the device state has to be told where the
    /// sequence got to, or the first decode token would convolve against zeros.
    pub fn set_state(&self, gpu: &Gpu, rows: &[f32]) -> Result<(), String> {
        let want = (self.l - 1) * self.e;
        if rows.len() != want {
            return Err(format!("conv state needs {want} values, got {}", rows.len()));
        }
        gpu.client.write(&self.state, Bytes::from_elems(rows.to_vec()));
        Ok(())
    }

    /// Read the state back. Costs a sync, so this is for the rare path where the host needs
    /// the history again (a multi-token forward after decoding has already advanced it).
    pub fn read_state(&self, gpu: &Gpu) -> Result<Vec<f32>, String> {
        let n = (self.l - 1) * self.e;
        let bytes = gpu.client.read(vec![self.state.clone()]).map_err(|e| format!("conv state read failed: {e:?}"))?;
        Ok(f32::from_bytes(&bytes[0])[..n].to_vec())
    }

    /// Zero the rolling state: a new sequence starts with no history.
    pub fn reset(&self, gpu: &Gpu) {
        let zeros = vec![0f32; (self.l - 1) * self.e];
        // overwrite in place so every handle already captured in a launch still refers to it
        gpu.client.write(&self.state, Bytes::from_elems(zeros));
    }

    /// One decode token: `in_proj -> conv -> out_proj`, all on the device. Returns the output
    /// handle and its length; nothing is read back, so the caller syncs once for the layer.
    ///
    /// Advances the rolling state by one token. `x` must be a single row (`m = 1`).
    pub fn launch(&self, gpu: &Gpu, x: &Handle) -> (Handle, usize) {
        let bcx = self.in_proj.launch(gpu, x, 1);
        let y = gpu.scratch(self.e * 4);
        unsafe {
            conv_decode_kernel::launch(
                &gpu.client,
                cube_count(self.e, gpu.cube_dim),
                CubeDim::new_1d(gpu.cube_dim as u32),
                BufferArg::from_raw_parts(bcx.clone(), 3 * self.e),
                BufferArg::from_raw_parts(self.conv.clone(), self.l * self.e),
                BufferArg::from_raw_parts(self.state.clone(), (self.l - 1) * self.e),
                BufferArg::from_raw_parts(y.clone(), self.e),
                BufferArg::from_raw_parts(self.dims.clone(), 2),
            );
        }
        let out = self.out_proj.launch(gpu, &y, 1);
        (out, self.e)
    }

    /// Upload `x`, run the block, read the result (one token, one round trip).
    pub fn forward(&self, gpu: &Gpu, x: &[f32]) -> Result<Vec<f32>, String> {
        let xh = gpu.upload_f32(x);
        let (out, n) = self.launch(gpu, &xh);
        Ok(gpu.read_all(vec![(out, n)])?.remove(0))
    }
}

/// A Q6_K matrix resident on the device, repacked (see [`repack_q6_k`]).
///
/// The one Q6_K tensor in LFM2.5 is `token_embd`, which is also the tied LM head: 128000 x
/// 2048, the largest dense matrix in the model and 55 ms of CPU per token on its own.
pub struct GpuQ6KMat {
    pub rows: usize,
    pub cols: usize,
    qv: Handle,
    sd: Handle,
    n_qv: usize,
    n_sd: usize,
    dims: Handle,
}

impl GpuQ6KMat {
    pub fn upload(gpu: &Gpu, w: QuantMat<'_>) -> Result<Self, String> {
        if w.ty != lb_gguf::GgmlType::Q6K {
            return Err(format!("expected Q6_K, got {}", w.ty.name()));
        }
        if w.cols % 256 != 0 {
            return Err(format!("Q6_K needs cols % 256 == 0, got {}", w.cols));
        }
        let n_super = w.rows * (w.cols / 256);
        let (qv, sd) = repack_q6_k(w.bytes, n_super);
        let dims = matvec_dims(w.rows, w.cols);
        Ok(Self {
            rows: w.rows,
            cols: w.cols,
            n_qv: qv.len(),
            n_sd: sd.len(),
            qv: gpu.upload_u32(qv),
            sd: gpu.upload_f32(&sd),
            dims: gpu.upload_u32(dims.to_vec()),
        })
    }

    pub fn bytes_on_device(&self) -> usize {
        (self.n_qv + self.n_sd) * 4
    }

    /// Launch `out = x @ W^T` for `m` rows of `x` already on the device.
    pub fn launch(&self, gpu: &Gpu, x: &Handle, m: usize) -> Handle {
        let n_out = m * self.rows;
        let out = gpu.scratch(n_out * 4);
        unsafe {
            q6_k_matvec_kernel::launch(
                &gpu.client,
                cube_count(n_out, gpu.per_cube()),
                CubeDim::new_1d(gpu.cube_dim as u32),
                BufferArg::from_raw_parts(self.qv.clone(), self.n_qv),
                BufferArg::from_raw_parts(self.sd.clone(), self.n_sd),
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

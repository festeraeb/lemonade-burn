//! Resident Q4_0 matrices and expert FFNs on one wgpu device.
//!
//! `Gpu` owns the compute client for the device chosen by `LB_WGPU_DEVICE`. A `GpuQ4Mat`
//! is a Q4_0 matrix uploaded once; `GpuExpert` is one SwiGLU expert (gate, up, down)
//! whose `ffn` runs three matvec launches and one `silu(g) * u` launch with the
//! intermediates staying on the device. Launches are asynchronous: `GpuExpert::launch`
//! returns the output handle and `Gpu::read_all` fetches a batch of outputs in one sync,
//! so a layer's resident experts cost one round trip, not one per expert.

use crate::{matvec_dims, q4_0_matvec_kernel, QuantMat, THREADS};
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::server::Handle;
use lb_ternary_gpu::pick_device;

/// `h[i] = silu(g[i]) * u[i]`.
#[cube(launch)]
fn silu_mul_kernel(g: &[f32], u: &[f32], h: &mut [f32]) {
    let i = ABSOLUTE_POS;
    if i < h.len() {
        let x = g[i];
        h[i] = x / (1.0f32 + Exp::exp(-x)) * u[i];
    }
}

pub struct Gpu {
    pub client: Client,
    pub name: String,
}

impl Gpu {
    /// Open the device named by `LB_WGPU_DEVICE` (see `lb_ternary_gpu::pick_device`).
    pub fn open() -> Result<Self, String> {
        let device = pick_device()?;
        let dev = cubecl::Device::Wgpu(device);
        let client = std::panic::catch_unwind(|| dev.client()).map_err(|_| "no wgpu adapter available".to_string())?;
        Ok(Self { client, name: format!("{dev:?}") })
    }

    pub fn upload_f32(&self, x: &[f32]) -> Handle {
        self.client.create_from_slice(f32::as_bytes(x))
    }

    /// Wait for everything submitted so far (a timing fence).
    pub fn sync(&self) -> Result<(), String> {
        cubecl::future::block_on(self.client.sync()).map_err(|e| format!("GPU sync failed: {e:?}"))
    }

    /// Read several f32 outputs back in one synchronisation.
    pub fn read_all(&self, handles: Vec<(Handle, usize)>) -> Result<Vec<Vec<f32>>, String> {
        let lens: Vec<usize> = handles.iter().map(|(_, n)| *n).collect();
        let hs: Vec<Handle> = handles.into_iter().map(|(h, _)| h).collect();
        let bytes = self.client.read(hs).map_err(|e| format!("GPU read failed: {e:?}"))?;
        Ok(bytes.iter().zip(lens).map(|(b, n)| f32::from_bytes(b)[..n].to_vec()).collect())
    }
}

/// A Q4_0 matrix resident on the device.
pub struct GpuQ4Mat {
    pub rows: usize,
    pub cols: usize,
    pub row_bytes: usize,
    words: Handle,
    n_words: usize,
    dims: Handle,
}

impl GpuQ4Mat {
    pub fn upload(gpu: &Gpu, w: QuantMat<'_>) -> Result<Self, String> {
        if w.ty != lb_gguf::GgmlType::Q4_0 {
            return Err(format!("expected Q4_0, got {}", w.ty.name()));
        }
        let mut bytes = w.bytes.to_vec();
        while bytes.len() % 4 != 0 {
            bytes.push(0);
        }
        let n_words = bytes.len() / 4;
        let words = gpu.client.create_from_slice(&bytes);
        let dims = matvec_dims(w.rows, w.cols, w.row_bytes());
        let dims = gpu.client.create_from_slice(u32::as_bytes(&dims));
        Ok(Self { rows: w.rows, cols: w.cols, row_bytes: w.row_bytes(), words, n_words, dims })
    }

    pub fn bytes_on_device(&self) -> usize {
        self.n_words * 4
    }

    /// Launch `out = x @ W^T` for `m` rows of `x` already on the device; returns the output handle.
    pub fn launch(&self, gpu: &Gpu, x: &Handle, m: usize) -> Handle {
        let n_out = m * self.rows;
        let out = gpu.client.empty(n_out * 4);
        unsafe {
            q4_0_matvec_kernel::launch(
                &gpu.client,
                CubeCount::Static(n_out as u32, 1, 1),
                CubeDim::new_1d(THREADS as u32),
                BufferArg::from_raw_parts(self.words.clone(), self.n_words),
                BufferArg::from_raw_parts(x.clone(), m * self.cols),
                BufferArg::from_raw_parts(self.dims.clone(), 4),
                BufferArg::from_raw_parts(out.clone(), n_out),
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
        let g = self.gate.launch(gpu, x, m);
        let u = self.up.launch(gpu, x, m);
        let h = gpu.client.empty(m * n_ff * 4);
        let threads = 64usize;
        unsafe {
            silu_mul_kernel::launch(
                &gpu.client,
                CubeCount::Static((m * n_ff).div_ceil(threads) as u32, 1, 1),
                CubeDim::new_1d(threads as u32),
                BufferArg::from_raw_parts(g, m * n_ff),
                BufferArg::from_raw_parts(u, m * n_ff),
                BufferArg::from_raw_parts(h.clone(), m * n_ff),
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

//! Ternary GEMM on the GPU (CubeCL, wgpu runtime). Same layout and exact integers as
//! `lb_ternary::ternary_gemm_i32(w, x, Kernel::Scalar)`: out[t * rows + r] = sum_k w[r,k] * x_q[t,k].

use cubecl::prelude::*;
use cubecl::wgpu::WgpuDevice;
use lb_ternary::{QuantActs, TernaryView};

/// One unit per output element (token t, row r).
/// `w`: packed rows as little-endian u32 words (kp / 16 words per row).
/// `x`: activations as i32, m rows of kp. `dims` = [rows, kp].
#[cube(launch)]
fn ternary_gemm_kernel(w: &[u32], x: &[i32], dims: &[u32], out: &mut [i32]) {
    let pos = ABSOLUTE_POS;
    if pos < out.len() {
        let rows = dims[0] as usize;
        let kp = dims[1] as usize;
        let t = pos / rows;
        let r = pos % rows;
        let words_per_row = kp / 16;
        let blocks = kp / 128;
        let mut acc = 0i32;
        for b in 0..blocks {
            for j in 0..32usize {
                let word = w[r * words_per_row + b * 8 + j / 4];
                let byte = (word >> (((j % 4) * 8) as u32)) & 255u32;
                for k in 0..4usize {
                    let code = ((byte >> ((2 * k) as u32)) & 3u32) as i32;
                    acc += (code - 1) * x[t * kp + b * 128 + k * 32 + j];
                }
            }
        }
        out[pos] = acc;
    }
}

/// A minimal `block_on` (the adapter list is an async call in wgpu 30).
fn block_on<F: std::future::Future>(f: F) -> F::Output {
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};
    struct Unpark(std::thread::Thread);
    impl Wake for Unpark {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(Unpark(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut f = std::pin::pin!(f);
    loop {
        if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::park();
    }
}

/// One Vulkan adapter as the wgpu runtime counts it.
#[derive(Clone, Debug)]
pub struct AdapterRow {
    /// Position among the discrete adapters (what `discrete:N` means *right now*; it moves when a
    /// card is added or removed, so name or bus is the stable way to pick).
    pub discrete_index: Option<usize>,
    pub name: String,
    /// PCI bus id as the driver reports it (`0000:21:00.0`), empty when it does not.
    pub bus: String,
    pub device_type: String,
}

/// The Vulkan adapters of this machine, in the order the CubeCL wgpu runtime indexes them
/// (it asks an instance limited to Vulkan, exactly as this does).
pub fn list_adapters() -> Vec<AdapterRow> {
    use cubecl::wgpu::wgpu;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..wgpu::InstanceDescriptor::new_without_display_handle() });
    let mut next_discrete = 0usize;
    block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN))
        .iter()
        .map(|a| {
            let i = a.get_info();
            let discrete = i.device_type == wgpu::DeviceType::DiscreteGpu;
            let row = AdapterRow { discrete_index: discrete.then_some(next_discrete), name: i.name.clone(), bus: i.device_pci_bus_id.clone(), device_type: format!("{:?}", i.device_type) };
            if discrete {
                next_discrete += 1;
            }
            row
        })
        .collect()
}

/// The discrete index `spec` names right now. `spec` is `discrete:N`, `name:<text>` (case-insensitive
/// substring of the adapter name; must match exactly one adapter, `name:<text>#K` takes the Kth match)
/// or `bus:<pci id>` (`0000:21:00.0`, or the short `21:00.0`).
pub fn resolve_discrete(spec: &str) -> Result<usize, String> {
    let (kind, val) = spec.split_once(':').ok_or_else(|| format!("device spec {spec:?}: use discrete:N, name:<text> or bus:<pci id>"))?;
    let rows = list_adapters();
    let describe = || rows.iter().map(|r| format!("{} [{}] {}", r.name, r.bus, r.discrete_index.map(|i| format!("discrete:{i}")).unwrap_or_else(|| r.device_type.clone()))).collect::<Vec<_>>().join("; ");
    match kind {
        "discrete" => val.parse().map_err(|_| format!("device spec {spec:?}: index must be a number")),
        "name" => {
            let (text, nth) = match val.rsplit_once('#') {
                Some((t, k)) => (t, k.parse::<usize>().map_err(|_| format!("device spec {spec:?}: #K must be a number"))?),
                None => (val, 0),
            };
            let want = text.to_ascii_lowercase();
            let hits: Vec<&AdapterRow> = rows.iter().filter(|r| r.discrete_index.is_some() && r.name.to_ascii_lowercase().contains(&want)).collect();
            if hits.len() > 1 && !val.contains('#') {
                return Err(format!("device spec {spec:?} matches {} adapters, add #K or use bus: ({})", hits.len(), describe()));
            }
            hits.get(nth).and_then(|r| r.discrete_index).ok_or_else(|| format!("device spec {spec:?}: no such adapter ({})", describe()))
        }
        "bus" => {
            let want = val.trim().to_ascii_lowercase();
            rows.iter()
                .find(|r| {
                    let b = r.bus.to_ascii_lowercase();
                    !b.is_empty() && (b == want || b.ends_with(&format!(":{want}")))
                })
                .and_then(|r| r.discrete_index)
                .ok_or_else(|| format!("device spec {spec:?}: no discrete adapter on that bus ({})", describe()))
        }
        _ => Err(format!("device spec {spec:?}: use discrete:N, name:<text> or bus:<pci id>")),
    }
}

/// The wgpu device for a spec: `default`, `discrete:N`, `integrated:N`, `name:<text>`, `bus:<pci id>`.
pub fn pick_device_spec(spec: &str) -> Result<WgpuDevice, String> {
    use cubecl::device::WgpuDeviceKind;
    if spec.starts_with("name:") || spec.starts_with("bus:") {
        return Ok(WgpuDevice::new(WgpuDeviceKind::DiscreteGpu(resolve_discrete(spec)?)));
    }
    let (kind, idx) = spec.split_once(':').unwrap_or((spec, "0"));
    let n: usize = idx.parse().map_err(|_| format!("LB_WGPU_DEVICE {spec:?}: index must be a number"))?;
    Ok(match kind {
        "" | "default" => WgpuDevice::default(),
        "discrete" => WgpuDevice::new(WgpuDeviceKind::DiscreteGpu(n)),
        "integrated" => WgpuDevice::new(WgpuDeviceKind::IntegratedGpu(n)),
        _ => return Err(format!("LB_WGPU_DEVICE {spec:?}: use default, discrete:N, integrated:N, name:<text> or bus:<pci id>")),
    })
}

/// The wgpu device from `LB_WGPU_DEVICE`: unset or "default" = wgpu's choice, "discrete:N" = the
/// Nth discrete GPU (the order changes when a card is added: prefer `name:WX 5100` or
/// `bus:0000:21:00.0`), "integrated:N" = the Nth integrated one.
pub fn pick_device() -> Result<WgpuDevice, String> {
    pick_device_spec(&std::env::var("LB_WGPU_DEVICE").unwrap_or_default())
}

pub fn ternary_gemm_i32_gpu(w: TernaryView<'_>, x: &QuantActs) -> Result<Vec<i32>, String> {
    let kp = lb_ternary::pack::padded_cols(w.cols);
    if x.kp != kp {
        return Err(format!("activation row length {} does not match padded weight cols {}", x.kp, kp));
    }
    let n_out = x.m * w.rows;
    if n_out == 0 {
        return Ok(Vec::new());
    }
    let device = pick_device()?;
    let client = std::panic::catch_unwind(|| cubecl::Device::Wgpu(device.clone()).client())
        .map_err(|_| "no wgpu adapter available (WgpuRuntime::client panicked)".to_string())?;

    let words: Vec<u32> = w.data[..w.rows * w.row_bytes()]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let acts: Vec<i32> = x.q.iter().map(|&v| v as i32).collect();
    let dims = [w.rows as u32, kp as u32];

    let w_h = client.create_from_slice(u32::as_bytes(&words));
    let x_h = client.create_from_slice(i32::as_bytes(&acts));
    let d_h = client.create_from_slice(u32::as_bytes(&dims));
    let o_h = client.empty(n_out * std::mem::size_of::<i32>());

    let threads = 64usize;
    let cubes = n_out.div_ceil(threads) as u32;
    unsafe {
        ternary_gemm_kernel::launch(
            &client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(threads as u32),
            BufferArg::from_raw_parts(w_h, words.len()),
            BufferArg::from_raw_parts(x_h, acts.len()),
            BufferArg::from_raw_parts(d_h, 2),
            BufferArg::from_raw_parts(o_h.clone(), n_out),
        );
    }
    let bytes = client.read_one(o_h).map_err(|e| format!("GPU read failed: {e:?}"))?;
    Ok(i32::from_bytes(&bytes).to_vec())
}

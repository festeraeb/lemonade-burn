//! Resident Q4_0 matrices and expert FFNs on one device (wgpu, or CUDA with the `cuda` feature).
//!
//! `Gpu` owns the compute client for the device chosen by `LB_CUDA_DEVICE` or `LB_WGPU_DEVICE`. A `GpuQ4Mat`
//! is a Q4_0 matrix uploaded once; `GpuExpert` is one SwiGLU expert (gate, up, down)
//! whose `ffn` runs three matvec launches and one `silu(g) * u` launch with the
//! intermediates staying on the device. Launches are asynchronous: `GpuExpert::launch`
//! returns the output handle and `Gpu::read_all` fetches a batch of outputs in one sync,
//! so a layer's resident experts cost one round trip, not one per expert.

use crate::{conv_decode_kernel, cube_count, matvec_dims, q4_0_gate_up_silu_kernel, q4_0_gate_up_silu_rows_kernel, q4_0_matvec_kernel, q4_0_matvec_rows_kernel, q4_0_repack_kernel, stream_fence_kernel, q4_0_routed_gate_up_silu_kernel, q4_0_routed_matvec_kernel, q6_k_matvec_kernel, repack_q4_0, repack_q6_k, QuantMat, REPACK_CUBE, THREADS};
use cubecl::bytes::Bytes;
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::server::Handle;
use lb_ternary_gpu::pick_device;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::Mutex;

/// Launch the prefill GEMM on `$gpu` with the kernel its card takes: the tensor-core form when
/// [`Gpu::tensor_cores`] (its input `$x` of `$n` f32 values converted to f16 first), else the f32
/// tiles. Every other argument is the same for both.
macro_rules! gemm_launch {
    ($gpu:expr, $count:expr, $dim:expr, $gq:expr, $gs:expr, $uq:expr, $us:expr, ($x:expr, $n:expr), $($arg:expr),+ $(,)?) => {
        if $gpu.tensor_cores {
            let xh = $gpu.to_f16(&$x, $n);
            crate::gemm::q4_0_gemm_tc_kernel::launch(&$gpu.client, $count, $dim, $gq, $gs, $uq, $us, BufferArg::from_raw_parts(xh, $n), $($arg),+, crate::gemm::tc_stage2())
        } else if $gpu.fp16_pipe {
            let xh = $gpu.to_f16(&$x, $n);
            crate::gemm::q4_0_gemm_h2_kernel::launch(&$gpu.client, $count, $dim, $gq, $gs, $uq, $us, BufferArg::from_raw_parts(xh, $n), $($arg),+)
        } else {
            crate::gemm::q4_0_gemm_kernel::launch(&$gpu.client, $count, $dim, $gq, $gs, $uq, $us, BufferArg::from_raw_parts($x.clone(), $n), $($arg),+)
        }
    };
}

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
    /// f16 tensor cores (cmma 16x16x16, f32 accumulate) reported by the runtime and planes of
    /// 32: the prefill GEMM takes `q4_0_gemm_tc_kernel` (`gemm.rs`). `LB_NO_TC=1` forces the f32
    /// GEMM on such a card (the A/B switch).
    pub tensor_cores: bool,
    /// fp16 multiplies at twice the fp32 rate and there are no tensor cores (sm_60, the P100): the
    /// prefill GEMM takes `q4_0_gemm_h2_kernel`. `LB_NO_H2=1` forces the f32 tiles.
    pub fp16_pipe: bool,
    /// Recycled scratch buffers, keyed by byte size.
    free: Mutex<HashMap<usize, Vec<Handle>>>,
    /// Scratch handed out since the last sync. Reusing one before the GPU has run would
    /// corrupt it, so these only return to `free` after `read_all` has synchronised.
    in_flight: Mutex<Vec<(usize, Handle)>>,
    /// Page-locked staging buffers the repack writes into directly (cuda runtime only; see
    /// `pinned.rs` for the measurement behind it). `None` on wgpu or with `LB_NO_PINNED=1`.
    #[cfg(feature = "cuda")]
    pinned: Option<crate::pinned::PinnedPool>,
    /// CUDA device index (driver order) when the cuda runtime is open; what the host-side
    /// driver calls (page-locking) need.
    #[cfg(feature = "cuda")]
    cuda_index: Option<usize>,
    /// Page-locked file mappings (`register_mapping`): a slice inside one is handed to the
    /// driver as it is, no host copy, and repacked on the device.
    #[cfg(feature = "cuda")]
    registered: Mutex<Vec<std::sync::Arc<crate::registered::HostRegistration>>>,
    /// Repack on the host even when the bytes are registered (`LB_HOST_REPACK=1`, or
    /// `set_host_repack`): the A/B switch for the device repack.
    host_repack: AtomicBool,
    /// Serve a miss's three matrices with three device calls instead of twelve
    /// (`upload_expert_batched`); `LB_NO_BATCH_MISS=1` or `set_batch_miss(false)` is the A/B
    /// switch back to one `create`, two `empty`s and a repack launch per matrix.
    batch_miss: AtomicBool,
    /// Small constant buffers (`dims`, kernel counts) keyed by their words. Every matrix of a
    /// shape shares one, so a miss uploads no tiny buffers (three per expert before).
    consts: Mutex<HashMap<Vec<u32>, Handle>>,
    /// Output rows per plane for the Q4_0 kernels (D2): 4 (default since the 2026-11-07 P100
    /// measurement) or 8 = `q4_0_*_rows_kernel`, 1 = the one-row kernels. `LB_Q4_ROWS`; live
    /// through `set_q4_rows` for A/Bs.
    q4_rows: AtomicUsize,
    /// Stage the activation row in shared memory in the rows kernels (`LB_Q4_XSHARED=1`).
    q4_xshared: AtomicBool,
    /// ... only for matrices up to this many columns (`LB_Q4_XSHARED_MAX_COLS`, default 4096:
    /// 16 KB of shared memory; the 11776-wide dense down would take 46 KB of the 48).
    pub xshared_max_cols: usize,
    /// X7a: the math the Q4_0 rows kernels run in (`Q4_MATH_*`), per launch, live. Chosen at
    /// open by [`Q4Math::pick`] from the probed card and `LB_Q4_MATH`; `set_q4_math` switches it.
    q4_math: AtomicU32,
    /// CUDA compute capability `(major, minor)` of the card, when the cuda runtime is open.
    pub compute_capability: Option<(u32, u32)>,
}

/// `LB_Q4_ROWS`: 4 (default), 8, or 1 for the one-row kernels; anything else is the default.
/// Measured 2026-11-07 (docs/WORK_LOG.md): 1.29x end to end in the device decode on a P100,
/// 1.16x / 1.31x on the GTX 1070's distinct-expert stream (wgpu / cuda).
fn q4_rows_from_env() -> usize {
    match std::env::var("LB_Q4_ROWS").ok().and_then(|v| v.trim().parse::<usize>().ok()) {
        Some(r @ (1 | 4 | 8)) => r,
        _ => 4,
    }
}

/// The device to open. `LB_CUDA_DEVICE=N` opens CUDA device N through the cubecl-cuda runtime,
/// counting in the driver's order (nvidia-smi's, not wgpu's `discrete:N` order); it needs the
/// `cuda` feature and is checked first so a CUDA run cannot silently fall through to wgpu.
/// Otherwise `LB_WGPU_DEVICE` picks a wgpu adapter (see `lb_ternary_gpu::pick_device`).
pub fn pick_cube_device() -> Result<cubecl::Device, String> {
    if let Ok(spec) = std::env::var("LB_CUDA_DEVICE") {
        let n: usize = spec.trim().parse().map_err(|_| format!("LB_CUDA_DEVICE {spec:?}: index must be a number"))?;
        return cubecl::Device::cuda(n).map_err(|e| format!("CUDA device {n}: {e} (build lb-q4-gpu with --features cuda; libcuda + libnvrtc on the library path)"));
    }
    Ok(cubecl::Device::Wgpu(pick_device()?))
}

/// X7a: how a Q4_0 rows kernel does its arithmetic (see the notes at `Q4_MATH_F32` in lib.rs).
/// The weights on the device are the same whatever the math, so this is a launch-time choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Q4Math {
    /// Reference: int-to-float dequant, f32 FMA.
    F32,
    /// F32 with the dequant done by a mantissa bit trick; bit-identical to F32.
    F32Magic,
    /// half2 dequant + `__hfma2` per block, f32 across blocks (Pascal sm_60: fast fp16, no dp4a).
    F16,
    /// int8 activations + `dp4a` on the nibbles (sm_61+ / Turing: dp4a; fp16 is slow on sm_61).
    Dp4a,
}

impl Q4Math {
    pub const ALL: [Q4Math; 4] = [Q4Math::F32, Q4Math::F32Magic, Q4Math::F16, Q4Math::Dp4a];

    pub fn code(self) -> u32 {
        match self {
            Q4Math::F32 => crate::Q4_MATH_F32,
            Q4Math::F32Magic => crate::Q4_MATH_F32M,
            Q4Math::F16 => crate::Q4_MATH_F16,
            Q4Math::Dp4a => crate::Q4_MATH_DP4A,
        }
    }

    pub fn from_code(c: u32) -> Self {
        match c {
            crate::Q4_MATH_F32M => Q4Math::F32Magic,
            crate::Q4_MATH_F16 => Q4Math::F16,
            crate::Q4_MATH_DP4A => Q4Math::Dp4a,
            _ => Q4Math::F32,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Q4Math::F32 => "f32",
            Q4Math::F32Magic => "f32m",
            Q4Math::F16 => "f16",
            Q4Math::Dp4a => "dp4a",
        }
    }

    /// `f32`, `f32m`, `f16`, `dp4a`; `None` for anything else (including `auto`).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f32" => Some(Q4Math::F32),
            "f32m" | "magic" => Some(Q4Math::F32Magic),
            "f16" | "half" => Some(Q4Math::F16),
            "dp4a" | "int8" => Some(Q4Math::Dp4a),
            _ => None,
        }
    }

    /// Whether the path gives the same f32 bits as the reference (the greedy sha must hold).
    pub fn exact(self) -> bool {
        matches!(self, Q4Math::F32 | Q4Math::F32Magic)
    }

    /// The card's fast math from its compute capability: sm_60 (P100) has 2x-rate fp16 and no
    /// dp4a -> F16; sm_61 (1070, P106, P40) has dp4a and 1/64-rate fp16 -> Dp4a; sm_62 (Jetson
    /// TX2) fast fp16, no dp4a -> F16; sm_70+ (Turing, Ampere, ...) dp4a -> Dp4a. Unknown (wgpu,
    /// no capability) -> F32.
    pub fn for_card(cc: Option<(u32, u32)>) -> Self {
        match cc {
            Some((6, 0)) | Some((6, 2)) => Q4Math::F16,
            Some((6, _)) => Q4Math::Dp4a,
            Some((m, _)) if m >= 7 => Q4Math::Dp4a,
            _ => Q4Math::F32,
        }
    }

    /// `LB_Q4_MATH`: unset or `f32` -> the reference (the default: every published sha was made
    /// with it); `auto` -> [`Q4Math::for_card`]; or `f32m`, `f16`, `dp4a` by name.
    pub fn pick(cc: Option<(u32, u32)>) -> Self {
        match std::env::var("LB_Q4_MATH") {
            Ok(v) if v.trim().eq_ignore_ascii_case("auto") => Self::for_card(cc),
            Ok(v) => Self::parse(&v).unwrap_or_else(|| {
                eprintln!("lb-q4-gpu: LB_Q4_MATH={v:?} not understood (f32|f32m|f16|dp4a|auto), using f32");
                Q4Math::F32
            }),
            Err(_) => Q4Math::F32,
        }
    }
}

/// Compute capability of CUDA device `index` (driver order), from the driver.
#[cfg(feature = "cuda")]
fn cuda_compute_capability(index: usize) -> Option<(u32, u32)> {
    use cudarc::driver::{result, sys::CUdevice_attribute::*};
    // SAFETY: plain read-only driver queries on a valid device ordinal.
    unsafe {
        result::init().ok()?;
        let dev = result::device::get(index as i32).ok()?;
        let major = result::device::get_attribute(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).ok()?;
        let minor = result::device::get_attribute(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR).ok()?;
        Some((major as u32, minor as u32))
    }
}

impl Gpu {
    /// Open the device named by `LB_CUDA_DEVICE` or `LB_WGPU_DEVICE` (see `pick_cube_device`).
    pub fn open() -> Result<Self, String> {
        Self::open_device(pick_cube_device()?)
    }

    /// Open CUDA device `index` (driver order inside `CUDA_VISIBLE_DEVICES`), whatever
    /// `LB_CUDA_DEVICE` says: a second card in the same process (work-to-weights, in-box
    /// expert cards).
    #[cfg(feature = "cuda")]
    pub fn open_cuda(index: usize) -> Result<Self, String> {
        let dev = cubecl::Device::cuda(index).map_err(|e| format!("CUDA device {index}: {e}"))?;
        Self::open_device(dev)
    }

    fn open_device(dev: cubecl::Device) -> Result<Self, String> {
        let client = std::panic::catch_unwind(|| dev.client()).map_err(|_| format!("{dev:?}: no client (adapter missing, or its runtime failed to initialise)"))?;
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
        #[cfg(feature = "cuda")]
        let pinned = match (&dev, std::env::var("LB_NO_PINNED").is_err()) {
            (cubecl::Device::Cuda(d), true) => match crate::pinned::PinnedPool::open(d.index) {
                Ok(p) => Some(p),
                Err(e) => {
                    eprintln!("lb-q4-gpu: pinned staging pool unavailable, uploads stay pageable: {e}");
                    None
                }
            },
            _ => None,
        };
        #[cfg(feature = "cuda")]
        let compute_capability = match &dev {
            cubecl::Device::Cuda(d) => cuda_compute_capability(d.index),
            _ => None,
        };
        #[cfg(not(feature = "cuda"))]
        let compute_capability: Option<(u32, u32)> = None;
        let q4_math = Q4Math::pick(compute_capability);
        let tensor_cores = {
            use cubecl::ir::{ElemType, FloatKind};
            let f16 = ElemType::Float(FloatKind::F16);
            let want = cubecl::ir::features::MmaConfig { a_type: f16, b_type: f16, cd_type: ElemType::Float(FloatKind::F32), m: 16, n: 16, k: 16 };
            props.features.matmul.cmma.contains(&want) && pmax == 32 && pmin == 32 && std::env::var("LB_NO_TC").map(|v| v.is_empty() || v == "0").unwrap_or(true)
        };
        let fp16_pipe = !tensor_cores && compute_capability == Some((6, 0)) && std::env::var("LB_NO_H2").map(|v| v.is_empty() || v == "0").unwrap_or(true);
        Ok(Self {
            fp16_pipe,
            tensor_cores,
            client,
            name: format!("{dev:?}"),
            cube_dim,
            use_plane,
            plane_size: (pmin, pmax),
            free: Mutex::new(HashMap::new()),
            in_flight: Mutex::new(Vec::new()),
            #[cfg(feature = "cuda")]
            pinned,
            #[cfg(feature = "cuda")]
            cuda_index: match &dev {
                cubecl::Device::Cuda(d) => Some(d.index),
                _ => None,
            },
            #[cfg(feature = "cuda")]
            registered: Mutex::new(Vec::new()),
            host_repack: AtomicBool::new(std::env::var_os("LB_HOST_REPACK").is_some()),
            batch_miss: AtomicBool::new(std::env::var_os("LB_NO_BATCH_MISS").is_none()),
            consts: Mutex::new(HashMap::new()),
            q4_rows: AtomicUsize::new(q4_rows_from_env()),
            q4_xshared: AtomicBool::new(std::env::var("LB_Q4_XSHARED").map(|v| v == "1").unwrap_or(false)),
            xshared_max_cols: std::env::var("LB_Q4_XSHARED_MAX_COLS").ok().and_then(|v| v.parse().ok()).unwrap_or(4096),
            q4_math: AtomicU32::new(q4_math.code()),
            compute_capability,
        })
    }

    /// The math code the Q4_0 rows kernels are launched with (`Q4_MATH_*`; see [`Q4Math`]).
    /// Only the rows kernels (`LB_Q4_ROWS` 4 or 8, plane ops) have the variants; the one-row
    /// kernels are always f32.
    pub fn q4_math(&self) -> u32 {
        self.q4_math.load(Ordering::Relaxed)
    }

    /// The math as an enum.
    pub fn q4_math_kind(&self) -> Q4Math {
        Q4Math::from_code(self.q4_math())
    }

    /// Switch the Q4_0 rows kernels' math, live (the next launch uses it; queued work is not
    /// affected). The device bytes are the same for every math, so nothing is re-uploaded.
    pub fn set_q4_math(&self, m: Q4Math) {
        self.q4_math.store(m.code(), Ordering::Relaxed);
    }

    /// Rows per plane the Q4_0 kernels use (1 = the one-row kernels).
    pub fn q4_rows(&self) -> usize {
        self.q4_rows.load(Ordering::Relaxed)
    }

    /// Switch the Q4_0 kernels' rows per plane: 4 or 8 take the rows kernels, anything else the
    /// one-row ones. Live (an A/B in one process); a kernel already queued is unaffected.
    pub fn set_q4_rows(&self, r: usize) {
        self.q4_rows.store(if r == 4 || r == 8 { r } else { 1 }, Ordering::Relaxed);
    }

    pub fn q4_xshared(&self) -> bool {
        self.q4_xshared.load(Ordering::Relaxed)
    }

    pub fn set_q4_xshared(&self, on: bool) {
        self.q4_xshared.store(on, Ordering::Relaxed);
    }

    /// Rows per plane for a matrix of `rows` rows, or `None` for the one-row kernel (knob at 1,
    /// no plane ops, or rows not a multiple).
    fn rows_for(&self, rows: usize) -> Option<usize> {
        let r = self.q4_rows();
        (self.use_plane && (r == 4 || r == 8) && rows % r == 0).then_some(r)
    }

    /// Whether the rows kernel stages `x` in shared memory: one input row, narrow enough.
    fn xshared_for(&self, n_x_rows: usize, cols: usize) -> bool {
        self.q4_xshared() && n_x_rows == 1 && cols <= self.xshared_max_cols
    }

    /// Whether a miss's three matrices go up as one batch (see `batch_miss`).
    pub fn batch_miss(&self) -> bool {
        self.batch_miss.load(Ordering::Relaxed)
    }

    /// Switch the batched miss path on or off (A/B switch; live).
    pub fn set_batch_miss(&self, on: bool) {
        self.batch_miss.store(on, Ordering::Relaxed);
    }

    /// Whether registered bytes are repacked on the host (see `host_repack`).
    pub fn host_repack(&self) -> bool {
        self.host_repack.load(Ordering::Relaxed)
    }

    /// Three Q4_0 matrices of one expert in **three** device calls: one `create_tensors` DMAs
    /// the three raw slices out of the page-locked mapping into one allocation, one
    /// `empty_tensors` reserves the packed words of all three, and one repack launch covers the
    /// lot (three launches if the allocator did not place the raw slices back to back). Each
    /// matrix then owns an offset view of the shared `qs`/`sc` buffers, which the kernels read
    /// exactly as they read a buffer of their own.
    ///
    /// Why: a miss paid twelve calls (`create`, two `empty`s and a launch per matrix), and the
    /// three `create`s alone measured 0.59 ms for 5.3 MB on c2 where the bus time is 0.47 ms
    /// (`docs/WORK_LOG.md`, 2026-11-03). `None` when the fast path does not apply (not all
    /// registered, odd or unequal block counts, host repack or batching switched off): the
    /// caller takes the per-matrix path.
    #[cfg(feature = "cuda")]
    fn upload_expert_batched(&self, mats: [&QuantMat<'_>; 3]) -> Option<[GpuQ4Mat; 3]> {
        use cubecl::server::MemoryLayoutDescriptor;
        if !self.batch_miss() || self.host_repack() {
            return None;
        }
        let nb = mats[0].rows * (mats[0].cols / 32);
        if nb == 0 || nb % 2 != 0 {
            return None;
        }
        for m in mats {
            if m.ty != lb_gguf::GgmlType::Q4_0 || m.cols % 32 != 0 || m.rows * (m.cols / 32) != nb {
                return None;
            }
        }
        let raw_len = nb * 18;
        let mut descs = Vec::with_capacity(3);
        for m in mats {
            let raw = self.registered_slice(&m.bytes[..raw_len])?;
            descs.push((MemoryLayoutDescriptor::contiguous([raw_len].into(), 1), raw));
        }
        let (n_qs, n_sc) = (nb * 4, nb / 2);
        // one submission, one allocation, three DMAs
        let raws = self.client.create_tensors(descs);
        // one submission, one allocation, for the six packed outputs
        let outs = self.client.empty_tensors(vec![
            MemoryLayoutDescriptor::contiguous([3 * n_qs * 4].into(), 1),
            MemoryLayoutDescriptor::contiguous([3 * n_sc * 4].into(), 1),
        ]);
        let (qs_all, sc_all) = (outs[0].memory.clone(), outs[1].memory.clone());
        let n = self.const_u32(&[(3 * n_sc) as u32]);
        let off = |h: &Handle| h.offset_start.unwrap_or(0);
        let back_to_back = (1..3).all(|i| off(&raws[i].memory) == off(&raws[0].memory) + (i * raw_len) as u64);
        if back_to_back {
            // one view over the three raw slices: start at the first, end where the last ends
            let mut span = raws[0].memory.clone();
            span.offset_end = raws[2].memory.offset_end;
            unsafe {
                q4_0_repack_kernel::launch(
                    &self.client,
                    cube_count(3 * n_sc, REPACK_CUBE),
                    CubeDim::new_1d(REPACK_CUBE as u32),
                    BufferArg::from_raw_parts(span, 3 * raw_len / 4),
                    BufferArg::from_raw_parts(qs_all.clone(), 3 * n_qs),
                    BufferArg::from_raw_parts(sc_all.clone(), 3 * n_sc),
                    BufferArg::from_raw_parts(n, 1),
                );
            }
        } else {
            let n1 = self.const_u32(&[n_sc as u32]);
            for i in 0..3 {
                let qs_i = qs_all.clone().offset_start((i * n_qs * 4) as u64).offset_end(((2 - i) * n_qs * 4) as u64);
                let sc_i = sc_all.clone().offset_start((i * n_sc * 4) as u64).offset_end(((2 - i) * n_sc * 4) as u64);
                unsafe {
                    q4_0_repack_kernel::launch(
                        &self.client,
                        cube_count(n_sc, REPACK_CUBE),
                        CubeDim::new_1d(REPACK_CUBE as u32),
                        BufferArg::from_raw_parts(raws[i].memory.clone(), raw_len / 4),
                        BufferArg::from_raw_parts(qs_i, n_qs),
                        BufferArg::from_raw_parts(sc_i, n_sc),
                        BufferArg::from_raw_parts(n1.clone(), 1),
                    );
                }
            }
        }
        let mut out = Vec::with_capacity(3);
        for (i, m) in mats.iter().enumerate() {
            let qs = qs_all.clone().offset_start((i * n_qs * 4) as u64).offset_end(((2 - i) * n_qs * 4) as u64);
            let sc = sc_all.clone().offset_start((i * n_sc * 4) as u64).offset_end(((2 - i) * n_sc * 4) as u64);
            let dims = matvec_dims(m.rows, m.cols);
            out.push(GpuQ4Mat { rows: m.rows, cols: m.cols, row_bytes: m.row_bytes(), n_qs, n_sc, qs, sc, dims: self.const_u32(&dims) });
        }
        // `raws` drops here: the raw allocation is released once the stream is past the repack
        let (d, u, g) = (out.pop()?, out.pop()?, out.pop()?);
        Some([g, u, d])
    }

    /// X7b on the device: IQ4_XS bytes (`raw`, whole super-blocks, already on the device) ->
    /// Q4_0 words in the device layout, encoded like the host transcode (plain by default,
    /// the scale search with `LB_TRANSCODE_SEARCH=1`). Returns
    /// `((qs, n_qs), (sc, n_sc))` as `upload_q4_0` does. `n_super` super-blocks = 8 * n_super
    /// Q4_0 blocks.
    pub fn transcode_iq4_xs_on_device(&self, raw: &Handle, n_super: usize) -> ((Handle, usize), (Handle, usize)) {
        self.transcode_iq4_xs_on_device_with(raw, n_super, crate::transcode::search_enabled())
    }

    /// As [`Self::transcode_iq4_xs_on_device`], with the encoder chosen explicitly (`search`).
    pub fn transcode_iq4_xs_on_device_with(&self, raw: &Handle, n_super: usize, search: bool) -> ((Handle, usize), (Handle, usize)) {
        let n_blocks = 8 * n_super;
        let (n_qs, n_sc) = (n_blocks * 4, n_blocks / 2);
        let qs = self.client.empty(n_qs * 4);
        let sc = self.client.empty(n_sc * 4);
        let levels: Vec<f32> = lb_gguf::types::IQ4_LEVELS.iter().map(|&l| l as f32).collect();
        let lv = self.upload_f32(&levels);
        let n = self.const_u32(&[n_sc as u32]);
        unsafe {
            crate::iq4_xs_to_q4_0_kernel::launch(
                &self.client,
                cube_count(n_sc, crate::TRANSCODE_CUBE),
                CubeDim::new_1d(crate::TRANSCODE_CUBE as u32),
                BufferArg::from_raw_parts(raw.clone(), n_super * 34),
                BufferArg::from_raw_parts(lv, 16),
                BufferArg::from_raw_parts(qs.clone(), n_qs),
                BufferArg::from_raw_parts(sc.clone(), n_sc),
                BufferArg::from_raw_parts(n, 1),
                search,
            );
        }
        ((qs, n_qs), (sc, n_sc))
    }

    /// A device buffer holding exactly `words`, shared by every caller asking for the same
    /// words (kernel `dims` and counts). Never freed: there are a few dozen distinct ones.
    pub fn const_u32(&self, words: &[u32]) -> Handle {
        let mut c = self.consts.lock().expect("consts");
        if let Some(h) = c.get(words) {
            return h.clone();
        }
        let h = self.upload_u32(words.to_vec());
        c.insert(words.to_vec(), h.clone());
        h
    }

    /// Repack Q4_0 on the host even for registered bytes (A/B switch; see `host_repack`).
    pub fn set_host_repack(&self, on: bool) {
        self.host_repack.store(on, Ordering::Relaxed);
    }

    /// Page-lock a read-only file mapping with the driver (`cuMemHostRegister`, read-only), so
    /// that every later `upload_q4_0` of a slice inside it DMAs the raw bytes straight out of
    /// the page cache and repacks on the device: no host copy, no host repack. Returns the
    /// bytes locked. Measured on c2 GPU2 (P100, driver 580): 48 ms per GB to register, then
    /// 11.2 GB/s per 1.77 MB slice against 4.6-4.9 GB/s from the unregistered mapping.
    ///
    /// # Safety
    /// `map` must stay mapped and unchanged for as long as this `Gpu` lives: uploads keep a
    /// raw pointer into it and the driver reads from it asynchronously. A `memmap2::Mmap` the
    /// caller holds for the model's lifetime satisfies this.
    #[cfg(feature = "cuda")]
    pub unsafe fn register_mapping(&self, map: &[u8]) -> Result<usize, String> {
        let dev = self.cuda_index.ok_or("register_mapping: not a CUDA device")?;
        // SAFETY: forwarded to the caller's contract.
        let reg = unsafe { crate::registered::HostRegistration::new(dev, map)? };
        let n = reg.len();
        self.registered.lock().expect("registered").push(std::sync::Arc::new(reg));
        Ok(n)
    }

    /// Bytes page-locked by `register_mapping` so far (0 without the cuda runtime).
    pub fn registered_bytes(&self) -> usize {
        #[cfg(feature = "cuda")]
        {
            self.registered.lock().expect("registered").iter().map(|r| r.len()).sum()
        }
        #[cfg(not(feature = "cuda"))]
        {
            0
        }
    }

    /// `bytes` as driver-ready `Bytes` when they lie inside a registered mapping.
    #[cfg(feature = "cuda")]
    fn registered_slice(&self, bytes: &[u8]) -> Option<Bytes> {
        let regs = self.registered.lock().expect("registered");
        let reg = regs.iter().find(|r| r.contains(bytes))?;
        Some(crate::registered::RegisteredSlice::bytes(reg.clone(), bytes))
    }

    /// The registration holding `bytes`, for benches that build the raw `Bytes` themselves.
    #[cfg(feature = "cuda")]
    pub fn registration_of(&self, bytes: &[u8]) -> Option<std::sync::Arc<crate::registered::HostRegistration>> {
        self.registered.lock().expect("registered").iter().find(|r| r.contains(bytes)).cloned()
    }

    /// Whether `upload_q4_0` of these bytes would take the zero-copy device-repack path.
    pub fn is_registered(&self, bytes: &[u8]) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.registered.lock().expect("registered").iter().any(|r| r.contains(bytes))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = bytes;
            false
        }
    }

    /// Outputs one cube produces with the current cube size and reduction.
    pub fn per_cube(&self) -> usize {
        if self.use_plane { self.cube_dim / (self.plane_size.1 as usize).max(1) } else { 1 }
    }

    pub fn upload_f32(&self, x: &[f32]) -> Handle {
        self.client.create(Bytes::from_elems(x.to_vec()))
    }

    /// Overwrite a device buffer with `x` in place (stream-ordered after what is queued).
    pub fn write_f32(&self, h: &Handle, x: &[f32]) {
        self.client.write(h, Bytes::from_elems(x.to_vec()));
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

    /// Whether uploads can be written straight into page-locked memory (`upload_u32_with`,
    /// `upload_q4_0`); false on wgpu, with `LB_NO_PINNED=1`, or when the pool failed to open.
    pub fn has_pinned(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.pinned.is_some()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    /// Bytes the pinned pool has page-locked so far (0 without one). A pool at its cap hands
    /// out nothing and uploads silently take the pageable path, so a bench prints this.
    pub fn pinned_allocated(&self) -> usize {
        #[cfg(feature = "cuda")]
        {
            self.pinned.as_ref().map(|p| p.allocated()).unwrap_or(0)
        }
        #[cfg(not(feature = "cuda"))]
        {
            0
        }
    }

    /// A page-locked buffer for `n` words from the pool (cuda runtime), or `None`: lets a
    /// caller time the fill and the upload apart (`upload_pinned`). `upload_u32_with` is the
    /// one-call form.
    #[cfg(feature = "cuda")]
    pub fn take_pinned(&self, n: usize) -> Option<crate::pinned::PinnedBuf> {
        self.pinned.as_ref().and_then(|p| p.take(n * 4))
    }

    /// Upload a buffer from `take_pinned`: the DMA runs straight from it.
    #[cfg(feature = "cuda")]
    pub fn upload_pinned(&self, buf: crate::pinned::PinnedBuf) -> Handle {
        self.client.create(buf.into_bytes())
    }

    /// `n` words produced by `fill`, uploaded from page-locked memory when the pool has it
    /// (no host copy between `fill` and the DMA), else from a fresh `Vec` as `upload_u32`.
    /// `fill` must assign every word: the buffer it gets may hold a previous upload.
    pub fn upload_u32_with(&self, n: usize, fill: impl FnOnce(&mut [u32])) -> Handle {
        #[cfg(feature = "cuda")]
        if let Some(mut buf) = self.pinned.as_ref().and_then(|p| p.take(n * 4)) {
            fill(buf.as_mut_u32());
            return self.client.create(buf.into_bytes());
        }
        let mut v = vec![0u32; n];
        fill(&mut v);
        self.upload_u32(v)
    }

    /// Repack `n_blocks` Q4_0 blocks and upload the device layout: `(qs, sc)` handles with
    /// their word counts. The repack writes into page-locked memory when the pool has it, so
    /// the only copy is the DMA; otherwise into `Vec`s, as before.
    pub fn upload_q4_0(&self, bytes: &[u8], n_blocks: usize) -> ((Handle, usize), (Handle, usize)) {
        let (n_qs, n_sc) = (n_blocks * 4, n_blocks.div_ceil(2).max(1));
        // zero-copy: raw bytes from the page-locked mapping, repacked by the device. An even
        // block count keeps the raw buffer 4-byte aligned (the kernel reads word pairs).
        #[cfg(feature = "cuda")]
        if n_blocks > 0 && n_blocks % 2 == 0 && !self.host_repack.load(Ordering::Relaxed) {
            if let Some(raw) = self.registered_slice(&bytes[..n_blocks * 18]) {
                let raw = self.client.create(raw);
                return self.repack_on_device(raw, n_blocks);
            }
        }
        #[cfg(feature = "cuda")]
        if let Some(pool) = self.pinned.as_ref() {
            if let (Some(mut qs), Some(mut sc)) = (pool.take(n_qs * 4), pool.take(n_sc * 4)) {
                crate::repack_q4_0_into(bytes, n_blocks, qs.as_mut_u32(), sc.as_mut_u32());
                return ((self.client.create(qs.into_bytes()), n_qs), (self.client.create(sc.into_bytes()), n_sc));
            }
        }
        let (qs, sc) = crate::repack_q4_0(bytes, n_blocks);
        ((self.upload_u32(qs), n_qs), (self.upload_u32(sc), n_sc))
    }

    /// Repack `n_blocks` (even) raw Q4_0 blocks already on the device (`raw`: their
    /// `n_blocks * 18` file bytes) into fresh `qs`/`sc` buffers with [`q4_0_repack_kernel`].
    /// Any runtime. The raw buffer is released when `raw` drops.
    pub fn repack_on_device(&self, raw: Handle, n_blocks: usize) -> ((Handle, usize), (Handle, usize)) {
        assert!(n_blocks > 0 && n_blocks % 2 == 0, "device repack needs an even block count, got {n_blocks}");
        let (n_qs, n_sc) = (n_blocks * 4, n_blocks / 2);
        let qs = self.client.empty(n_qs * 4);
        let sc = self.client.empty(n_sc * 4);
        let n = self.const_u32(&[n_sc as u32]);
        unsafe {
            q4_0_repack_kernel::launch(
                &self.client,
                cube_count(n_sc, REPACK_CUBE),
                CubeDim::new_1d(REPACK_CUBE as u32),
                BufferArg::from_raw_parts(raw, n_blocks * 18 / 4),
                BufferArg::from_raw_parts(qs.clone(), n_qs),
                BufferArg::from_raw_parts(sc.clone(), n_sc),
                BufferArg::from_raw_parts(n, 1),
            );
        }
        ((qs, n_qs), (sc, n_sc))
    }

    /// Upload the raw file bytes of `n_blocks` (even) Q4_0 blocks through the ordinary path
    /// and repack on the device: the any-runtime form of the zero-copy path, for tests and
    /// for sources that are not a registered mapping.
    pub fn upload_q4_0_raw(&self, bytes: &[u8], n_blocks: usize) -> ((Handle, usize), (Handle, usize)) {
        let words: Vec<u32> = bytes[..n_blocks * 18].chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let raw = self.upload_u32(words);
        self.repack_on_device(raw, n_blocks)
    }

    /// Read `n` u32 words of a buffer back (synchronises).
    pub fn read_u32(&self, h: &Handle, n: usize) -> Result<Vec<u32>, String> {
        let bytes = self.client.read(vec![h.clone()]).map_err(|e| format!("GPU read failed: {e:?}"))?;
        Ok(u32::from_bytes(&bytes[0])[..n].to_vec())
    }

    /// Copy a device buffer of f32 to another `Gpu` of the same runtime, **device to device and
    /// stream ordered on both sides** when the runtime has a transport between its devices
    /// (cubecl-cuda: NCCL send/recv on the communication streams, the receiving stream waits on
    /// the device; no host sync). Without one cubecl copies through the host (synchronously):
    /// [`Gpu::device_transport`] says which.
    pub fn copy_to(&self, h: &Handle, dst: &Gpu) -> Handle {
        let mut c = self.client.clone();
        c.to_client(h.clone(), &dst.client, cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F32))
    }

    /// Whether [`Gpu::copy_to`] goes device to device (cubecl-cuda with NCCL loadable).
    pub fn device_transport(&self) -> bool {
        self.client.has_device_transport()
    }

    /// Wait for this device and recycle the scratch handed out since its last read: the end of
    /// a token for a card that never reads in the middle of one (in-box expert cards).
    pub fn sync_recycle(&self) -> Result<(), String> {
        self.sync()?;
        self.recycle();
        Ok(())
    }

    /// `x` (`n` f32) as a new f16 scratch buffer (`n` halves), for the tensor-core GEMM's input.
    pub fn to_f16(&self, x: &Handle, n: usize) -> Handle {
        let y = self.scratch(n.div_ceil(2) * 4);
        unsafe {
            crate::gemm::f32_to_f16_kernel::launch(
                &self.client,
                cube_count(n, 256),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(x.clone(), n),
                BufferArg::from_raw_parts(y.clone(), n),
                BufferArg::from_raw_parts(self.upload_u32(vec![n as u32]), 1),
            );
        }
        y
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

    /// A second device stream for work that must not queue behind the caller's (the expert
    /// staging ring's prefetch copies, `lb-lfm2` `stage.rs`). CubeCL 0.11 gives every OS thread
    /// its own stream and orders work across streams itself: a kernel on the main stream that
    /// binds a buffer written on this one waits for it on the device (`cuStreamWaitEvent`), so
    /// the host never blocks on a copy it issued here, and a read on the main stream does not
    /// wait for this stream's queue. See [`Gpu::on_stream`].
    pub fn side_stream(&self) -> SideStream {
        SideStream(cubecl_environment::stream::StreamId::allocate())
    }

    /// Run `f` with every device call it makes submitted on `s` instead of the calling
    /// thread's stream (restored afterwards, also on unwind).
    pub fn on_stream<T>(&self, s: &SideStream, f: impl FnOnce() -> T) -> T {
        s.0.executes(f)
    }
}

/// A device stream other than the calling thread's (see [`Gpu::side_stream`]).
#[derive(Clone, Copy, Debug)]
pub struct SideStream(cubecl_environment::stream::StreamId);

/// A Q8_0 matrix resident on the device (X7b follow-up; see `repack_q8_0`). Rows kernel only
/// (4 rows a plane, f32 math, plane ops needed). Rows are padded to a multiple of 4 with zero
/// blocks: `launch` writes `[m, rows_pad]`, `matvec` returns `[m, rows]`.
pub struct GpuQ8Mat {
    pub rows: usize,
    pub rows_pad: usize,
    pub cols: usize,
    qs: Handle,
    sc: Handle,
    n_qs: usize,
    n_sc: usize,
    dims: Handle,
}

impl GpuQ8Mat {
    /// Upload `w` as Q8_0: a Q8_0 matrix as it is, anything else lb-gguf can dequantize re-encoded
    /// to Q8_0 (`lb_gguf::types::quantize_q8_0`, the quality-preserving transcode target).
    pub fn upload(gpu: &Gpu, w: QuantMat<'_>) -> Result<Self, String> {
        if w.cols % 32 != 0 || !gpu.use_plane {
            return Err(format!("GpuQ8Mat needs cols % 32 == 0 and plane ops ({}x{})", w.rows, w.cols));
        }
        let rows_pad = w.rows.div_ceil(4) * 4;
        let mut bytes: Vec<u8> = if w.ty == lb_gguf::GgmlType::Q8_0 {
            w.bytes[..w.rows * w.cols / 32 * 34].to_vec()
        } else {
            let mut out = Vec::with_capacity(rows_pad * w.cols / 32 * 34);
            let mut buf = vec![0f32; w.cols];
            for r in 0..w.rows {
                w.dequant_row(r, &mut buf);
                out.extend_from_slice(&lb_gguf::types::quantize_q8_0(&buf));
            }
            out
        };
        bytes.resize(rows_pad * w.cols / 32 * 34, 0);
        let n_blocks = rows_pad * w.cols / 32;
        let (qs, sc) = crate::repack_q8_0(&bytes, n_blocks);
        let (n_qs, n_sc) = (qs.len(), sc.len());
        Ok(Self { rows: w.rows, rows_pad, cols: w.cols, qs: gpu.upload_u32(qs), sc: gpu.upload_u32(sc), n_qs, n_sc, dims: gpu.const_u32(&matvec_dims(rows_pad, w.cols)) })
    }

    pub fn bytes_on_device(&self) -> usize {
        (self.n_qs + self.n_sc) * 4
    }

    /// `out = x @ W^T` for `m` rows of `x`; the output is `[m, rows_pad]`.
    pub fn launch(&self, gpu: &Gpu, x: &Handle, m: usize) -> Handle {
        let n_out = m * self.rows_pad;
        let out = gpu.scratch(n_out * 4);
        unsafe {
            crate::q8_0_matvec_rows_kernel::launch(
                &gpu.client,
                cube_count(n_out / 4, gpu.per_cube()),
                CubeDim::new_1d(gpu.cube_dim as u32),
                BufferArg::from_raw_parts(self.qs.clone(), self.n_qs / 4),
                BufferArg::from_raw_parts(self.sc.clone(), self.n_sc),
                BufferArg::from_raw_parts(x.clone(), m * self.cols / 4),
                BufferArg::from_raw_parts(self.dims.clone(), 2),
                BufferArg::from_raw_parts(out.clone(), n_out),
                4usize,
            );
        }
        out
    }

    pub fn matvec(&self, gpu: &Gpu, x: &[f32], m: usize) -> Result<Vec<f32>, String> {
        let xh = gpu.upload_f32(x);
        let out = self.launch(gpu, &xh, m);
        let padded = gpu.read_all(vec![(out, m * self.rows_pad)])?.remove(0);
        Ok(padded.chunks_exact(self.rows_pad).flat_map(|r| r[..self.rows].iter().copied()).collect())
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
    /// Upload `w`; a non-Q4_0 matrix is transcoded to Q4_0 first (X7b, `transcode.rs`).
    pub fn upload(gpu: &Gpu, w: QuantMat<'_>) -> Result<Self, String> {
        if w.cols % 32 != 0 {
            return Err(format!("Q4_0 needs cols % 32 == 0, got {}", w.cols));
        }
        let bytes = crate::transcode::q4_0_bytes(&w)?;
        let n_blocks = w.rows * (w.cols / 32);
        let ((qs, n_qs), (sc, n_sc)) = gpu.upload_q4_0(&bytes, n_blocks);
        let dims = matvec_dims(w.rows, w.cols);
        Ok(Self { rows: w.rows, cols: w.cols, row_bytes: w.cols / 32 * 18, n_qs, n_sc, qs, sc, dims: gpu.const_u32(&dims) })
    }

    pub fn bytes_on_device(&self) -> usize {
        (self.n_qs + self.n_sc) * 4
    }

    /// The packed words as the device holds them, `(qs, sc)`; synchronises. For parity checks
    /// between upload paths (the batched expert path hands out offset views of shared buffers).
    pub fn words(&self, gpu: &Gpu) -> Result<(Vec<u32>, Vec<u32>), String> {
        Ok((gpu.read_u32(&self.qs, self.n_qs)?, gpu.read_u32(&self.sc, self.n_sc)?))
    }

    /// Launch `out = x @ W^T` for `m` rows of `x` already on the device; returns the output handle.
    pub fn launch(&self, gpu: &Gpu, x: &Handle, m: usize) -> Handle {
        let n_out = m * self.rows;
        let out = gpu.scratch(n_out * 4);
        if let Some(r) = gpu.rows_for(self.rows) {
            let xsh = gpu.xshared_for(m, self.cols);
            unsafe {
                q4_0_matvec_rows_kernel::launch(
                    &gpu.client,
                    cube_count(n_out / r, gpu.per_cube()),
                    CubeDim::new_1d(gpu.cube_dim as u32),
                    BufferArg::from_raw_parts(self.qs.clone(), self.n_qs / 4),
                    BufferArg::from_raw_parts(self.sc.clone(), self.n_sc),
                    BufferArg::from_raw_parts(x.clone(), m * self.cols / 4),
                    // not routed: `routes` is never read, any real buffer will do
                    BufferArg::from_raw_parts(self.dims.clone(), 2),
                    BufferArg::from_raw_parts(self.dims.clone(), 2),
                    BufferArg::from_raw_parts(out.clone(), n_out),
                    r,
                    xsh,
                    if xsh { self.cols / 4 } else { 1 },
                    false,
                    gpu.q4_math(),
                );
            }
            return out;
        }
        unsafe {
            // `from_raw_parts` counts ELEMENTS of the kernel's slice type, not u32 words:
            // `qs`/`qv` are `&[V4U]` and `x` is `&[V4F]`, so the word counts divide by 4. Both
            // divide exactly by construction (a Q4_0 block repacks to 4 u32 words, and `cols` is
            // a multiple of 32) -- assert it so a future shape change fails here instead of
            // silently handing the kernel a length 4x too long.
            debug_assert_eq!(self.n_qs % 4, 0);
            debug_assert_eq!((m * self.cols) % 4, 0);
            q4_0_matvec_kernel::launch(
                &gpu.client,
                cube_count(n_out, gpu.per_cube()),
                CubeDim::new_1d(gpu.cube_dim as u32),
                BufferArg::from_raw_parts(self.qs.clone(), self.n_qs / 4),
                BufferArg::from_raw_parts(self.sc.clone(), self.n_sc),
                BufferArg::from_raw_parts(x.clone(), m * self.cols / 4),
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

    /// Tiled GEMM (`gemm.rs`) for `m` rows of `x`: each weight tile read once per 64 rows.
    /// The prefill path's choice for large `m`; same result as [`GpuQ4Mat::launch`] to rounding.
    pub fn launch_gemm(&self, gpu: &Gpu, x: &Handle, m: usize) -> Handle {
        self.gemm_with(gpu, None, x, m)
    }

    /// `silu(x @ gate^T) * (x @ up^T)` through the tiled GEMM, `[m, rows]`.
    pub fn launch_gemm_gate_up(gpu: &Gpu, gate: &GpuQ4Mat, up: &GpuQ4Mat, x: &Handle, m: usize) -> Handle {
        assert!(gate.rows == up.rows && gate.cols == up.cols, "gate/up shape mismatch");
        gate.gemm_with(gpu, Some(up), x, m)
    }

    fn gemm_with(&self, gpu: &Gpu, up: Option<&GpuQ4Mat>, x: &Handle, m: usize) -> Handle {
        let n_out = m * self.rows;
        let out = gpu.scratch(n_out * 4);
        let dims = gpu.upload_u32(vec![self.rows as u32, self.cols as u32, m as u32, 0]);
        let (uq, us, nuq, nus) = match up {
            Some(u) => (u.qs.clone(), u.sc.clone(), u.n_qs, u.n_sc),
            None => (self.qs.clone(), self.sc.clone(), self.n_qs, self.n_sc),
        };
        unsafe {
            gemm_launch!(
                gpu,
                CubeCount::Static(self.rows.div_ceil(crate::gemm::TILE) as u32, m.div_ceil(crate::gemm::TILE) as u32, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(self.qs.clone(), self.n_qs),
                BufferArg::from_raw_parts(self.sc.clone(), self.n_sc),
                BufferArg::from_raw_parts(uq, nuq),
                BufferArg::from_raw_parts(us, nus),
                (x, m * self.cols),
                // dense: routes / perm / tiles are never read; bind any real buffer
                BufferArg::from_raw_parts(dims.clone(), 4),
                BufferArg::from_raw_parts(dims.clone(), 4),
                BufferArg::from_raw_parts(dims.clone(), 4),
                BufferArg::from_raw_parts(dims, 4),
                BufferArg::from_raw_parts(out.clone(), n_out),
                false,
                up.is_some(),
                false,
            );
        }
        out
    }

    /// `silu(x @ gate^T) * (x @ up^T)` in one dispatch (`q4_0_gate_up_silu_kernel`), for a dense
    /// SwiGLU whose gate and up are separate resident matrices of one shape. Returns `[m, rows]`.
    pub fn launch_gate_up_silu(gpu: &Gpu, gate: &GpuQ4Mat, up: &GpuQ4Mat, x: &Handle, m: usize) -> Handle {
        assert!(gate.rows == up.rows && gate.cols == up.cols, "gate/up shape mismatch");
        let n_h = m * gate.rows;
        let h = gpu.scratch(n_h * 4);
        if gpu.rows_for(gate.rows).is_some() {
            gpu.gate_up_rows(gate, up, x, m, &h);
            return h;
        }
        unsafe {
            debug_assert_eq!(gate.n_qs % 4, 0);
            debug_assert_eq!((m * gate.cols) % 4, 0);
            q4_0_gate_up_silu_kernel::launch(
                &gpu.client,
                cube_count(n_h, gpu.per_cube()),
                CubeDim::new_1d(gpu.cube_dim as u32),
                BufferArg::from_raw_parts(gate.qs.clone(), gate.n_qs / 4),
                BufferArg::from_raw_parts(gate.sc.clone(), gate.n_sc),
                BufferArg::from_raw_parts(up.qs.clone(), up.n_qs / 4),
                BufferArg::from_raw_parts(up.sc.clone(), up.n_sc),
                BufferArg::from_raw_parts(x.clone(), m * gate.cols / 4),
                BufferArg::from_raw_parts(gate.dims.clone(), 2),
                BufferArg::from_raw_parts(h.clone(), n_h),
                gpu.use_plane,
            );
        }
        h
    }
}

impl Gpu {
    /// The rows variant of the fused gate/up/silu into `h` (`m * gate.rows`), two resident matrices.
    fn gate_up_rows(&self, gate: &GpuQ4Mat, up: &GpuQ4Mat, x: &Handle, m: usize, h: &Handle) {
        let r = self.rows_for(gate.rows).expect("caller checked");
        let n_h = m * gate.rows;
        let xsh = self.xshared_for(m, gate.cols);
        unsafe {
            q4_0_gate_up_silu_rows_kernel::launch(
                &self.client,
                cube_count(n_h / r, self.per_cube()),
                CubeDim::new_1d(self.cube_dim as u32),
                BufferArg::from_raw_parts(gate.qs.clone(), gate.n_qs / 4),
                BufferArg::from_raw_parts(gate.sc.clone(), gate.n_sc),
                BufferArg::from_raw_parts(up.qs.clone(), up.n_qs / 4),
                BufferArg::from_raw_parts(up.sc.clone(), up.n_sc),
                BufferArg::from_raw_parts(x.clone(), m * gate.cols / 4),
                BufferArg::from_raw_parts(gate.dims.clone(), 2),
                BufferArg::from_raw_parts(gate.dims.clone(), 2),
                BufferArg::from_raw_parts(h.clone(), n_h),
                r,
                xsh,
                if xsh { gate.cols / 4 } else { 1 },
                false,
                self.q4_math(),
            );
        }
    }

    /// The short-conv middle for one token on device buffers: `bcx` is `in_proj`'s output
    /// (`[b | c | x]`, `3e`), `taps` `[l, e]`, `state` `[l - 1, e]` (advanced in place). Returns
    /// `y` (scratch, `e`), the input of `out_proj`. See `conv_decode_kernel`.
    pub fn conv_decode(&self, bcx: &Handle, taps: &Handle, state: &Handle, e: usize, l: usize) -> Handle {
        assert!(l >= 2, "l_cache {l}");
        let y = self.scratch(e * 4);
        unsafe {
            conv_decode_kernel::launch(
                &self.client,
                cube_count(e, self.cube_dim),
                CubeDim::new_1d(self.cube_dim as u32),
                BufferArg::from_raw_parts(bcx.clone(), 3 * e),
                BufferArg::from_raw_parts(taps.clone(), l * e),
                BufferArg::from_raw_parts(state.clone(), (l - 1) * e),
                BufferArg::from_raw_parts(y.clone(), e),
                BufferArg::from_raw_parts(self.const_u32(&[e as u32, l as u32]), 2),
            );
        }
        y
    }

    /// [`Gpu::read_raw`] without recycling the scratch pool: for a read in the middle of a token
    /// whose scratch buffers (the router input, the last FFN output) are still needed afterwards.
    /// The token's last `read_raw` recycles them.
    pub fn read_raw_keep(&self, handles: Vec<(Handle, usize)>) -> Result<Vec<Vec<u8>>, String> {
        let lens: Vec<usize> = handles.iter().map(|(_, n)| *n).collect();
        let hs: Vec<Handle> = handles.into_iter().map(|(h, _)| h).collect();
        let bytes = self.client.read(hs).map_err(|e| format!("GPU read failed: {e:?}"))?;
        Ok(bytes.iter().zip(lens).map(|(b, n)| b[..n].to_vec()).collect())
    }

    /// Overwrite a whole device buffer of `u32`s, stream-ordered after what is queued.
    pub fn write_u32(&self, h: &Handle, v: &[u32]) {
        self.client.write(h, Bytes::from_elems(v.to_vec()));
    }

    /// Read buffers of any element type back in **one** synchronisation, then recycle this
    /// round's scratch (as [`Gpu::read_all`]). `want` is the byte count to keep from each.
    pub fn read_raw(&self, handles: Vec<(Handle, usize)>) -> Result<Vec<Vec<u8>>, String> {
        let lens: Vec<usize> = handles.iter().map(|(_, n)| *n).collect();
        let hs: Vec<Handle> = handles.into_iter().map(|(h, _)| h).collect();
        let bytes = self.client.read(hs).map_err(|e| format!("GPU read failed: {e:?}"))?;
        let out = bytes.iter().zip(lens).map(|(b, n)| b[..n].to_vec()).collect();
        self.recycle();
        Ok(out)
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
            dims: gpu.const_u32(&[e as u32, l as u32]),
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
            dims: gpu.const_u32(&dims),
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
            // `from_raw_parts` counts ELEMENTS of the kernel's slice type, not u32 words:
            // `qs`/`qv` are `&[V4U]` and `x` is `&[V4F]`, so the word counts divide by 4. Both
            // divide exactly by construction (a Q4_0 block repacks to 4 u32 words, and `cols` is
            // a multiple of 32) -- assert it so a future shape change fails here instead of
            // silently handing the kernel a length 4x too long.
            debug_assert_eq!(self.n_qv % 4, 0);
            debug_assert_eq!((m * self.cols) % 4, 0);
            q6_k_matvec_kernel::launch(
                &gpu.client,
                cube_count(n_out, gpu.per_cube()),
                CubeDim::new_1d(gpu.cube_dim as u32),
                BufferArg::from_raw_parts(self.qv.clone(), self.n_qv / 4),
                BufferArg::from_raw_parts(self.sd.clone(), self.n_sd),
                BufferArg::from_raw_parts(x.clone(), m * self.cols / 4),
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
    /// Upload the three matrices: as one batch when the bytes are page-locked and the
    /// shapes allow it (`Gpu::upload_expert_batched`, cuda), else one matrix at a time.
    pub fn upload(gpu: &Gpu, gate: QuantMat<'_>, up: QuantMat<'_>, down: QuantMat<'_>) -> Result<Self, String> {
        #[cfg(feature = "cuda")]
        if let Some([gate, up, down]) = gpu.upload_expert_batched([&gate, &up, &down]) {
            return Ok(Self { gate, up, down });
        }
        Self::upload_per_matrix(gpu, gate, up, down)
    }

    /// One matrix at a time, whatever the runtime: the path for experts that stay resident.
    ///
    /// Measured 2026-11-04 on c4 GPU 3 (`docs/WORK_LOG.md`): loading all 2432 experts of the
    /// 24B through the batched path put 16,267 MiB on the card against 15,291 MiB this way
    /// (+0.4 MiB an expert from how the memory pool rounds the larger shared allocations),
    /// and the first token then ran out of device memory. The batched path is a win per miss
    /// (transient), not per resident.
    pub fn upload_per_matrix(gpu: &Gpu, gate: QuantMat<'_>, up: QuantMat<'_>, down: QuantMat<'_>) -> Result<Self, String> {
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
        if gpu.rows_for(n_ff).is_some() {
            gpu.gate_up_rows(&self.gate, &self.up, x, m, &h);
            let out = self.down.launch(gpu, &h, m);
            return (out, m * self.down.rows);
        }
        unsafe {
            // `from_raw_parts` counts ELEMENTS of the kernel's slice type, not u32 words:
            // `qs`/`qv` are `&[V4U]` and `x` is `&[V4F]`, so the word counts divide by 4. Both
            // divide exactly by construction (a Q4_0 block repacks to 4 u32 words, and `cols` is
            // a multiple of 32) -- assert it so a future shape change fails here instead of
            // silently handing the kernel a length 4x too long.
            debug_assert_eq!(self.gate.n_qs % 4, 0);
            debug_assert_eq!(self.up.n_qs % 4, 0);
            debug_assert_eq!((m * self.gate.cols) % 4, 0);
            q4_0_gate_up_silu_kernel::launch(
                &gpu.client,
                cube_count(n_h, gpu.per_cube()),
                CubeDim::new_1d(gpu.cube_dim as u32),
                BufferArg::from_raw_parts(self.gate.qs.clone(), self.gate.n_qs / 4),
                BufferArg::from_raw_parts(self.gate.sc.clone(), self.gate.n_sc),
                BufferArg::from_raw_parts(self.up.qs.clone(), self.up.n_qs / 4),
                BufferArg::from_raw_parts(self.up.sc.clone(), self.up.n_sc),
                BufferArg::from_raw_parts(x.clone(), m * self.gate.cols / 4),
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

/// Every expert of one MoE layer, concatenated so the layer is **2 dispatches, not 2 per expert**.
///
/// A dispatch on a loaded device measures 0.108 ms, so a top-4 layer costs 8 of them and the 38
/// MoE layers of a token cost 304 -- 37.6 ms, 42% of the matmul budget. Here the layer's experts
/// share one buffer per projection and the routing arrives as data, so the cost stops scaling
/// with top-k. It also collapses ~14,600 separate buffers to ~114, which should return a chunk
/// of the 1.5 GB gap between the weights (13.66 GB) and what nvidia-smi reports (15.2 GB).
pub struct GpuExpertLayer {
    n_experts: usize,
    /// gate/up rows (n_ff_exp) and cols (n_embd); `down` is the transpose.
    n_ff: usize,
    n_embd: usize,
    gate: (Handle, usize, Handle, usize),
    up: (Handle, usize, Handle, usize),
    down: (Handle, usize, Handle, usize),
    gu_dims: Handle,
    dn_dims: Handle,
    bytes: usize,
}

impl GpuExpertLayer {
    /// Concatenate `experts` (gate, up, down per expert, all Q4_0 and all the same shape).
    ///
    /// The scale buffer can be concatenated verbatim only because `q4_row_partial` reads
    /// `sc[gb / 2]` from the GLOBAL block index: a layer whose per-expert block count were odd
    /// would put an expert boundary mid-u32 and shift every scale after it. Asserted here rather
    /// than left to a wrong answer.
    pub fn upload<'a>(gpu: &Gpu, experts: impl IntoIterator<Item = (QuantMat<'a>, QuantMat<'a>, QuantMat<'a>)>) -> Result<Self, String> {
        let mut shape: Option<(usize, usize)> = None;
        let (mut gq, mut gs, mut uq, mut us, mut dq, mut ds) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut n_experts = 0usize;
        for (g, u, d) in experts {
            for m in [&g, &u, &d] {
                if m.ty != lb_gguf::GgmlType::Q4_0 && !crate::transcode::transcodable(m.ty) {
                    return Err(format!("GpuExpertLayer needs Q4_0 (or a transcodable type), got {}", m.ty.name()));
                }
                if m.cols % 32 != 0 {
                    return Err(format!("Q4_0 needs cols % 32 == 0, got {}", m.cols));
                }
            }
            if g.rows != u.rows || g.cols != u.cols || d.rows != g.cols || d.cols != g.rows {
                return Err("GpuExpertLayer: gate/up must share a shape and down must be its transpose".into());
            }
            match shape {
                None => shape = Some((g.rows, g.cols)),
                Some(s) if s == (g.rows, g.cols) => {}
                Some(s) => return Err(format!("GpuExpertLayer: expert shape {:?} does not match {:?}", (g.rows, g.cols), s)),
            }
            for (m, q, s) in [(&g, &mut gq, &mut gs), (&u, &mut uq, &mut us), (&d, &mut dq, &mut ds)] {
                let nb = m.rows * (m.cols / 32);
                if nb % 2 != 0 {
                    return Err(format!("GpuExpertLayer: {} blocks per expert is odd; concatenated scales would shift", nb));
                }
                let (qw, sw) = repack_q4_0(&crate::transcode::q4_0_bytes(m)?, nb);
                q.extend_from_slice(&qw);
                s.extend_from_slice(&sw);
            }
            n_experts += 1;
        }
        let (n_ff, n_embd) = shape.ok_or("GpuExpertLayer: no experts")?;
        let bytes = (gq.len() + gs.len() + uq.len() + us.len() + dq.len() + ds.len()) * 4;
        let (ngq, ngs, nuq, nus, ndq, nds) = (gq.len(), gs.len(), uq.len(), us.len(), dq.len(), ds.len());
        Ok(Self {
            n_experts,
            n_ff,
            n_embd,
            gate: (gpu.upload_u32(gq), ngq, gpu.upload_u32(gs), ngs),
            up: (gpu.upload_u32(uq), nuq, gpu.upload_u32(us), nus),
            down: (gpu.upload_u32(dq), ndq, gpu.upload_u32(ds), nds),
            gu_dims: gpu.const_u32(&[n_ff as u32, n_embd as u32, 0]),
            dn_dims: gpu.const_u32(&[n_embd as u32, n_ff as u32, 0]),
            bytes,
        })
    }

    /// One layer's experts from the GGUF stacks as they are on disk: `gate` and `up` are
    /// `[n_experts * n_ff, n_embd]`, `down` `[n_experts * n_embd, n_ff]`, each one contiguous
    /// `QuantMat` (`ExpertMats` in `lb-lfm2`: expert `i` starts at `i * per_expert`, so the
    /// experts are already concatenated). Uploaded through `Gpu::upload_q4_0`, so on the cuda
    /// runtime with a registered mapping the raw bytes are DMA'd and repacked on the device.
    ///
    /// The per-expert block count must be even (concatenated scales, see [`Self::upload`]).
    pub fn upload_stacked(gpu: &Gpu, gate: QuantMat<'_>, up: QuantMat<'_>, down: QuantMat<'_>, n_experts: usize) -> Result<Self, String> {
        if n_experts == 0 {
            return Err("GpuExpertLayer: no experts".into());
        }
        for m in [&gate, &up, &down] {
            if m.ty != lb_gguf::GgmlType::Q4_0 && !crate::transcode::transcodable(m.ty) {
                return Err(format!("GpuExpertLayer needs Q4_0 (or a transcodable type), got {}", m.ty.name()));
            }
            if m.cols % 32 != 0 || m.rows % n_experts != 0 {
                return Err(format!("GpuExpertLayer: [{}, {}] is not {n_experts} Q4_0 experts", m.rows, m.cols));
            }
        }
        let (n_ff, n_embd) = (gate.rows / n_experts, gate.cols);
        if up.rows != gate.rows || up.cols != gate.cols || down.rows != n_experts * n_embd || down.cols != n_ff {
            return Err("GpuExpertLayer: gate/up must share a shape and down must be its transpose".into());
        }
        let mut parts = Vec::with_capacity(3);
        for m in [&gate, &up, &down] {
            let nb_e = (m.rows / n_experts) * (m.cols / 32);
            if nb_e % 2 != 0 {
                return Err(format!("GpuExpertLayer: {nb_e} blocks per expert is odd; concatenated scales would shift"));
            }
            let ((qs, n_qs), (sc, n_sc)) = gpu.upload_q4_0(&crate::transcode::q4_0_bytes(m)?, n_experts * nb_e);
            parts.push((qs, n_qs, sc, n_sc));
        }
        let bytes = parts.iter().map(|p| (p.1 + p.3) * 4).sum();
        let down = parts.pop().expect("three");
        let up = parts.pop().expect("three");
        let gate = parts.pop().expect("three");
        Ok(Self {
            n_experts,
            n_ff,
            n_embd,
            gate,
            up,
            down,
            gu_dims: gpu.const_u32(&[n_ff as u32, n_embd as u32, 0]),
            dn_dims: gpu.const_u32(&[n_embd as u32, n_ff as u32, 0]),
            bytes,
        })
    }

    /// Expert `e` as an ordinary [`GpuExpert`] whose matrices are offset views of this layer's
    /// buffers: no copy, no extra device memory. The per-expert path (`GpuExpert::launch`) and
    /// the routed layer path then read the very same words, so an A/B between them compares
    /// kernels, never two uploads.
    ///
    /// A view's byte offset must be a multiple of 256 (wgpu's storage-binding offset alignment;
    /// the 24B's experts are 1,572,864 + 196,608 bytes a matrix, so they are).
    pub fn expert(&self, gpu: &Gpu, e: usize) -> Result<GpuExpert, String> {
        if e >= self.n_experts {
            return Err(format!("expert {e} >= {}", self.n_experts));
        }
        let n = self.n_experts;
        let view = |p: &(Handle, usize, Handle, usize), rows: usize, cols: usize| -> Result<GpuQ4Mat, String> {
            let (n_qs, n_sc) = (p.1 / n, p.3 / n);
            if p.1 % n != 0 || p.3 % n != 0 || (n_qs * 4) % 256 != 0 || (n_sc * 4) % 256 != 0 {
                return Err(format!("expert views need 256-byte aligned slices ({} qs / {} sc words per expert)", n_qs, n_sc));
            }
            let qs = p.0.clone().offset_start((e * n_qs * 4) as u64).offset_end(((n - 1 - e) * n_qs * 4) as u64);
            let sc = p.2.clone().offset_start((e * n_sc * 4) as u64).offset_end(((n - 1 - e) * n_sc * 4) as u64);
            Ok(GpuQ4Mat { rows, cols, row_bytes: cols / 32 * 18, qs, sc, n_qs, n_sc, dims: gpu.const_u32(&matvec_dims(rows, cols)) })
        };
        Ok(GpuExpert { gate: view(&self.gate, self.n_ff, self.n_embd)?, up: view(&self.up, self.n_ff, self.n_embd)?, down: view(&self.down, self.n_embd, self.n_ff)? })
    }

    /// The layer on **device-resident** routes: slots `base .. base + slots` of `routes` (a
    /// `[n_route_slots, 2]` u32 buffer of `[expert, token]` pairs that a kernel wrote, e.g.
    /// `Gpu::router_topk`). Nothing is checked on the host because nothing is on the host: the
    /// router kernel only ever writes expert ids below its `n_expert`, which `launch_device`
    /// requires to equal this layer's. Two dispatches. Returns `[slots, n_embd]`.
    pub fn launch_device(&self, gpu: &Gpu, x: &Handle, n_tokens: usize, routes: &Handle, n_route_slots: usize, base: usize, slots: usize) -> (Handle, usize) {
        assert!(slots > 0 && base + slots <= n_route_slots, "slots {base}+{slots} of {n_route_slots}");
        let gu = gpu.const_u32(&[self.n_ff as u32, self.n_embd as u32, base as u32]);
        let dn = gpu.const_u32(&[self.n_embd as u32, self.n_ff as u32, base as u32]);
        self.launch_with(gpu, x, n_tokens, routes, 2 * n_route_slots, slots, &gu, &dn)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_with(&self, gpu: &Gpu, x: &Handle, n_tokens: usize, rh: &Handle, n_route_words: usize, slots: usize, gu_dims: &Handle, dn_dims: &Handle) -> (Handle, usize) {
        let n_h = slots * self.n_ff;
        let h = gpu.scratch(n_h * 4);
        if let (Some(r), Some(_)) = (gpu.rows_for(self.n_ff), gpu.rows_for(self.n_embd)) {
            // every slot of a decode token reads token 0, so x can sit in shared memory
            let xsh = gpu.xshared_for(n_tokens, self.n_embd);
            unsafe {
                q4_0_gate_up_silu_rows_kernel::launch(
                    &gpu.client,
                    cube_count(n_h / r, gpu.per_cube()),
                    CubeDim::new_1d(gpu.cube_dim as u32),
                    BufferArg::from_raw_parts(self.gate.0.clone(), self.gate.1 / 4),
                    BufferArg::from_raw_parts(self.gate.2.clone(), self.gate.3),
                    BufferArg::from_raw_parts(self.up.0.clone(), self.up.1 / 4),
                    BufferArg::from_raw_parts(self.up.2.clone(), self.up.3),
                    BufferArg::from_raw_parts(x.clone(), n_tokens * self.n_embd / 4),
                    BufferArg::from_raw_parts(rh.clone(), n_route_words),
                    BufferArg::from_raw_parts(gu_dims.clone(), 3),
                    BufferArg::from_raw_parts(h.clone(), n_h),
                    r,
                    xsh,
                    if xsh { self.n_embd / 4 } else { 1 },
                    true,
                    gpu.q4_math(),
                );
            }
            let n_out = slots * self.n_embd;
            let out = gpu.scratch(n_out * 4);
            unsafe {
                q4_0_matvec_rows_kernel::launch(
                    &gpu.client,
                    cube_count(n_out / r, gpu.per_cube()),
                    CubeDim::new_1d(gpu.cube_dim as u32),
                    BufferArg::from_raw_parts(self.down.0.clone(), self.down.1 / 4),
                    BufferArg::from_raw_parts(self.down.2.clone(), self.down.3),
                    BufferArg::from_raw_parts(h.clone(), n_h / 4),
                    BufferArg::from_raw_parts(rh.clone(), n_route_words),
                    BufferArg::from_raw_parts(dn_dims.clone(), 3),
                    BufferArg::from_raw_parts(out.clone(), n_out),
                    r,
                    false,
                    1,
                    true,
                    gpu.q4_math(),
                );
            }
            return (out, n_out);
        }
        unsafe {
            debug_assert_eq!(self.gate.1 % 4, 0);
            debug_assert_eq!((n_tokens * self.n_embd) % 4, 0);
            q4_0_routed_gate_up_silu_kernel::launch(
                &gpu.client,
                cube_count(n_h, gpu.per_cube()),
                CubeDim::new_1d(gpu.cube_dim as u32),
                BufferArg::from_raw_parts(self.gate.0.clone(), self.gate.1 / 4),
                BufferArg::from_raw_parts(self.gate.2.clone(), self.gate.3),
                BufferArg::from_raw_parts(self.up.0.clone(), self.up.1 / 4),
                BufferArg::from_raw_parts(self.up.2.clone(), self.up.3),
                BufferArg::from_raw_parts(x.clone(), n_tokens * self.n_embd / 4),
                BufferArg::from_raw_parts(rh.clone(), n_route_words),
                BufferArg::from_raw_parts(gu_dims.clone(), 3),
                BufferArg::from_raw_parts(h.clone(), n_h),
                gpu.use_plane,
            );
        }
        let n_out = slots * self.n_embd;
        let out = gpu.scratch(n_out * 4);
        unsafe {
            debug_assert_eq!(self.down.1 % 4, 0);
            debug_assert_eq!(n_h % 4, 0);
            q4_0_routed_matvec_kernel::launch(
                &gpu.client,
                cube_count(n_out, gpu.per_cube()),
                CubeDim::new_1d(gpu.cube_dim as u32),
                BufferArg::from_raw_parts(self.down.0.clone(), self.down.1 / 4),
                BufferArg::from_raw_parts(self.down.2.clone(), self.down.3),
                BufferArg::from_raw_parts(h.clone(), n_h / 4),
                BufferArg::from_raw_parts(rh.clone(), n_route_words),
                BufferArg::from_raw_parts(dn_dims.clone(), 3),
                BufferArg::from_raw_parts(out.clone(), n_out),
                gpu.use_plane,
            );
        }
        (out, n_out)
    }

    /// The layer for a whole prefill chunk through the tiled GEMM (`gemm.rs`): slots grouped by
    /// expert on the host (`gemm::group_slots` over the chunk's `routes`, uploaded as `perm` and
    /// `tiles`), each expert's weight tiles read once per 64 of its slots. `routes` is the
    /// `[expert, token]` buffer the router wrote (`n_slots` pairs), `x` `[n_tokens, n_embd]`.
    /// Returns `[n_slots, n_embd]` in slot order, as [`GpuExpertLayer::launch_device`] does.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_grouped(&self, gpu: &Gpu, x: &Handle, n_tokens: usize, routes: &Handle, n_slots: usize, perm: &Handle, tiles: &Handle, n_tiles: usize) -> (Handle, usize) {
        let n_h = n_slots * self.n_ff;
        let h = gpu.scratch(n_h * 4);
        let gu = gpu.const_u32(&[self.n_ff as u32, self.n_embd as u32, 0, 0]);
        let dn = gpu.const_u32(&[self.n_embd as u32, self.n_ff as u32, 0, 0]);
        unsafe {
            gemm_launch!(
                gpu,
                CubeCount::Static(self.n_ff.div_ceil(crate::gemm::TILE) as u32, n_tiles as u32, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(self.gate.0.clone(), self.gate.1),
                BufferArg::from_raw_parts(self.gate.2.clone(), self.gate.3),
                BufferArg::from_raw_parts(self.up.0.clone(), self.up.1),
                BufferArg::from_raw_parts(self.up.2.clone(), self.up.3),
                (x, n_tokens * self.n_embd),
                BufferArg::from_raw_parts(routes.clone(), 2 * n_slots),
                BufferArg::from_raw_parts(perm.clone(), n_slots),
                BufferArg::from_raw_parts(tiles.clone(), 3 * n_tiles),
                BufferArg::from_raw_parts(gu, 4),
                BufferArg::from_raw_parts(h.clone(), n_h),
                true,
                true,
                true,
            );
        }
        let n_out = n_slots * self.n_embd;
        let out = gpu.scratch(n_out * 4);
        unsafe {
            gemm_launch!(
                gpu,
                CubeCount::Static(self.n_embd.div_ceil(crate::gemm::TILE) as u32, n_tiles as u32, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(self.down.0.clone(), self.down.1),
                BufferArg::from_raw_parts(self.down.2.clone(), self.down.3),
                BufferArg::from_raw_parts(self.down.0.clone(), self.down.1),
                BufferArg::from_raw_parts(self.down.2.clone(), self.down.3),
                (h, n_h),
                BufferArg::from_raw_parts(routes.clone(), 2 * n_slots),
                BufferArg::from_raw_parts(perm.clone(), n_slots),
                BufferArg::from_raw_parts(tiles.clone(), 3 * n_tiles),
                BufferArg::from_raw_parts(dn, 4),
                BufferArg::from_raw_parts(out.clone(), n_out),
                true,
                false,
                false,
            );
        }
        (out, n_out)
    }

    pub fn n_experts(&self) -> usize {
        self.n_experts
    }

    pub fn bytes_on_device(&self) -> usize {
        self.bytes
    }

    /// Run the whole layer in **two** dispatches.
    ///
    /// `routes` is `[expert_id, token_index]` per slot, a slot being one (expert, token) pair the
    /// router chose. `x` holds `n_tokens` activation rows of `n_embd`. Returns the output handle
    /// and its length, `slots * n_embd`, in the same slot order as `routes`.
    pub fn launch(&self, gpu: &Gpu, x: &Handle, n_tokens: usize, routes: &[u32]) -> Result<(Handle, usize), String> {
        if routes.len() % 2 != 0 {
            return Err("GpuExpertLayer::launch: routes must be [expert, token] pairs".into());
        }
        let slots = routes.len() / 2;
        if slots == 0 {
            return Err("GpuExpertLayer::launch: no slots".into());
        }
        if let Some(bad) = routes.chunks(2).find(|c| c[0] as usize >= self.n_experts) {
            return Err(format!("GpuExpertLayer::launch: expert {} >= {}", bad[0], self.n_experts));
        }
        if let Some(bad) = routes.chunks(2).find(|c| c[1] as usize >= n_tokens) {
            return Err(format!("GpuExpertLayer::launch: token {} >= {n_tokens}", bad[1]));
        }
        let rh = gpu.upload_u32(routes.to_vec());
        let out = self.launch_with(gpu, x, n_tokens, &rh, routes.len(), slots, &self.gu_dims, &self.dn_dims);
        // the routes buffer must outlive the launches: it is dropped after they are queued,
        // which the runtime orders after them
        Ok(out)
    }
}

/// A fixed bank of device slots for experts of one shape (gate/up `[n_ff, n_embd]`, down its
/// transpose, Q4_0), allocated **once**, plus a few fixed staging buffers for the raw file bytes of
/// an expert being filled. Filling a slot is a DMA of the three raw slices into a staging buffer
/// (straight from the page-locked mapping when it is registered) and one repack launch per
/// projection into the slot; nothing is allocated per fill, so the device bytes are exactly
/// [`ExpertSlots::bytes`] for the life of the bank, whatever the cache does.
///
/// Why (MoE exp. #2, `docs/STAGE_RING.md`): experts uploaded one by one and kept took 7 MiB pool
/// slices plus a transient raw buffer each while being counted as 5.31 MB, and device memory grew
/// until the card ran out.
///
/// **Layout (engine step E2).** The slots are one [`GpuExpertLayer`] with `n_experts = n_slots`: one
/// contiguous `qs` and one `sc` buffer per projection. So the routed kernels address a slot exactly
/// as they address an expert of a stacked layer, from a routes buffer holding slot numbers
/// ([`ExpertSlots::bank`], `GpuExpertLayer::launch_device`), and the per-slot [`GpuExpert`] views
/// serve the host-routed prefill path from the same memory.
///
/// All work is on the calling stream, in order: a slot refilled after kernels that read its old
/// contents were queued is written only after those kernels ran, and a staging buffer is rewritten
/// only after the repack that read it.
pub struct ExpertSlots {
    bank: GpuExpertLayer,
    slots: Vec<GpuExpert>,
    staging: Vec<Handle>,
    next_stage: AtomicUsize,
    /// File bytes of one matrix (`nb * 18`).
    raw_len: usize,
    n_sc: usize,
    /// Output of [`ExpertSlots::fence`]; nothing reads it.
    fence_out: Handle,
    /// Device bytes held: the slots plus the staging buffers.
    pub bytes: usize,
}

impl ExpertSlots {
    /// Device bytes of one slot for three matrices of `nb` Q4_0 blocks each (repacked: 16 bytes of
    /// `qs` and 2 of `sc` a block, the same 18 a block as the file).
    pub fn slot_bytes(nb: usize) -> usize {
        3 * (nb * 4 + nb / 2) * 4
    }

    /// Device bytes of the staging buffers.
    pub fn staging_bytes(nb: usize, n_staging: usize) -> usize {
        n_staging * 3 * nb * 18
    }

    /// `n_slots` slots for experts whose three matrices are `shapes` (rows, cols) in the order gate,
    /// up, down, Q4_0, all with the same even block count; `n_staging` raw staging buffers.
    pub fn new(gpu: &Gpu, n_slots: usize, shapes: [(usize, usize); 3], n_staging: usize) -> Result<Self, String> {
        let nb = shapes[0].0 * shapes[0].1 / 32;
        for (r, c) in shapes {
            if c % 32 != 0 || r * c / 32 != nb {
                return Err(format!("expert slots need three Q4_0 matrices of one block count, got {shapes:?}"));
            }
        }
        if nb == 0 || nb % 2 != 0 {
            return Err(format!("expert slots need an even block count, got {nb}"));
        }
        let (n_ff, n_embd) = shapes[0];
        if shapes[1] != shapes[0] || shapes[2] != (n_embd, n_ff) {
            return Err(format!("expert slots need gate = up and down their transpose, got {shapes:?}"));
        }
        let (n_qs, n_sc) = (nb * 4, nb / 2);
        let raw_len = nb * 18;
        let n_staging = n_staging.max(1);
        // Allocated once, in CubeCL's persistent pool: exact-fit slices that the pool never sweeps.
        // The default adaptive pool carves from pages sized to the largest request seen, which left
        // ~2 GB of a P100 unused around 5 GB of slots (measured, `docs/ENGINE_E1_CACHE.md`).
        let (parts, staging): (Vec<(Handle, Handle)>, Vec<Handle>) = gpu.client.memory_persistent_allocation((), |_| {
            let parts = (0..3).map(|_| (gpu.client.empty(n_slots * n_qs * 4), gpu.client.empty(n_slots * n_sc * 4))).collect();
            let staging = (0..n_staging).map(|_| gpu.client.empty(3 * raw_len)).collect();
            (parts, staging)
        });
        let proj = |i: usize| (parts[i].0.clone(), n_slots * n_qs, parts[i].1.clone(), n_slots * n_sc);
        let bank = GpuExpertLayer {
            n_experts: n_slots,
            n_ff,
            n_embd,
            gate: proj(0),
            up: proj(1),
            down: proj(2),
            gu_dims: gpu.const_u32(&[n_ff as u32, n_embd as u32, 0]),
            dn_dims: gpu.const_u32(&[n_embd as u32, n_ff as u32, 0]),
            bytes: n_slots * Self::slot_bytes(nb),
        };
        let slots: Vec<GpuExpert> = (0..n_slots).map(|s| bank.expert(gpu, s)).collect::<Result<_, _>>()?;
        let bytes = bank.bytes + n_staging * 3 * raw_len;
        let fence_out = gpu.upload_u32(vec![0u32]);
        Ok(Self { bank, slots, staging, next_stage: AtomicUsize::new(0), raw_len, n_sc, fence_out, bytes })
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// The expert held in `slot` (whatever was filled last).
    pub fn expert(&self, slot: usize) -> &GpuExpert {
        &self.slots[slot]
    }

    /// The slots as one stacked layer: `launch_device` on it takes routes holding slot numbers.
    pub fn bank(&self) -> &GpuExpertLayer {
        &self.bank
    }

    /// Fill `slot` with the expert whose raw Q4_0 file bytes are `raws` (gate, up, down; each the
    /// matrix's whole byte range). Queued on the calling stream; returns at once.
    pub fn fill(&self, gpu: &Gpu, slot: usize, raws: [&[u8]; 3]) -> Result<(), String> {
        let st = &self.staging[self.next_stage.fetch_add(1, Ordering::Relaxed) % self.staging.len()];
        let ex = &self.slots[slot];
        let n = gpu.const_u32(&[self.n_sc as u32]);
        for (j, (raw, m)) in raws.iter().zip([&ex.gate, &ex.up, &ex.down]).enumerate() {
            if raw.len() < self.raw_len {
                return Err(format!("expert slot fill: matrix {j} has {} bytes, need {}", raw.len(), self.raw_len));
            }
            let raw = &raw[..self.raw_len];
            let dst = st.clone().offset_start((j * self.raw_len) as u64).offset_end(((2 - j) * self.raw_len) as u64);
            #[cfg(feature = "cuda")]
            let reg = if gpu.host_repack() { None } else { gpu.registered_slice(raw) };
            #[cfg(not(feature = "cuda"))]
            let reg: Option<Bytes> = None;
            match reg {
                Some(b) => gpu.client.write(&dst, b),
                None => {
                    let words: Vec<u32> = raw.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                    gpu.client.write(&dst, Bytes::from_elems(words));
                }
            }
            unsafe {
                q4_0_repack_kernel::launch(
                    &gpu.client,
                    cube_count(self.n_sc, REPACK_CUBE),
                    CubeDim::new_1d(REPACK_CUBE as u32),
                    BufferArg::from_raw_parts(dst, self.raw_len / 4),
                    BufferArg::from_raw_parts(m.qs.clone(), m.n_qs),
                    BufferArg::from_raw_parts(m.sc.clone(), m.n_sc),
                    BufferArg::from_raw_parts(n.clone(), 1),
                );
            }
        }
        Ok(())
    }

    /// Device bytes of `n` prefetch landing buffers (see [`ExpertSlots::land`]).
    pub fn landing_bytes(nb: usize, n: usize) -> usize {
        n * 3 * nb * 18
    }

    /// Bytes of one expert's raw file slices (the size of a landing buffer).
    pub fn raw_expert_bytes(&self) -> usize {
        3 * self.raw_len
    }

    /// Allocate `n` landing buffers on `side` (persistent pool, once) for [`ExpertSlots::land`].
    pub fn landing_buffers(&self, gpu: &Gpu, side: &SideStream, n: usize) -> Vec<Handle> {
        let size = 3 * self.raw_len;
        gpu.on_stream(side, || gpu.client.memory_persistent_allocation((), |_| (0..n).map(|_| gpu.client.empty(size)).collect()))
    }

    /// Prefetch: copy an expert's raw Q4_0 slices (gate, up, down) into the landing buffer `lb`,
    /// queued on `side` so the copy overlaps the main stream's kernels. Returns a sentinel handle
    /// created on `side` **after** the copies: CubeCL's cross-stream ordering works by buffer cursors
    /// set when a buffer is allocated, not when it is written, so a kernel binding the (long-lived)
    /// landing buffer would not wait for this copy; a freshly allocated sentinel has a cursor after
    /// the copies, and a main-stream kernel that binds it waits on the device for them. The sentinel
    /// doubles as the repack's block-count argument, so the kernel signature does not change.
    pub fn land(&self, gpu: &Gpu, side: &SideStream, lb: &Handle, raws: [&[u8]; 3]) -> Result<Handle, String> {
        for (j, raw) in raws.iter().enumerate() {
            if raw.len() < self.raw_len {
                return Err(format!("expert landing: matrix {j} has {} bytes, need {}", raw.len(), self.raw_len));
            }
        }
        Ok(gpu.on_stream(side, || {
            for (j, raw) in raws.iter().enumerate() {
                let raw = &raw[..self.raw_len];
                let dst = lb.clone().offset_start((j * self.raw_len) as u64).offset_end(((2 - j) * self.raw_len) as u64);
                #[cfg(feature = "cuda")]
                let reg = if gpu.host_repack() { None } else { gpu.registered_slice(raw) };
                #[cfg(not(feature = "cuda"))]
                let reg: Option<Bytes> = None;
                match reg {
                    Some(b) => gpu.client.write(&dst, b),
                    None => {
                        let words: Vec<u32> = raw.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                        gpu.client.write(&dst, Bytes::from_elems(words));
                    }
                }
            }
            gpu.upload_u32(vec![self.n_sc as u32])
        }))
    }

    /// Prefetch into a spare slot: copy the raw slices into `raw_buf` and repack them into slot `phys`,
    /// all on `side`, so neither the bus nor the repack is on the main stream. `phys` must be a slot no
    /// kernel can be reading (one that no expert maps to). Returns a sentinel allocated after the
    /// repack; [`ExpertSlots::fence`] on the main stream waits for it before the slot is read.
    pub fn land_spare(&self, gpu: &Gpu, side: &SideStream, raw_buf: &Handle, phys: usize, raws: [&[u8]; 3]) -> Result<Handle, String> {
        for (j, raw) in raws.iter().enumerate() {
            if raw.len() < self.raw_len {
                return Err(format!("expert landing: matrix {j} has {} bytes, need {}", raw.len(), self.raw_len));
            }
        }
        let ex = &self.slots[phys];
        Ok(gpu.on_stream(side, || {
            let n = gpu.const_u32(&[self.n_sc as u32]);
            for (j, (raw, m)) in raws.iter().zip([&ex.gate, &ex.up, &ex.down]).enumerate() {
                let raw = &raw[..self.raw_len];
                let dst = raw_buf.clone().offset_start((j * self.raw_len) as u64).offset_end(((2 - j) * self.raw_len) as u64);
                #[cfg(feature = "cuda")]
                let reg = if gpu.host_repack() { None } else { gpu.registered_slice(raw) };
                #[cfg(not(feature = "cuda"))]
                let reg: Option<Bytes> = None;
                match reg {
                    Some(b) => gpu.client.write(&dst, b),
                    None => {
                        let words: Vec<u32> = raw.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                        gpu.client.write(&dst, Bytes::from_elems(words));
                    }
                }
                unsafe {
                    q4_0_repack_kernel::launch(
                        &gpu.client,
                        cube_count(self.n_sc, REPACK_CUBE),
                        CubeDim::new_1d(REPACK_CUBE as u32),
                        BufferArg::from_raw_parts(dst, self.raw_len / 4),
                        BufferArg::from_raw_parts(m.qs.clone(), m.n_qs),
                        BufferArg::from_raw_parts(m.sc.clone(), m.n_sc),
                        BufferArg::from_raw_parts(n.clone(), 1),
                    );
                }
            }
            gpu.upload_u32(vec![self.n_sc as u32])
        }))
    }

    /// Make the calling stream wait on the device for the side-stream work that produced `sentinel`.
    pub fn fence(&self, gpu: &Gpu, sentinel: &Handle) {
        unsafe {
            stream_fence_kernel::launch(&gpu.client, cube_count(1, 1), CubeDim::new_1d(1), BufferArg::from_raw_parts(sentinel.clone(), 1), BufferArg::from_raw_parts(self.fence_out.clone(), 1));
        }
    }

    /// Fill `slot` from a landing buffer filled by [`ExpertSlots::land`]: repack only (three launches
    /// on the calling stream), the bus already paid. `sentinel` is the handle `land` returned; binding
    /// it makes the calling stream wait on the device for the copy if it has not landed yet.
    pub fn repack_landed(&self, gpu: &Gpu, slot: usize, lb: &Handle, sentinel: &Handle) {
        let ex = &self.slots[slot];
        for (j, m) in [&ex.gate, &ex.up, &ex.down].into_iter().enumerate() {
            let src = lb.clone().offset_start((j * self.raw_len) as u64).offset_end(((2 - j) * self.raw_len) as u64);
            unsafe {
                q4_0_repack_kernel::launch(
                    &gpu.client,
                    cube_count(self.n_sc, REPACK_CUBE),
                    CubeDim::new_1d(REPACK_CUBE as u32),
                    BufferArg::from_raw_parts(src, self.raw_len / 4),
                    BufferArg::from_raw_parts(m.qs.clone(), m.n_qs),
                    BufferArg::from_raw_parts(m.sc.clone(), m.n_sc),
                    BufferArg::from_raw_parts(sentinel.clone(), 1),
                );
            }
        }
    }
}

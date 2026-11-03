//! Page-locked host staging buffers the engine fills in place (cuda runtime only).
//!
//! Why this exists, measured 2026-11-03 on cesarops2 GPU2 (Tesla P100, Gen3 x16) with a timer
//! probe inside cubecl-server's `write_to_gpu` and cubecl-cuda's `copy_to_device` (raw outputs:
//! c2 `~/lb-measure/out/probe-b2/`, table in `docs/WORK_LOG.md`): a 5.31 MB `create(Bytes)` of a
//! fresh `Vec` spent 2.90 ms in `cuMemAllocHost` (a cold page-lock on every call, because the
//! drop queue holds the earlier staging slices until a 64 MiB flush), 1.29 ms in the host memcpy
//! into that staging slice (4.1 GB/s on c2's Xeon E5-2450 v2), 55 us enqueueing, while the DMA
//! itself took 0.51 ms (10.3 GB/s). CubeCL's driver path reaches the bus; what is lost is the
//! serial host copy and the page-lock in front of it, both of which `Staging::of` skips when the
//! `Bytes` it is handed already reports `AllocationProperty::Pinned`.
//!
//! So: a small pool of page-locked buffers on the device's primary context (the one cubecl-cuda
//! runs on), handed out as slices the caller writes directly (the repack assigns every word),
//! wrapped as pinned `Bytes` for `Client::create`, and taken back when CubeCL's drop queue drops
//! them after its fence. The memcpy is gone and each page-lock is paid once per slot.
//!
//! Not a CubeCL patch: it only uses the public `AllocationController` trait and
//! `Bytes::from_controller`.
use cubecl::bytes::{AccessError, AccessPolicy, AllocationController, AllocationProperty, Bytes};
use cudarc::driver::{result, sys};
use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::sync::{Arc, Mutex};

/// Slot sizes round up to this, so a handful of size classes serve every upload and a slot
/// freed by one matrix fits the next of the same shape.
const GRANULE: usize = 256 * 1024;
/// Page-locked memory one pool may hold, MB (`LB_PINNED_POOL_MB` overrides). Past it `take`
/// returns `None` and the caller uses the old pageable path, so the pool can never exhaust
/// the host. 256 MB covers CubeCL's drop queue (64 MiB per batch, two batches held) with room.
const DEFAULT_CAP_MB: usize = 256;

struct Slot {
    ptr: *mut c_void,
    cap: usize,
}
// SAFETY: page-locked host memory is valid from any thread; the pool serialises the slot list.
unsafe impl Send for Slot {}

struct State {
    free: HashMap<usize, Vec<Slot>>,
    allocated: usize,
}

struct Inner {
    ctx: sys::CUcontext,
    dev: sys::CUdevice,
    cap: usize,
    state: Mutex<State>,
}
// SAFETY: a CUcontext handle is a process-wide token; the driver serialises access to it.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

impl Drop for Inner {
    fn drop(&mut self) {
        // SAFETY: the context was retained in `open`; every slot was allocated on it and
        // is no longer referenced (all `PinnedBuf`s hold an `Arc` to this, so none is alive).
        unsafe {
            let _ = result::ctx::set_current(self.ctx);
            let state = self.state.get_mut().unwrap_or_else(|e| e.into_inner());
            for slot in state.free.values_mut().flatten() {
                let _ = result::free_host(slot.ptr);
            }
            let _ = result::primary_ctx::release(self.dev);
        }
    }
}

/// A pool of page-locked host buffers on one CUDA device's primary context.
#[derive(Clone)]
pub struct PinnedPool(Arc<Inner>);

impl PinnedPool {
    /// Open the pool on CUDA device `device` (driver order, as `LB_CUDA_DEVICE` counts).
    pub fn open(device: usize) -> Result<Self, String> {
        let cap = std::env::var("LB_PINNED_POOL_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_CAP_MB) << 20;
        // SAFETY: plain driver calls; the primary context is reference counted and shared with
        // cubecl-cuda, which retains it the same way (`runtime.rs`), so memory page-locked here
        // is valid for the stream CubeCL copies on.
        let (dev, ctx) = unsafe {
            result::init().map_err(|e| format!("cuInit: {e}"))?;
            let dev = result::device::get(device as i32).map_err(|e| format!("cuDeviceGet({device}): {e}"))?;
            let ctx = result::primary_ctx::retain(dev).map_err(|e| format!("cuDevicePrimaryCtxRetain({device}): {e}"))?;
            (dev, ctx)
        };
        Ok(Self(Arc::new(Inner { ctx, dev, cap, state: Mutex::new(State { free: HashMap::new(), allocated: 0 }) })))
    }

    /// Bytes currently page-locked by this pool (free and handed out).
    pub fn allocated(&self) -> usize {
        self.0.state.lock().unwrap_or_else(|e| e.into_inner()).allocated
    }

    /// A page-locked buffer of `len` bytes, zero-filled the first time its slot is allocated,
    /// or `None` when the pool is at its cap (the caller falls back to pageable memory).
    pub fn take(&self, len: usize) -> Option<PinnedBuf> {
        let cap = len.div_ceil(GRANULE).max(1) * GRANULE;
        let mut st = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = st.free.get_mut(&cap).and_then(|v| v.pop()) {
            return Some(PinnedBuf { pool: self.0.clone(), slot: Some(slot), len });
        }
        if st.allocated + cap > self.0.cap {
            return None;
        }
        // SAFETY: the context is this pool's; a page-lock needs it current on the calling
        // thread, which may be any engine thread. Zeroing makes every byte initialised for
        // the lifetime of the slot, so handing out `&mut [u32]` over it is sound.
        let ptr = unsafe {
            result::ctx::set_current(self.0.ctx).ok()?;
            let ptr = result::malloc_host(cap, 0).ok()?;
            std::ptr::write_bytes(ptr as *mut u8, 0, cap);
            ptr
        };
        st.allocated += cap;
        Some(PinnedBuf { pool: self.0.clone(), slot: Some(Slot { ptr, cap }), len })
    }
}

/// One page-locked buffer on loan from a [`PinnedPool`]; back in the pool when dropped,
/// which for a buffer uploaded through `Client::create` is when CubeCL's drop queue releases it.
pub struct PinnedBuf {
    pool: Arc<Inner>,
    slot: Option<Slot>,
    len: usize,
}
// SAFETY: see `Slot`; the buffer is only ever written by its holder.
unsafe impl Send for PinnedBuf {}
unsafe impl Sync for PinnedBuf {}

impl PinnedBuf {
    fn ptr(&self) -> *mut u8 {
        self.slot.as_ref().expect("slot present until drop").ptr as *mut u8
    }

    /// The buffer as `len / 4` words (`len` must be a multiple of 4; page-locked memory is
    /// page aligned, so the cast is fine).
    pub fn as_mut_u32(&mut self) -> &mut [u32] {
        assert_eq!(self.len % 4, 0, "pinned buffer of {} bytes viewed as u32", self.len);
        // SAFETY: the slot is zero-initialised on allocation, at least `len` bytes, aligned,
        // and exclusively borrowed through `&mut self`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr() as *mut u32, self.len / 4) }
    }

    /// Hand the buffer to CubeCL as pinned `Bytes`: `write_to_gpu` DMAs straight from it.
    pub fn into_bytes(self) -> Bytes {
        let len = self.len;
        // SAFETY: the controller reports the whole slot, initialised, and keeps it alive.
        unsafe { Bytes::from_controller(Box::new(self), len) }
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.take() {
            self.pool.state.lock().unwrap_or_else(|e| e.into_inner()).free.entry(slot.cap).or_default().push(slot);
        }
    }
}

impl AllocationController for PinnedBuf {
    fn alloc_align(&self) -> usize {
        // cuMemHostAlloc returns page-aligned memory; 4096 is what we rely on.
        4096
    }

    fn property(&self) -> AllocationProperty {
        AllocationProperty::Pinned
    }

    unsafe fn memory_mut(&mut self, _policy: AccessPolicy) -> Result<&mut [MaybeUninit<u8>], AccessError> {
        let cap = self.slot.as_ref().expect("slot present until drop").cap;
        // SAFETY: the slot is live, `cap` bytes, exclusively borrowed.
        Ok(unsafe { std::slice::from_raw_parts_mut(self.ptr() as *mut MaybeUninit<u8>, cap) })
    }

    fn memory(&self, _policy: AccessPolicy) -> Result<&[MaybeUninit<u8>], AccessError> {
        let cap = self.slot.as_ref().expect("slot present until drop").cap;
        // SAFETY: as `memory_mut`, without the write.
        Ok(unsafe { std::slice::from_raw_parts(self.ptr() as *const MaybeUninit<u8>, cap) })
    }
}

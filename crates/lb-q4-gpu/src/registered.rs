//! A read-only file mapping page-locked with the driver, so the device DMAs tensor bytes
//! straight out of the page cache (cuda runtime only).
//!
//! Measured 2026-11-03 on cesarops2 GPU2 (Tesla P100, driver 580.178.04,
//! `examples/register_bench.rs`, raw lines in `docs/WORK_LOG.md`): `cuMemHostRegister` with
//! `CU_MEMHOSTREGISTER_READ_ONLY` on the whole 12.9 GB GGUF mapping succeeds in 0.60 s
//! (48 ms/GB; both `HOST_REGISTER_SUPPORTED` and `READ_ONLY_HOST_REGISTER_SUPPORTED` are 1
//! there), after which a 1.77 MB expert-matrix slice copies at 11.2 GB/s (0.157 ms) against
//! 4.6-4.9 GB/s (0.36-0.39 ms) from the same mapping unregistered. Unregister: 0.1 s.
//!
//! The slice goes to CubeCL as `Bytes` reporting `AllocationProperty::Pinned`, which its
//! write path hands to the driver as it is (`Staging::of`: pinned is never re-staged). Only
//! the public `AllocationController` trait is used; the mapping is never written (`memory_mut`
//! refuses), so a read-only `MAP_SHARED` file mapping is sound to expose this way.
use cubecl::bytes::{AccessError, AccessPolicy, AllocationController, AllocationProperty, Bytes};
use cudarc::driver::{result, sys};
use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::sync::Arc;

const PAGE: usize = 4096;

/// One page-locked range of host memory, unregistered on drop.
pub struct HostRegistration {
    start: usize,
    len: usize,
    ctx: sys::CUcontext,
    dev: sys::CUdevice,
}
// SAFETY: the range is plain memory any thread may read; the context handle is a process-wide token.
unsafe impl Send for HostRegistration {}
unsafe impl Sync for HostRegistration {}

impl HostRegistration {
    /// Page-lock the pages of `map` (rounded out to page boundaries, so `map` must be a
    /// mapping, not a slice of some larger heap object) on CUDA device `device`'s primary
    /// context, read-only.
    ///
    /// # Safety
    /// `map` must stay mapped and unchanged until this registration is dropped.
    pub unsafe fn new(device: usize, map: &[u8]) -> Result<Self, String> {
        let start = map.as_ptr() as usize & !(PAGE - 1);
        let end = (map.as_ptr() as usize + map.len()).div_ceil(PAGE) * PAGE;
        // SAFETY: driver calls on the primary context, retained for the registration's lifetime.
        unsafe {
            result::init().map_err(|e| format!("cuInit: {e}"))?;
            let dev = result::device::get(device as i32).map_err(|e| format!("cuDeviceGet({device}): {e}"))?;
            let ctx = result::primary_ctx::retain(dev).map_err(|e| format!("cuDevicePrimaryCtxRetain({device}): {e}"))?;
            result::ctx::set_current(ctx).map_err(|e| format!("cuCtxSetCurrent: {e}"))?;
            let r = sys::cuMemHostRegister_v2(start as *mut c_void, end - start, sys::CU_MEMHOSTREGISTER_READ_ONLY);
            if r != sys::CUresult::CUDA_SUCCESS {
                let _ = result::primary_ctx::release(dev);
                return Err(format!("cuMemHostRegister({} MB, read-only): {r:?}", (end - start) >> 20));
            }
            Ok(Self { start, len: end - start, ctx, dev })
        }
    }

    /// Bytes page-locked.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether `bytes` lies wholly inside this range.
    pub fn contains(&self, bytes: &[u8]) -> bool {
        let b = bytes.as_ptr() as usize;
        b >= self.start && b + bytes.len() <= self.start + self.len
    }
}

impl Drop for HostRegistration {
    fn drop(&mut self) {
        // SAFETY: registered in `new` on this context; every `RegisteredSlice` holds an `Arc`
        // to this, so no upload still references the range.
        unsafe {
            let _ = result::ctx::set_current(self.ctx);
            let _ = sys::cuMemHostUnregister(self.start as *mut c_void);
            let _ = result::primary_ctx::release(self.dev);
        }
    }
}

/// A slice of a [`HostRegistration`], as CubeCL `Bytes` the driver can DMA from directly.
pub struct RegisteredSlice {
    reg: Arc<HostRegistration>,
    ptr: *const u8,
    len: usize,
}
// SAFETY: read-only view of page-locked memory kept alive by `reg`.
unsafe impl Send for RegisteredSlice {}
unsafe impl Sync for RegisteredSlice {}

impl RegisteredSlice {
    /// `bytes` (inside `reg`) as pinned `Bytes`.
    pub fn bytes(reg: Arc<HostRegistration>, bytes: &[u8]) -> Bytes {
        assert!(reg.contains(bytes), "slice outside its registration");
        let len = bytes.len();
        // SAFETY: the controller reports exactly `len` initialised bytes that `reg` keeps locked.
        unsafe { Bytes::from_controller(Box::new(Self { reg, ptr: bytes.as_ptr(), len }), len) }
    }
}

impl AllocationController for RegisteredSlice {
    fn alloc_align(&self) -> usize {
        // the true alignment of the slice start (tensor offsets are 2-byte aligned at worst)
        1usize << (self.ptr as usize).trailing_zeros().min(12)
    }

    fn property(&self) -> AllocationProperty {
        AllocationProperty::Pinned
    }

    unsafe fn memory_mut(&mut self, _policy: AccessPolicy) -> Result<&mut [MaybeUninit<u8>], AccessError> {
        Err(AccessError::Read("read-only file mapping".into()))
    }

    fn memory(&self, _policy: AccessPolicy) -> Result<&[MaybeUninit<u8>], AccessError> {
        let _ = &self.reg;
        // SAFETY: `ptr..ptr+len` is inside the live, locked mapping.
        Ok(unsafe { std::slice::from_raw_parts(self.ptr as *const MaybeUninit<u8>, self.len) })
    }
}

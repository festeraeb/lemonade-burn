//! The wgpu runtime's counterpart of `registered.rs`: the model's read-only file mapping handed to
//! CubeCL as `Bytes` without a host copy (no page-locking, wgpu has none).
//!
//! Measured on c1 (E5-2680 v2): the first touch of a fresh page costs ~3 us there, so the host
//! copy an expert miss made before `client.write` (a fresh 5.3 MB `Vec`, 1300 faults) was most of
//! its 20 ms; the wgpu write then copies once, into its own persistent staging memory.
//! The caller promises the mapping outlives the `Gpu` (it is the model's `Arc<Gguf>` mapping).
use cubecl::bytes::{AccessError, AccessPolicy, AllocationController, AllocationProperty, Bytes};
use std::mem::MaybeUninit;

/// A range the caller keeps alive and never writes.
#[derive(Clone, Copy)]
pub struct Mapping {
    pub start: usize,
    pub len: usize,
}

impl Mapping {
    pub fn contains(&self, bytes: &[u8]) -> bool {
        let p = bytes.as_ptr() as usize;
        p >= self.start && p + bytes.len() <= self.start + self.len
    }
}

struct BorrowedSlice {
    ptr: *const u8,
    len: usize,
}
// SAFETY: a read-only view of memory the registering caller keeps alive.
unsafe impl Send for BorrowedSlice {}
unsafe impl Sync for BorrowedSlice {}

/// `bytes` (inside a registered mapping) as `Bytes` that borrow it.
pub fn bytes_of(bytes: &[u8]) -> Bytes {
    let len = bytes.len();
    // SAFETY: the controller reports exactly `len` initialised bytes of a mapping that outlives the Gpu.
    unsafe { Bytes::from_controller(Box::new(BorrowedSlice { ptr: bytes.as_ptr(), len }), len) }
}

impl AllocationController for BorrowedSlice {
    fn alloc_align(&self) -> usize {
        1usize << (self.ptr as usize).trailing_zeros().min(12)
    }
    fn property(&self) -> AllocationProperty {
        AllocationProperty::Other
    }
    unsafe fn memory_mut(&mut self, _policy: AccessPolicy) -> Result<&mut [MaybeUninit<u8>], AccessError> {
        Err(AccessError::Read("read-only file mapping".into()))
    }
    fn memory(&self, _policy: AccessPolicy) -> Result<&[MaybeUninit<u8>], AccessError> {
        // SAFETY: inside the live mapping (caller's contract).
        Ok(unsafe { std::slice::from_raw_parts(self.ptr as *const MaybeUninit<u8>, self.len) })
    }
}

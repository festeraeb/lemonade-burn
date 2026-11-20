use super::{
    graph::{GraphRecording, ReplayDispatch, ReplayTask, WgpuGraph},
    mem_manager::{self, AuxiliaryMemory},
    poll::WgpuPoll,
    timings::{QueryProfiler, TimestampQuerySetBudget, TimestampSampling},
};
use crate::compute::copies::WgpuCopies;
use crate::compute::device_poison::PoisonWatch;
use crate::{
    WgpuResource, WgpuStorage,
    controller::WgpuAllocController,
    schedule::{Addresses, ScheduleTask},
};
use core::iter;
#[cfg(renderdoc)]
use core::{cell::LazyCell, ptr::null};
use cubecl_common::{
    bytes::Bytes,
    profile::{ProfileDuration, TimingMethod},
};
use cubecl_core::{
    CubeCount, MemoryConfiguration,
    server::{BufferBinding, DeviceCaptures, IoError, ProfileError, ProfilingToken, ServerError},
    zspace::Shape,
};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::future::{self, DynFut};
use cubecl_environment::stream::StreamId;
#[cfg(renderdoc)]
use cubecl_environment::sync::Mutex;
use cubecl_ir::MemoryDeviceProperties;
use cubecl_server::memory_management::relocation::{
    RelocatableStream, RelocationNeed, RelocationReason,
};
use cubecl_server::memory_management::{Cleanup, MemoryManagement};
use cubecl_server::{
    logging::ServerLogger,
    memory_management::{ErrorGraph, FailureId, ManagedMemoryHandle, SharedMemoryBindings},
    metadata_cache::{MetadataCachePolicy, MetadataInfoCache},
    stream::{StreamCapture, StreamMemory},
    timestamp_profiler::TimestampProfiler,
};
#[cfg(renderdoc)]
use renderdoc::{RenderDoc, V100};
use std::{future::Future, num::NonZero, pin::Pin, sync::Arc};
use wgpu::ComputePipeline;

#[cfg(renderdoc)]
thread_local! {
    static RENDERDOC: LazyCell<Option<Mutex<RenderDoc<V100>>>> = LazyCell::new(|| RenderDoc::new().ok().map(Mutex::new));
}

#[derive(Debug)]
enum Timings {
    /// Device timing the stream has not claimed yet. A query-set budget slot is claimed by the
    /// stream's first window: Metal caps the slots per device below the stream count, and a
    /// stream that never profiles must not hold one another stream then cannot get.
    Unclaimed {
        budget: Arc<TimestampQuerySetBudget>,
        sampling: TimestampSampling,
    },
    // Boxed: `QueryProfiler` is much larger than `TimestampProfiler`
    // (clippy::large_enum_variant).
    Device(Box<QueryProfiler>),
    System(TimestampProfiler),
}

impl Timings {
    fn system() -> Self {
        if cfg!(target_family = "wasm") {
            // On WASM, there's not much we can do here anymore. This should be very rare however,
            // all modern GPU's support timestamp queries.
            panic!(
                "Cannot profile on web assembly without timestamp_query feature as it requires blocking."
            );
        }
        Timings::System(TimestampProfiler::default())
    }
}

/// Persistent upload ring (vendored patch, off unless `CUBECL_WGPU_RING_MB` > 0).
///
/// `queue.write_buffer` makes a fresh staging buffer per call inside wgpu-core (an allocation, a
/// first-touch of its pages and a free after the submission), and cubecl submits the queue before
/// every write. This ring owns a fixed set of `MAP_WRITE | COPY_SRC` buffers that stay allocated:
/// a write memcpys into a mapped slot, unmaps it and records `copy_buffer_to_buffer` into the
/// stream's own encoder (so it is ordered with the compute passes, no queue submit before it); after
/// the submission that used the slot, `map_async(Write)` hands it back.
#[derive(Debug)]
struct UploadRing {
    bufs: Vec<wgpu::Buffer>,
    slot_bytes: u64,
    min_bytes: usize,
    /// Slots mapped and free to write.
    ready: Arc<std::sync::Mutex<Vec<usize>>>,
    /// Slots written and referenced by the encoder that has not been submitted yet.
    in_encoder: Vec<usize>,
    /// `CUBECL_WGPU_RING_COPY=nt`: copy with non-temporal 16-byte stores (the mapped memory of a
    /// discrete card's upload heap is write-combined; `rep movsb` into it is slow on older Xeons).
    nt_copy: bool,
    /// `CUBECL_WGPU_RING_THREADS`: threads that share one slot's memcpy (the caller plus helpers).
    /// One thread copies into a discrete card's write-combined upload memory at 1-3 GB/s on an
    /// E5 v2 Xeon; the line-fill buffers, not the DRAM, are the limit, so more threads add up.
    threads: usize,
    /// Counters for `CUBECL_WGPU_RING_STATS`.
    writes: u64,
    bytes: u64,
    waits: u64,
    wait_us: u64,
    memcpy_us: u64,
}

impl UploadRing {
    fn from_env(device: &wgpu::Device) -> Option<Self> {
        let mb: u64 = std::env::var("CUBECL_WGPU_RING_MB").ok()?.trim().parse().ok()?;
        if mb == 0 {
            return None;
        }
        let slot_kb: u64 = std::env::var("CUBECL_WGPU_RING_SLOT_KB")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(2048);
        let min_kb: usize = std::env::var("CUBECL_WGPU_RING_MIN_KB")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(64);
        let slot_bytes = (slot_kb.max(4) * 1024).next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
        let n = ((mb * 1024 * 1024) / slot_bytes).max(2) as usize;
        let bufs: Vec<wgpu::Buffer> = (0..n)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("CubeCL upload ring"),
                    size: slot_bytes,
                    usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: true,
                })
            })
            .collect();
        let ready = Arc::new(std::sync::Mutex::new((0..n).collect::<Vec<_>>()));
        Some(Self {
            bufs,
            slot_bytes,
            min_bytes: min_kb * 1024,
            ready,
            in_encoder: Vec::new(),
            nt_copy: std::env::var("CUBECL_WGPU_RING_COPY").map(|v| v == "nt").unwrap_or(false),
            threads: std::env::var("CUBECL_WGPU_RING_THREADS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1usize).clamp(1, 16),
            writes: 0,
            bytes: 0,
            waits: 0,
            wait_us: 0,
            memcpy_us: 0,
        })
    }
}

/// Copy `src` into `dst` on `threads` threads (the caller is one of them), in 64-byte multiples.
fn copy_parallel(dst: &mut [u8], src: &[u8], threads: usize, nt: bool) {
    let n = src.len();
    let one = |d: &mut [u8], s: &[u8]| {
        if nt {
            copy_stream(d, s)
        } else {
            d.copy_from_slice(s)
        }
    };
    if threads <= 1 || n < 256 * 1024 {
        return one(dst, src);
    }
    let part = (n / threads).next_multiple_of(4096).min(n);
    std::thread::scope(|sc| {
        let mut d_rest = dst;
        let mut s_rest = src;
        let mut mine: Option<(&mut [u8], &[u8])> = None;
        while !s_rest.is_empty() {
            let take = part.min(s_rest.len());
            let (d, dr) = d_rest.split_at_mut(take);
            let (s, sr) = s_rest.split_at(take);
            d_rest = dr;
            s_rest = sr;
            if s_rest.is_empty() {
                mine = Some((d, s));
            } else {
                sc.spawn(move || one(d, s));
            }
        }
        if let Some((d, s)) = mine {
            one(d, s);
        }
    });
}

/// Copy `src` into `dst` (same length) with non-temporal stores where the CPU is x86_64.
#[cfg(target_arch = "x86_64")]
fn copy_stream(dst: &mut [u8], src: &[u8]) {
    use std::arch::x86_64::*;
    assert_eq!(dst.len(), src.len());
    let n = src.len();
    let (d, s) = (dst.as_mut_ptr(), src.as_ptr());
    let mut i = 0usize;
    // SAFETY: SSE2 is baseline on x86_64; all accesses stay inside both slices; the streaming
    // store needs a 16-byte aligned destination, which is checked (else the plain copy runs).
    unsafe {
        if (d as usize) % 16 == 0 {
            while i + 64 <= n {
                let a = _mm_loadu_si128(s.add(i) as *const __m128i);
                let b = _mm_loadu_si128(s.add(i + 16) as *const __m128i);
                let c = _mm_loadu_si128(s.add(i + 32) as *const __m128i);
                let e = _mm_loadu_si128(s.add(i + 48) as *const __m128i);
                _mm_stream_si128(d.add(i) as *mut __m128i, a);
                _mm_stream_si128(d.add(i + 16) as *mut __m128i, b);
                _mm_stream_si128(d.add(i + 32) as *mut __m128i, c);
                _mm_stream_si128(d.add(i + 48) as *mut __m128i, e);
                i += 64;
            }
        }
        if i < n {
            core::ptr::copy_nonoverlapping(s.add(i), d.add(i), n - i);
        }
        _mm_sfence();
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn copy_stream(dst: &mut [u8], src: &[u8]) {
    dst.copy_from_slice(src);
}

#[derive(Debug)]
pub struct WgpuStream {
    /// The memory every buffer a user allocates lives in.
    pub memory: MemoryManagement<WgpuStorage>,
    /// The memories the stream keeps for its own reads and launches.
    pub auxiliary: AuxiliaryMemory,
    pub device: wgpu::Device,
    /// Whether the device is poisoned, shared with every other stream on the device.
    poison: PoisonWatch,
    compute_pass: Option<wgpu::ComputePass<'static>>,
    timings: Timings,
    tasks_count: usize,
    tasks_max: usize,
    queue: wgpu::Queue,
    encoder: wgpu::CommandEncoder,
    poll: WgpuPoll,
    submission_load: SubmissionLoad,
    /// Number of consecutive `write_buffer` calls without a `queue.submit()`.
    /// Used to prevent wgpu staging buffer pool exhaustion during bulk writes
    /// (e.g. model loading with hundreds of tensors).
    pending_write_count: usize,
    /// Cross-stream input bindings for tasks recorded but not yet submitted.
    /// Kept alive here until the next `flush` ties their release to the submission's completion.
    /// See [`ScheduleTask::Execute::pins`](crate::schedule::ScheduleTask).
    shared_bindings: SharedMemoryBindings,
    /// Reusable per-launch info uniforms (kernel shapes/strides/scalars), keyed
    /// by the exact info words they were built from — same scheme as the CUDA
    /// and HIP servers. A hit reuses the already-uploaded uniform buffer, so a
    /// stable-shape launch costs no uniform reservation and no
    /// `queue.write_buffer`. The cached [`ManagedMemoryHandle`] keeps the slice
    /// reserved past the per-flush release in
    /// [`AuxiliaryMemory::release_uniforms`].
    pub(crate) info_cache: MetadataInfoCache<(ManagedMemoryHandle, WgpuResource)>,
    /// This stream's position in the graph-capture lifecycle (see
    /// [`StreamCapture`]). Enforces the ordered `graph_prepare` →
    /// `begin_capture` → `end_capture` transitions; while recording, enqueued
    /// launches append to `recording` instead of the encoder.
    pub(crate) capturing: StreamCapture,
    /// The launches recorded since `begin_capture`, drained into a
    /// [`WgpuGraph`] at `end_capture`.
    recording: GraphRecording,
    /// Vendored patch: persistent upload ring for host writes (see [`UploadRing`]).
    ring: Option<UploadRing>,
    /// Vendored patch: wait for read-backs and syncs on the calling thread instead of the poll
    /// thread (`CUBECL_WGPU_POLL_CALLER=1`).
    poll_caller: bool,
}

impl StreamMemory for WgpuStream {
    fn failure(&self, binding: &BufferBinding) -> Option<FailureId> {
        self.memory.failure(&binding.memory, binding.range())
    }

    fn taint(&mut self, binding: &BufferBinding, failure: FailureId, failures: &mut ErrorGraph) {
        self.memory
            .taint(&binding.memory, binding.range(), failure, failures)
    }

    fn written(&mut self, binding: &BufferBinding, failures: &mut ErrorGraph) {
        self.memory
            .written(&binding.memory, binding.range(), failures)
    }
}

impl RelocatableStream for WgpuStream {
    fn recording(&self) -> bool {
        self.capturing.device().any_recording()
    }

    fn has_outdated(&self) -> bool {
        self.memory.has_outdated()
    }

    fn relocation_need(&self) -> RelocationNeed {
        self.memory.relocation_need()
    }

    fn bytes_allocated(&self) -> u64 {
        self.memory.bytes_allocated() + self.auxiliary.bytes_allocated()
    }

    fn relocate(&mut self, reason: RelocationReason, failures: &mut ErrorGraph) {
        let mut copier = WgpuCopies::new(self.device.clone(), self.queue.clone());
        self.memory.relocate(&mut copier, reason, failures);
    }

    fn cleanup_memory(&mut self, failures: &mut ErrorGraph) {
        self.memory.cleanup(Cleanup::Explicit, failures);
        self.auxiliary.cleanup_uniforms();
    }
}

impl WgpuStream {
    /// The resource the bytes `binding` names live in.
    pub(crate) fn resource(&mut self, binding: BufferBinding) -> Result<WgpuResource, IoError> {
        self.memory
            .get_resource(binding.memory, binding.offset_start, binding.offset_end)
    }

    /// Creates a new WGPU stream.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        timing_method: TimingMethod,
        timing_budget: Arc<TimestampQuerySetBudget>,
        timestamp_sampling: TimestampSampling,
        tasks_max: usize,
        logger: Arc<ServerLogger>,
        use_vulkan_compiler: bool,
        captures: &DeviceCaptures,
        poison: PoisonWatch,
    ) -> Self {
        let timings = match timing_method {
            TimingMethod::Device => Timings::Unclaimed {
                budget: timing_budget,
                sampling: timestamp_sampling,
            },
            TimingMethod::System => Timings::system(),
        };

        #[cfg(renderdoc)]
        RENDERDOC.with(|renderdoc| {
            if let Some(renderdoc) = &**renderdoc {
                let mut renderdoc = renderdoc.lock();
                renderdoc.start_frame_capture(null(), null());
            }
        });

        let poll = WgpuPoll::new(device.clone());
        let device_for_ring = device.clone();

        let memory = mem_manager::main_memory(
            &device,
            &memory_properties,
            memory_config,
            logger.clone(),
            use_vulkan_compiler,
        );
        let auxiliary =
            AuxiliaryMemory::new(&device, &memory_properties, logger, use_vulkan_compiler);

        Self {
            memory,
            auxiliary,
            compute_pass: None,
            timings,
            encoder: {
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("CubeCL Tasks Encoder"),
                })
            },
            device,
            poison,
            queue,
            tasks_count: 0,
            tasks_max,
            poll,
            submission_load: SubmissionLoad::default(),
            pending_write_count: 0,
            shared_bindings: SharedMemoryBindings::default(),
            // Tighter entry cap than the default policy: the uniforms pool is
            // bucketed exclusive pages with a 32 KiB minimum, so every cached
            // entry pins a whole page (512 × 32 KiB ≈ 16 MiB worst case) —
            // unlike CUDA/HIP where an entry is a small dynamic-pool slice.
            info_cache: MetadataInfoCache::new(MetadataCachePolicy::new(512, 2048)),
            capturing: StreamCapture::new(captures),
            recording: GraphRecording::default(),
            ring: UploadRing::from_env(&device_for_ring),
            poll_caller: std::env::var("CUBECL_WGPU_POLL_CALLER").map(|v| v == "1").unwrap_or(false),
        }
    }

    /// Refuse `operation` while a capture is recording on this stream.
    ///
    /// A software graph records dispatches and nothing else, so everything a
    /// launch is not — a read, a sync, a profile, a host write, a replay — has
    /// no recorded form and must not silently do nothing. One place builds the
    /// refusal so every caller reports the same thing; whether that refusal is
    /// returned to the caller or landed on the buffers it left untouched is
    /// the caller's call.
    ///
    /// # Errors
    ///
    /// Fails while [`StreamCapture::is_recording`] holds, naming `operation`.
    pub(crate) fn reject_while_recording(&self, operation: &str) -> Result<(), ServerError> {
        if !self.capturing.is_recording() {
            return Ok(());
        }
        Err(ServerError::graph_state(format!(
            "{operation}: a wgpu capture window records dispatches only, so this operation \
             cannot be part of a graph"
        )))
    }

    /// Queue an error raised inside a capture window, for the stream that owns
    /// the capture to surface when it calls `end_capture`.
    ///
    /// Only the capture's own flush may take it: several logical streams share
    /// this backend stream, and neither `flush` nor `sync` is refused while a
    /// capture records, so a shared entry would be drained by whichever
    /// neighbour flushes first — failing that neighbour on work it never
    /// issued, and leaving `end_capture` to seal a graph with the failed
    /// dispatch silently missing.
    ///
    /// Falls back to shared outside a window, which the callers make
    /// unreachable: each raises its error only while recording.
    ///
    /// Only the report lives here. What the rejected work was going to write
    /// is tainted at the raise site, on the allocations themselves.
    fn capture_error(&mut self, error: ServerError) {
        match self.capturing.is_recording() {
            true => self.capturing.fail(error),
            // Outside a window there is nothing to doom and nothing to claim
            // that the raise site has not claimed already, so the log is the
            // whole report.
            false => log::warn!("{error}"),
        }
    }

    /// Mark every open profile invalid: a failure inside a profiling window
    /// invalidates the measurement. A no-op with no profile open.
    pub fn profile_failure(&mut self, error: &ServerError) {
        self.profile_error(error.into());
    }

    /// Enqueue a [`ScheduleTask`] on this stream.
    ///
    /// # Arguments
    ///
    /// * `task` - The task to execute.
    pub fn enqueue_task(&mut self, task: ScheduleTask, failures: &mut ErrorGraph) {
        match task {
            ScheduleTask::Write {
                data,
                buffer,
                handle,
            } => {
                // Defensive: the server already rejects writes while recording,
                // and `begin_capture` drains the queue, so none should reach here.
                if let Err(err) = self.reject_while_recording("write") {
                    // Taint what this stream's own manager holds; a destination
                    // owned by another stream is the server-side rejection's to
                    // taint, and that rejection comes first.
                    let failure = failures.insert(err.clone());
                    self.memory
                        .taint(&handle.memory, handle.range(), failure, failures);
                    failures.prune(failure);
                    self.capture_error(err);
                    return;
                }
                if self.ring_write(&buffer, &data) {
                    return;
                }
                // It is important to flush before writing, as the write operation is inserted
                // into the QUEUE not the encoder. We want to make sure all outstanding work
                // happens _before_ the write operation.
                self.submit();
                if !self.write_mapped(&buffer, &data) {
                    self.write_to_buffer(&buffer, &data);
                }
            }
            ScheduleTask::Execute {
                pipeline,
                count,
                resources,
                mut shared_inputs,
            } => {
                // The capture lifecycle drives the info cache: while a graph is
                // prepared or recording, every buffer is cached, none is
                // evicted, and touched entries are pinned to the graph being
                // built (see [`StreamCapture::cache_mode`]). Set on the
                // launch path, before anything resolves an info buffer, as the
                // hardware backends do.
                self.info_cache.mode(self.capturing.cache_mode());

                if self.capturing.is_recording() {
                    // Cross-stream input pins belong to the graph for its whole
                    // lifetime, not to the next submission.
                    self.recording
                        .shared
                        .bindings
                        .append(&mut shared_inputs.bindings);
                    let (resources, custom_handles, addresses) = resources.into_resources(self);
                    self.record_pipeline(pipeline, &resources, &custom_handles, addresses, &count);
                    return;
                }
                // Drain into the stream's pending pins; the handle returns its buffer to the
                // server pool when it drops at the end of this arm.
                self.shared_bindings
                    .bindings
                    .append(&mut shared_inputs.bindings);
                let (resources, custom_handles, addresses) = resources.into_resources(self);
                self.register_pipeline(pipeline, &resources, &custom_handles, addresses, &count);
            }
        }
    }

    /// Read multiple buffers lazily to [Bytes], potentially using pinned memory.
    ///
    /// # Arguments
    ///
    /// * `self` - The current stream.
    /// * `descriptors` - A vector of copy descriptors specifying the source data.
    ///
    /// # Returns
    ///
    /// A [Result] containing a vector of [Bytes] with the copied data, or an [`IoError`] if any copy fails.
    pub fn read_resources(
        &mut self,
        descriptors: Vec<(WgpuResource, Shape, usize)>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        self.compute_pass = None;
        let mut staging_info = Vec::with_capacity(descriptors.len());
        let mut callbacks = Vec::with_capacity(descriptors.len());

        for (resource, shape, elem_size) in descriptors {
            let size = shape.iter().product::<usize>() * elem_size;

            // Zero-sized resources don't need a GPU copy.
            if resource.size == 0 {
                staging_info.push(None);
                continue;
            }

            // Copying into a buffer has to be 4 byte aligned. We can safely do so, as
            // memory is 32 bytes aligned (see WgpuStorage).
            let align = wgpu::COPY_BUFFER_ALIGNMENT;
            let aligned_len = resource.size.div_ceil(align) * align;
            let (staging, binding) = self.auxiliary.reserve_staging(aligned_len).unwrap();

            self.tasks_count += 1;
            self.encoder.copy_buffer_to_buffer(
                &resource.buffer,
                resource.offset,
                &staging.buffer,
                0,
                aligned_len,
            );
            staging_info.push(Some((staging, binding, size)));
        }

        // Flush all commands to the queue, so GPU gets started on copying to the
        // staging buffer. This is also where accumulated stream errors resurface:
        // a kernel that failed at launch (e.g. a compilation error) never wrote
        // the buffers this read is about to return, so returning bytes instead of
        // the error would silently hand back stale memory.
        if let Err(err) = self.flush(stream_id) {
            return Box::pin(async move { Err(err) });
        }
        // A poisoned device completes the copy as if it ran.
        // Return the poisoning instead.
        if let Err(err) = self.poison.check() {
            return Box::pin(async move { Err(err) });
        }

        for entry in staging_info.iter() {
            if let Some((staging, _binding, _size)) = entry {
                let (sender, receiver) = cubecl_environment::future::channel::bounded(1);
                staging
                    .buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |v| {
                        // This might fail if the channel is closed (eg. the future is dropped).
                        // This is fine, just means results aren't needed anymore.
                        let _ = sender.try_send(v);
                    });

                callbacks.push(Some(receiver));
            } else {
                callbacks.push(None);
            }
        }

        let poll = if self.poll_caller { None } else { Some(self.poll.start_polling()) };
        let caller_device = if self.poll_caller { Some(self.device.clone()) } else { None };
        let poison = self.poison.clone();

        Box::pin(async move {
            // vendored patch: wait on this (the calling) thread, one wake-up instead of two
            if let Some(device) = &caller_device {
                if let Err(e) = device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                }) {
                    log::warn!("wgpu: caller-side wait for a read failed ({e})");
                }
            }
            for receiver in callbacks.iter().flatten() {
                let mapped = receiver.recv().await;
                // The device can be poisoned while the copy is in flight.
                poison.check()?;
                match mapped {
                    Ok(Ok(())) => {}
                    Ok(Err(err)) => {
                        return Err(ServerError::Generic {
                            reason: format!("the staging buffer could not be mapped: {err}"),
                            backtrace: BackTrace::capture(),
                        });
                    }
                    Err(_) => {
                        return Err(ServerError::Generic {
                            reason: "the staging buffer's map never completed".into(),
                            backtrace: BackTrace::capture(),
                        });
                    }
                }
            }

            // Can stop polling now.
            core::mem::drop(poll);

            let result = {
                staging_info
                    .into_iter()
                    .map(|entry| {
                        if let Some((staging, binding, size)) = entry {
                            let controller =
                                Box::new(WgpuAllocController::init(binding, staging.buffer));
                            // SAFETY: The binding has initialized memory for at least `size` bytes.
                            unsafe { Bytes::from_controller(controller, size) }
                        } else {
                            Bytes::from_bytes_vec(vec![])
                        }
                    })
                    .collect()
            };

            Ok(result)
        })
    }

    /// Device timing needs a counter sample buffer per query set, capped per device on Metal.
    /// A stream claims a budget slot at its first window (lock-free); if none is free, it falls
    /// back to the system timer so the device never exceeds the hardware limit.
    fn claim_timings(&mut self) {
        let Timings::Unclaimed { budget, sampling } = &self.timings else {
            return;
        };
        self.timings = match budget.try_acquire() {
            true => Timings::Device(Box::new(QueryProfiler::new(
                &self.queue,
                &self.device,
                budget.clone(),
                *sampling,
            ))),
            false => Timings::system(),
        };
    }

    // Bit silly but needed to make the borrow checker happy.
    fn system_profiler(&mut self) -> &mut TimestampProfiler {
        let Timings::System(timing) = &mut self.timings else {
            panic!("Unexpected timings type");
        };
        timing
    }

    pub fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        self.claim_timings();
        match &self.timings {
            Timings::System(_) => cubecl_environment::future::block_on(self.sync(stream_id))?,
            Timings::Device(query) if query.sampling().drains_before_window() => {
                self.flush(stream_id)?;
                // Waits like system timing's sync, for the reason `drains_before_window` gives:
                // work still running when the window's first pass starts is timed as its own.
                if let Err(err) = self.device.poll(wgpu::PollType::wait_indefinitely()) {
                    log::warn!("waiting for the work ahead of a profiled window: {err}");
                }
            }
            Timings::Device(_) | Timings::Unclaimed { .. } => self.flush(stream_id)?,
        }

        match &mut self.timings {
            Timings::Unclaimed { .. } => unreachable!("claimed above"),
            Timings::System(_) => {
                let profiler = self.system_profiler();
                Ok(profiler.start())
            }
            Timings::Device(query) => {
                self.compute_pass = None;
                let token = query.start_profile();
                Ok(token)
            }
        }
    }

    pub fn profile_error(&mut self, error: ProfileError) {
        match &mut self.timings {
            Timings::Unclaimed { .. } => {}
            Timings::Device(profiler) => {
                profiler.error(error);
            }
            Timings::System(profiler) => {
                profiler.error(error);
            }
        }
    }

    pub fn end_profile(
        &mut self,
        token: ProfilingToken,
        stream_id: StreamId,
    ) -> Result<ProfileDuration, ProfileError> {
        match &mut self.timings {
            Timings::Unclaimed { .. } => Err(ProfileError::NotRegistered {
                backtrace: BackTrace::capture(),
            }),
            Timings::System(..) => {
                // Nb: WASM _has_ to use device timing and will panic here if query timestamps are not supported.
                let result = future::block_on(self.sync(stream_id));
                let profiler = self.system_profiler();

                if let Err(err) = result {
                    profiler.error(ProfileError::Server(Box::new(err)));
                }
                profiler.stop(token)
            }
            Timings::Device(..) => {
                let poll = self.poll.start_polling();
                self.compute_pass = None;

                // Submit commands needed for profiling.
                let readback = {
                    let Timings::Device(timing) = &mut self.timings else {
                        return Err(ProfileError::Unknown {
                            reason: "Unexpected timings type".to_string(),
                            backtrace: BackTrace::capture(),
                        });
                    };
                    timing.stop_profile_setup(token, &self.device)?
                };

                // This flushes the queue to execute the encoder write command to write the
                // timings.
                self.tasks_count += 1;
                let result = self.flush(stream_id);

                let Timings::Device(timing) = &mut self.timings else {
                    return Err(ProfileError::Unknown {
                        reason: "Unexpected timings type".to_string(),
                        backtrace: BackTrace::capture(),
                    });
                };

                match result {
                    Ok(_) => {
                        let map_buffer = readback.map(|readback| {
                            if timing.sampling().resolves_after_completion() {
                                // Blocks like system timing's sync at a window's end. Resolving
                                // later without waiting is not an option: the window's query
                                // sets return to the pool here, and a pass that reuses one
                                // before the resolve runs would overwrite the window's ends.
                                if let Err(err) =
                                    self.device.poll(wgpu::PollType::wait_indefinitely())
                                {
                                    log::warn!("waiting for a profiled window to complete: {err}");
                                }
                            }
                            self.queue.submit([readback.commands]);
                            readback.map_buffer
                        });
                        timing.stop_profile(map_buffer, poll)
                    }
                    Err(err) => {
                        // Dropped rather than mapped: the flush this window
                        // needed has failed, so a map requested here may never
                        // complete, and the map's callback is what releases the
                        // poll handle. The query sets were already given back in
                        // `stop_profile_setup`, so there is nothing else to undo.
                        drop(readback);
                        drop(poll);
                        Err(ProfileError::Server(Box::new(err)))
                    }
                }
            }
        }
    }

    /// Drop `token`'s window without measuring it.
    ///
    /// Neither arm syncs or flushes, which is the whole difference from
    /// [`end_profile`](Self::end_profile): the device arm hands the query set
    /// back, the system arm forgets the start instant, and the work in the
    /// window carries on as if it had never been bracketed.
    pub fn abandon_profile(&mut self, token: ProfilingToken) {
        match &mut self.timings {
            Timings::Unclaimed { .. } => {}
            Timings::System(profiler) => profiler.abandon(token),
            Timings::Device(timing) => timing.abandon_profile(token),
        }
    }

    pub fn sync(
        &mut self,
        stream_id: StreamId,
    ) -> Pin<Box<dyn Future<Output = Result<(), ServerError>> + Send + 'static>> {
        let error_scope = self.device.push_error_scope(wgpu::ErrorFilter::Internal);

        let flush_error = self.flush(stream_id).err();

        let queue = self.queue.clone();
        let error_future = error_scope.pop();
        let poll = if self.poll_caller { None } else { Some(self.poll.start_polling()) };
        let caller_device = if self.poll_caller { Some(self.device.clone()) } else { None };
        let poison = self.poison.clone();

        Box::pin(async move {
            let (sender, receiver) = cubecl_environment::future::channel::bounded::<()>(1);
            queue.on_submitted_work_done(move || {
                // Signal that we're done.
                let _ = sender.try_send(());
                core::mem::drop(poll);
            });
            if let Some(device) = &caller_device {
                if let Err(e) = device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                }) {
                    log::warn!("wgpu: caller-side wait for a sync failed ({e})");
                }
            }
            let _ = receiver.recv().await;

            poison.check()?;

            if let Some(error) = error_future.await {
                return Err(ServerError::Generic {
                    reason: format!("{error}"),
                    backtrace: BackTrace::capture(),
                });
            }

            match flush_error {
                Some(err) => Err(err),
                None => Ok(()),
            }
        })
    }

    pub(crate) fn create_uniform(&mut self, data: &[u8]) -> WgpuResource {
        let (handle, resource) = self.auxiliary.reserve_uniform(data.len() as u64);
        // A uniform created inside a recording window (e.g. a Vulkan address
        // buffer) is referenced by the recorded task on every replay, so it is
        // pinned to the recording instead of released on the next flush.
        if self.capturing.is_recording() {
            self.recording.uniform_pins.push(handle);
        }
        self.write_to_buffer(&resource, data);
        resource
    }

    /// Stage the metadata info `words` into a uniform, reusing a cached one
    /// when a launch has already staged these exact words. The info is
    /// read-only metadata (no buffer bindings), so sharing it across launches —
    /// even of different kernels — is sound; see
    /// [`MetadataInfoCache`](cubecl_server::metadata_cache::MetadataInfoCache).
    /// `words` is taken by value so a miss hands it to the cache as the key
    /// without cloning. A hit's buffer bytes always equal the key bytes, so it
    /// is byte-identical to what the miss path would have built and uploaded.
    pub(crate) fn info_uniform(&mut self, words: Vec<u64>) -> WgpuResource {
        let size = core::mem::size_of_val(words.as_slice());
        if !self.info_cache.should_cache(size) {
            return self.create_uniform(bytemuck::cast_slice(&words));
        }
        if let Some((_handle, resource)) = self.info_cache.get(&words) {
            return resource;
        }
        let (handle, resource) = self.auxiliary.reserve_uniform(size as u64);
        self.write_to_buffer(&resource, bytemuck::cast_slice(&words));
        self.info_cache.insert(words, (handle, resource.clone()));
        resource
    }

    /// Write `data` into `resource` through the upload ring. Returns `false` to fall back to the
    /// queue path (ring off, small write, odd size or alignment, destination too small).
    #[cfg(not(target_family = "wasm"))]
    fn ring_write(&mut self, resource: &WgpuResource, data: &[u8]) -> bool {
        let copy_align = wgpu::COPY_BUFFER_ALIGNMENT;
        let Some(ring) = self.ring.as_ref() else {
            return false;
        };
        if data.len() < ring.min_bytes
            || (data.len() as u64) % copy_align != 0
            || resource.offset % copy_align != 0
            || data.len() as u64 > resource.size
        {
            return false;
        }
        let slot_bytes = ring.slot_bytes as usize;
        let mut done = 0usize;
        while done < data.len() {
            let n = (data.len() - done).min(slot_bytes);
            let slot = self.ring_acquire();
            let t0 = std::time::Instant::now();
            let ring = self.ring.as_mut().expect("ring");
            {
                let mut view = ring.bufs[slot].slice(0..n as u64).get_mapped_range_mut().expect("ring slot is mapped");
                let dst = view.slice(..).as_raw_element_ptr();
                // SAFETY: the view maps `n` writable bytes
                let dst = unsafe { core::slice::from_raw_parts_mut(dst.as_ptr(), n) };
                copy_parallel(dst, &data[done..done + n], ring.threads, ring.nt_copy);
            }
            ring.bufs[slot].unmap();
            ring.memcpy_us += t0.elapsed().as_micros() as u64;
            ring.writes += 1;
            ring.bytes += n as u64;
            if ring.writes % 300 == 0 && std::env::var_os("CUBECL_WGPU_RING_STATS").is_some() {
                eprintln!(
                    "wgpu-ring: {} writes {:.1} MB, memcpy {:.2} ms total ({:.2} GB/s), {} waits {:.1} ms",
                    ring.writes,
                    ring.bytes as f64 / 1e6,
                    ring.memcpy_us as f64 / 1e3,
                    ring.bytes as f64 / ring.memcpy_us.max(1) as f64 / 1e3,
                    ring.waits,
                    ring.wait_us as f64 / 1e3
                );
            }
            // a copy cannot be recorded inside an open compute pass
            self.compute_pass = None;
            let ring = self.ring.as_mut().expect("ring");
            self.encoder.copy_buffer_to_buffer(
                &ring.bufs[slot],
                0,
                &resource.buffer,
                resource.offset + done as u64,
                n as u64,
            );
            ring.in_encoder.push(slot);
            self.tasks_count += 1;
            done += n;
        }
        true
    }

    #[cfg(target_family = "wasm")]
    fn ring_write(&mut self, _resource: &WgpuResource, _data: &[u8]) -> bool {
        false
    }

    /// A mapped, free ring slot; submits the pending encoder and waits when every slot is busy.
    #[cfg(not(target_family = "wasm"))]
    fn ring_acquire(&mut self) -> usize {
        let mut waited: Option<std::time::Instant> = None;
        loop {
            let got = {
                let ring = self.ring.as_ref().expect("ring");
                ring.ready.lock().expect("ring ready").pop()
            };
            if let Some(slot) = got {
                if let Some(t) = waited {
                    let ring = self.ring.as_mut().expect("ring");
                    ring.waits += 1;
                    ring.wait_us += t.elapsed().as_micros() as u64;
                }
                return slot;
            }
            waited.get_or_insert_with(std::time::Instant::now);
            if !self.ring.as_ref().expect("ring").in_encoder.is_empty() {
                // slots held by the unsubmitted encoder only come back after it is submitted
                self.submit();
            }
            if let Err(e) = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            }) {
                log::warn!("wgpu: ring wait failed ({e})");
            }
        }
    }

    /// After a queue submit: ask for every slot the submission used to be mapped again; the
    /// mapping completes once the GPU is done with it.
    #[cfg(not(target_family = "wasm"))]
    fn ring_arm(&mut self) {
        let Some(ring) = self.ring.as_mut() else {
            return;
        };
        for slot in ring.in_encoder.drain(..) {
            let ready = ring.ready.clone();
            ring.bufs[slot]
                .slice(..)
                .map_async(wgpu::MapMode::Write, move |r| {
                    if r.is_ok() {
                        ready.lock().expect("ring ready").push(slot);
                    }
                });
        }
    }

    #[cfg(target_family = "wasm")]
    fn ring_arm(&mut self) {}

    /// Copies into a mapped buffer, skipping staging. Returns `false` to fall back to the queue.
    ///
    /// Not ordered by the queue, so call it only after a submit, never for uniforms.
    #[cfg(not(target_family = "wasm"))]
    fn write_mapped(&mut self, resource: &WgpuResource, data: &[u8]) -> bool {
        let Some(host_ptr) = resource.host_ptr else {
            return false;
        };
        // The copy below is unchecked; the queue path validates.
        if data.len() as u64 > resource.size {
            return false;
        }

        // A pooled range may have belonged to a tensor whose kernels are still running, so
        // the copy needs an idle queue. Never drain one to get it: the stall costs a
        // pipelined loop the CPU/GPU overlap it lives on, and it buys nothing on a bulk
        // upload, which finds the queue idle anyway because nothing is queued behind what
        // it is loading. So take the mapping when the queue is already free, and leave the
        // write to the queue when it is not.
        match self.device.poll(wgpu::PollType::Poll) {
            Ok(status) if status.wait_finished() => {}
            Ok(_) => return false,
            Err(e) => {
                log::warn!("wgpu: poll before a mapped write failed ({e})");
                return false;
            }
        }

        // SAFETY: The range is in bounds and the queue is idle.
        unsafe {
            let dst = host_ptr.0.as_ptr().add(resource.offset as usize);
            core::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
        true
    }

    #[cfg(target_family = "wasm")]
    fn write_mapped(&mut self, _resource: &WgpuResource, _data: &[u8]) -> bool {
        false
    }

    // Nb: this function submits a command to the _queue_ not to the encoder,
    // so you have to be really careful about the ordering of operations here.
    // Any buffer which has outstanding (not yet flushed) compute work should
    // NOT be copied to.
    fn write_to_buffer(&mut self, resource: &WgpuResource, data: &[u8]) {
        // Nothing to write for zero-sized resources.
        if resource.size == 0 {
            return;
        }

        // Copying into a buffer has to be 4 byte aligned. We can safely do so, as
        // memory is also aligned (see WgpuStorage). Per the WebGPU spec, this
        // just has to be a multiple of 4: https://www.w3.org/TR/webgpu/#dom-gpuqueue-writebuffer
        let copy_align = wgpu::COPY_BUFFER_ALIGNMENT;
        let size = resource.size.next_multiple_of(copy_align);

        if size == data.len() as u64 {
            // write_buffer is the recommended way to write this data, as:
            // - On WebGPU, from WASM, this can save a copy to the JS memory.
            // - On devices with unified memory, this could skip the staging buffer entirely.
            self.queue
                .write_buffer(&resource.buffer, resource.offset, data);
        } else {
            // For sizes not aligned we need to only write a part of the staging buffer, do this
            // with `write_buffer_with`.
            let mut buffer = self
                .queue
                .write_buffer_with(
                    &resource.buffer,
                    resource.offset,
                    NonZero::new(size).unwrap(),
                )
                .expect("Internal error: Failed to call `write_buffer_with`, this likely means no staging buffer could be allocated.");
            buffer.slice(0..data.len()).copy_from_slice(data);
        }

        self.pending_write_count += 1;

        // Prevent wgpu staging buffer pool exhaustion during bulk writes (e.g. model
        // loading with hundreds of tensors). queue.write_buffer() is async — wgpu
        // copies data into an internal staging buffer, then transfers to GPU on the
        // next queue.submit(). Without periodic submits, hundreds of writes accumulate
        // and staging buffers get recycled before the GPU copy completes, silently
        // corrupting early tensors.
        // See: https://github.com/tracel-ai/cubecl/issues/1120
        const MAX_PENDING_WRITES: usize = 64;

        if self.pending_write_count >= MAX_PENDING_WRITES {
            // Submit a fresh, empty command buffer to flush all pending write_buffer work.
            // wgpu flushes its internal staging-buffer copies on any queue.submit(),
            // so we don't need to touch the main compute encoder here.
            let write_flush_encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("CubeCL Write Flush Encoder"),
                    });
            let index = self.queue.submit([write_flush_encoder.finish()]);

            // Wait for the GPU to finish processing these writes before continuing.
            #[cfg(not(target_family = "wasm"))]
            if let Err(e) = self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: None,
            }) {
                log::warn!("wgpu: write flush poll failed ({e})");
            }

            self.pending_write_count = 0;
        }
    }

    fn flush_if_needed(&mut self) {
        // Flush when there are too many tasks, or when too many handles are locked.
        // Locked handles should only accumulate in rare circumstances (where uniforms
        // are being created but no work is submitted).
        if self.tasks_count >= self.tasks_max {
            self.submit();
        }
    }

    /// Submit the queued work and surface nothing.
    ///
    /// For the pooled paths that flush the stream without any logical stream
    /// asking — a full task queue, the ordering barrier before a write, the
    /// scheduler aligning streams. Whatever is queued stays queued, for the
    /// flush of the stream that owns it.
    pub fn submit(&mut self) {
        self.submit_tasks();
        self.collect_validation_errors();
    }

    /// Submit the queued work, then report the device fault if one arrived.
    /// A launch failure is not the flush's to report: it lives on the buffers
    /// the launch left unwritten, and surfaces on any read, sync or check of
    /// them.
    pub fn flush(&mut self, _owner: StreamId) -> Result<(), ServerError> {
        self.submit();
        Ok(())
    }

    fn submit_tasks(&mut self) {
        if self.tasks_count == 0 {
            self.shared_bindings.clear();
            return;
        }

        // End the current compute pass.
        self.compute_pass = None;

        // Submit the pending actions to the queue. This will _first_ submit the
        // pending uniforms copy operations, then the main tasks.
        let tasks_encoder = {
            std::mem::replace(&mut self.encoder, {
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("CubeCL Tasks Encoder"),
                    })
            })
        };

        // This will _first_ fire off all pending write_buffer work.
        let index = self.queue.submit([tasks_encoder.finish()]);
        self.ring_arm();

        // Release cross-stream input pins.
        if !self.shared_bindings.is_empty() {
            let pins = core::mem::take(&mut self.shared_bindings);
            self.queue.on_submitted_work_done(move || {
                drop(pins);
            });
        }

        self.submission_load
            .regulate(&self.device, self.tasks_count, index);

        // The main pool's pages are cleaned up by the server before it
        // reserves, and only while no stream records.
        self.auxiliary.release_uniforms();

        #[cfg(renderdoc)]
        RENDERDOC.with(|renderdoc| {
            if let Some(renderdoc) = &**renderdoc {
                let mut renderdoc = renderdoc.lock();
                renderdoc.end_frame_capture(null(), null());
                renderdoc.start_frame_capture(null(), null());
            }
        });

        self.tasks_count = 0;
        self.pending_write_count = 0;
    }

    /// Drain the driver's validation canary into the log.
    ///
    /// The driver reports these against the device and not against the launch
    /// that caused them, so there is no buffer to claim: whatever the rejected
    /// work was going to write was claimed at the raise site, and this is the
    /// diagnostic for the cases that reach nobody else.
    fn collect_validation_errors(&mut self) {
        #[cfg(feature = "deny-validation-errors")]
        {
            let validation_errors = wgpu_hal::VALIDATION_CANARY.get_and_reset();
            for err in validation_errors.into_iter() {
                log::warn!("wgpu validation: {err}");
            }
        }
    }

    /// Start a new compute pass if needed. The `forget_lifetime` allows
    /// storing this with a 'static lifetime, but the compute pass must
    /// be dropped before the encoder. This isn't unsafe - it's still checked at runtime.
    ///
    /// An associated function over the individual fields (rather than a
    /// `&mut self` method) so callers keep access to their other fields while
    /// the returned pass borrows `compute_pass`.
    fn current_pass<'a>(
        compute_pass: &'a mut Option<wgpu::ComputePass<'static>>,
        timings: &mut Timings,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
    ) -> &'a mut wgpu::ComputePass<'static> {
        compute_pass.get_or_insert_with(|| {
            let writes = match timings {
                Timings::Device(query_time) => query_time.register_profile_device(device),
                Timings::Unclaimed { .. } | Timings::System(_) => None,
            };
            encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: writes,
                })
                .forget_lifetime()
        })
    }

    /// Record one launch into the in-progress recording instead of encoding
    /// it: the counterpart of [`register_pipeline`](Self::register_pipeline)
    /// while a capture is recording. Everything a replay needs is resolved
    /// here, once — the bind group is built, the indirect-dispatch buffer is
    /// resolved — so replaying is nothing but re-encoding prebuilt state.
    fn record_pipeline(
        &mut self,
        pipeline: Arc<ComputePipeline>,
        resources: &[WgpuResource],
        custom_resources: &[WgpuResource],
        immediates: Option<Addresses>,
        dispatch: &CubeCount,
    ) {
        // An empty dispatch is a no-op on the normal path; record nothing.
        if dispatch.is_empty() {
            return;
        }

        let bind_group = (!resources.is_empty()).then(|| {
            let entries = resources
                .iter()
                .enumerate()
                .map(|(i, r)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: r.as_wgpu_bind_resource(),
                })
                .collect::<Vec<_>>();
            let group_layout = pipeline.get_bind_group_layout(0);
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &group_layout,
                entries: &entries,
            })
        });

        if let CubeCount::Dynamic(binding) = dispatch {
            self.capturing
                .touch(binding.stream, binding.memory.descriptor().location());
        }
        let dispatch = match dispatch.clone() {
            CubeCount::Static(x, y, z) => ReplayDispatch::Static(x, y, z),
            CubeCount::Dynamic(binding) => match self.resource(binding) {
                Ok(resource) => ReplayDispatch::Dynamic(resource),
                Err(err) => {
                    // The recording is now incomplete; `end_capture` sees the
                    // doomed window and rejects the capture. It taints nothing
                    // itself: rejecting the capture is what taints the buffers,
                    // and it taints every one the recording was given, this
                    // launch's included.
                    self.capture_error(err.into());
                    return;
                }
            },
        };

        self.recording.tasks.push(ReplayTask {
            pipeline,
            bind_group,
            immediates,
            transitions: custom_resources.to_vec(),
            dispatch,
        });
    }

    /// Move the in-progress recording out of the stream (leaving it empty),
    /// for `end_capture` to seal into a [`WgpuGraph`].
    pub(crate) fn take_recording(&mut self) -> GraphRecording {
        core::mem::take(&mut self.recording)
    }

    /// Re-encode a captured graph's tasks — one dispatch per recorded launch,
    /// prebuilt state only — and let the normal `tasks_max`/submission-load
    /// batching decide when to submit. Fire-and-forget like a launch: the
    /// work lands on this stream's encoder in recorded order.
    pub(crate) fn replay_graph(&mut self, graph: &WgpuGraph) {
        // Consecutive tasks often share a pipeline (decode loops); skip the
        // redundant `set_pipeline`. Pass state does not survive a flush, so
        // the tracking resets whenever the pass was closed.
        let mut last_pipeline: Option<&Arc<ComputePipeline>> = None;

        for task in graph.tasks.iter() {
            if self.compute_pass.is_none() {
                last_pipeline = None;
            }
            let pass = Self::current_pass(
                &mut self.compute_pass,
                &mut self.timings,
                &mut self.encoder,
                &self.device,
            );

            if !last_pipeline.is_some_and(|prev| Arc::ptr_eq(prev, &task.pipeline)) {
                pass.set_pipeline(&task.pipeline);
                last_pipeline = Some(&task.pipeline);
            }
            if let Some(bind_group) = &task.bind_group {
                pass.set_bind_group(0, bind_group, &[]);
            }
            if let Some(immediates) = &task.immediates {
                pass.set_immediates(0, bytemuck::cast_slice(immediates));
            }
            if !task.transitions.is_empty() {
                let buffer_transitions =
                    task.transitions
                        .iter()
                        .map(|resource| wgpu::BufferTransition {
                            buffer: &resource.buffer,
                            state: wgpu::BufferUses::STORAGE_READ_WRITE,
                        });
                pass.transition_resources(buffer_transitions, iter::empty());
            }
            match &task.dispatch {
                ReplayDispatch::Static(x, y, z) => pass.dispatch_workgroups(*x, *y, *z),
                ReplayDispatch::Dynamic(resource) => {
                    pass.dispatch_workgroups_indirect(&resource.buffer, resource.offset)
                }
            }

            self.tasks_count += 1;
            self.flush_if_needed();
        }
    }

    fn register_pipeline(
        &mut self,
        pipeline: Arc<ComputePipeline>,
        resources: &[WgpuResource],
        custom_resources: &[WgpuResource],
        addresses: Option<Addresses>,
        dispatch: &CubeCount,
    ) {
        if dispatch.is_empty() {
            return;
        }

        let entries = resources
            .iter()
            .enumerate()
            .map(|(i, r)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: r.as_wgpu_bind_resource(),
            })
            .collect::<Vec<_>>();

        let pass = Self::current_pass(
            &mut self.compute_pass,
            &mut self.timings,
            &mut self.encoder,
            &self.device,
        );

        self.tasks_count += 1;

        pass.set_pipeline(&pipeline);

        if !resources.is_empty() {
            let group_layout = pipeline.get_bind_group_layout(0);
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &group_layout,
                entries: &entries,
            });

            pass.set_bind_group(0, &bind_group, &[]);
        }

        if let Some(addresses) = addresses {
            pass.set_immediates(0, bytemuck::cast_slice(&addresses));
        }

        if !custom_resources.is_empty() {
            let buffer_transitions =
                custom_resources
                    .iter()
                    .map(|resource| wgpu::BufferTransition {
                        buffer: &resource.buffer,
                        state: wgpu::BufferUses::STORAGE_READ_WRITE,
                    });
            pass.transition_resources(buffer_transitions, iter::empty())
        }

        match dispatch.clone() {
            CubeCount::Static(x, y, z) => {
                pass.dispatch_workgroups(x, y, z);
            }
            CubeCount::Dynamic(binding) => {
                let res = self
                    .memory
                    .get_resource(binding.memory, binding.offset_start, binding.offset_end)
                    .unwrap();
                pass.dispatch_workgroups_indirect(&res.buffer, res.offset);
            }
        }

        self.flush_if_needed();
    }
}

#[cfg(not(target_family = "wasm"))]
mod __submission_load {
    #[derive(Default, Debug)]
    pub enum SubmissionLoad {
        Init {
            last_index: wgpu::SubmissionIndex,
            tasks_count_submitted: usize,
        },
        #[default]
        Empty,
    }

    impl SubmissionLoad {
        pub fn regulate(
            &mut self,
            device: &wgpu::Device,
            tasks_count: usize,
            mut index: wgpu::SubmissionIndex,
        ) {
            match self {
                SubmissionLoad::Init {
                    last_index,
                    tasks_count_submitted,
                } => {
                    *tasks_count_submitted += tasks_count;

                    // Enough to keep the GPU busy.
                    //
                    // - Too much can hang the GPU and create slowdown.
                    // - Too little and GPU utilization is really bad.
                    //
                    // TODO: Could be smarter and dynamic based on stats.
                    const MAX_TOTAL_TASKS: usize = 512;

                    if *tasks_count_submitted >= MAX_TOTAL_TASKS {
                        core::mem::swap(last_index, &mut index);
                        if let Err(e) = device.poll(wgpu::PollType::Wait {
                            submission_index: Some(index),
                            timeout: None,
                        }) {
                            log::warn!(
                                "wgpu: requested wait timed out before the submission was completed during sync. ({e})"
                            )
                        }
                        *tasks_count_submitted = 0;
                    }
                }
                SubmissionLoad::Empty => {
                    *self = Self::Init {
                        last_index: index,
                        tasks_count_submitted: 0,
                    }
                }
            }
        }
    }
}

#[cfg(target_family = "wasm")]
mod __submission_load_wasm {
    #[derive(Default, Debug)]
    pub struct SubmissionLoad;

    impl SubmissionLoad {
        pub fn regulate(
            &mut self,
            _device: &wgpu::Device,
            _tasks_count: usize,
            _index: wgpu::SubmissionIndex,
        ) {
            // Nothing to do.
        }
    }
}

#[cfg(not(target_family = "wasm"))]
use __submission_load::*;
#[cfg(target_family = "wasm")]
use __submission_load_wasm::*;

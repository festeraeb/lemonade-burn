# Vendored cubecl-wgpu 0.11.2 (small patch, all behind env switches, off by default)

Only `src/compute/stream.rs` differs from the crates.io 0.11.2 source. What to upstream:

1. **Upload ring** (`CUBECL_WGPU_RING_MB=N`, slot size `CUBECL_WGPU_RING_SLOT_KB`, default 2048, minimum write
   `CUBECL_WGPU_RING_MIN_KB`, default 64). `queue.write_buffer` makes a fresh staging buffer per call inside
   wgpu-core and cubecl submits the queue before each write. The ring keeps N `MAP_WRITE|COPY_SRC` buffers
   mapped, memcpys into a slot, unmaps, records `copy_buffer_to_buffer` into the stream's own encoder (ordered
   with the compute passes, no submit before it) and re-maps the slot with `map_async(Write)` after the submit.
   `CUBECL_WGPU_RING_THREADS=T` shares one slot's memcpy over T threads (one thread into write-combined upload
   memory is 1-3 GB/s on an E5 v2 Xeon; 4 threads reach ~4 GB/s), `CUBECL_WGPU_RING_COPY=nt` uses 16-byte
   non-temporal stores (no gain measured), `CUBECL_WGPU_RING_STATS=1` prints counters.
2. **Read-back / sync wait on the calling thread** (`CUBECL_WGPU_POLL_CALLER=1`): `read_resources` and `sync`
   call `device.poll(Wait)` in the returned future instead of waking the `WgpuPoll` thread and then a channel.

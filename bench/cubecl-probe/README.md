# CubeCL host-to-device write-path probe (2026-11-03)

Where the ~3x between CubeCL's `create(Bytes)` and raw `cudaMemcpy` goes, measured on
cesarops2 GPU2 (Tesla P100, Gen3 x16). Answer and table: `docs/WORK_LOG.md`, section
"Probe: where the copy path loses 3x". Raw per-call lines: c2 `~/lb-measure/out/probe-b2/`.

- `cubecl-probe.patch`: timers in cubecl-server 0.11.2 `write_to_gpu` (reserve pinned, host
  memcpy, enqueue, drop-queue flush) and cubecl-cuda 0.11.2 `copy_to_device` (the
  `memcpy_htod_async` call, whether the host pointer is page-locked, and with
  `LB_CUBECL_PROBE_SYNC=1` a stream sync right after it, which isolates the DMA), plus the
  two `alloc`s (`cuMemAllocHost`, `cuMemAlloc`). One JSON line per call on stderr when
  `LB_CUBECL_PROBE=1`. Local measurement only: never sent upstream.
- `config.toml`: cargo `[patch.crates-io]` pointing at the patched copies. Rebuild:
  copy the two crates from the cargo registry to `/data/cubecl-probe/`, `git apply` the patch,
  then `cargo --config bench/cubecl-probe/config.toml build --release -p lb-q4-gpu --features cuda --example copy_bench`
  and `git checkout Cargo.lock` afterwards.
- `c2_probe_run.sh`: the three passes on c2 (probe, probe + sync, no probe).
- `c2_pinned_run.sh`: the fix's bench (`pinned.rs` pool on / off, `LB_BENCH_WARM=40`) and
  the cuda parity test on the same card.
- `c2_pool_sha_ab.sh`: same-config greedy-sha A/B of the pool (on, off, on again) with on-demand misses, sized to fit beside a live seat on the card.

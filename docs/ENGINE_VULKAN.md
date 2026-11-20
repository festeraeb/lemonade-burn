# ENGINE_VULKAN: the engine on CubeCL's wgpu/Vulkan runtime (c1: AMD cards)

Branch `fable/engine-123-vulkan-2026-11-15` (off `fable/engine-123-predict-2026-11-15`, 7576600). Date 2026-11-15.
Raw results: `/mnt/fast/eng123/vulkan-results/` (t440) and `/mnt/models/vk/results/` (c1). Tags: [measured] run here,
[estimate] derived from measured pieces, [read] from source.

Test box: c1 (HP ML350p Gen8), 2 x E5-2680 v2 (no AVX2), 149 GB RAM visible at test time, kernel 7.x, amdgpu + RADV (Mesa, Vulkan 1.4),
no ROCm. Vulkan order at test time (read `vulkaninfo` every time, it changes when a card is added): `discrete:0` RX 6700 XT 12 GB
(wave 32/64), `discrete:1` Radeon Pro WX 5100 8 GB (Polaris, wave 64, PCIe 3 x16), `discrete:2` RX 570 4 GB (Polaris). The 6700 XT
holds the `granite-8b-c1` seat, so all throughput numbers below are from the WX 5100 and RX 570 (**correction, phase 2: the WX 5100 also held the live seat `granite-3b-c1`, 4.3 GB of its VRAM, so the WX 5100 numbers in sections 3-5 were taken on a shared card; see section 10**); the 6700 XT got only the
small kernel tests, run beside the live seat. Pinned versions: cubecl 0.11.2, wgpu 30.0.1, burn 0.22.1.

## 0. What happened before

The engine started on wgpu (Step 0 on c3's WX 5100, 2026-10-31, `docs/LFM2_CUBECL.md`, `WORK_LOG.md`); CUDA (`cubecl-cuda`) was added beside
it on 2026-11-03 (commit 603512f) because of the wgpu copy path (1-4.7 GB/s and no pinned memory). Everything since (E1 cache, E2 router,
sync cost, CPU misses, predictor) was developed and measured on CUDA/P100 only. So the question was never "can it run on Vulkan": the
`gpu` feature builds with no CUDA. It is "does the E2 cache path still work and what does it cost there". Answer: it works and is
numerically right, and it is 20-30x slower than the P100 at the same cache budget, for reasons that are mostly wgpu's write and
read-back paths, not the kernels.

## 1. What depends on CUDA, and what does not

Build: `cargo build --release --features gpu -p lb-lfm2 --bins` is the wgpu build (default features, WGSL compiler); `--features cuda`
adds `cubecl/cuda` + `cudarc` (dynamic loading, nothing linked). New: `--features vulkan` on `lb-q4-gpu` selects cubecl's SPIR-V
compiler for Vulkan (section 5).

CUDA-only code, all behind `#[cfg(feature = "cuda")]` [read]:

| what | where | what it does | wgpu status |
|---|---|---|---|
| `cudarc` page-locked staging pool | `lb-q4-gpu/src/pinned.rs`, `Gpu::take_pinned/upload_pinned` | repack target in pinned host memory | absent; wgpu has no pinned memory |
| `cuMemHostRegister(READ_ONLY)` of the GGUF mapping | `registered.rs`, `Gpu::register_mapping`, `moe.rs open_gpu` | expert bytes DMA from the page cache, repack on device | **added a wgpu counterpart** (`borrowed.rs`: borrowed `Bytes`, no pinning) |
| compute capability query | `store.rs cuda_compute_capability`, `Q4Math::pick`, `fp16_pipe` | picks dp4a (sm_61+) / half2 (sm_60) Q4 math | None on wgpu: always `F32` math |
| `Gpu::open_cuda(i)`, second device in one process | `store.rs` | prefill card (`device_prefill.rs attach_prefill_card`), in-box expert cards (`inbox.rs open_card`), `lb-expert-host` (`LB_CUDA_DEVICE`) | **wgpu opens exactly one adapter per process (`LB_WGPU_DEVICE=discrete:N`)**: no multi-card residency on wgpu |
| CUDA graphs | `client.start_capture/stop_capture`, `CubeGraph`, `LB_DD_GRAPH`, `--ab-graph` | replay a token | cubecl-wgpu has `graph.rs` but its header says WebGPU has no re-submittable command buffer; default off in the engine anyway |
| examples `register_bench`, `miss_bench`, `stage_bench` | `lb-q4-gpu/examples` | CUDA path benches | `required-features = ["cuda"]` (sync_floor was too; fixed) |

Goes through CubeCL's backend-neutral layer and runs unchanged on wgpu [measured, parity below]: every `#[cube]` kernel (Q4_0/Q6_K/Q8_0
matvec and rows kernels, gate-up-silu, repack, router_topk, add_rms_norm, qk_norm_rope, attn_decode, weighted sum, conv decode, slot
translate), plane ops (`plane_sum`, plane size read at run time: 64 on Polaris, 32/64 on the 6700 XT, 32 on NVIDIA), shared memory,
the memory pool and persistent allocation (`ExpertSlots`), `client.read/write`, `read_begin/read_finish`, the `flush_queue` trick
(`Client::profile_abandon` compiles and is harmless on wgpu; it does nothing useful there, see 4), scratch recycling, and the E2
layer itself. The launch trace's per-launch device times work on wgpu (timestamp queries).

CubeCL 0.11.2 runtimes [read, Cargo.toml features of `cubecl`]: `cuda`, `hip`/`rocm` (needs ROCm; c1 has none and gfx900 is dropped from
current ROCm), `wgpu` (WGSL compiler; Vulkan, DX12, Metal, GL adapters), `vulkan` = `wgpu-spirv` (wgpu runtime but kernels compiled to
SPIR-V by `cubecl-spirv`, Vulkan only), `metal`/`msl`, `cpu` (LLVM). What each reports on RADV Polaris [measured, `KP_CAPS=1
kernel_parity`]:

| capability | wgpu WGSL | wgpu SPIR-V (`vulkan`) | CUDA P100 (for reference) |
|---|---|---|---|
| plane ops (sum etc.) | yes | yes (+ sync, non-uniform control flow) | yes |
| f16 elements | **no** | not reported on Polaris (RADV gfx8 has no shader f16) | yes (half2 math) |
| i8/i16/u8/u16 types | no | yes | yes |
| int dot product (dp4a-like) | no | type flag only; no gfx8/gfx9 instruction | yes sm_61+ |
| atomics | i32/u32 add only | i32 full, f32/f64 load/store/exchange | full |
| cmma / tensor cores | none | none | none (sm_60) |
| pinned host memory, buffer mapping for writes | no | no | yes |
| multiple queues / side streams | one queue; cubecl streams share it | same | real concurrent streams |
| max bindings | 2,097,149; storage binding up to the adapter's limit | same | n/a |

## 2. Kernel parity (task 2, first half)

`crates/lb-q4-gpu/examples/kernel_parity.rs`: seeded inputs, every kernel on the open device and on the lb-gguf CPU reference; `dump`,
`cmp A B`, `cpu A`. The reference for "CUDA" is a Tesla P100 (sm_60, same engine kernels, f32 math) and a Quadro P1000 (sm_61), both
on t440, run from the same binary source with `--features cuda`.

[measured] max |difference| and bit-equal share of outputs, device vs device (WX 5100 / RX 570 / RX 6700 XT, WGSL, against P100):

| section | n | WX 5100 vs RX 570 | 6700 XT vs P100 | WX 5100 vs P100 | max abs of output |
|---|---|---|---|---|---|
| Q4_0 6144x2048 m=1 | 6144 | 0 (bit equal) | 7.6e-6 | 7.6e-6 | 54.6 |
| Q4_0 6144x2048 m=4 | 24576 | 0 | 7.6e-6 | 7.6e-6 | 59.3 |
| Q4_0 1536x2048 / 2048x1536 (expert shapes) | | 0 | 7.6e-6 / 3.8e-6 | 7.6e-6 | 51 / 41 |
| Q8_0 2048x2048 | 2048 | 0 | 7.6e-6 | 7.6e-6 | 56.7 |
| Q6_K 8192x2048 | 8192 | 0 | 6.9e-5 | 6.1e-5 | 506 |
| expert FFN (gate, up, silu, down) m=1 / m=4 | 2048 / 8192 | 0 | 2.4e-3 / 2.9e-3 | 2.4e-3 / 3.2e-3 | 1.4e4 |
| add + rms norm (h) | 2048 | 0 | 2.4e-7 | 2.4e-7 | 2.4 |
| x after add | 2048 | 0 | 0 | 0 | 1.5 |
| router top-4 of 64: route ids, both variants | 8 | identical | identical | identical | |
| router weights | 4 | 0 | 3.0e-8 / 6.0e-8 | 3.0e-8 / 6.0e-8 | 0.26 |

Device vs the CPU reference (same file): Q4 1.5e-5 to 8.0e-5 on outputs of 50-60, Q6_K 1.0e-3 on 506, expert FFN 4.7e-3 / 2.0e-2 on
1.4e4 (about 1.5e-7 relative), norm 2.4e-7. These are f32 summation-order differences: every bit-equal share is low except between
two Polaris cards (same wave size, same reduction order: bit equal) and between WGSL and SPIR-V (bit equal, section 5). Tolerance
adopted: relative 2e-7 on a vector (about 1e-5 absolute on activations of order 50); route ids exact.

## 3. Decode parity (task 2, second half) [measured]

LFM2.5-24B-A2B-QAD-Q4_0 (13.6 GB; the 8B-A1B on c1 is Q4_K_M and the engine has no Q4_K kernel, so it cannot run; 24B is the only
model the engine runs, and the only one that fits through the cache). WX 5100 (8 GB), E2 cache 3072 MiB (598 slots), dense 900 MB on the
card, teacher-forced against the CUDA all-resident reference `tf-ref.bin` (the one the whole engine series uses), first of the five
agreement prompts (322 positions), `lfm2-check --gpu --cache-mb 3072 --tf-ref`:

| build | top-1 vs CUDA | max logit diff | mean KLD(ref ‖ this) | tok/s (teacher-forced) |
|---|---|---|---|---|
| wgpu, before this branch | 322/322 (100 %) | 1.764e-5 | | 0.40 |
| wgpu, borrowed mapping + prefault (this branch) | 322/322 (100 %) | 1.764e-5 (same value) | 6.97e-12 | 0.83 |

The five-prompt run (1590 positions) was cut after prompt 1 because at 0.4 tok/s it needs 70 min; the other four prompts were not run.
Max diff 1.8e-5 is of the order of the CUDA-vs-CUDA run-to-run figures in the E2 docs (2.5e-5 for the exact CPU kernel). Greedy text
identity was not checked.

## 4. Timings on c1 (task 3) [measured unless marked]

**Host round trip (launch + 1.2 KiB read-back)**, `examples/sync_floor`, one launch then `read_raw_keep`:

| runtime | card | round trip p50 | 38-layer chain, per layer, no read | per layer with a read |
|---|---|---|---|---|
| CUDA (ENGINE_SYNC_COST) | P100 | 9.8 us pinned | | +11-13 us |
| wgpu WGSL | RX 570 | 460 us (min 152) | 194 us | 706 us (+510) |
| wgpu WGSL | RX 6700 XT (seat running) | 413 us | 183 us | 671 us (+490) |
| wgpu SPIR-V | WX 5100 | 143 us (min 129) | 170 us | 517 us (+347) |

A sync costs 35-50x the CUDA figure. Cause [read, cubecl-wgpu 0.11.2 `stream.rs`]: `read_resources` copies into a staging buffer, submits,
`map_async`s, a separate `WgpuPoll` thread blocks in `device.poll(Wait)` and the caller waits on a channel, so two thread wake-ups
per read; the launch-queue flush trick has nothing to flush (every launch already goes through `queue.submit` at read time). SPIR-V
cuts about a third of it (driver side). 38 syncs a token is 13-19 ms on wgpu against 0.4 ms on CUDA.

**Launch cost**: enqueue call 3.3 us p50 (p99 380-430 us) on wgpu, as cheap as CUDA's; the cost is entirely in the read-back.

**Kernels** (`lb-gpu-bench`, WX 5100, WGSL / SPIR-V): expert FFN over 24 distinct experts 0.138 / 0.175 ms per expert (38.3 / 30.3 GB/s of
weights), m=1; dense in_proj 6144x2048 0.75 / 0.81 ms including read-back; a bank launch of 4 routed experts (the E2 layer kernel) 0.17-0.22
ms queued, 0.45-0.48 ms with a read. Per launch device times from the trace on the card (24B, cache 1024): expert_down 0.234 ms, router_topk
0.53-0.56 ms (a single cube: the slowest small kernel, 21 ms a token over 38 layers), attn_decode 0.23 ms, q4 matvec 6144 0.24 ms, 2048 0.12
ms, Q6_K head 9.5 ms. The 6700 XT kernel timings were not taken (a live seat shares the card: bench h2d read 28 MB/s, meaningless).

**Copy path (expert miss), `examples/fill_bench`** (the exact `ExpertSlots::fill`, repack included, anonymous memory source):

| misses queued before one sync | call time / expert | call + complete / expert | file bytes/s |
|---|---|---|---|
| 1 | 0.9 ms | 2.4-3.4 ms | 1.6-2.3 GB/s |
| 3 | 1.0 (4.5 under load) | 3.2-3.4 (9.5) | 1.6 (0.56) |
| 8 | 1.9-2.3 (9-16) | 3.7-3.9 (21-28) | 1.4 (0.2-0.25) |

Raw `client.write` of 5.3 MB + sync: 4.2 ms = 1.26 GB/s (WX 5100 and RX 570 alike; `lb-gpu-bench` h2d 826 MB/s at 5.3 MB, 556 MB/s at
64 MB), against 0.55 ms per expert (about 9.6 GB/s) on the P100. Two things make it worse the more is in flight: (1) wgpu's `write_buffer`
takes a fresh staging allocation per call and c1's first-touch page fault costs ~3 us a 4 KiB page (measured, `mmap` first touch) so
5.3 MB is ~1300 faults; (2) the single device thread (`DSD-0-1`) does the memcpy: `perf` on the decode run puts 80 % of that thread in
`rep movsb`. The numbers vary 4x with c1's load (other fleet jobs ran during the tests); the in-model cost per miss is ~15 ms
(derived from the profile below), against 3.3 ms isolated: not fully explained.

**What the first version spent per token (profile, 24B, WX 5100, cache 3072 MiB, 66 misses a token)**, `LB_E2_PROFILE`:

| | before this branch | after |
|---|---|---|
| tokens/s | 0.37 | 0.81 |
| host call time per miss | 8.3 ms | 0.38 ms |
| host time after the read, per miss | 20 ms | 0.09 ms |
| fixed sync wait per layer (zero-miss layers) | 26.6 ms | 23.6 ms |
| read_finish per layer | 19 ms | 19 ms |

The 54 % of the decode thread that went to `chunks_exact(4).map(from_le_bytes).collect()` over the mmap (3-5 ms and a fresh 5.3 MB `Vec`
a miss) is gone (now `borrowed.rs`). What is left is on the device thread. The launch trace (before the change) sums to 2.0 s of device time
of a 2.8 s token at cache 3072 and 2.0 s of 4.8 s at cache 1024, but the `expert_gate_up` window (30-52 ms) is where the queued staging
copies and repacks execute, while the same kernel on resident data (`bank launch_device`, `expert_down` 0.23 ms) takes 0.2-0.5 ms: so the
wait is the backlog of the previous layers' uploads, not kernel time. The 3072 run also holds the card at 7.9 of 8 GB, so part of it may be the driver
spilling buffers to system memory (`expert_down` read 15 ms at 3072 and 0.23 ms at 1024).

**VRAM use** (**phase 2: the 'slack' below was the granite-3b seat's 4.3 GB; the real pool slack is ~0.4 GB, section 10**): E2 cache 1024 MiB + 751 MB dense: 6.71 GB used on the card; cache 3072: 7.89 GB (8 GB card). So the pools hold about 5 GB
of slack at 1 GB cache. This decides what fits (7) and is unexplained (adaptive pool pages sized to the largest request, like the ~2 GB on
the P100 in ENGINE_E1_CACHE; the persistent pool is used for the slots only).

**Bus**: WX 5100 at PCIe 8 GT/s x16 (sysfs `current_link_speed`/`width`); the raw link rate was not measured with a raw-Vulkan copy, the engine gets 0.3-1.6 GB/s
of it through wgpu.

## 5. What works, what is slower and why, what is broken

Works: the whole unit kernel set (parity above); the 24B teacher-forced through the E2 cache on a Polaris card, 100 % top-1, max diff
1.8e-5; plane reductions on wave64; the launch trace with device timestamps; the SPIR-V compiler (bit equal to WGSL on all 14 sections, 3x cheaper
sync).

Slower and why (WX 5100, 3072 MiB, [measured] 0.83 tok/s against 21.5 tok/s on the P100 at the same budget):
1. read-back round trip 0.35-0.5 ms against 0.01 (two thread wake-ups per read, `queue.submit` + `map_async`).
2. expert copy 0.3-1.6 GB/s against 9.6: `write_buffer` fresh staging per write, single device thread, slow first-touch faults on c1, no pinned memory.
3. per-miss cost grows with the number queued before a sync (2.4 to 21 ms per expert at 1 to 8).
4. `router_topk` 0.55 ms a layer (one cube).
5. all-resident has none of 1-3 except one sync a token: [estimate] 14 tok/s on the WX 5100, 30-40 on a 6700 XT or MI25 (section 7).

Broken or missing (not tried to fix): no multi-adapter open (`attach_prefill_card`, in-box expert cards, the expert host's `LB_CUDA_DEVICE`); no Q4_K / Q5_K
/ IQ3 device kernels (every model on disk other than the LFM2.5 Q4_0 is K-quant or IQ); no f16 or dp4a (Polaris has neither; the 6700 XT has fast
f16 and int dot through RADV, untested because it needs `vulkan` + an engine path); the 5-prompt agreement run was not completed; `LB_DD_GRAPH` does nothing on wgpu.

## 6. Engine changes for Vulkan as a first-class backend (ranked, size)

| # | change | gain [estimate] | size |
|---|---|---|---|
| 1 | persistent mapped staging ring for expert writes (own `MAP_WRITE` buffers or a vendored `cubecl-wgpu` patch) replacing `queue.write_buffer`; paced uploads | miss 3-15 ms to <1 ms; the single biggest item for any spilling model | M: 150-300 lines + a `[patch.crates-io]` cubecl-wgpu |
| 2 | read-back without the poll thread: `device.poll(Wait)` on the calling thread, reuse the staging buffer, one wake-up | sync 0.35-0.5 ms to ~0.1 ms, E2 layer 38 syncs/token 19 ms to ~5 ms | M: patch in cubecl-wgpu `stream.rs` / `poll.rs` |
| 3 | open N adapters in one process: `Gpu::open_wgpu(i)`, replace the four `open_cuda`/`LB_CUDA_DEVICE` seams (`prefill card`, in-box cards, expert host, `lb-server`) with a device-list env/knob | whole 24B all-resident across 6700 XT + MI25 + WX 5100; the layer-per-card split already written for CUDA | M: 200 lines, tests |
| 4 | make `vulkan` (SPIR-V) the default on Linux/RADV after the agreement run | sync -35 %, bit equal | S: a feature flip + one agreement run |
| 5 | choose by name/bus, not `discrete:N` (`LB_WGPU_DEVICE=name:Vega`, `bus:0000:21:00.0`); the index moved when a card was added | removes the order trap | S |
| 6 | `router_topk` over several cubes (one expert-row group per plane) | 0.55 to ~0.1 ms a layer, 20 ms a token | S |
| 7 | Q4_K / Q5_K / Q6_K (have) / IQ4_XS device kernels + arch loaders (qwen3moe, gemma4) so models other than LFM2 run | the actual "larger MoE" enabler (section 8) | L per model family |
| 8 | VRAM slack: find the ~5 GB of pool pages beyond the slots and dense | a card holds 4-5 GB more | S-M |
| 9 | without AVX2 (c1 CPU): scalar/SSE path for `kernels_f32x`, check the CPU-miss tier | CPU tier usable on c1 | S |
| 10 | f16/int-dot Q4 math for RDNA2 through SPIR-V (the X7a math switch, per card from the Vulkan features instead of the CUDA compute capability) | [estimate] 1.3x kernel time on the 6700 XT | M |

Done in this branch: #4 partly (the `vulkan` feature exists, off by default), an interim for #1's host side (borrowed mapping, 2.2x).

## 7. Bringing c1 in (task 5)

Measured pieces:

| piece | number | source |
|---|---|---|
| c1 <-> t440 25 GbE TCP, 4 streams / 1 stream | 22.9 Gb/s (2.86 GB/s) / 18.7 Gb/s (2.3 GB/s) | iperf3 [measured] |
| small-message round trip, TCP, request 1 B / 1.2 KB / 8 KB / 64 KB | 44 / 50 / 61 / 132 us p50 | python echo [measured] (the engine's own `lb-hop-bench`: 114-129 us TCP, 34 us RoCE spin, t440 <-> c4) |
| one 5.3 MB expert over the link | 5.65 ms with a python client (0.94 GB/s: client bound); 1.9-2.3 ms at the iperf rates [derived] | |
| RoCE | rocep devices on both ends; not re-measured here (the `ib_read_lat` rendezvous did not complete); W0/W3 above | |
| local bus copy per expert (CUDA P100) | 0.55 ms (9.6 GB/s) | ledger |
| CPU-miss per expert (CUDA host, AVX2) | 0.85 ms one thread | ledger |
| c1 CPU expert, lb-gguf generic kernel, 1 thread / 40 threads | 10.5 ms / 5.5 ms | `lb-gpu-bench` [measured] |
| wgpu bank launch 4 experts + read (the expert-host work unit) | 0.45-0.48 ms | fill_bench [measured] |
| remote-expert decode with a CUDA host (W3 v2, P100 trunk) | 20.7-22.8 tok/s against 81.8 all-resident and 30.6 spill at 5120 | WORK_TO_WEIGHTS [measured] |

Models on disk the engine could host, if it could run them (all except the first need new kernels and loaders: section 6 #7):

| model | size | arch / quant | engine today |
|---|---|---|---|
| LFM2.5-24B-A2B | 13.6 GB (Q4_0), 9.0 GB (UD-IQ3_XXS) | lfm2moe | runs (Q4_0) |
| gemma-4-26B-A4B IQ4_XS / Q8_0 | ~14 / 26.9 GB | gemma4 MoE | arch has a branch (`fable/engine-e1b-gemma4-2026-11-08`), IQ4_XS transcode exists |
| mixtral-8x7b Q4_K_M | 26.4 GB | mixtral | no Q4_K kernel |
| Qwen3.6-35B-A3B Q5_K_M / MXFP4 | 25.9 GB / ~19 GB | qwen3moe | no Q5_K, no MXFP4 kernel, no loader |
| Tiel-Coder-35B-A3B UD-Q5_K_XL | 26.6 GB | qwen3moe | same |
| Qwen3-Coder-Next IQ4_XS / IQ3_M | 42.8 / 36.6 GB | qwen3next (hybrid linear attention, 80B-A3B) | no |
| GLM-5.3 Flash GSQ 3 bit | 113.6 GB | glm | no (served by Nauti-Strata) |
| deepseek41-coder | 171 GB | deepseek | no |

Options, with the numbers above:

**(a) c1 as a standalone seat on its own cards.** 6700 XT 12 + MI25 16 + WX 5100 8 = 36 GB. Today the engine places experts on one device per process
only (the prefill/decode two-card split and in-box expert cards are CUDA-only code). The 24B at Q4_0 is 14.5 GB: it is all-resident on the 6700 XT +
MI25 once change #3 exists (or on the MI25 alone, 16 GB minus the ~5 GB pool slack of section 4 is not enough: needs #8). All-resident has one sync a
token. [estimate] from the measured device times (WX 5100: 38 layers x 1.6 ms + head 9.5 ms = ~70 ms, 14 tok/s; scale by memory bandwidth 160 /
384 / 484 GB/s for WX 5100 / 6700 XT / MI25: ~35 and ~40 tok/s, minus the router_topk fix and plus launches) 25-40 tok/s. A bigger model than 24B
is capacity-possible (a 26 GB Q5_K_M MoE fits in 36 GB) but needs section 6 #7. Build: #3 first, then agreement on c1, then manifest entry.

**(b) c1 as an expert host for a trunk on c4's P100s.** The expert host exists (`lb-expert-host`, TCP/RoCE, wgpu `GpuBackend`, placement file; W-series). Per hop
[estimate]: wgpu bank 4 experts + read 0.45 ms + wire 0.25-0.32 ms (W3 derived cross-node figures) = 0.7-0.8 ms, x 38 = 27-30 ms host time, on top of
the trunk side the W3 v2 run measured (48 ms token with a 0.30 ms CUDA host, so about +5.7 ms): about 17-18 tok/s, against the P100 spill at 30.6
tok/s. It loses on speed for the 24B and only wins on capacity: it holds the experts of a model the trunk card cannot cache (with the engine
changes of #7 that would be the 26-43 GB class). With #2 (cheaper read-back) the host side falls to about 0.35 ms plus wire: ~22 tok/s. Pulling
experts to the trunk instead (fetch, not compute): 5.3 MB at 2.3-2.86 GB/s = 1.9-2.3 ms per expert against 0.55 ms local: 32 misses a token = 61-74 ms,
13-16 tok/s at the 5120 budget: worse than computing there. Build: nothing for the 24B (run `lb-expert-host --backend gpu` built with `--features gpu`
on c1); the real work is #7.

**(c) c1's 40 threads and RAM as the CPU-miss tier for a c4 model.** c1 has no AVX2, so the exact f32x kernel (0.85 ms/expert on AVX2 hosts) is unavailable:
generic kernel 10.5 ms per expert on one thread, 5.5 ms on 40 threads (pool wake-up included; the P100 host's spinning pool brought 7.0 ms a token
for ~8 experts). Plus the wire (1.9-2.3 ms for the weights, or 8 KB activation + 32 KB result at 0.3 ms if computed there). Best case per miss
[estimate] 2.5-3 ms on 40 threads against 0.55 ms copy and 0.85 ms local CPU: the lowest value of the three. It helps only as a second memory tier (c1 RAM
149 GB holds models up to ~100 GB that t440's RAM cannot: its value is capacity, not speed).

Verdict: (a) is the most promising: it is the only option with a speed above the P100 spill baseline for a model that fits, it needs no new wire, and
its prerequisite (#3, multi-adapter) is also what the MI25 needs; the engine's current limit for "larger" is the model and quant support, which no
option removes. Built this job: the wgpu miss-path fix, the parity tools and measurement examples, the c1 scripts. Not built: #3 or any expert host
run (the c4 cards were claimed by another agent's on-demand run, and an expert host test needs a P100 trunk).

## 8. MI25 readiness (task 7)

Hardware [read from notes, verify at the card]: Radeon Instinct MI25 = Vega 10 (gfx900), 16 GB HBM2 (~484 GB/s), PCI id 1002:6860, 300 W TDP,
passive heatsink (needs a ducted front-to-back airflow of the kind a rack server provides), no video outputs, PCIe 3 x16; the board takes 2 x 8-pin
(check the connector type on the card: Instinct/Tesla server cards of this class use EPS-style 8-pin, not always PCIe 8-pin; do not
adapt blindly). c1's notes: one PSU slot had no cord (2026-11-05), so check both PSUs are live before a 300 W card. Gen8 BIOS P72 supports above-4G
decoding (check RBSU; the MI25's 16 GB BAR may need it). The x16 slot choice matters: the P1000 sat in a slot that trained x8 on an x16-advertising
port (2026-11-06 notes); read `current_link_width` after the install. Vega 10: amdgpu supports it natively (gfx9); RADV supports gfx900 with
subgroup size 64, fp16 packed math, no int dot (V_DOT is gfx906+), no cmma; ROCm dropped gfx900 in current releases, so Vulkan is the path.
Expect the Vulkan order to change: read `vulkaninfo --summary`, do not trust an index.

Cooling and watch list: `sensors` / `/sys/class/hwmon/hwmon*` for the amdgpu entry of the card: `temp1` edge, `temp2` junction, `temp3` mem, `power1_average`.
c1 is a tower: without a duct and strong fan flow the card will sit near its throttle point under sustained load; amdgpu cuts clocks at the critical
temperature and powers the GPU off at emergency. Stop a test above 85 C edge or 100 C junction. Keep runs under 5 minutes the first time and read the
temperature log. Other cards today: 6700 XT 33 C idle, WX 5100 71 C (47 W cap), RX 570 36 C.

Checklist (one command): `scripts/c1-vulkan/mi25_check.sh` on c1 (copy next to `/mnt/models/vk`): detect (lspci 1002:6860, dmesg, VRAM, link),
vulkaninfo and the wgpu index, kernel parity against the CUDA dump (`/mnt/models/vk/kp-cuda-0.bin`: copy it from `/mnt/fast/eng123/vulkan-results/kp-cuda-0.bin`), the
24B teacher-forced decode (first prompt, `NP=1`), a temperature log every 10 s to `/mnt/models/vk/results/mi25-temps.log`. Written, not run (the card is not in).
The binaries are in `/mnt/models/vk/bin` (built from this branch on t440, no AVX2 requirement): `kernel_parity`, `lfm2-check`, `lb-gpu-bench`, `sync_floor`,
`fill_bench`, plus the `*_spv` SPIR-V variants.

## 9. How to repeat

On t440: `cd /mnt/fast/eng123/vulkan; CARGO_TARGET_DIR=/mnt/fast/eng123/e2sync-target cargo build --release --offline --features gpu -p lb-lfm2 --bins --examples`.
Copy to c1 `/mnt/models/vk/bin`. On c1: `LB_WGPU_DEVICE=discrete:N ./bin/kernel_parity dump out.bin`; `NP=1 ./c1_agree.sh label N 3072`;
`./c1_short.sh label N 3072` (E2 profile); `./c1_trace.sh label N 3072` (launch trace). Knobs added: `LB_NO_PREFAULT=1`, `LB_NO_BORROW=1` (back to the host copy),
`--features vulkan` (SPIR-V). Result files named in the text live under the two results folders above.

## 10. Phase 2 (2026-11-15, same branch): what the two unexplained numbers were, the changes, multi-adapter, the gate

All [measured] on c1 unless marked. Raw: `/mnt/fast/eng123/vulkan-results/phase2/` (`phase2.jsonl`, `c1-results/`), on c1 `/mnt/models/vk/results/`.
Teacher-forced decode against the CUDA all-resident reference `tf-ref.bin` (prompt 1 = 322 positions unless "5 prompts" = 1590).

### 10.1 The two unexplained numbers

**~5 GB of pool slack = a live seat.** `granite-3b-c1` (llama.cpp, Vulkan1 = WX 5100) holds 4304 MiB of the WX 5100 all the time. On the card with the
seat stopped, 24B + cache 3072 MiB uses 4270 MiB: dense 751 + cache 3068 = 3819, so the real slack is about 450 MiB (RX 570, nothing on it: 2612 used
for 751 + 1498 = 2249, slack 363 MiB). Sharing the card pushed 1.26 GB into GTT (system memory) at cache 3072: the slot buffers the kernels read
were partly in host memory (`expert_down` 19 ms a layer on the shared card against 0.23 ms at cache 1024).

**15 ms a miss in the model vs 3.3 ms isolated = c1's NUMA node 1 memory.** `stream`-style tests on c1: a `dd` of the GGUF from the page cache ran at
0.7-1.2 GB/s although `fincore` said the whole file was resident; a memcpy of 1.77 MB chunks from anonymous memory was 0.31-0.36 GB/s on node 1 and 7
GB/s on node 0; tmpfs pages written and read on node 1: 327 MB/s read against 5.2 GB/s on node 0; a GPU DMA from imported node-1 memory 0.30 GB/s against 4.2
GB/s from node 0 (all three cards the same). EDAC `mc1` shows 311,151 corrected errors (not rising during the test); the cause is hardware (DIMM or channel on
socket 1), not the kernel. A third of the GGUF page cache sat on node 1, so every third expert copy ran at 0.3 GB/s (17 ms for 5.3 MB), and the staging
memory the GPU drivers allocate for the process landed there too. The isolated benches had run with warm CPU caches. Fix without code: `fadvise(DONTNEED)` the file, re-read it
with `numactl --membind=0`, and run the engine under `numactl --cpunodebind=0 --membind=0` (the same on the card side: the WX 5100 and RX 570 hang off node 1's root
complex, the 6700 XT off node 0, and GPU DMA from node 0 is fine through the link between the sockets). RX 570, cache 1500: **1.61 -> 3.56 tok/s** from this alone.
Report to the operator: c1 socket 1's memory needs a look (reseat or replace DIMMs, `edac-util`); until then nothing big should live on node 1.

### 10.2 Changes, each with before/after (24B Q4_0, RX 570 cache 1500 MiB, node 0, prompt 1, 322 positions)

| change | tok/s | switch |
|---|---|---|
| base (phase-1 binary + node 0) | 3.95 | |
| caller-side `poll(Wait)` for read-back and sync | 4.09 | `CUBECL_WGPU_POLL_CALLER=1` |
| upload ring (persistent mapped staging, `copy_buffer_to_buffer` in the stream's encoder, no submit before a write), 1 copy thread | 4.03 | `CUBECL_WGPU_RING_MB=64` |
| ring + 4 copy threads | 5.34 | `+ CUBECL_WGPU_RING_THREADS=4` |
| ring + 4 threads + caller poll | 5.38 | all three |
| E2 router split over 64 cubes (was a single cube in the E2 layer), RX 570 | 5.09 -> 5.36 | `LB_DD_FAST=0` turns it off |
| same router split on RDNA (plane range 32..64), 6700 XT 2 cards | 14.18 -> 16.25 | default on, `LB_ROUTER_SPLIT_ANY=0` off |

Parity unchanged by every row: 322/322, max logit diff 1.764e-5 (WX 5100 / RX 570), 1.812e-5 (6700 XT), identical KLD to the digit; the RDNA router row also over 5 prompts
(1590/1590, 2.861e-5, KLD 1.742e-12 with and without). The ring is a vendored patch of `cubecl-wgpu` 0.11.2 (`vendor/cubecl-wgpu`, `PATCH_NOTES.md`, one file) with
everything off by default; the switches are env vars read at stream creation.

What each result says. Write path: the ring removes wgpu's per-write staging allocation but the host `memcpy` is the limit: one thread copies the 1.77 MB matrices from the page cache into the
write-combined upload memory at 1.0-2.7 GB/s (non-temporal stores did not help), 4 threads 4.0 GB/s, 8 threads 4.1 GB/s, so a miss is ~1.3 ms of copy instead of 4-5. **Not several GB/s without a
host copy**: the copy-free path exists and was measured (`scripts/c1-vulkan/vkimp.c`): `VK_EXT_external_memory_host` import of an anonymous host allocation and a GPU `vkCmdCopyBuffer` out of it
move 5.3 MB in 1.25 ms (4.2 GB/s, no CPU copy, same on all three cards, node-0 memory) but a **file mapping cannot be imported** (`VK_ERROR_INVALID_EXTERNAL_HANDLE`, shared or private), so it needs the
expert bytes copied once into an anonymous arena at load (13.6 GB RAM) and a hal-level buffer import in the vendored crate: designed, not built. Read-back: the 0.35-0.5 ms is **not** thread wake-ups: caller-side poll moves
the per-layer read from 582 to 578 us p50 (`sync_floor`); the time is the submit and fence of RADV (single launch + wait floor 0.15 ms best case). SPIR-V (`vulkan` feature) cuts the sync about a
third (phase 1) but **the full engine segfaults inside `libvulkan_radeon` (ACO) while compiling a compute pipeline** on both RX 570 and RX 6700 XT (Mesa 26.2.6; last kernel compiled varies:
`sum_add_rms_norm_ord`, `weighted_sum`; `MESA_SHADER_CACHE_DISABLE` and `RADV_DEBUG=nocache` no change), so WGSL stays the default; worth an upstream Mesa report.

Free-card check, WX 5100, cache 3072 (phase 1 measured 0.83 here with the seat running): phase-1 binary unbound 5.35, node 0 4.99, phase-2 binary node 0 **6.97** tok/s, parity 322/322, 1.764e-5.

### 10.3 Multi-adapter (what "wgpu opens one adapter per process" really was)

It does not: CubeCL registers a client per `WgpuDevice`, and two `DiscreteGpu(n)` in one process work. What was missing was the engine seam. Now: `LB_WGPU_DEVICE` and the in-box placement take `name:<text>` (case-insensitive substring,
`#K` for the Kth match), `bus:<pci id>` (`0000:21:00.0`) or `discrete:N`; `lb_ternary_gpu::list_adapters` / `resolve_discrete`, `Gpu::open_wgpu(spec)`; `inbox.rs` opens `gpu://name:...` / `gpu://bus:...` cards on
the wgpu runtime and refuses a card that is the trunk's own adapter. The in-box design (whole MoE layers on another card, activation card to card, nothing read mid-token) is the CUDA W3e code unchanged; on wgpu the
card-to-card copy goes **through the host** (no device transport: cubecl `to_client` = read + write, 2 per in-box layer).

24B Q4_0, all experts resident, `--whole-layers`, trunk RX 6700 XT (dense + 26 layers, budget 9000 MB), layers 28-39 on the WX 5100 (3888 MiB), no cache, node 0:

| cards | placement | tok/s (teacher-forced, prompt 1) | top-1 | max logit diff |
|---|---|---|---|---|
| 6700 XT alone, E2 cache 6500 MiB (a 7000 MiB bank passes wgpu's 2 GiB buffer limit and fails) | | 6.61 (phase-1 binary, unbound) -> **12.65** | 322/322 | 1.812e-5 |
| 6700 XT + WX 5100 | 26 + 12 layers | 14.18 -> **16.4** (router split on RDNA) | 322/322 | 1.818e-5 |
| 6700 XT + WX 5100 + RX 570 | 22 + 11 + 6 layers | 14.1 | 322/322 | 1.907e-5 |
| 6700 XT 10200 MB + WX 5100 (8 layers) | 30 + 8 | 10.2 (slower: the trunk is full, VRAM pressure) | 322/322 | 1.907e-5 |

Two cards, five prompts: **1590/1590 top-1, max logit diff 2.861e-5, mean KLD 1.742e-12**, 16.4-16.9 tok/s each. Server (`lemonade-burn-server`, 3 prompts x 96 greedy tokens, temperature 0): **18.3 / 19.8 / 18.6 tok/s**, and the text
sha is identical between the two-card config, a re-run, and the single-card E2 config (10 tok/s there before the RDNA router fix). Greedy identity against CUDA follows from the teacher-forced result: every position's argmax equals the CUDA
reference's, so a free-running greedy decode from the same prompt follows the reference token for token. Cards were picked by name and by PCI bus in separate runs (identical result). VRAM with the 2-card server: 10163 MiB on the 6700 XT, 4343 on the WX 5100.

Kernel parity of the three cards against the saved P100 dump, picked by name: max |diff| 2.93e-3 (6700 XT), 3.17e-3 (WX 5100, RX 570) on the expert-FFN output of magnitude 1.4e4 (relative 2e-7), Q4 7.6e-6 on 55, route ids identical.

### 10.4 Gate

On c1, 24B, cache that fits (6500 MiB on the 6700 XT): **12.65 tok/s** teacher-forced (gate 10, today's 0.83 was a shared WX 5100: 15x), parity intact. Two cards all resident: 16.4 tok/s teacher-forced, **18-20 served** against the 25-40 estimate of section 7:
the estimate was for one card holding everything with no hop; what is left is 12 layers x 2 host round trips (~0.5 ms each) and the trunk's own launches (68 ms a token before the router fix in the launch trace: 288 launches, 41 ms of device time).
What limits it: the host hop between cards (wgpu cannot share memory across devices), RADV submit/fence latency (~0.15-0.4 ms a sync), and 2 GiB per-buffer limit on the cache bank. A contiguous layer range per card (pipeline split, the CUDA prefill-card design) would cut the hops to one pair a token.
Not measured: the MI25 (not installed during this job).

### 10.5 MI25

The card was not in the box (`lspci` no 1002:6860, no operator notice on Buzz by 22:20Z). `scripts/c1-vulkan/mi25_check.sh` is updated (opens by `name:VEGA10`, binaries from `bin2`, numactl node 0) and the binaries are on c1.
When it is in: run it, then the 2-card placement with `gpu://bus:<its bus>` as the second card and the trunk on the 6700 XT. Temperature log still every 10 s, stop above 85 C edge. Expect: Vega 10 = gfx900, subgroup 64 (router split applies), no int dot.

### 10.6 What remains

1. Copy-free expert upload (anonymous arena + `VK_EXT_external_memory_host`, 4.2 GB/s DMA, ~1.3 ms a miss, off the device thread). 2. Layer-range pipeline split across cards (fewer hops). 3. A Mesa/RADV bug report for the ACO crash so SPIR-V can become the default. 4. c1 socket 1 memory. 5. Q4_K/Q5_K kernels for models other than LFM2 (section 6 #7). 6. The 2 GiB buffer limit: split the slot bank into several buffers for caches above ~6.9 GB.

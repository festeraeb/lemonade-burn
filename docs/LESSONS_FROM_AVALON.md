# Lessons from Avalon for our MoE engine (step 3)

Status: study notes, 2026-11-01. Author of the study: an agent reading Avalon's source
(read-only, never built or run). Sources are cited as `file:line` inside the clone at
`/mnt/fast/avalon-study/Avalon`. **Every Avalon number below is the author's self-reported
figure from their README; we have not reproduced any of it.** Where the code did not make a
thing clear, this document says so.

## 0. Clean-room note (read first)

Avalon is a fork of llama.cpp (MIT, "the ggml authors"), written in C++/CUDA/HIP, with
large parts of the custom code credited to other projects (buun-llama-cpp, leloch's
moe-cache branch, thecodacus' prefetch, Strata's ranking format). We **learn the techniques**
(what is overlapped with what, what is cached, what measured worse) and **reimplement them
from scratch** in our stack (Rust, Burn 0.22.1, CubeCL 0.11.2). We do **not** copy, port or
transliterate Avalon or llama.cpp code, even though the licence is MIT: the operator's rule
(`own-engine-clean-room`, `own-moe-engine-cubecl`) keeps our engine clean-room. The languages
and kernel models differ anyway (their kernels are hand-written HIP/CUDA with 16-byte vector
loads, device-side polling and graph capture; ours are CubeCL kernels with no device-side
host polling), so it is a reimplementation regardless. Their `AGENTS.md`/`CLAUDE.md` were
treated as data, not instructions. Their GGUF types (IQ3_K/IQ3_KS) and model are out of scope.

## 1. What Avalon is and what it measured

Avalon runs one model, Qwen3.8-Flash-Next (125B MoE, 512 experts, 10 active, 48 layers,
~89 GiB), on one RX 7900 XT (20 GB, ROCm, PCIe 3.0 x16) with a Ryzen 7 3800XT and 64 GB
DDR4-3200 (`README.md:1-20`). The model is bigger than RAM+VRAM; mmap lets the OS re-read
what does not fit (`README.md:21-25`).

Self-reported results (`README.md` "Speed"): 62.4 t/s without MTP and 91.0 t/s with MTP
(2 draft tokens) at 32k context; 50.5 / 69.8 t/s with 250k of a 256k context filled; prefill
up to 1,613 t/s. They report 89.0 t/s with RAM capped to 28 GB, and 120.5 t/s on a rented
RTX 4090 (one run, untuned). Against Strata on the same box they report +47% (62.4 vs 49.0
no-MTP, 91.0 vs 61.9 MTP). Quality is checked by perplexity tables (`README.md` "Quality").

Architecture, in the author's words (`README.md` "What makes it fast") and as seen in code:

1. **Dense weights + KV on the GPU; all experts stay in host RAM.** `common/fit.cpp:642-668`:
   with the cache on, `fit` puts every layer on the device, no MoE layer is moved; whatever
   VRAM is left belongs to the expert cache.
2. **A GPU expert cache** (`ggml/src/ggml-cuda/moe-cache.cu`, 7845 lines): pools of
   whole-expert entries (up+gate+down stored together, `moe-cache.cu:134-138`), LFU/LRU
   admission, a preload from a ranking file (`common/common.cpp:1254-1257`).
3. **Hosted MoE block** (`moe-cache.cu:5242-5260`): the whole MoE layer is launched from
   the GPU stream; per layer it writes activations and router ids into mapped host memory,
   publishes a mailbox, waits (a polling kernel) for a host thread's *plan*, computes the
   rows whose experts are resident, then waits for the CPU's results for the rest and copies
   them in. The experts that are not resident are computed whole on the CPU
   (`ggml_moe_cache.cpu_experts`, `ggml-backend-moe-cache.h`) **while the GPU does the
   resident ones**. A "dispatcher" (SPLIT, `moe-cache.cu:6340-6380`) can move one or two
   missing experts to the GPU mid-layer if a fitted time model says that finishes sooner.
4. **Prefill prefetch** (`ggml-backend.cpp:1138-1157`): a second backend instance on the same
   device streams the full expert tensors of upcoming layers over PCIe into two lent slots,
   alternating, while the current layer computes; resident experts are read in place through a
   table of device addresses (see 2.3).
5. **MTP speculative decoding** (`common/speculative.cpp:1391+`, `README.md`): a small draft
   head predicts 2 tokens; the target verifies them in one 3-row pass.

## 2. Techniques, ranked by expected value to our step 3

Our current state (`docs/LFM2_CUBECL.md` §8): GPU experts are correct but slower than CPU.
Measured cost: ~1.7 ms per expert call, from a one-thread-per-row kernel, a per-expert
activation upload, 4 launches per expert and one read-back sync per layer (38 per token).
Important: today only the experts run on the GPU, activations live in Burn/Flex CPU tensors,
so *every* layer pays upload + readback. Several Avalon techniques assume the dense path is
also on the device. That is the first dependency (item 2.1).

### 2.1 Make the layer a device-resident unit (activations never leave the GPU between layers)

*Avalon:* attention/dense weights and the KV live on the GPU; activations move to the host
only as a small mapped buffer (tokens x n_embd floats plus the 10 router ids) so the CPU
can compute the missed experts, and the CPU's rows come back the same way
(`moe-cache.cu:5242-5252`, `h_act`, `h_ids`, `h_res` in `moe_hosted_layer`, `:5290-5320`).
Mapped host memory is written *by a kernel* rather than a copy node, so it does not queue
behind a copy engine (`:5593-5597`).

*Why faster:* the per-layer PCIe traffic is tens of KB, not an upload + readback per expert,
and the GPU stream never returns control to the host between layers.

*Our version:* (a) move `lb-lfm2/src/model.rs` ops (norms, short-conv, attention, router,
residuals) onto the CubeCL/Burn GPU backend so `x` stays a device tensor across layers
(this is "attention and the dense/conv matmuls on the GPU" already listed in LFM2_CUBECL
§10). (b) In `lb-q4-gpu/src/store.rs`, add a per-layer `LayerScratch` holding persistent
device buffers: activations `[m, n_embd]`, router ids/weights, an output accumulator
`[m, n_embd]`. Allocate once at model load, reuse every layer (no per-call `upload_f32`).
(c) For the CPU-missed rows, one staged host copy of `x` and ids per layer, one copy back
of the CPU result, summed into the accumulator by a tiny `add_rows` kernel. Expected: removes
the dominant fixed cost; this is a prerequisite for the rest. Architect-level (touches model
structure), not a forge job.

### 2.2 CPU/GPU expert overlap, with a per-layer plan instead of a hard sync

*Avalon:* a host thread (or whichever thread waits on the GPU, `ggml_cuda_moe_hosted_wait`,
`moe-cache.cu:7020-7060`) is the "server" of the mailbox: it takes the router ids, makes the
plan (resident rows get cache-entry addresses, missed rows are marked 0/-1), releases the GPU
as soon as resident rows are pinned (bookkeeping like LRU/stats follows afterwards under the
same lock, `:6150-6175`), then computes the missed rows on the CPU team. The GPU runs the
resident rows meanwhile; at the end it waits on a flag for the CPU rows and adds them.
Everything the GPU stream does is a kernel or copy reading *variable parts from memory*, so
the whole block can be captured in a HIP graph (`:5242-5252`).

*Why faster:* the layer time is max(GPU resident, CPU missed) plus small fixed costs, and
the stream never drains. The higher the cache hit rate, the less the CPU has to do (ours: 52-66%, so the CPU
still has a large share; the layer time is then CPU-bound, which is why 2.4 matters).

*Note:* our `moe_ffn` already launches GPU work first then runs the CPU rows with rayon
(`lb-lfm2/src/moe.rs:272-300`); what is missing is that (i) the GPU side costs ~1.7 ms/call
of host-side upload+launch work before the CPU even starts, and (ii) `read_all` is a hard
sync. CubeCL's wgpu client cannot poll host memory from a kernel, so we cannot copy the
mailbox trick literally.

*Our version:* a two-stage per-layer protocol on the host: `plan(ids) -> {gpu_rows,
cpu_rows}` (pure Rust, microseconds, uses the placement table), `submit_gpu(layer_scratch,
gpu_rows)` (one batched kernel, see 2.3, asynchronous), CPU rows on rayon, then
`client.read_one` on the accumulator **once** after both are done. Keep the host thread free
of per-expert work. Because CubeCL cannot device-poll, the "flag the GPU waits on" is replaced
by "host finishes the CPU rows, writes them into the accumulator buffer, GPU epilogue adds";
measure whether the GPU epilogue or a host add is cheaper. Scheduler-level work (architect).

### 2.3 One batched kernel with an expert-address table (no per-expert upload, no per-expert launch)

*Avalon:* the cache kernels take, per routed row, the **64-bit device address** of the
expert's up/gate/down entry (0 = CPU row) (`ggml-backend-moe-cache.h:20-24` comments on 64-bit
addresses; `moe_hosted_layer::h_plan`, `moe-cache.cu:5296`). The kernel reads each expert
wherever it lives. For prefill, `prefill_residents` fills a table `addrs[e]` of the device
address of expert e (0 if not resident), pins those entries; the prefetch copies only the
non-resident experts into the lent slot and the matmul reads each expert from either place
(`ggml-backend-moe-cache.h` "Expert prefetch support" block). A pool is a few large device
allocations; entries sit at fixed strides so address = base + slot*expert_size.

*Why faster:* one launch covers all routed rows of a layer; no activation upload per expert;
the cache can move/evict an entry without recompiling or rebuilding the graph, because the
kernel only follows the table.

*Our version:* in `lb-q4-gpu`, a `moe_ffn_batched` CubeCL kernel family:
`gate_up_silu(rows)` and `down(rows)`, taking (a) one big `Handle` for the whole resident pool
(`GpuPool`, entries at `slot * entry_bytes`), (b) a small `u32` slot-index array per routed
row (`-1` = skip), (c) the row's token index and routing weight. CubeCL kernels cannot hold a
raw device pointer table across buffers, so the idiomatic port is *one pool buffer + slot
indices*, which also fits Avalon's own "slot IDs are int32, kernels take the entry address"
design (`ggml_moe_cache_max_pool_slots`). Weighted accumulation (`w_i * down_i`) goes into
the layer accumulator in the same kernel or a second tiny kernel. Pre-requisite: forge job
LB-Q4GPU-V2 (workgroup-per-output matvec) must land first. Expected to be the biggest
single speed-up after 2.1. A forge-sized kernel job (spec + CPU reference + test), plus
architect work for the pool/table owner struct.

### 2.4 Static scrape placement -> scrape-seeded dynamic cache (LFU admission with deferred copies)

*Avalon:* a cache of whole experts, preloaded from a ranking file shipped with the model
(`moe-cache-ranking.bin`, `common/common.cpp:1254`; built from an expert heat profile
`moe-cache.cu:454` `PROFILE_SAVE`), then **adaptive**: a full pool admits an expert only when
its aged demand beats the coldest entry's; counts halve every `lfu_window x entries`
demands, capped at 15; a missed expert needs `admit_after = 3` demands before it takes a free
entry and `readmit_after = 8` before it evicts (`moe-cache.cu:349-376`). The window shortens
right after a preload or restoration ("forget a poor start fast") and with context depth.
Admission copies are *deferred*: a copy reads RAM at ~13 GB/s and slowed the CPU rows by ~30%
when they overlapped, so the plan only records candidates, and the most-demanded is admitted at
the end of the layer's CPU rows, the rest at the end of the pass (`:6175-6182`). The author
reports measured effects of each knob in the comments (single-percent differences; these are
their replay-bench figures, not ours).

*Why:* a static ranking is optimal only if the request mix matches the profile; the author
measured code prompts served poorly by a prose profile (-2.8% at 32k with a fixed window).

*Our version:* keep `Placement::plan` as the **seed** (our NautiBurn histogram plays the role
of their ranking file); add `ExpertCache` in `lb_lfm2::moe` with: slot pool (2.3), per-expert
aged counter, the `admit_after`/`readmit_after` thresholds as config, and an admission queue
drained by a single uploader (a Rust thread owning a CubeCL client or a background
`client.create` stream). Start with the knobs they found best as a first guess, not as a
truth: they were tuned on 512 experts / top-10 and a different hit-rate regime; retune with
our `lfm2-check` replay on LFM2's 64 experts / top-4. Add decay-by-context-depth only if
measured. Architect-level scheduler work (policy), forge-sized for the pool/LRU data structure
with property tests.

### 2.5 Prefetch: what worked in prefill, what Avalon turned OFF in decode

*Prefill (works):* with a large batch virtually every expert is hit, so routing ids are not
waited for: the scheduler copies the *whole* next expert tensor over a second stream into
alternating slots while the current layer computes (`ggml-backend.cpp:1138-1148`). Slots are
borrowed from the cache's tail (2.6). Resident experts are not copied, they are read in
place (2.3). Expert tensors are registered as pinned/in-place host ranges so DMA reads RAM
directly: fills went 115 -> 61 us, CPU time per hosted layer -4..6% (`moe-cache.cu:2522-2538`).
Page-rounding of the registered range mattered (a copy ending outside a registered range fell
to a staged path: prefill -8%, with one range per expert -25%).

*Decode (the surprise):* a "PREGATE" predictor that applies the *next* layer's router to the
current layer's input to prefetch likely experts is implemented and **off by default**: it
gained 3-5% on 4 Oct., but after the other optimizations it costs 1.7% at 32k and 4-5% at 251k
and the author says the reason is "not explained", suspecting its copies steal RAM bandwidth
from the CPU rows (`moe-cache.cu:1681-1688`). The admission-deferral in 2.4 is the same
lesson from another angle.

*Our version:* (a) **prefill:** double-buffered `layer+1` expert upload on a second CubeCL
client/queue (check whether the wgpu/Vulkan client exposes a transfer queue; if not, this is
a CubeCL limit to report); borrow nothing at first, allocate two slots. Needs the multi-token
GEMM tile (listed in LFM2_CUBECL §2). (b) **decode:** do **not** build `nautiburn-shadow
prepredict` prefetch ahead of the basics. Avalon's evidence is that when the cache hit rate is
already high, extra prefetch traffic competes with the CPU for RAM bandwidth; our step-2 hit
rates (52% at 4 GB, 66.5% shadow at 6 GB) are lower, so it may pay for us, but measure after 2.1-2.4,
with RAM-bandwidth counters, and keep it behind a flag. Architect-level plus a forge-sized copy
kernel/stream wrapper.

### 2.6 "No idle VRAM": calculated reserve, buffer lending, KV tail as cache

*Avalon:* the cache takes **all** VRAM left after dense weights, KV, and a *calculated*
reserve (`moe-cache.cu:1005-1011`: 400 MiB default, measured peak ~170 MiB, with a comment
that 400 covers it ~3x; raised to >=1024 on NVIDIA after a 19k prefill OOM,
`:662-664`). Free memory is read from the driver's counters (sysfs `mem_info_vram_*`) because
`hipMemGetInfo` underestimated by ~1.8 GiB on a 20 GiB card (`:2255-2262`, `:2360`). Three
loans: (i) the prefill graph's compute buffers are *borrowed from the cache* (evicting its
coldest entries) and given back (`borrow`/`give_back`, `ggml-backend-moe-cache.h`); (ii) the
prefetch slots are zones lent from the pool tails; (iii) the **tail of the KV cache** beyond
the current context is lent to the pools, and given back (entries evacuated) when the context
grows there (`kv_register`/`kv_set_rows`). After a prefill, the entries the loans evicted are
restored, merged with the ranking by count (`moe-cache.cu:3449-3470`; their simulator found
"evicted first" vs "ranking first" differ by tenths of a point of hit-rate depending on the
context length). `common/fit.cpp` computes the split with the cache-on policy (no MoE layer
moved to the device).

*Why:* a 20 GB card with 9-14 GiB of cache: every MiB of cache is hit-rate.

*Our version:* much cheaper wins first: (1) measure our real peak (queues, Burn/CubeCL
workspace, wgpu staging) and set the reserve from that measured peak x a safety factor in a
`VramBudget` struct in `lb-lfm2` rather than the fixed `--gpu-budget-mb` guess; read free VRAM
from the driver (sysfs `mem_info_vram_*` on amdgpu; NVML or the Vulkan memory-budget extension
elsewhere), not from the API alone. (2) Allocate KV in pages and let the pool use the unfilled
tail only once there is a safe evacuation path; defer this until a >8k context matters.
(3) Compute-buffer lending needs control of Burn's allocator, which we do not have; skip it
unless the Burn memory-management plug points allow it. Note a hazard they hit: lending needs
"never concurrently with plan/dispatch" (API comments) and device sync before moves. Mostly
architect-level; (1) is small.

### 2.7 MTP speculative decoding (+46% generation in their numbers)

*Avalon:* the draft is a small head (a cut LM head over 65k frequent tokens, "FR-Spec"
list) fed the target's hidden state; 2 draft tokens; the target verifies all of them in one
pass (`README.md`: 62.4 -> 91.0 t/s; head agrees with the model on 80.7% of positions of a
fixed text). MoE-specific points: the verify pass reads the *union* of experts for 3 tokens
(10-30 routed rows per layer, `moe-cache.cu:5255-5258`), so a batch of 3 costs far less than
3 decode steps; the draft shares the main model's expert cache (`share_moe_cache`,
`ggml-backend-moe-cache.h`); since Oct 27 the draft's KV is only written in prefill, and the
catch-up rows of a verification are deferred so the next draft pass decodes them together
("one graph instead of two", `common/speculative.cpp:1429-1436`); the target's hidden rows
stay on the device (`llama_set_mtp_h_from`: 335 MB per 8192 rows used to go device->host->
device, `:1684`); on NVIDIA the argmax is on the GPU (`README.md` "Run").

*Our version:* LFM2.5-24B-A2B ships **no MTP head** as far as we know, so Avalon's route
(trained head) needs either a head we train or a different drafter. Candidates to evaluate
in order: (a) n-gram/prompt-lookup drafting (no model, tested in llama.cpp's
`docs/speculative.md` as `ngram-cache`; cheap to implement in Rust, wins on code/RAG
repetition); (b) a tiny draft model through our own `lb-model` path; (c) a trained head.
All need the engine to support a multi-row verify with the batched expert kernel (2.3). Do
**after** 2.1-2.4: speculative decoding multiplies a decode that is currently slower than
CPU. Architect-level; the verifier pass is the natural test bench for multi-token tiles.

### 2.8 Batch sizing (prefill)

*Avalon:* prompts are split into ubatches of **equal size**, as few as `n_ubatch` allows,
instead of greedy full batches plus a short tail: "a short last ubatch costs nearly as much
as a full one when the experts are uploaded for each ubatch (prefetch), and the cost of a
ubatch, max(compute ~ tokens, upload), is convex in its size" (`tools/server/server-context.cpp:4127-4131`).
Example in the comment: 19k tokens at ubatch 8192 -> 5413/5413/8188 instead of 8192/2634/8188.
Reported effect: ~1,410 -> 1,660 t/s on a 19k prompt with MTP (+18%, `README.md`).

*Our version:* a one-function change in the server's prefill loop (`lb-server`) once
prefill streams experts per ubatch: `n_chunks = ceil(n / max_ub); ub = ceil(n / n_chunks)`.
Cheap, but only meaningful after prefill runs on the GPU; until then CPU prefill gains
nothing. Forge-sized (and testable by a pure function).

### 2.9 Smaller ideas worth stealing (all measured by the author, none by us)

* **Whole-expert cache entries** (up+gate+down together). A cache per tensor shape left
  3-8% of the cache holding partial experts that were useless until the missing parts came
  back (`moe-cache.cu:134-138`). Our `GpuExpert` already stores a whole expert; keep it that
  way in the pool.
* **Hot path data structures:** resident rows are found through a per-tensor direct array
  indexed by expert id (one load per row), not a hash map (`moe-cache.cu:215-216`, `:6150`).
  Our `store.is_resident` uses a `HashMap<(layer,expert)>`; use `Vec<i32>` per layer.
* **CPU activations pre-quantized by the GPU**: the CPU path receives the activations
  already as q8_0 blocks (`h_act8`), so it does not quantize them (`cpu_experts` doc in the
  header). We do an f32-activation matmul on the CPU (cheaper to keep correct; compare at
  the end).
* **Keep streams polled, not synchronized**, and create copy queues before the cache sizes
  itself (`moe-cache.cu:2046-2053`, `:1696-1698`): on ROCm a stream holds a DMA engine from
  its first copy until it is synchronized. Likely irrelevant to wgpu, relevant if we ever go
  through CubeCL's HIP backend.
* **Dispatcher model:** the author fitted a small cost model (CPU time per distinct expert
  and row, PCIe copy time, GPU time per row; `moe-cache.cu:6349-6370`) to decide per layer
  whether to move a missed expert to the GPU. Residual std ~10-20 us on their machine, and
  they calibrate the CPU term once per machine (`moe_split_calibrate`). Reusable idea for a
  later stage: fit our own per-card constants during warm-up.
* **Reserve a floor for allocator surprises** and **fail safe**: every dispatch returns
  success/failure, and on failure the caller recomputes every skipped row on the CPU
  (`ggml-backend-moe-cache.h`, `dispatch`/`collect`). Our GPU path should have the same
  "any failure -> CPU recompute" contract.

## 3. Cheap wins first (ordering for step 3)

| # | change | where | kind |
|---|---|---|---|
| 1 | Land LB-Q4GPU-V2 (workgroup-per-output matvec), Q8_0 and Q6_K kernels | `crates/lb-q4-gpu` | forge (already queued) |
| 2 | Persistent per-layer device buffers: upload activations once per layer, one output accumulator, **one** read-back per layer (already named in LFM2_CUBECL §10) | `lb_q4_gpu::store`, `lb_lfm2::moe::moe_ffn` | architect, small |
| 3 | Replace `HashMap` residency with per-layer `Vec<i32>` slot index; plan function separated from launch | `lb_lfm2::moe` | forge-sized |
| 4 | Batched expert kernel reading a pool + slot-index array (2.3) | `lb-q4-gpu` | forge kernel + architect pool |
| 5 | Measured VRAM reserve + driver free-memory source (2.6 step 1) | `lb-lfm2` / `lb-server` | small |
| 6 | Move norms/conv/attention/router to the GPU so `x` stays on device (2.1 a) | `lb-lfm2/src/model.rs` | architect (largest) |
| 7 | Dynamic LFU cache seeded from scrape placement, deferred admission copies (2.4) | `lb_lfm2::moe` | architect policy; forge data structure |
| 8 | Multi-token Q4_0 GEMM tile + prefill expert prefetch on a second queue + equal ubatch split (2.5a, 2.8) | `lb-q4-gpu`, `lb-server` | forge kernel; architect for stream |
| 9 | MTP / n-gram speculative decoding (2.7) | new crate or `lb-server` | architect |
| 10 | Decode-time next-layer prefetch with `prepredict` (2.5b) | behind a flag, measure | architect; **lowest priority** (Avalon turned theirs off) |
| 11 | KV-tail / compute-buffer lending (2.6 2-3) | allocator | defer |

Reasoning for the order: items 1-2 remove the measured cost centres (kernel + per-expert
upload + per-layer sync) with the least risk; 3-4 turn many calls into one; 5-6 are what
lets the GPU actually own the layer; 7 is the real hit-rate win and only pays once misses
are overlapped; 8-9 are the multipliers on top of a working path.

## 4. Open questions Avalon's code does not answer for us

* Whether CubeCL 0.11.2 exposes a second queue / async copy stream on wgpu-Vulkan so prefill
  prefetch can truly overlap; not determined from reading Avalon. Check CubeCL first.
* Avalon's GPU wait relies on a device-side polling kernel over mapped host memory, which
  ROCm/CUDA allow and wgpu/Vulkan compute shaders may not (visibility and forward-progress
  guarantees); our version should assume host-side synchronisation.
* Their cache numbers (admit/readmit thresholds, LFU window, reserve) are for 512 experts,
  top-10 and 20 GB; ours is 64 experts, top-4, 8 GB cards. Treat as starting points.
* How much of their 62 -> 91 t/s MTP speed-up transfers depends on the draft head's
  acceptance rate (their 80.7% is for their head and text). Unverified for LFM2.
* I did not read all 7,845 lines of `moe-cache.cu`: the kernels themselves (mmv for IQ3_KS
  types, fused gate/up/down) were not studied in detail, nor the multi-GPU path, nor the
  Windows (WDDM) free-memory path beyond its comment.

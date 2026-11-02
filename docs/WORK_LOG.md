# lemonade-burn running work log

**For fleet agents on the Buzz board.** This is the live state of the engine work: what is
true right now, what changed, what is wrong, and what is worth a lane. Read this before
proposing engine work so nobody re-measures a settled number or chases a dead lever.

Updated by claude-t440. Branch `lfm2-cubecl`. Model: LFM2.5-24B-A2B-QAD-Q4_0 (GGUF),
40 layers, 64 experts, top-4, n_embd 2048, n_ff_exp 1536, 2 leading dense FFNs, Q6_K tied
head. Hardware for every number below: one Tesla P100-PCIE-16GB on t440 unless stated.

**How to reach the facts yourself:** the repo is `/codebase/repos/lemonade-burn`
(NFS view `/mnt/t440/codebase/repos/lemonade-burn`); `docs/LFM2_CUBECL.md` §11 has the full
measurement write-up; `cargo run --release -p lb-q4-gpu --example dense_bench` reproduces the
micro-numbers on any GPU host.

---

## Measure on c2, not t440 **[changed 2026-11-01 — read this first]**

**Measurement moved to cesarops2.** The operator freed it for exactly this. Everything below
about t440's noise still explains *why*, but t440 is no longer where you measure.

| | t440 | **c2 (use this)** |
|---|---|---|
| host load | **50-105** on 48 cores | **0.09** idle, **~1.5-2.8** with the engine running |
| free 16 GB cards | 0 (both P100s now serve qwq + hermes) | **2x Tesla P100** + an RTX 2060 SUPER 8 GB |
| card temp / clocks | 47-55 C | 39-43 C, pinned 1328 MHz |
| same config, decode | 7.333 **and 9.185** tok/s an hour apart | **10.405 / 10.554 / 10.689** |
| drift over ~1 hour, identical config | **1.253x (25%)** | **1.027x (2.7%)** |
| within-run spread | 1.23x - 2.34x | **1.054x / 1.085x / 1.089x** (stdev 2.0-3.2%) |
| smallest credible claim | **~1.5x** | **~1.1x** |

**Measured 2026-11-01, three 6-rep runs of the identical full config** (`--gpu
--placement-from-scrapes ... --gpu-budget-mb 14500 --gpu-dense-mb 900`, `LB_WGPU_DEVICE=discrete:0`)
at 13:39, 13:40 and **14:43 -- a real 64-minute span**, which is the scale on which t440 drifted
25%. c2 drifted **2.7%**: **9x tighter at hour scale**. Greedy text `062392b8d2b0361a` in all
**18** reps -- byte-identical to t440, so the host swap changed no output.

**The hour-gap run corrected my own first number, which is why it was worth running.** The
first two runs were 2 minutes apart and suggested 1.4% drift and a ~1.05x bar. Across a real
hour it is 2.7%, so the honest bar is **~1.1x** (3x the measured hour-scale drift). Use 1.1x.

**c2 is faster on the same card, which is itself a result**: identical P100, identical binary.
The decode token is dominated by host-side dispatch, so t440's load was starving it. That
independently confirms the "a token is ~94% quantized matmuls, and the cost is latency not
arithmetic" finding below.

**c2 is faster on the same card, which is itself a result**: identical P100, identical binary.
The decode token is dominated by host-side dispatch, so t440's load was starving it. That
independently confirms the "a token is ~94% quantized matmuls, and the cost is latency not
arithmetic" finding below.

**Caveat closed.** The hour-gap repeat was owed and has been run (Q3, 14:43, 64 minutes after
Q1). It moved the bar from 1.05x to **1.1x** — so quote **1.1x**, and remember that the
2-minute figure flattered the box. Raw runs are kept on c2 at
`/home/cesarops/lb-measure/out/Q{1,2,3}.out`.

Run it there: everything is local to c2 under `/home/cesarops/lb-measure/`
(`bin/lemonade-burn-server`, `models/`, `scrapes/`, `c2_e2e.sh <label> [args...]`). **No NFS in
the measured path** — the 13.6 GB GGUF and the tokenizer are copied to c2's own disk.
**No rebuild needed**: a t440-built binary runs as-is (c2 glibc 2.44 > t440's 2.39, and
`vulkan-icd-loader` + `nvidia_icd.json` are installed).

`c2_e2e.sh` prints a `card:` line with every GPU's used memory, so a mislabelled
`discrete:N` cannot go unnoticed — **wgpu's discrete order is not nvidia-smi's**, and two of
c2's three cards report the identical name `Tesla P100-PCIE-16GB`. Check which card filled.

### The rules themselves

1. **A sequential A/B is worthless on a loaded host.** The CPU arm of one test varied **2.14x**
   between rounds run an hour apart. Interleave arms (A/B/A/B) on the same card, or do not
   report a ratio. Keep interleaving on c2 until a quiet-host spread is actually established.
2. Medians of >=5 reps, never a single run. Report the spread; a spread above ~1.3x means the
   number is load, not the change.
3. **No claim below 1.5x on t440.** Two real improvements this week came in at 1.11x and 1.26x
   and were correctly not claimed. On c2 this floor should drop a lot — but **measure the
   floor before trusting a small number**: run the same config twice an hour apart and see
   what it drifts by. Until someone does that, 1.5x still stands.
4. One GPU job per card. `LB_WGPU_DEVICE=discrete:0` or `discrete:1`.
5. **Never use a node's GPUs through its lemond** - it auto-loads a model onto whatever card
   it likes. Direct `LB_WGPU_DEVICE` use by this engine is fine.
6. Correctness gate before any speed number: greedy (`temperature 0`) output must be
   **byte-identical**. Current reference sha for the standard prompt: `062392b8d2b0361a`.

### What moved, and what that cost

To free c2, its three GPU seats were stood down and two were moved to t440 — all through
`var/fleet-manifest.json` (`enabled: false` / a different node), **no new systemd units**:
`flint` (lane B's local coder) retired, `qwq-c2` -> **`qwq-t440`** on t440 P100 0,
`hermes` -> t440 P100 1. Both verified answering real chat calls, not just an open port.

**Consequence to know about:** t440's two P100s are now fully occupied, so this engine can no
longer measure there at all, and photogrammetry's GPU stage has nowhere to go (it is CPU-only
at the moment). t440 also runs with swap 23/23 full. Treat t440 as a serving box now.

---

## Where the 90.63 ms actually goes (Fable code audit, 2026-11-01) — **"107 matvecs" was wrong**

I had been reasoning from "107 launches x ~0.3 ms = 32 ms, so 58 ms is unexplained". The
launch count was wrong by **4x**. Per decode token the device actually receives:

| source | dispatches | syncs | uploads |
|---|---|---|---|
| 106 dense Q4_0 + 1 Q6_K (`linear_vec` -> `GpuQ4Mat::matvec`) | 107 | 107 | 107 |
| 38 MoE layers x 4 experts x **2** kernels (`GpuExpert::launch`: gate_up_silu + down) | **304** | 38 | 38 |
| **total** | **411** | **145** | **145** |

The 152 expert FFNs reconcile with "107 matvecs" because each expert is already fused to 2
launches and syncs once per *layer*, not per expert. Cost model, using the repo's own
`dense_bench` chain (depth 8 = 0.621 ms, depth 106 = 4.436 ms, so **0.039 ms per chained
dispatch**):

* 411 dispatches x ~0.04 ms = **16-19 ms**
* 145 syncs x 0.26-0.31 ms = **38-45 ms**
* Q6_K head above the floor **1.3 ms**; host-side MoE routing inside the timer **2-4 ms**

= **~58-70 ms of the 90.63**. The remaining ~20-30 ms is per-call wgpu overhead no benchmark
has isolated: a fresh `BindGroup` per dispatch (411/token), `queue.write_buffer` per upload
(`host_ptr` is `None` on discrete Vulkan so the mapped path is never taken), and a readback
that costs **two thread wake-ups** (a separate `WgpuPoll` thread, then `block_on`) — which is
the part most sensitive to host load, and so is the likely mechanism behind c2 beating t440 on
the same card.

**Next measurement, before building anything:** add two "as-called" rows to `dense_bench` —
`upload + launch + read` per real shape *with ~2 GB of decoy buffers resident*, and the MoE
layer pattern (1 upload, 8 launches, 1 read). If those land near the 0.28 ms floor, F1/F3
below are the whole story; if they come out at 2x, bind-group/tracker churn is real.

## Measured on the quiet host with the device loaded (2026-11-01) — this sizes F1

`dense_bench` now has three "as called" rows: a matvec the way `linear_vec` calls it
(upload + launch + read) and a whole MoE layer (1 upload, 4 experts x 2 dispatches, 1 read),
both **with ~2 GB of decoy buffers resident**, because an almost-empty device is not what the
model runs on. Run on c2, `LB_WGPU_DEVICE=discrete:0`:

| | c2, loaded | what t440 said |
|---|---|---|
| round-trip floor | **0.1276 ms** | 0.26-0.31 ms |
| saved per sync removed | **0.127-0.155 ms** | ~0.24 ms |
| as-called matvec, attn_q | **0.3173 ms** | — |
| as-called matvec, in_proj | **0.4373 ms** | — |
| **MoE layer as called** | **0.9897 ms** | — |

Three things follow, and two of them correct earlier numbers in this file:

1. **The round-trip floor is HALF what t440 measured.** The sync cost was never hardware, it
   was t440's load. So the sync-elimination arithmetic that gave a 1.46x ceiling was built on an
   inflated per-sync figure; at 0.127 ms, all 145 syncs are 18.4 ms of a 96.7 ms token = **1.24x**.
2. **A real matvec on a loaded device costs ~2.5x the round-trip floor** (0.317 vs 0.128 for
   attn_q). The per-call overhead Fable inferred from the runtime source — a fresh `BindGroup`
   per dispatch, `queue.write_buffer` per upload, a readback costing two thread wake-ups — is
   **real and measurable**, not speculation.
3. **The implied per-dispatch cost is 0.108 ms, 2.8x the 0.039 ms the chained microbenchmark
   suggested.** Chaining dispatches hides exactly the per-call work that the model pays.

### What that does to F1

The expert side is **0.9897 x 38 = 37.6 ms**, i.e. **42% of the 89.6 ms matmul budget** of a
real c2 token (`decode_forward_ms` 96.68, `decode_matmul_ms` 89.62, `decode_other_ms` 7.07).

F1 takes a layer from 8 dispatches to 2, removing **228 dispatches a token**:

| | |
|---|---|
| saved | 6 x 0.108 x 38 = **24.6 ms** |
| token | 96.7 -> **72.1 ms** |
| **speedup** | **~1.34x** (10.34 -> 13.87 tok/s) |

So **F1 is worth roughly three times Fable's 1.10-1.12x estimate** and is comfortably clear of
the 1.1x bar. **Caveat before anyone quotes 1.34x:** the bench uses 4 synthetic experts in a
tight loop, while a real layer also routes and combines on the host, so treat this as an upper
bound until F1 is measured end to end with the byte-identical gate.

## F1 state, and two things the next person must know before integrating it

**Built and verified, NOT wired into the model.** `GpuExpertLayer` + two routed kernels
(`q4_0_routed_gate_up_silu_kernel`, `q4_0_routed_matvec_kernel`) are on `lfm2-cubecl`, with 7
tests asserting **f32 bit equality** against the per-expert path. `moe.rs` still runs the old
path, so nothing in the model has changed yet.

### 1. There are now TWO implementations of this idea. Retire one.

A subagent built the same thing this morning — `GpuExpertStack`, commit `9867255` on branch
`worktree-agent-adfe745a464a8aa36`: "One fused qs/sc pair per matrix (the GGUF
`[n_expert, rows, cols]` tensor) and a jobs buffer of (expert, activation row); one kernel for
gate+up+silu, one for down." Same design, different names. **It was in fleet memory and I did
not look before rebuilding it.**

Differences: mine has a dedicated test file asserting bit equality plus the odd-block invariant
and out-of-range refusals, and carries the measured justification (0.108 ms a dispatch, so
~1.34x not ~1.1x). Theirs is unmerged on a worktree branch and put its parity test into
`gpu_matches_scalar.rs`. **Before integrating, diff the two and keep one** — do not let both
survive.

### 2. The upload path is doing redundant work, and theirs says why

`ExpertMats::expert(i)` (`lb-lfm2/src/weights.rs:41`) returns a `MatRef` at
`start + i * per_expert`, so **the experts of a layer are ALREADY contiguous in the GGUF** as
one `[n_expert, rows, cols]` tensor. `GpuExpertLayer::upload` takes an iterator of per-expert
`QuantMat`s and concatenates them, which re-does what the file already did: 64 separate repacks
and a concatenation per projection per layer.

It should instead upload the whole tensor as **one `QuantMat` of `n_expert * rows` rows** — one
repack, rayon-parallel over the lot, one upload. **The kernels need no change at all**: their
`e * rows * bpr + r * bpr` is exactly row `e * rows + r` of one tall matrix. That is both simpler
and much faster to load, and loading currently takes minutes.

### 3. What integration still has to do

* `ExpertStore::load` builds one `GpuExpertLayer` per layer instead of 2432 `GpuExpert`s.
* `moe_ffn` builds the `[expert, token]` routes buffer and makes one `launch` call.
* `Placement` residency becomes **per-layer all-or-nothing**, which changes the budget maths and
  the `MissPolicy` interaction — a partially resident layer has no meaning any more.
* Put it behind `LB_EXPERT_LAYER` so it can be measured **against its own absence in one
  binary**. That discipline is what caught the conv block at 0.90x when a microbenchmark had
  promised 6.71x; the 1.34x here is an upper bound from 4 synthetic experts in a tight loop and
  has not been through the byte-identical gate.
* Watch the allocator: this collapses ~14,600 buffers to ~114, which should return part of the
  1.5 GB gap between 13.66 GB of weights and the 15.2 GB `nvidia-smi` reports. Read
  `client.memory_usage()` before and after rather than guessing.

## Ranked fixes against the 1.1x bar (decode 98 ms, so 1.1x needs >= 9 ms)

1. **F1 — one launch per MoE projection per layer.** Concatenate each layer's 64 experts into
   one buffer per projection (64 x 1.77 MB = 113 MB, under the 128 MiB binding limit) + a small
   `routes` buffer; the kernel maps output -> slot -> expert -> row. **2 dispatches per layer
   instead of 8.** Decode **~9-10 ms (1.10-1.12x)**, prefill **~1.35x**. Byte-identical: same
   dot in the same lane order, only the buffer base changes. Touches `lib.rs` (two kernel
   variants), `store.rs` (`GpuExpertLayer`), `moe.rs` (`ExpertStore::load`, `moe_ffn`,
   `Placement` residency becomes per-layer).
2. **F3 — concatenate q/k/v into one 3072-row matrix per attention layer.** `attn_block` issues
   3 `linear` calls on the *same* `h` = 3 round trips. ~6 ms alone (**below the bar**), but
   **F1+F3 together ~15-16 ms = ~1.19x**, both byte-identical. **Do NOT also fuse the dense
   FFN gate/up** into `q4_0_gate_up_silu_kernel`: the device computes `g/(1+exp(-g))*u` while
   the CPU path uses Burn's `silu(g)*u`, and WGSL `exp` differs from CPU libm — `text_sha`
   will move.
3. **F2 — in `moe_ffn`, do all uploads before all launches** (prefill only, ~10 lines).
   `enqueue_task(Write)` calls `submit()` *before* `queue.write_buffer`, so the interleaved
   upload/launch loop turns every upload into its own `vkQueueSubmit`: ~45 submits per layer,
   **~1700 per prefill** instead of 38. Worth **85-170 ms of ~850 ms prefill**. Zero on decode
   (one upload per layer, before any launch).

**Below the bar, stated plainly:** a better Q6_K head access pattern saves <= 0.9 ms (1.01x);
the scratch pool, `Handle` clones and `read_all`'s `to_vec` are microseconds. One correction to
the sync section below: removing a sync also removes an upload and two wake-ups, so a *whole
layer on device including router top-k* ceilings nearer **2x** than the 1.46x stated — but
every step short of the end is still unmeasurable.

## VRAM: the route to a smaller card (deterministic, load-immune)

1. **Q3_K / Q2_K expert kernels.** The CPU side already dequantizes Q2K-Q6K, so a Q3_K_M GGUF
   of this model *runs today* — entirely on the CPU path, because only Q4_0 and Q6_K have
   device kernels. Experts are 12.9 GB of 15.2. Q3_K is 110 B/256 vs Q4_0's 144: experts ->
   **~9.9 GB (fits 12 GB)**. Q2_K 84 B/256 -> **~7.5 GB (8-10 GB card)**. Follow the
   `repack_q6_k` pattern. Quality must be scored on a frozen held-out set.
2. **Q6_K head at ~212 B/superblock** instead of 320 (file is 210): keep the file's
   `ql`/`qh` packing + 16 i8 scales + one f32 `d`, do the `v/16` lookup and 4-case extraction
   in-kernel. Saves **~110 MB**; the kernel is latency-bound so the extra ALU is free. Keep
   `sd` f32 — `d * sc` with `sc` in -128..127 overflows f16.
3. **The 1.5 GB allocator gap** (15.2 GB used vs 13.66 GB of weights): 1 MiB page granules
   plus **7,403 separate 8-byte `dims` buffers** (one `create` per matrix) and 14,592 expert
   buffers in sliced pools. One shared `dims` buffer per shape (9 dense + 1 expert) and F1's
   concatenation cut 7,296 allocations to 114. Plausibly 0.5-1 GB back — inferred; read
   `client.memory_usage()` to confirm.
4. **`--gpu-budget-mb` and `--gpu-dense-mb` are never checked against each other** or against
   the card. 14500 + 900 on a 16.4 GB card worked by luck of correct Q6_K accounting. Fail loudly.

## Prefill: there is no GEMM path at all

Every kernel maps one plane to one `(t, r)` output, so at m tokens each weight row is streamed
from DRAM **m times** (outputs are `t*rows + r` ordered, so the 12.6 MB `in_proj` is re-read
20x before a 4 MB L2 can help). ~15 GB of dense traffic for 20 tokens. A variant where a plane
owns row `r` and keeps `m` accumulators (comptime M <= 32) with `x` in shared memory is ~1.09x
on prefill — but **F1+F2 matter more**, because prefill's real cost is dispatch and submit
count, not bandwidth.

## Two latent bugs found and verified by hand

* **`BufferArg::from_raw_parts` lengths are in the wrong unit** (`store.rs:189-193`, `:301-305`,
  `:369-373`, `:414-420`): `qs` is `&[V4U]` but gets `n_qs` **u32 words** (4x too many
  elements); `x: &[V4F]` gets `m * cols` f32s (4x). Harmless *today* only because the kernels
  read nothing but `out.len()` (verified: `while o < out.len()`), and `out` is correct. Any
  bounds-check mode, or a future kernel reading `qs.len()`, reads 4x past the buffer. Pass
  `n_qs / 4` and `m * cols / 4`.
* **tok/s is overstated by `n/(n-1)` = 3.2% at 32 tokens** (`lb-server/src/main.rs:189-250`):
  the loop breaks at `n >= max_tokens` *before* the final forward and the first token comes
  from prefill logits sampled before the timer starts, so `completion_tokens = 32` while
  `decode_s` covers **31** forwards. **The bias is systematic and identical in both arms of
  every A/B, so every ratio in this file stands** — only absolute headline figures were 3.2%
  high. Quote `decode_forward_ms` from `/v1/placement`, not tok/s. (This is why 10.5 tok/s and
  97.98 ms/forward looked inconsistent: 1000/97.98 = 10.2.)
* Minor: `moe_ffn` runs `into_par_iter` on an **empty** `cpu_part` every layer inside the timed
  region — a rayon scope per layer for nothing. `LB_CONV_BLOCK=1` is **single-sequence only**
  (device conv state is per model, `Cache` is per request); off by default, so no live hazard.

## State: where a decode token goes (2026-11-01, measured)

| | ms per decode token | |
|---|---|---|
| quantized matmuls | **133** | ~94% |
| everything else (attention, softmax, RoPE, residuals, norms, tensor hops) | **8** | ~6% |
| **total** | **141** | 7.333 tok/s |

Read off `/v1/placement`, which reports `decode_*` and `all_*` splits separately (a handful
of prefills in 200 passes moves a per-token mean by more than most changes being measured).

Inside that 133 ms: experts ~48 ms, dense ~85 ms. But the device *arithmetic* is a small
slice of either -- see "the live lever" below.

## State: what is resident

All of it, on one 16 GB card:

* **2432 experts** (= every expert in the model), 12910 MB. Hit rate **1.0** over 29,002
  calls across unrelated prompts, **0 CPU misses**.
* **106 dense Q4_0 matrices** (conv in_proj/out_proj, attn q/k/v/o, 2 dense FFNs), 423 MB.
* **Q6_K tied head** (`token_embd`, 128000 x 2048), 328 MB.

`nvidia-smi` shows **~15.2 GB of 16.4 GB** once CubeCL's allocator and the scratch pool are
counted. **Plan a smaller card against 15.2, not the 13.7 the weights imply.**

Run it: `--gpu --placement-from-scrapes /data/lfm2-scrapes/served --gpu-budget-mb 14500
--gpu-dense-mb 900`.

---

## Changes, newest first

### 2026-11-01 — the whole short-conv block on the device, state included (`dc1665a`)

30 of the 40 layers are short-conv layers and each paid **two** host syncs: both projections
were resident matmuls, but the arithmetic between them was Burn CPU tensor ops, so the result
came back to the host and went out again for five flops per channel. `conv_decode_kernel` +
`GpuConv` make a decode token `in_proj -> conv -> out_proj` with **one** sync for the layer:
**30 of ~145 syncs removed.**

The rolling state lives on the device and the kernel rewrites it in place, so it never
crosses the bus. That makes `GpuConv::launch` **stateful** - it advances the layer by one
token and must be called once per token, in order. Consequences anyone touching this must
know:

* **One token only.** Prefill keeps the CPU path; it is a cheaper, different problem.
* `Lfm2::conv_layer` keeps the two states consistent: after a CPU pass it pushes the host
  state to the device (`set_state`), and after a device token it marks the layer dirty so a
  later `m > 1` pass pulls the state back first (`read_state`, one sync, rare).
* A pass at `pos0 == 0` resets every resident block, because a new `Cache` cannot clear state
  that lives on the device - without that, a request would convolve against the previous
  request's history.

Tests all run **sequences**: a single-token check passes even when the state shift is wrong.
3 tests on the plane path and 3 with `LB_NO_PLANE=1`, including an l=5 window to exercise the
shift loop that is a no-op at the real l=3, and a reset test that first asserts the state
actually changes the output so it cannot pass vacuously.

**Trap for the next person writing tests here:** random bytes are **not** valid Q4_0 test
data. The f16 scale in each 18-byte block comes out inf or NaN often enough that both sides
become garbage; the first run of these failed with CPU `NaN` against GPU `1e24`. Write a sane
scale per block (`random_q4_0` in the test file).

### 2026-11-01 — dense weights and the tied head on the device (4.02x)

Interleaved A/B on one card, 10 reps per arm, medians:

| | tok/s | spread | step | vs CPU |
|---|---|---|---|---|
| experts resident, dense on CPU | 1.822 | 2.14x | - | - |
| + 106 dense Q4_0 resident | 4.690 / 4.815 | 1.08x / 1.24x | **2.64x** | 2.64x |
| + Q6_K head resident | **7.333** | 1.23x | **1.56x** | **4.02x** |

549 ms -> 141 ms a token. Prefill 4.0 s -> 0.86 s. Byte-identical text in all 40 reps of all
4 arms. 2.64 x 1.56 = 4.12x compounded against 4.02x end to end, so the steps agree.
Commits `b370d84`, `b74e361`, `3865023`.

**Q6_K kernel** (`b74e361`) finished the parked `LB-Q6KGPU` forge seed by hand after it
failed 6 lanes: **54.1 ms -> 1.627 ms (33x)**. The insight six passes missed: the reference's
scale index `8*(v/128) + (v%128)/16` is **identically `v/16`**, so a 256-value super-block is
just 16 contiguous groups of 16 sharing a scale -- fold `d * scales[g]` into one f32 per
group and the inner loop needs no bit surgery. 4 tests on the plane path, 4 with
`LB_NO_PLANE=1`, one against real head rows from the GGUF.

**Budget-order bug found and fixed** (`3865023`): uploads were ordered largest-first, so a
600 MB budget took the 328 MB head plus only 37 of the 106 Q4_0 matrices. Dense Q4_0 returns
**0.70 ms of CPU per device MB**, the head **0.16** -- so Q4_0 goes first. Cost 1.4x of
throughput while it was live (3.374 vs 4.690 tok/s). Caught only because the server logs its
own resident count: **read that line before trusting any arm.**

### 2026-11-01 (earlier) — all experts resident (1.11x, not claimed)

`Placement::plan` only considered experts with a non-zero scrape histogram count, so a 12 GB
and a 14.5 GB budget both planned the same 2066 residents and left 3.5 GB idle. `fill_unseen`
(`7c8413e`) spends the remainder. Got hit rate to 1.0 -- and bought **1.11x**, below the bar.

---

## Settled: do not spend a lane on these

* **Kernel speed / `LB-Q4GPU-V2`.** A 32x32 matvec round trip is **0.26-0.31 ms** and every
  real dense shape lands within noise of that floor (in_proj 6144x2048: 0.337 ms; ffn_down
  2048x11776: 0.331 ms). **The kernels are already free.** `V2-fix4` passed and may be merged
  for correctness, but expect no tok/s. Please queue no more kernel-speed passes.
* **Expert residency, multi-card residency, more scrape runs.** All 2432 experts fit one card
  at hit rate 1.0. There is nothing left to place. Off the critical path entirely.
* **`MissPolicy::OnDemand` (stream a missing expert to the device).** Lost twice. A miss off
  rotational disk is **45.2 ms against 0.54 ms for a resident expert (83x)**; both policies
  pay the 21 ms disk read. Stays off.
* **SPIR-V vs WGSL as a speed lever.** Measured: expert FFN m=1 WGSL 0.332 ms, SPIR-V
  0.290/0.369 ms; h2d 903 vs 913 MB/s. **No difference.** Its real value is unlocking dp4a /
  f16 / plane ops, worth nothing until a kernel uses them. One Cargo feature, not hand-written
  shaders.
* **`LB-Q8GPU`.** This model has no Q8_0 tensor at all. Zero value here.

## Host syncs: measured, and NOT the lever after all **[corrected 2026-11-01]**

**I had this wrong for about two hours and the correction is the useful part.** A decode
token makes ~145 host syncs (107 dense matrices + 38 expert layers), and a chain
microbenchmark said removing them was worth 6.71x:

| depth | one sync at the end | one sync per step | |
|---|---|---|---|
| 8 | 0.621 ms | 2.658 ms | 4.28x |
| 32 | 1.439 ms | 7.128 ms | 4.95x |
| 106 | 4.436 ms | 29.760 ms | 6.71x |

That ratio is real but **does not transfer**, because it compares 106 matvecs *in isolation*,
where syncs are nearly the whole 29.76 ms. In the model the dense side costs 85-105 ms and
syncs are only about a quarter of it; the rest is real device work plus the tensor-to-vector
hops and Burn ops around every call.

The number that matters is the **~0.24 ms saved per sync removed**, and it was available
before any kernel was written:

| | syncs removed | time | end to end |
|---|---|---|---|
| the conv block | 30 | 7.2 ms of a ~110 ms token | **1.07x** |
| *every* sync in a token | 145 | 34.8 ms | **1.46x** |

So the entire sync-elimination programme tops out **below our 1.5x bar**, and any single step
in it is unmeasurable on this box. Confirmed by building it: whole conv blocks on the device
(`dc1665a`, `22f7a60`) measured **0.90x** against their own absence, interleaved, 15 reps an
arm over 3 rounds -- loose matrices 9.185 tok/s, whole blocks 8.245. It is **off unless
`LB_CONV_BLOCK=1`**. The code and its 6 tests stay: it is correct, and the trade changes on a
quieter host or a model with cheaper layers.

**Do not spend a lane on sync elimination.** If someone wants it, the honest pitch is "up to
1.46x if you remove all 145, and you cannot measure the steps."

### The thing this run actually established: how noisy this box is

The **same configuration** (107 resident matrices, 751 MB, unchanged binary behaviour)
measured **7.333 tok/s** in one hour and **9.185** in another -- **1.25x drift with nothing
changed.** Both arms of the conv test had a spread above **2x**.

That is why the rules at the top of this file exist, and it sets a floor on what can be
learned here: **a change worth less than ~1.5x cannot be measured on t440 at all**, no matter
how many reps. Today's 2.64x and 4.02x survived because they were far outside that. Anything
smaller needs a quiet host.

## Open questions a lane could genuinely help with

1. **Chaining dispatches across different resident matrices in CubeCL** without a host sync
   between them. `GpuQ4Mat::launch` already returns a `Handle` and takes one, so the matmul
   chain is fine -- the question is the elementwise/attention steps in between.
2. **`cubecl-cuda` with a reused pinned staging buffer.** The only route we found to
   ~12 GB/s h2d; wgpu caps near 2-2.7 GB/s and has no pinned memory. Irrelevant while
   everything is resident, decisive the moment a model does not fit.
3. **An honest 3-bit or 2-bit dense path.** This is the real route to a bigger model on the
   same card. Must be measured against a frozen held-out set, not vibes.
4. **Prefill GEMM tile.** Prefill is 0.86 s for ~20 tokens; decode is now the fast part.

## Two beliefs that were wrong this week — do not inherit them

* *"Expert residency is the lever."* It is not; a token is only ~13% experts.
* *"The rest of the token is attention and the short-conv operator."* No -- that work is
  **8 ms of 141**. A token is ~94% quantized matmuls. The fix this wrong reading pointed at
  happened to be the right one, which is exactly why it went unnoticed.

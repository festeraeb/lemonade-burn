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

## The two routed-expert implementations, resolved (2026-11-02)

Both exist and **each has what the other lacks**, so neither should be merged as-is. A `git merge`
of `9867255` onto `lfm2-cubecl` produces 8 interleaved conflicts across `lib.rs` and `store.rs`
(the two sides use different buffer conventions in similarly-shaped code) — resolving those by hand
is how you ship the bug below. Port deliberately instead.

**`GpuExpertStack` (branch `feat/expert-stack-alt`, commit 9867255) has the better design.**
`StackMat::upload(gpu, w: QuantMat, n_expert)` takes **one** matrix spanning
`[n_expert * rows, cols]` — the GGUF tensor as it already is on disk — and repacks over the whole
run. `GpuExpertLayer` on `lfm2-cubecl` instead takes an iterator of per-expert `QuantMat`s and
concatenates them, which is wasted work: `ExpertMats::expert(i) == start + i * per_expert` means
the experts are **already contiguous**.

**But `GpuExpertStack` has the `from_raw_parts` element-count bug.** `StackMat` stores
`n_qs: qs.len()` where `qs: Vec<u32>`, while `q4_0_stack_matvec_kernel` takes `qs: &[V4U]` and
`x: &[V4F]`. It then calls `BufferArg::from_raw_parts(self.down.qs, self.down.n_qs)` and
`from_raw_parts(h, n_h)` with **no `/4`** — a 4x too-large element count on both, letting the kernel
index past the logical end. This is the same class of bug fixed on `lfm2-cubecl` in
`fix/buffer-arg-element-counts`, where the counts became `self.n_qs / 4` and `n_h / 4` behind
`debug_assert_eq!(.. % 4, 0)`.

Its single parity test passes anyway, which is the lesson worth keeping: an over-large element count
usually does **not** fault on wgpu — the binding is byte-sized by the handle, so the kernel simply
may index beyond where the data logically ends, and whether that is observed depends on launch
geometry. One happy-path parity test cannot see it. `lfm2-cubecl` has 7 bit-equality tests
(`expert_layer_matches_per_expert.rs`) covering the last expert in the buffer, a repeated expert, a
repeated token, interleaved tokens, out-of-range refusals and the even-block invariant — those are
the ones that would catch it.

**So the resolution is:** keep the stack *design* (one contiguous `QuantMat` per projection, a jobs
buffer of `(expert, row)`), apply the `/4` element counts and the `debug_assert`s to it, delete the
concat-based `GpuExpertLayer` and its `q4_0_routed_*` kernels, and re-point the 7 tests at the stack
API. Then integrate behind `LB_EXPERT_LAYER` so the A/B is same-binary, and gate on the greedy sha
`062392b8d2b0361a` with the 1.1x c2 bar — t440's 25% drift cannot support the projected 1.34x.

## cubecl-cuda measured on c2 (2026-11-03): copy gate fails, decode 1.47x, parity 22/22

Blocker B1 of `~/agentnotes/fable-vs-gemini-engine-compare-2026-11-02.md`: nobody had measured
cubecl-cuda h2d on our cards (the "~12 GB/s" above is raw `cudaMemcpy`; the 46x loss was wgpu).
Commit 603512f adds a `cuda` feature (`LB_CUDA_DEVICE=N`, driver order) beside the wgpu path and
`examples/copy_bench.rs`. Built on t440 with `CUDARC_CUDA_VERSION=12080` (without it cudarc falls
back to CUDA 13, which dropped Pascal); run on **cesarops2**, load 0.1-0.5, GPU2 = Tesla P100 at
bus `0000:0D:00.0`, Gen3 x16, driver 580.178.04. Both runtimes were confirmed on that bus by
`nvidia-smi --query-compute-apps` sampled mid-run (wgpu `discrete:2` happens to be the same card).
c2 has no CUDA toolkit: `libnvrtc.so.12.8.93`, `libnvrtc-builtins` and `nvvm/libdevice/libdevice.10.bc`
from t440's 12.8 are bundled in `~/lb-measure/{lib,cuda-path}` on c2 (`LD_LIBRARY_PATH`, `CUDA_PATH`);
without libdevice every kernel that calls `exp` (the silu-fused ones) fails to compile for sm_60.

**Copy path, c2 GPU2, medians of 20 reps after a warm-up, MB/s** (raw `bench/raw-h2d-bwtest.cu`
on the same card: pinned 11,845 at 5.3 MB / 12,030 at 64 MB; pageable 6,901 / 5,939; the x4 P100 GPU0
is 3,159 pinned):

| path (as the engine would use it) | cuda 5.3 MB | wgpu 5.3 MB | cuda 64 MB | wgpu 64 MB |
|---|---|---|---|---|
| `create(Bytes::from_elems(vec))` fresh Vec (= `Gpu::upload_u32`) | 1,090 | 4,002 | 2,948 | 2,145 |
| `Client::staging` into pooled pinned memory, then `create` | 2,416 | 3,966 | 3,691 | 1,777 |
| `write` into a handle allocated once, fresh Vec | 3,325 | 3,243 | 2,449 | 1,964 |
| `write` into a handle allocated once, pinned-staged | 3,229 | 3,239 | 3,705 | 1,852 |
| `create_from_slice` (double host copy; what lb-gpu-bench measured) | 506 | 504 | 708 | 668 |
| `GpuExpert::upload` minus repack (3 x 1.77 MB; what an on-demand miss pays) | **4,705** | 3,508 | - | - |

Activation handoff through the host (`read_all` + `upload_f32` + sync): m=1 **51.6 us** cuda vs
130.9 wgpu; m=4 62.6 vs 144.1; m=32 156 vs 270.

**Gate (>= 6 GB/s at 5.3 MB): FAIL on every CubeCL path, both runtimes.** Best 5.3 MB number is the
expert upload at 4.7 GB/s on cuda (1.34x wgpu, above the 1.1x bar, real). Readings: (a) the 46x was
`create_from_slice` + t440's host load, and is ~0.5 GB/s here on both runtimes; (b) the runtime switch
does not reach the bus - pre-allocated, pinned-staged `write` still sits at ~3.3 GB/s against 11.8 raw,
so the remaining ~3x is inside CubeCL's per-command write path on both runtimes, and three 1.77 MB
writes (0.38 ms each) beat one 5.3 MB write (1.6 ms) - size-dependent, not per-call. Where that time
goes is **unmeasured**; the next probe is a cudarc-level timer around `memcpy_htod_async` inside
cubecl-cuda's `copy_to_device` (an upstream-side question, not engine code). On-demand misses are
still a non-starter at these rates for the streaming design; static residency is unaffected.

**Parity under cuda on sm_60: 22/22** (gpu_matches_scalar 8, q6k 4, expert_layer 7, conv_block 3;
q8_0's 3 are `#[ignore]`), the same set wgpu passes on the same card. Answers B9: the plane-op
kernels compile and match under the cuda runtime.

**End to end, same binary, same card, `c2_e2e.sh` (32 tokens, temperature 0, 3 reps + warm-up),
2432 experts + 107 dense matrices resident, hit rate 1.0, 0 CPU calls:**

| runtime | decode tok/s (reps) | decode forward ms | prefill s | greedy sha |
|---|---|---|---|---|
| cuda, `LB_CUDA_DEVICE=2` | **16.35** (18.11 / 15.90 / 16.35; warm-up 16.00) | 61.8 (matmul 55.3) | 0.52-0.60 | `062392b8d2b0361a` x4 |
| wgpu, `discrete:2` | 11.09 (11.09 / 11.49 / 11.10; warm-up 11.72) | 90.4 (matmul 84.0) | 0.64-0.70 | `062392b8d2b0361a` x4 |

**1.47x decode on the cuda runtime with byte-identical output**, above c2's 1.1x bar (rep spreads
1.14x and 1.04x, non-overlapping). Only 3 reps a run and one run each; a repeat an hour later would
close it the way the 11-01 bar was closed. The gain is on the resident path (dispatch/sync cost, not
copies), consistent with "a token is ~145 host syncs" above. Raw outputs: c2
`~/lb-measure/out/cuda-b1/`, kit in `~/lb-measure/cuda-bin/`, driver `~/lb-measure/cuda_bench.sh`.

## Probe: where the copy path loses 3x, and the fix (2026-11-03, later the same day)

The 11-03 table above left the ~3x between CubeCL's `create(Bytes)` and raw `cudaMemcpy`
unexplained. Answered with a timer probe inside cubecl-server 0.11.2 `write_to_gpu` and
cubecl-cuda 0.11.2 `copy_to_device` (local patch, `bench/cubecl-probe/`, never upstream),
run on **cesarops2 GPU2** (Tesla P100 at bus `0D:00.0`, Gen3 x16, driver 580.178.04) beside an
idle seat on the same card (0 % util at every start, 5 GB free, which is enough for the blobs).
Host: Xeon E5-2450 v2 (Ivy Bridge, 2.5 GHz), load 0.1-0.8. Raw per-call lines: c2
`~/lb-measure/out/probe-b2/`; medians over the 20 timed reps after one warm-up.

**What the probe measured, 5.31 MB writes, microseconds per call** (`copy_to_device`'s host
pointer was page-locked in 100 % of calls, so the driver side is not it):

| path | reserve pinned | host memcpy into it | enqueue (`memcpy_htod_async`) | DMA (stream sync right after) | whole `write_to_gpu` |
|---|---|---|---|---|---|
| `create(Bytes)` of a fresh Vec, first 21 calls | **2,896** (= one `cuMemAllocHost` of 7.34 MB per call, 2.8-3.1 ms each) | **1,288** | 55 (10) | 516 | 4,248 |
| `write` into a pre-allocated handle, pool warm | 18 | **840** | 52 (8) | 514 | 910 |
| `Bytes` already pinned (`client.staging` first) | 0 | 0 | 67 (11) | 514 | **70** |
| 64 MB fresh Vec | 5 | **17,950** | 125 (28) | 6,418 | 18,236 |
| 1.57 MB expert matrix (3 per expert) | 2 | 246 | 53 (8) | 136 | 310 |

Reading: CubeCL's driver path reaches the bus (5.3 MB in 0.51 ms = **10.3 GB/s**, 64 MB in
6.4 ms = 10.5, 1.57 MB in 0.136 ms = 11.6). Everything lost is host-side and in front of the
DMA: (1) `Staging::of` stages every Native `Bytes` under 100 MB through a pooled pinned slice,
and that slice is only released by the drop queue two flushes (~2 x 64 MiB of uploads) later,
so a cold run page-locks a fresh 7.34 MB page per call at 2.9 ms; (2) the host memcpy into the
slice runs at 4-6 GB/s on this Xeon (0.84-1.29 ms for 5.3 MB, 18 ms for 64 MB), serialised
before a 0.51 ms DMA. The size dependence in the earlier table (three 1.77 MB writes beating
one 5.3 MB write) is this memcpy. Neither is the wgpu runtime's problem specifically; wgpu has
the same shape without pinned memory at all. The probe itself changed the headline numbers by
< 3 % (probe / probe+sync / no-probe runs in `probe-b2`).

**Fix, engine side, no CubeCL patch** (`crates/lb-q4-gpu/src/pinned.rs`, `cuda` feature): a
pool of page-locked buffers on the device's primary context (the one cubecl-cuda retains),
handed out as `&mut [u32]` the repack writes into directly (`repack_q4_0_into` assigns every
word, so a reused slot needs no zeroing), wrapped as `Bytes` with `AllocationProperty::Pinned`
through the public `AllocationController` trait, so `write_to_gpu` skips staging and DMAs from
it; the slot comes back to the pool when CubeCL's drop queue drops the `Bytes` after its fence.
`Gpu::upload_q4_0` uses it for every Q4_0 matrix (residents at load and on-demand misses);
`LB_NO_PINNED=1` disables it, `LB_PINNED_POOL_MB` caps it (256 default; past the cap uploads
silently take the old path, and `copy_bench` prints the pool's size so that cannot hide).
Off the `cuda` feature nothing changes.

**Re-measured, plain build, c2 GPU2, `LB_BENCH_WARM=40` so the pools are past their warm-up
(a cold pool pays the page-lock the probe showed; the first 21-rep run without warm-up was
*slower* with the pool, 4.1 ms, for exactly that reason), 20 timed reps, two pool runs and one
`LB_NO_PINNED=1` run (`~/lb-measure/out/pinned-b7/`):**

| | pool run A | pool run B | no pool |
|---|---|---|---|
| copy by itself: pinned 5.31 MB buffer filled before the clock, `create` + `sync` | **583 us** (465-693) = **9.1 GB/s** | **563 us** (469-647) = **9.4 GB/s** | n/a |
| same, 64 MB | 6.16 ms = 10.9 GB/s | 6.20 ms = 10.8 GB/s | n/a |
| fill 5.3 MB into a pool buffer, alone | 0.27 ms (Vec fill 0.41) | 0.27 | - |
| `h2d_expert_upload` whole (3 repacks + 3 uploads + sync), ms | 1.83 (1.57-1.93) | 1.88 (1.67-2.20) | 2.14 (1.93-2.51) |
| same, earlier run `pinned-b5` | 1.40 (1.22-1.97) | 1.61 (1.33-1.89) | 2.16 (1.95-2.69) |
| engine's generic `create(Bytes)` path, untouched, GB/s | 2.8 | 3.3 | 2.9 |

**Gate >= 6 GB/s at 5.3 MB: PASS on the copy itself** (9.1-9.4 GB/s at median, 7.7 GB/s at
the slowest rep, against the card's 10.3 GB/s DMA). The one-call `h2d_pinned_fill_5mb` number
(4.4-4.5 GB/s net) is lower only because it fills the *next* buffer inside the clock into a
cache-cold slot and subtracts the hot 0.27 ms fill; the split (`create` returns in 2.6 us, the
whole wait is `sync`) is the direct measurement. What an on-demand miss now pays end to end is
1.4-1.9 ms per expert instead of 2.1-2.2 (the repack, 0.26 ms a matrix, is now the larger
half), and the per-token arithmetic in `cubecl-copy-path-46x-loss` is on real bandwidth.

**Parity and sha.** `gpu_matches_scalar` under cuda on GPU2 with the pool on: 8/8 (twice).
`repack_q4_0_into` has a unit test against the previous accumulate-into-zeros form on dirty
slices, odd/even counts and chunk boundaries. End to end on the same card, same config, pool on
(x6 completions) vs `LB_NO_PINNED=1` (x3): greedy sha **`062392b8d2b0361a` in all nine**, the
same as the all-resident reference. The config had to fit beside the live seat on GPU2
(`--gpu-budget-mb 3200 --gpu-dense-mb 600 --miss-policy on-demand`, 602 residents, every other
expert uploaded on demand through the pool each token), and there the pool is worth
**5.48-5.51 tok/s against 3.82-3.85 (1.43x)**; the all-resident 16.3 tok/s configuration was
not re-run (needs ~13.7 GB on a free card) and is unaffected in principle, since residents only
upload at load. Scripts: `bench/cubecl-probe/c2_pinned_run.sh`, `c2_pool_sha_ab.sh`.

**Not done / caveats.** 64 MB uploads only reuse a slot because `flush_after` fires per call;
with the 256 MB default cap a run mixing 5 MB and 64 MB uploads hits the cap and falls back
(seen in `pinned-b4`: 267 MB allocated, 64 MB section at no-pool speed). The engine never
uploads 64 MB blobs, so the default stands; raise `LB_PINNED_POOL_MB` for benches. The pool
holds ~130-360 MB of page-locked host memory in steady state (the drop queue's two batches);
on a 46 GB host that is fine, on a small box it is a knob. The `h2d_expert_upload` gain varied
between runs (1.15x and 1.5x); the e2e 1.43x is the number to quote, taken at c2's 2.7 % drift.

**Scraper check (memory `moe-scraper-rule`), same day.** lemonade-burn already carries the
NautiBurn writer (`lb-lfm2/src/moe.rs` `Scraper`, env `NAUTIBURN_SCRAPE_DIR` / `_MODEL_ID` /
`_CLASS_ID`, default class `lb-lfm2`); `lb-server` flushes it once per request in
`after_request`. Verified on c2 with the on-demand config above: two completions wrote
`run-1793682926-0.json` and `run-1793682933-1.json` (schema 1, model `LFM2.5-24B-A2B-QAD-Q4_0`,
2,014 activations = 53 positions x 38 MoE layers, top-4 `expert_ids`). Two gaps, named: the
row's `phase` is always `"prefill"` and `token_ids` holds only the prompt although the
activations cover prefill *and* decode (one row per request); and no lemonade-burn seat is
running anywhere on the fleet, so nothing emits today, while the ingest side
(`nautiburn-moe-consume.service`, `/mnt/fast/sidechannel.sqlite` at 0 bytes) is down
regardless of the writer.

**All-resident reference re-run (same day, after the coordinator freed c2 GPU2; strand
stopped, card at 5 MiB, qwq on GPU0 untouched).** Same binary, `--gpu-budget-mb 14500
--gpu-dense-mb 900`, 2432 experts + 107 dense resident, 32 tokens, temperature 0, 3 reps +
warm-up per server, `~/lb-measure/out/ref-full-pool.out`, `ref2-full-nopool.out`,
`ref2-full-pool-rep2.out`:

| server | decode tok/s (warm-up; reps) | load (incl. 12.9 GB upload) | greedy sha |
|---|---|---|---|
| pool on | 16.2; 15.9 / 19.5 / 15.5 | **8.3 s** | `062392b8d2b0361a` x4 |
| `LB_NO_PINNED=1` | 16.4; 23.1 / 21.5 / 16.5 | 10.8 s | `062392b8d2b0361a` x4 |
| pool on, again | 18.0; 16.5 / 17.4 / 18.7 | **8.3 s** | `062392b8d2b0361a` x4 |

Sha unchanged in all 12 completions against the reference. Decode spreads (15.5-23.1) overlap
and all are at or above the 16.3 of commit 603512f: **no decode change all-resident**, as
expected, since residents upload once at load; the host was not at its usual quiet (load
0.35-1.4 from these back-to-back servers), so no decode delta under ~1.5x is claimable from
this run. What the pool does buy here is the **load: 8.3-8.9 s against 10.8-11.5 s** (two
servers each), the upload of the 2432 residents. Lesson for the harness: two servers started
back to back on one card OOM'd (`out of device memory allocating 2097152 bytes`, both pool
on and off) because the previous 13 GB context was still tearing down; `c2_e2e.sh` callers
must wait for the card to read empty first.

## The repack lever: misses now DMA raw bytes from the page cache and repack on the device (2026-11-03, evening)

Raw outputs on c2: `~/lb-measure/out/repack-*` (host repack), `devrepack-b1/` (parity, miss bench,
nine e2e runs, `devrepack-ab.log`), `register_bench` lines in `docs` below. Scripts:
`bench/cubecl-probe/c2_devrepack_ab.sh`; benches `examples/repack_bench.rs`, `miss_bench.rs`,
`register_bench.rs`. Host: cesarops2 GPU2 (P100, Gen3 x16), beside the live llama-server seat on
the same card (so the on-demand config, 602 residents, as in the pool A/B), load 0.05-9 during
the back-to-back servers; the e2e deltas below are far above the 1.4-2.7 % this box drifts.

**First, a correction to this morning's split.** The "repack 0.26 ms a matrix" quoted above came
from `copy_bench`, which timed the *allocating* `repack_q4_0` (a fresh `Vec` per call: page
faults) and subtracted it. The pool path never paid that. Measured properly (`repack_bench`,
`repack_q4_0_into` into reused words, 40-rep medians, three runs each, output FNV identical):

| repack of one expert (3 matrices), ms | 16 threads | 8 | 1 |
|---|---|---|---|
| old inner loop (byte-wise shifts) | 0.297 / 0.297 / 0.301 | 0.303 / 0.302 / 0.315 | 1.793 / 1.795 / 1.797 |
| new (SSSE3: one `pshufb` 4x4 byte transpose + nibble packs per block; scalar twin for the unit test) | **0.193 / 0.191 / 0.191** | **0.182 / 0.206 / 0.183** | **1.039 / 1.040 / 1.040** |
| allocating `repack_q4_0`, either loop | 0.64-0.73 | 0.64-0.73 | 1.27-1.58 |

So the host repack was 0.10 ms a matrix, is now 0.063 (1.57x; 1.73x single-threaded), and the
scalar bit-trick fallback is *slower* than the old loop (3.48 ms at 8 threads) so it is only the
non-x86 path. That is a small lever: the miss's real cost was elsewhere.

**Where a miss actually goes (`miss_bench`, 30-rep medians after 40 warm, `LB_PINNED_POOL_MB=768`,
expert slices carved from the model's largest Q4_0 tensor so the bytes are the file's):**

| per expert, ms | run 1 | run 2 | run 3 (bench fixed, see note) | run 4 |
|---|---|---|---|---|
| host path, whole (`GpuExpert::upload` + sync; repack into pool, six `create`s) | 1.310 / 1.353 | 1.343 / 1.359 | 1.441 / 1.409 | 1.366 / 1.366 |
| host repack alone (three `repack_q4_0_into` into pool slots) | 0.203 | 0.200 | 0.215 | 0.203 |
| host DMA alone (six `create`s of pre-filled pool buffers + sync) | (timed the fill too) | (same) | 0.588 | 0.605 |
| **device path, whole (three raw `create`s from the registered mapping + repack kernel + sync)** | **0.635 / 0.629** | **0.642 / 0.666** | **0.653 / 0.666** | **0.657 / 0.643** |
| device DMA alone (three raw `create`s + sync) | 0.590 | 0.577 | 0.588 | 0.589 |

Reading: on the host path, repack (0.20) plus DMA (0.59) leave ~0.6 ms that is in neither alone
(the interleaving of repack and `create`, or the pool's take between them); not chased, because
the device path makes it moot. On the device path the repack kernel plus freeing costs 0.05 ms
over the DMA, and the DMA of 3 x 1.77 MB raw at 0.59 ms is 9.0 GB/s, the same bus speed the copy
gate measured this morning. Register of the whole 12.9 GB mapping: 737 ms.

**What was built (`lb-q4-gpu`, cuda feature; nothing changes off it):**
- `registered.rs`: `cuMemHostRegister(CU_MEMHOSTREGISTER_READ_ONLY)` on the GGUF's `Mmap`
  (`Gguf::mapping()`), once, at `ExpertStore::open_gpu` (`LB_NO_REGISTER=1` skips it). A slice
  inside it becomes CubeCL `Bytes` reporting `Pinned` through the public `AllocationController`
  trait (`memory_mut` refuses: the mapping is read-only), so `write_to_gpu` hands the pointer
  to the driver as it is. `register_bench` on GPU2: both driver attributes 1, 1 GB in 44.5 ms,
  whole file 602.7 ms (47.7 ms/GB), unregister 103 ms; 1.77 MB slice copies **11.25 GB/s
  (0.157 ms)** registered vs 4.6-4.9 GB/s (0.36-0.39 ms) from the same mapping unregistered.
- `q4_0_repack_kernel` (`lib.rs`): one unit per block pair reads nine raw words and writes
  eight `qs` words and one `sc` word; the same arithmetic as the host `pack_block`.
  `Gpu::repack_on_device`, `upload_q4_0_raw` (any runtime, for tests), `read_u32`.
- `Gpu::upload_q4_0` takes the device path when the bytes lie in a registered mapping and the
  block count is even (every matrix in this model); `LB_HOST_REPACK=1` / `set_host_repack`
  is the A/B switch in one process. Residents at load take the same path.
- `Gpu::const_u32`: kernel `dims`/count buffers shared per value, so a miss uploads no tiny
  buffers (it uploaded three before).
- `copy_bench` now times the reused-buffer repack, so its `copy_ms` no longer hides page faults.

**Parity:** `tests/device_repack_matches_host.rs` (device words == host words at 2, 64, 510,
4096 and 98,304 blocks) passes on GPU2; `gpu_matches_scalar` 8/8 on the same card; the host
unit test (SIMD, scalar and reference agree on dirty slices across chunk boundaries) passes.

**End to end, same binary, same card, same on-demand config as the pool A/B (`--gpu-budget-mb
3200 --gpu-dense-mb 600 --miss-policy on-demand`, 602 residents, 9,465 misses and 8,682 hits per
32-token completion), three interleaved rounds x 2 timed completions:**

| server | decode tok/s (6 completions) | `on_demand_s` per run -> per miss | load | greedy sha |
|---|---|---|---|---|
| previous build (`fbdec14`, pool, old repack) | 5.13 / 5.18 / 5.47 / 5.43 / 5.25 / 5.28 | 10.97 / 9.57 / 10.48 s -> 1.01-1.16 ms | 2.4-2.5 s | `062392b8d2b0361a` x9 |
| this build, `LB_NO_REGISTER=1` (SIMD host repack, pool) | 5.43 / 5.44 / 5.56 / 5.56 / 5.59 / 5.56 | 9.31 / 9.00 / 8.96 s -> 0.95-0.98 ms | 2.3 s | `062392b8d2b0361a` x9 |
| **this build, mapping registered (device repack)** | **7.40 / 7.43 / 7.54 / 7.53 / 7.57 / 7.48** | **2.66 / 2.65 / 2.65 s -> 0.280 ms** | 2.2 s (+0.74 s register) | `062392b8d2b0361a` x9 |

**Gate, as set before the run: a served miss at no more than half the previous cost per expert,
output sha unchanged.** PASS: 0.28 ms against 1.01-1.16 (3.6-4.1x) by the engine's own counter;
the bench's per-expert-synced whole is 0.63-0.67 against 1.31-1.44 (2.1x). Decode in this config
**1.40x** over the previous build (7.40-7.57 vs 5.13-5.47), sha identical in all 27 completions.
The SIMD host repack on its own is 0.95-0.98 vs 1.01-1.16 ms a miss and 5.43-5.59 vs 5.13-5.47
tok/s: real on the counter, but not a decode delta this box can swear to; it stays as the
fallback (wgpu, `LB_NO_REGISTER`, odd block counts).

**Not done / caveats.** The all-resident 16 tok/s reference was not re-run (needs the card
free; residents upload once so decode is unaffected in principle; load should drop since the
2432 uploads now skip the host repack, unmeasured). The page-lock holds the whole GGUF (13.58 GB
here) in RAM for the model's lifetime: fine on 46-48 GB hosts, a knob (`LB_NO_REGISTER`) on a
small one; it pins pages the hot mapping would hold anyway. `cuMemHostRegister` is per process
and per device context: a second `Gpu` on another card would register again (not exercised).
Load went 2.3 -> 2.2 s plus 0.74 s to register at this 602-resident size, i.e. slightly longer
end to end at small residency; at full residency the 2432 skipped host repacks should repay it
(unmeasured). The pool path's unexplained ~0.6 ms stays unexplained.

**Next lever on a miss, measured not guessed:** `create` count. Three raw `create`s cost 0.59 ms
for 5.3 MB (0.197 each, of which 0.157 is bus time); the host path's six cost about the same
per call whatever the size (the 196 KB `sc` upload costs what the 1.57 MB `qs` one does). One
`create` per expert (gate/up/down are separate tensors, so that needs a gather on the host or
three-handle kernels) or a batched write in CubeCL would take a miss toward the 0.47 ms of
pure bus time; and layer-ahead prefetch (stage 5) now has a 0.28 ms miss to hide instead of 1.1.

## c4 is the measurement host now; the llama.cpp gap on the same model and card (Fable, 2026-11-04)

**Host change.** c2's P100s went to forge seats, so lemonade-burn measures on **cesarops4om GPU 3**
(Tesla P100-PCIE-16GB, bus AF:00.0, Gen3 x16; `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3
LB_CUDA_DEVICE=0`). Everything local under `/mnt/models/lb-measure/` (`models/`, `bin/`, `lib/`
= CUDA 12.8 nvrtc from c2, `scrapes/`, `c4_e2e.sh`, `c4_llama.sh`). c4 is a shared box: the `kat`
seat on GPU 0, `injector-qwen35` on GPU 2 and the CPU-only `flint` seat (40 threads, bursts) run
beside these numbers, which is why load is printed with every one. Load 2-3 idle, 7-12 while
flint answers the forge; the lemonade-burn server itself runs 80 rayon threads here.

**Step 1: where we stand against llama.cpp, like for like.** Same GGUF
(`LFM2.5-24B-A2B-QAD-Q4_0`, 13.58 GB), same card, same prompt, 32 greedy tokens through each
server's `/v1/completions`, two interleaved rounds, 5 timed completions a round after one warmup.
llama.cpp is the fleet's own build (`20261027-nautiburn-flint-cuda12.8-sm60_61_75_86-cpuall`,
`-ngl 99 -fa on -c 2048`, 13.36 GB on the card). lemonade-burn is HEAD `eec4500` (cuda feature).

| engine, config | decode tok/s, round 1 | round 2 | prefill s | load (1 min) | greedy sha |
|---|---|---|---|---|---|
| **llama.cpp**, all layers on the card | 66.16 / 66.21 / 66.17 / 66.21 / 66.18 | 66.73 / 66.73 / 66.70 / 66.73 / 66.72 | 0.209 | 2.4-2.5, 8.0 | `136ea2355506fcef` x10 |
| **lemonade-burn**, all 2432 experts + 751 MB dense resident (`--gpu-budget-mb 14500 --gpu-dense-mb 900`) | 18.37 / 22.50 / **27.68** / 22.67 / 24.27 | 18.21 / 17.93 / 18.69 / 18.75 / 18.79 | 0.55-0.65 | 2.2-2.8, 6.5-6.8 | `062392b8d2b0361a` x10 |
| lemonade-burn, 602 residents, misses copied in (`3200/600 --miss-policy on-demand`) | 7.93 / 7.74 / 7.65 | 7.91 / 7.66 / 8.04 | 1.14-1.18 | 2.9-9.2, 10-11 | `062392b8d2b0361a` x6 |

So on this model **llama.cpp decodes 66.2-66.7 tok/s against lemonade-burn's 18-19 (3.5x), or
18-28 in a round where the box was quieter** (round 1's 27.7 is a single completion; round 2 at
load 6.5-6.8 held 17.9-18.8 with a 1.05x spread, which is the number to quote). The two engines'
greedy texts share the first 60+ characters and diverge later (different f32 summation order in
the kernels: sha differs, both self-consistent across every rep). Per decode token lemonade-burn
reports 48.4-54.4 ms forward of which 41.8-46.5 ms is "matmul" time (`/v1/placement`): with
every expert resident that is the 411 dispatches + 145 syncs of the audit above, not the misses.
The on-demand config costs 131 ms a token, of which `on_demand_s` says 4.16 s / 12620 misses =
**0.33 ms a miss** on this card (0.28 on c2's slot).

Caveat for the KAT/Gemma numbers in the task brief (31.7 and 24.6 tok/s): those are different,
larger-active models (A3B / A4B); this table is the only like-for-like pair we have.

**Miss floor on c4 before any change** (`miss_bench`, HEAD build, GPU 3, 3 runs, load 1.95-2.0):

| per expert, ms (median of 30 after 40 warm) | run 1 | run 2 | run 3 |
|---|---|---|---|
| device path, whole (3 `create`s + 6 `empty`s + 3 repack launches + sync) | 0.927 / 0.960 (rep2) | 0.946 / 0.959 | 0.947 / 0.925 |
| device DMA alone (3 raw `create`s + sync) | 0.843 | 0.704 | 0.812 |
| host path, whole | 1.843 / 2.003 | 1.947 / 2.112 | 1.981 / 1.988 |
| host repack alone | 0.471 | 0.443 | 0.555 |
| host DMA alone (6 pool `create`s) | 0.896 | 0.885 | 0.916 |

c4's slot moves 5.3 MB in 0.70-0.84 ms (6.3-7.5 GB/s) against c2's 0.59 (9.0 GB/s), same Gen3
x16: a slower root complex on this box. The whole minus the DMA leaves **~0.12-0.24 ms** for the
nine remaining calls, which is the most the create-count lever can return here. Run-to-run
spread of the device whole: 0.925-0.960 (3.8%).

**Gate for the create-count change, written before the run:** (a) the parity check in
`miss_bench` passes (batched words == per-matrix words, batched expert output == per-matrix
output, bit for bit); (b) `device_batched_whole` is below `device_whole` by at least 0.08 ms
(twice the spread) in each of three runs, both arms interleaved inside the same run; (c) greedy
sha `062392b8d2b0361a` in every e2e completion, both arms, on-demand and all-resident configs;
(d) e2e `on_demand_s` per miss no worse than the per-matrix arm. Anything short of all four is
reported as not passing.

### Step 2, the create-count lever: built, parity passes, the miss gets 2-7% faster, which is under the gate

**What was built** (`lb-q4-gpu` `store.rs`, cuda feature): `Gpu::upload_expert_batched` serves a
miss's three matrices with **three** device calls instead of twelve: one `client.create_tensors`
DMAs the three raw slices out of the page-locked mapping into one allocation (three
`memcpy_htod_async` on the stream, one submission), one `client.empty_tensors` reserves the packed
`qs`/`sc` words of all three, and one `q4_0_repack_kernel` launch covers the lot when the
allocator placed the raw slices back to back (it does: 1,769,472-byte slices, 256-aligned; three
launches otherwise). Each `GpuQ4Mat` then owns an offset view (`Handle::offset_start/offset_end`)
of the shared buffers. `GpuExpert::upload` takes it when the bytes are registered, the block
counts are equal and even, and neither `LB_HOST_REPACK` nor `LB_NO_BATCH_MISS` is set; residents
at load go through the same call. `miss_bench` grew `device_batched_*` sections and a parity
check (batched words == per-matrix words for gate/up/down, and the same expert run on the same
input gives bit-identical output).

**Measured, c4 GPU 3, `miss_bench` new build, three runs, both arms interleaved in each run
(medians of 30 after 40 warm, ms per expert; load 2.8-5.3):**

| run | batched, whole | per-matrix, whole | diff | batched rep2 | per-matrix rep2 | diff | DMA alone | parity |
|---|---|---|---|---|---|---|---|---|
| 1 | **0.909** | 0.928 | -0.019 | **0.890** | 0.961 | -0.071 | 0.724 | ok |
| 2 | **0.914** | 0.937 | -0.023 | **0.809** | 0.835 | -0.026 | 0.718 | ok |
| 3 | **0.915** | 0.948 | -0.033 | **0.774** | 0.798 | -0.024 | 0.706 | ok |

Batched is lower in **6 of 6** interleaved pairs, by 0.019-0.071 ms (2-7%). The gate asked for
0.08 ms in every run; **it does not pass (b)**. (a) passes. Reading: nine of the twelve calls
were cheap; the cost that is not bus time (whole minus DMA alone: 0.17-0.24 ms) stays, so it
is not the call count. Candidates for that remainder, unmeasured: the allocation (the memory
pool's reserve for a fresh 5.3 MB + 4.2 MB on every miss, released after the sync), the three
separate `memcpy_htod_async` (each a driver round trip even inside one submission), and the
drop of the raw buffer. A true single copy would need the three slices contiguous in the file,
which gate/up/down are not. The path stays in as the default because it is never slower and
the parity is proven, with `LB_NO_BATCH_MISS=1` / the live `batch_miss` knob as the switch back.

## Device-resident decode (D1) and rows-per-plane kernels (D2): built, parity passes on the 1070, speed unproven until D3 (2026-11-07)

Branch `fable/device-decode-2026-11-07` (off `fable/tunables-2026-11-04`), commits `5fedfe2` (D1
kernels), `53b4780` (the token loop), `4b76e80` (D2). Spec: `docs/DEVICE_DECODE_PLAN.md`. Test card:
**cesarops1's GTX 1070** (8 GB, nvidia-smi 0, `LB_WGPU_DEVICE=discrete:0` / `LB_CUDA_DEVICE=0`
with c4's CUDA 12.8 nvrtc in `/mnt/models/lb-dd/lib`), model on c1's local disk
(`/mnt/models/lfm25/`). No P100 was touched. Everything below is off by default.

### What was built

* **Kernels** (`lb-q4-gpu/src/decode.rs`): `add_rms_norm` (residual add fused into the next
  pre-norm), `qk_norm_rope` (per-head rms + NEOX RoPE + KV append), `attn_decode` (online
  softmax, one cube per query head, GQA), `router_topk` (`route_sigmoid`'s rule: rank by
  `p + bias`, lower id on ties, `/ max(sum, 2^-14)`), `weighted_sum` (ascending expert id, the
  order `moe_ffn` adds in). `GpuExpertLayer` gained `upload_stacked` (the GGUF stack as one
  `QuantMat` per projection), `expert(e)` offset views and `launch_device` on device-written
  routes at a slot base.
* **The token loop** (`lb-lfm2/src/device_decode.rs`), `LB_DEVICE_DECODE=1` or the live knob
  `device_decode`: embedding upload, then per layer norm / conv or attention / norm / dense FFN
  or router + stacked experts + weighted sum, final norm, head, and **one read** for the
  logits plus every MoE layer's routes and weights; the scraper is fed from those routes in
  layer order. About **388 launches and 1 sync** a token when everything is resident (30 conv
  x 3 + 10 attention x 6 + 80 norms + 2 dense FFN x 2 + 38 MoE x 4 + final norm + head; the
  plan guessed ~430).
* **State**: `Cache` keeps a device copy (KV per attention layer, conv state per conv layer) and
  `host_len`; a device token uploads the host's state when its copy is behind, a host pass reads
  the device's back first when the host is behind. The two paths can alternate token by token.
* **Residency**: `LB_EXPERT_LAYERS` (unset: follows `LB_DEVICE_DECODE`) uploads a MoE layer whose
  experts are all placed as one stacked buffer per projection; the old path's per-expert entries
  are views into it, so **both arms read the same device memory**. `Placement::whole_layers` /
  `--whole-layers` places whole layers in order within a budget.
* **Fallback**: a token takes the old path, unchanged, when a dense matrix or the head is not
  resident, `LB_CONV_BLOCK` is on, or a MoE layer is not stacked-resident (reason logged once,
  `device_decode_fallbacks` in `/v1/placement`). `LB_DD_HOST_MOE=1` instead runs such a layer on
  the host path *inside* the device token (one extra sync for that layer): the only way a card
  smaller than the model runs the device path. `LB_DD_ROUTER_SEQ=1` computes router logits in
  the host's summation order.
* **D2**: `q4_0_matvec_rows_kernel` / `q4_0_gate_up_silu_rows_kernel`, r = 4 or 8 rows per plane,
  every load of a block issued before the FMAs, one `plane_sum` per 4 rows, optional x in shared
  memory; one comptime `routed` flag so dense, per-expert and stacked-layer paths all switch.
  `LB_Q4_ROWS` (default **1 = the old kernels**), `LB_Q4_XSHARED`, `LB_Q4_XSHARED_MAX_COLS`.

### Where the code disagreed with the plan (the code won)

1. **The model never used `GpuExpertLayer`.** Residents were one `GpuExpert` per (layer, expert)
   in a `HashMap`; the layer type existed only in tests, host-concatenating per-expert repacks.
   Device routes cannot pick a per-expert buffer, so the stacked layer had to become a residency
   mode (`LB_EXPERT_LAYERS`) with per-expert views for the old path.
2. **`GpuConv` is not the conv path to reuse**: it exists only under `LB_CONV_BLOCK` (off,
   measured 0.90x) and owns its own copies of the projections. The device token uses the loose
   resident `in_proj`/`out_proj` + `conv_decode` + its own state, and declines when
   `LB_CONV_BLOCK` is on (two device conv states would disagree).
3. **RoPE from the host's f64 table**, uploaded once per cache size, not cos/sin in-kernel: the
   rotation is then the host's to the bit. No per-token host table, which was the point.
4. Position values go in one per-token buffer (`TokenParams`) written once a token: a
   `const_u32` per position would have left a buffer behind for every position ever decoded.

### Measured: correctness (GTX 1070)

Kernel tests (`decode_kernels_match_host.rs`, 7): **7/7 wgpu, 7/7 cuda**; worst |gpu-host|
4.8e-7 (q after norm+RoPE), attention 1-257 positions at 32:8 / 4:2 / 8:8 heads within 1e-5
relative; router selection exact on 20 random routers, an exact tie across the top-k cut, a
bias-forced expert and the 2^-14 floor, both logit variants. Stacked layer + views + device
routes **bit-identical** to the per-expert path. D2 (`rows_kernels_match_one_row.rs`, 3): every
arm **bit-identical** to the one-row kernel on both runtimes; the decode, expert_layer and
gpu_matches_scalar suites pass at `LB_Q4_ROWS` 1, 4 and 8 on both.

Real model, `lfm2-check --ab-device` (old arm = knob off, device arm = on, same process, same
residency): dense 751 MB + head, MoE layers 2-17 stacked on the card (16 of 38, 5436 MB), the
other 22 on the host inside the device token (`LB_DD_HOST_MOE=1`). Teacher-forced token by token
over the three prompts (13 + 29 + 44 = 86 positions), then 32 greedy tokens of c4_e2e's prompt:

| runtime | top-1 old==device | max \|logit diff\| | route activations differing | alternating arm | greedy 32 | sha both arms |
|---|---|---|---|---|---|---|
| wgpu | **86/86** | 3.60e-5 | **0 of 3268** | 86/86 (2.48e-5) | identical | `062392b8d2b0361a` |
| cuda | **86/86** | 3.95e-5 | **0 of 3268** | 86/86 (3.00e-5) | identical | `062392b8d2b0361a` |
| cuda, `LB_DD_ROUTER_SEQ=1` | 86/86 | 2.24e-5 | 0 | 86/86 (3.56e-5) | identical | `062392b8d2b0361a` |

`062392b8d2b0361a` is the old path's greedy sha on c4's P100 as well. The route comparison is the
scrape file's content (layer, position, experts) compared in process. **Fallback case**
(partial residency, `LB_DD_HOST_MOE` off): 13/13 tokens declined, logits **bit-identical**
(max diff 0), routes identical. **lb-server**, `device_decode` posted live between
completions: sha `062392b8d2b0361a` x4, 62 device tokens, 0 fallbacks; and all 30 completions
of the e2e table below give the same sha.

### Measured: D2 kernel A/B (GTX 1070, roofline 256 GB/s)

`examples/rows_bench.rs`, arms interleaved in one process on one upload after a 2 s warm-up,
median of 30 reps per arm, **3 runs** (separate processes), median [min-max] of the 3. "chain" =
16 launches of the matrix, one readback, per launch (a single launch + readback is mostly
round-trip latency: on wgpu every dense shape sat at 0.16-0.46 ms whatever the kernel). Load
0.23-0.51 (wgpu runs), 0.67-0.89 (cuda runs).

| | rows1 (old) | rows4 | rows8 | rows4 + x shared |
|---|---|---|---|---|
| **wgpu distinct24_m1**, ms/expert | 0.1032 [0.1004-0.1033], 51.4 GB/s | **0.0887** [0.0885-0.0907], 59.9 GB/s, **1.16x** | 0.1148, 46.2, 0.90x | 0.1050, 50.6, 0.98x |
| **cuda distinct24_m1**, ms/expert | 0.0670 [0.0670-0.0670], 79.2 GB/s | **0.0511** [0.0511-0.0516], 103.9 GB/s, **1.31x** | 0.0850, 62.4, 0.79x | 0.0525, 101.2, 1.28x |
| cuda in_proj 6144x2048 chain | 83.7 GB/s | 118.0, 1.41x | 117.7, 1.41x | 116.4, 1.39x |
| cuda out_proj 2048x2048 chain | 50.6 [rows1 range 0.0331-0.0467 ms: bimodal] | 68.4, 1.35x | 68.0, 1.35x | 67.5, 1.34x |
| cuda kv 512x2048 chain | 35.9 GB/s | 33.6, **0.93x** | 34.1, 0.95x | 33.5, 0.93x |
| cuda ffn_down 2048x11776 chain | 91.5 GB/s | **158.4, 1.73x** | 122.0, 1.33x | 157.8, 1.72x |
| cuda ffn_up 11776x2048 chain | 91.3 GB/s | 138.0, 1.51x | 137.8, 1.51x | 136.8, 1.50x |
| wgpu in_proj chain | 60.8 GB/s | 77.5, 1.27x | 76.8, 1.26x | 67.8, 1.12x |
| wgpu ffn_down chain | 80.2 GB/s | 86.4, 1.08x | 103.8, 1.29x | 87.0, 1.08x |
| wgpu ffn_up chain | 74.7 GB/s | 102.5, 1.37x | 102.2, 1.37x | 87.2, 1.17x |
| wgpu kv 512 chain | 15.9 GB/s | 15.2, 0.96x | 11.2, 0.70x | 11.6, 0.73x |

Reading: **rows4 is the pick** (rows8 loses on the expert FFN, where gate + up hold 16 partial
vectors a lane; x in shared memory adds nothing). On the quantity the plan named, distinct24_m1,
rows4 is **1.16x (wgpu) and 1.31x (cuda)**: run-to-run spread is under 2% on this quiet box, so
the gain is real here, but it is **under 1.5x and it is one card**; the best kernel reaches 158
GB/s = **62% of the 1070's roofline** (ffn_down chain), the expert FFN 104 GB/s = 41%. The
512-row K/V matrices get slower (too few row groups to fill the card): a per-shape r is the
obvious follow-up. On cuda a *single* launch + readback of the big shapes is bimodal (ffn_up
rows1 0.366 ms vs 0.148 per launch in a chain; ffn_down rows4 0.100): the host sync's wake-up
quantises, so those single-launch ratios (3.3-3.7x) are not quoted. Default stays `LB_Q4_ROWS=1`
until rows4 wins on a P100 too.

### Measured: end to end on the 1070 (not a speed claim)

lb-server, whole-layer residency as above (22 of 38 MoE layers on c1's Xeon E5-2680, 32 threads,
in every config), one server per config, 1 warm-up + 3 completions of 32 tokens, rounds
interleaved:

| config | wgpu tok/s | cuda tok/s | load (1 min) |
|---|---|---|---|
| old path (`LB_DEVICE_DECODE=0`) | 8.87 / 8.95 / 8.94 | 10.94 / 10.72 / 11.10 | 2.7-10.4 |
| device decode, rows1 | 10.92-11.94 (2 rounds) | 13.19-13.40 | 10-15 |
| device decode, rows4 | 11.52-12.17 | 12.97-13.83 | 9-18 |

`/v1/placement`: 124 device tokens a config, 0 fallbacks, 80-91 ms a device token. The load is
the host MoE layers' own rayon threads. This config is CPU-bound by construction, so it says
nothing about the P100: rows4 vs rows1 does not show at all, and device vs old (1.2-1.3x) is
mostly 22 per-layer round trips becoming cheaper ones. **No tok/s claim comes from this table.**

### What did not pass or was not done

* **No all-resident run.** The 1070 cannot hold the model, so the path the plan is about (zero
  host layers, one sync a token) has never run end to end on the real model; it is the same code
  minus the host-layer branch, and every kernel in it is tested, but its speed is unmeasured.
* **Stacked residency memory on a 16 GB card is unknown.** The per-expert all-resident config
  used 15,291 MiB of the P100's 16,280; the batched-miss path once cost +0.4 MiB an expert from
  pool rounding and ran out. Stacked layers are 3 x ~113 MB allocations a layer instead of 192
  small ones: it may cost more or less. D3 must look before it measures.
* D2 under 1.5x on distinct24 on this card; nothing claimed for the P100.
* The plan's in-kernel cos/sin was not built (table instead, see above).
* lfm2-check `--ab-device` compares against our old path, not the llama.cpp oracle (the old
  path's 75/86 against the oracle is unchanged by construction when top-1 is equal everywhere).

### What Phase D3 must run on a P100 (c4 GPU 3, after asking on Buzz #human-questions)

1. Build the cuda lb-server from this branch (`CUDARC_CUDA_VERSION=12080`), into
   `/mnt/models/lb-measure/bin/`.
2. **Memory first**: `LB_EXPERT_LAYERS=1` with `--placement-from-scrapes <scrapes>
   --gpu-budget-mb 14500 --gpu-dense-mb 900`: every layer must come up stacked
   (`uploaded 2432 experts`, `/v1/placement`), and `nvidia-smi` used vs the per-expert
   15,291 MiB. If it does not fit, that is the first finding.
3. **Parity on the card**: `lfm2-check --ab-device --tokenizer .../tokenizer.json --gpu
   --placement-from-scrapes ... --gpu-budget-mb 14500 --gpu-dense-mb 900` with
   `LB_EXPERT_LAYERS=1` and *without* `LB_DD_HOST_MOE` (so a token that is not fully resident
   falls back and shows as `fallbacks`): 86/86, routes 0 differing, greedy identical, and
   `device tokens == positions`.
4. **Speed**: `c4_e2e.sh` / `c4_llama.sh`, 2 interleaved rounds x 5 completions, arms
   (a) `LB_DEVICE_DECODE=0 LB_Q4_ROWS=1` (the 18-19 tok/s baseline), (b) `LB_DEVICE_DECODE=1
   LB_Q4_ROWS=1`, (c) `LB_DEVICE_DECODE=1 LB_Q4_ROWS=4`, (d) `LB_DEVICE_DECODE=0 LB_Q4_ROWS=4`,
   `LB_EXPERT_LAYERS=1` in all four so residency is identical; gate: sha `062392b8d2b0361a` in
   every completion, `device_decode_fallbacks` 0 in (b)/(c), load printed, quieter round quoted.
5. `examples/rows_bench.rs` 3 runs on the P100 (roofline 732 GB/s), to decide `LB_Q4_ROWS`'s
   default and whether the 512-row shapes need their own r.

## D3 on the P100: device decode 79-82 tok/s, llama.cpp 82.0 in the same session; defaults flipped (2026-11-07, 18:10-18:24Z)

**Card.** c4's Tesla P100 at **AF:00.0**, pinned by UUID (`CUDA_VISIBLE_DEVICES=GPU-fd9b03e5-aed9-cbca-a57b-efc90d61addd`,
`LB_CUDA_DEVICE=0`). **c4's card set has changed since 11-04:** `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3`
now resolves to the **RTX 2060 at D8:00.0** (PCI order today: 18:00 P100, 3B:00 T10, AF:00 P100, D8:00 2060), so
`c4_e2e.sh` / `c4_llama.sh`'s default `CUDA_DEV=3` is stale and points at the wrong card. Every run below set
`CUDA_DEV` to the UUID (`/mnt/models/lb-measure/d3_common.sh`). Binaries built on t440 from this branch,
`bin/lemonade-burn-server-dd` (sha256 `895a2c15581eaa53`, `LB_*` set per arm), `-dd2` (the new defaults),
`lfm2-check-dd`, `rows_bench-dd`; the old binaries are untouched. Le Chonk was stopped for the window; c4's
CPU seats ran beside (load 1.1-2.5 for the e2e, 5-10 for rows_bench). Raw: `/mnt/models/lb-measure/out/d3-*`.

**Step 2, memory.** Same placement (2432 experts from the 2 scrape files + 751 MB dense + head):
`LB_EXPERT_LAYERS=1` (stacked) **13,775 MiB** at load / 13,825 after 186 decode tokens; per-expert **15,291 /
15,809 MiB**. Stacking saves ~1.5 GB (the ~14,600 separate buffers' pool overhead), so it fits with room.

**Step 3, parity on the card** (`lfm2-check --ab-device`, everything resident, no `LB_DD_HOST_MOE`):
**86/86 top-1, max |logit diff| 2.34e-5, 0 of 3268 route activations differing**, alternating device/old every
token 86/86 (2.09e-5), greedy 32 tokens identical (sha `062392b8d2b0361a` both arms), **every token on the
device path, 0 fallbacks**.

**Step 4, end to end** (`c4_e2e.sh` protocol: same GGUF, prompt, 32 greedy tokens, one warmup + 5 timed
completions, two interleaved rounds, llama.cpp first in each round, lemonade-burn arms a-b-c-d then d-c-b-a;
`--gpu --placement-from-scrapes scrapes --gpu-budget-mb 14500 --gpu-dense-mb 900 --miss-policy on-demand`,
`LB_EXPERT_LAYERS=1` in every lemonade-burn arm so residency is identical):

| engine, config | decode tok/s, round 1 | round 2 | prefill s | load (1 min) | greedy sha |
|---|---|---|---|---|---|
| **llama.cpp** (fleet build 20261027, `-ngl 99 -fa on -c 2048`, no scrape dir) | 82.09 / 82.07 / 82.13 / 82.13 / 82.14 | 82.00 / 82.05 / 81.99 / 82.02 / 82.08 | 0.202-0.204 | 1.16, 1.53 | `eccfc8c1a26ab916` x10 |
| llama.cpp, same build, `NAUTIBURN_SCRAPE_DIR` set (its scraper on; one session, 2 timed) | 66.35 / 66.41 | - | 0.209 | 1.56 | `136ea2355506fcef` x3 |
| (a) lemonade-burn old path, rows1 (`LB_DEVICE_DECODE=0 LB_Q4_ROWS=1`) | 20.06 / 17.63 / 23.93 / 21.00 / 18.96 | 19.13 / 22.61 / 21.69 / 21.43 / 18.89 | 0.54-0.63 | 1.13-1.35, 1.69-2.20 | `062392b8d2b0361a` x10 |
| (b) **device decode**, rows1 | 61.27 / 61.31 / 61.46 / 61.34 / 61.41 | 61.37 / 61.32 / 61.37 / 61.36 / 61.43 | 0.60-0.64 | 1.50-1.62, 1.87-1.94 | `062392b8d2b0361a` x10 |
| (c) **device decode, rows4** | 79.51 / 82.51 / 79.81 / 79.35 / 79.11 | 82.17 / 79.22 / 79.22 / 79.13 / 79.42 | 0.48-0.64 | 1.37-1.42, 1.71 | `062392b8d2b0361a` x10 |
| (d) old path, rows4 (`LB_DEVICE_DECODE=0 LB_Q4_ROWS=4`) | 28.51 / 18.49 / 16.10 / 18.39 / 18.34 | 18.18 / 18.21 / 21.16 / 18.13 / 19.45 | 0.49-0.60 | 1.48-1.56, 1.79-1.89 | `062392b8d2b0361a` x10 |
| (c) with lemonade-burn's scraper on (`NAUTIBURN_SCRAPE_DIR` set) | 81.09 / 78.61 / 77.19 / 80.69 / 78.74 | - | 0.54-0.63 | 7.23-7.77 | `062392b8d2b0361a` x5 |

`/v1/placement`: arms b and c **186 device tokens of 186 decode passes, 0 fallbacks**; per device token
**20.74 / 18.86 ms (rows1)** and **17.04 / 15.67 ms (rows4)**; arms a and d 0 device tokens, decode forward
49.9 / 48.6 and 54.2 / 54.4 ms.

Reading:
* **Device decode alone is 3.0x** (b vs a: 61.3-61.5 against 17.6-23.9), and **rows4 on top of it is 1.29x**
  (c vs b: 79.1-82.5 against 61.3-61.5; neither arm's range comes near the other's). Together **3.7-4.0x the old
  path** (round medians 79.5 / 79.4 against 20.1 / 21.4). On the old path rows4 does nothing visible (d vs a,
  both noisy): the old path is bound by its 145 round trips, not the kernel, which is exactly what the plan
  predicted for each lever alone.
* **Against llama.cpp on the same card and GGUF: 79.4-79.5 against 82.0, 0.97x** (scrape dirs unset on
  both). The 11-04 bar of 66.2-66.7 was llama.cpp **with its built-in NautiBurn scraper on** (`baseline.sh`
  exported `NAUTIBURN_SCRAPE_DIR`): reproduced here, same binary, 66.4 tok/s and the 11-04 sha
  `136ea2355506fcef`. Without the scraper it is 82 tok/s and a different text, so the scraper changes
  llama.cpp's arithmetic as well as its speed (presumably a different MoE path; not investigated). Ours
  with the scraper on is 77.2-81.1 (load 7, something else started on c4) and the same sha: our scrape is
  free, theirs costs 19%. Like for like, scrapers off: **0.97x llama.cpp**; scrapers on: **1.2x**.
* Prefill is unchanged by any of this (0.48-0.64 s against llama.cpp's 0.20; not in the plan).
* Different sha from llama.cpp (`062392b8...` vs `eccfc8c1...`): the two engines sum in different orders;
  ours is the same in every arm and every completion, which is the gate.

**Step 5, kernels on the P100** (`rows_bench`, 3 runs, load 5.1-10.5, roofline 732 GB/s; median [min-max]):

| | rows1 | rows4 | rows8 | rows4 + x shared |
|---|---|---|---|---|
| distinct24_m1, ms/expert | 0.0618 [0.0613-0.0624], 85.9 GB/s | 0.0583 [0.0472-0.0583], 91.1, 1.06x | 0.0573, 92.6, 1.08x | 0.0581, 91.3, 1.06x |
| in_proj 6144x2048 chain | 99.7 GB/s | 132.0, 1.32x | 131.8, 1.32x | 131.2, 1.32x |
| out_proj 2048x2048 chain | 80.1 GB/s | 79.2, 0.99x | 77.2, 0.96x | 79.3, 0.99x |
| kv 512x2048 chain | 32.9 GB/s | 35.2, 1.07x | 20.1, 0.61x | 22.0, 0.67x |
| ffn_down 2048x11776 chain | 116.6 GB/s | **203.9, 1.75x** | 170.6, 1.46x | 171.0, 1.46x |
| ffn_up 11776x2048 chain | 116.5 GB/s | **204.8, 1.76x** | 171.2, 1.47x | 171.1, 1.47x |

On the busier box the per-expert number is bimodal (rows4's best run 0.0472 = 112 GB/s) and not conclusive by
itself; the big matrices are, and the e2e above is the measurement that matters. The best kernel is at 28% of
the P100's 732 GB/s, so there is still room in the kernel: at 15.7 ms a token for ~1.56 GB the device token
streams ~100 GB/s overall, about what llama.cpp's 12.2 ms does (~128 GB/s).

**Defaults changed (this commit)**, because both levers won on both cards with every gate passed:
* `LB_DEVICE_DECODE` **on unless set to 0** (live knob `device_decode` as before). A token whose weights are
  not all resident still takes the old path (0 fallbacks here; the fallback is bit-identical on the 1070).
* `LB_EXPERT_LAYERS` follows it, so **stacked residency is the default** (13.8 GB instead of 15.3 on the 24B);
  `LB_EXPERT_LAYERS=0` keeps the per-expert upload.
* `LB_Q4_ROWS` **defaults to 4**; `LB_Q4_ROWS=1` is the one-row kernel.

Checked on the card with the new binary (`-dd2`), no `LB_*` set: **79.54 / 79.95 / 84.58 / 83.07 / 80.54
tok/s**, prefill 0.46-0.64, load 2.0-2.1, 13,775 MiB, 186 device tokens, 0 fallbacks, sha `062392b8d2b0361a`
x6. With every opt-out (`LB_DEVICE_DECODE=0 LB_Q4_ROWS=1 LB_EXPERT_LAYERS=0`): **18.45 / 18.60 / 19.10 /
18.74 / 24.25**, 15,291 MiB, 0 device tokens, same sha: the old engine, unchanged.

**Still open.** The 512-row and 2048x2048 shapes get nothing from rows4 (too few row groups for 56 SMs: a
per-shape r or split-K is next); the kernel sits at 28% of HBM2 so a further 1.5x is not ruled out; prefill
is 2.5-3x llama.cpp's; `c4_e2e.sh` / `c4_llama.sh` should get the AF:00.0 UUID as their default `CUDA_DEV`
(left unchanged here: not in the repo, and they were to be kept as they are); and why llama.cpp's scraper
changes its output text is not ours to fix but should be told to whoever owns that build.

## MoE exp. #2, expert staging ring under VRAM pressure: gate fails, sha holds, NUMA is worth 4x (Fable, 2026-11-08)

Branch `fable/stage-ring-2026-11-08`; full write-up, tables and resume commands in `docs/STAGE_RING.md`.
LFM2.5-24B-A2B Q4_0 with a 5120 MiB expert budget on 8 GB cards, interleaved 5 rounds, 288 decode tokens
a session:

* **RTX 2060 SUPER (c2, PCH slot, Gen2 x4, 1.45 GB/s):** (a) heat residency + on-demand 3.13 tok/s; (b4)
  + ring, top-4 guesses 2.67 (0.85x); (b) + ring, 8 guesses 1.60 (0.51x); llama.cpp `--moe-cache-mib 5120`
  6.75 (2.16x). Spread under 3 % in every arm. Each arm's token time is its bus bytes / 1.45 GB/s + ~30 ms.
* **GTX 1070 (c1, Gen3 x8, host at load 35):** a ~7.7, b4 ~7.9 (paired median ~1.05x, 0.60-1.70), b ~5.6,
  llama.cpp ~12 (collapsed in 2 of 5 rounds). Noisy; no claim beyond "the ring breaks even at best".
* **Gate (1.3x over (a), >= llama.cpp): FAIL** on both cards. **Same greedy text with the ring on and off:
  PASS**, 30/30 sessions, and the sonar prompt gives the all-resident P100 sha `062392b8d2b0361a` at 41.6 %
  residency because `moe_ffn` now sums experts in ascending id whatever served them.
* The `mid1` guess works (ring raises the hit rate 0.48 -> 0.87 / 0.95; staged-expert precision 82 % conv
  targets, 65 % attention targets at 4 guesses); it loses because the token is copy-bound and the guess adds
  bytes. llama.cpp's win is its dynamic cache's hit rate (0.73 vs our static 0.48), not overlap.
* **c1 side finding:** the page-locked GGUF split across two NUMA nodes made each miss 7.7 ms instead of
  1.5; reloading the file's page cache on the GPU's node took arm (a) from 1.59 to 6.6 tok/s.
* **Exp. #2b, Tesla P100 Gen3 x16 (c2 GPU2, 7.8-8.4 GB/s):** (a) 10.18 tok/s, (b4) 11.80 (**1.16x**), (b4m,
  margin >= 0.0054) 9.88 (0.97x), (b) 9.77 (0.96x), llama.cpp 15.72 (1.54x over a paired 10.20). Gates fail
  here too (best 1.16x; 0.75x of llama.cpp); sha holds (55 sessions, three cards). The token is ~half copies,
  ~half host round trips; the CPU guess costs 8-11 ms of an 86-104 ms token.

## Engine step E1: dynamic SLRU expert cache passes its gate on both c2 cards (Fable, 2026-11-08)

Branch `fable/engine-e1-cache-2026-11-08`; write-up `docs/ENGINE_E1_CACHE.md`. LFM2.5 Q4_0, 5120 MiB experts:
P100 x16 15.47 tok/s vs llama.cpp 15.87 (0.975x), 2060 SUPER 7.37 vs 6.75 (1.09x); hit rate 0.854 on both
(static plan 0.482, llama.cpp 0.72-0.74); sha `062392b8d2b0361a` at 3072 and 5120 MiB; device memory flat over a
50-pass soak. The static plan's device memory grew 6769 -> 7661 MiB in the same runs; the slot bank does not.

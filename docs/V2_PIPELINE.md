# V2: the layer pipeline, its simulator, and self-calibration

Status 2026-11-01: **measurements taken on two cards, simulator built and committed
(`crates/lb-calibrate`, 9 tests), first tuned results and the feasibility verdict below.**
No multi-card prototype was run on hardware (and per the operator's steer, none is needed
yet: the pipeline is simulated from measured per-card timings). Every number is labelled
**[measured]** (command and card given), **[simulated]** (the model in `sim.rs` on the real
scrape traces with measured inputs), or **[assumed]**.

Related: `docs/LFM2_CUBECL.md` (steps 1-2, what runs today), `docs/LESSONS_FROM_AVALON.md`
(the techniques learned, clean-room).

## 1. The operator's v2 idea, restated and corrected

**Idea:** a multi-GPU *layer pipeline* that stages expert weights a layer ahead: big expert
weights are prestaged across several cards; the small activation moves card to card to
where the weights already are; while card 1 computes layer L, cards 2 and 3 are loaded
with the predicted experts for L+1, L+2; a card that finishes evacuates what the lookahead
window no longer needs and takes a future layer's experts; experts are processed on
several cards in parallel.

**What holds, what does not (from the model and the measurements):**

* *"Move the work to the weights"* holds: a token's activation is 8 KB; one expert is
  5.3 MB. Moving activations (0.13 ms per handoff on the 1070 **[measured]**) is always
  cheaper than moving experts (4.8 ms per expert on the same card **[measured]**).
* *Staging hidden behind compute* holds **only when there are other cards computing
  while this one loads**, and only if the link is fast enough for the staged set (§5).
  A single card cannot hide its own staging: a wgpu copy and the compute share one queue
  and, more simply, there is nothing else to do; the rolling single-card case stalls
  96-99% of its time **[simulated]**.
* *Parallel expert processing across cards* does **not** help a single decode stream: the
  layers of one token are sequential, so at any moment exactly one card computes. Cards
  add **VRAM** (bigger resident sets, higher hit rate) and **overlap for staging**; they
  add compute parallelism only with several streams or during prefill (not modelled yet).
* *Prediction* works and is stable: the hot set learned on one set of prompts covers
  62.5% of the expert demand of held-out prompts at k=24/64 per layer, 74.9% at k=32
  **[measured on the traces]** (the coordinator's independent script found 64.8% at
  top-24 held out; same picture).
* The surprise: on our hardware **static residency** (every layer's staged set resident
  for good, nothing copied per token) is both feasible (k=24 for all 38 MoE layers is
  4.8 GB) and far better than rolling; and once the sets are static, the bus is
  irrelevant and the two things that decide throughput are the **expert kernel speed**
  and the **hit rate** (misses run on the CPU, overlapped). The v2 pipeline's value on this
  fleet is therefore: more cards = more static VRAM = higher hit rate, plus a path to
  rolling only on x16 links with small k.

## 2. Measurements **[measured 2026-11-01]**

`lb-gpu-bench` (crate `lb-q4-gpu`, `LB_WGPU_DEVICE` picks the card; one card at a time,
no seat touched; `LB_BENCH_REPS=20`, medians). Expert = LFM2.5 shapes (gate/up
1536x2048, down 2048x1536, Q4_0, 5.31 MB). Kernel = the new one-cube-per-output
`q4_0_matvec_kernel` (64 units split the row's blocks, shared-memory reduction; all
6 lb-q4-gpu tests pass on both cards).

| metric | c4 GTX 1070 (`discrete:2`, PCIe x4 slot) | c3 Radeon Pro WX 5100 (`discrete:1`, PCIe 2.0 x4) |
|---|---|---|
| expert FFN, m=1, 4 experts launched then one read-back, per expert | **0.430 ms** | 0.964 ms |
| expert FFN, m=1, single (upload x + 4 launches + read-back) | 0.485 ms | 1.221 ms |
| expert FFN, m=4, per expert (batched) | 0.961 ms | 2.114 ms |
| host->device, 5.3 MB expert blob | **1101 MB/s** (4.8 ms) | 540 MB/s (9.8 ms) |
| host->device, 64 MB blob | 664 MB/s | 385 MB/s |
| activation handoff via host (read back + re-upload 8 KB) | 129 us | 394 us |
| same, 32 KB / 256 KB | 158 / 453 us | 446 / 944 us |
| dense in_proj 6144x2048 (launch + read-back) | 0.381 ms | 0.975 ms |
| dense 2048x2048 | 0.189 ms | 0.535 ms |
| dense 512x2048 | 0.112 ms | 0.276 ms |
| CPU expert, all threads (lb-gguf path) | 1.088 ms (80 thr, load ~24) | 12.65 ms (8 thr) |
| CPU expert, 1 thread | 13.4 ms | 12.67 ms |
| CPU Q6_K head 128000x2048 | 18.3 ms | 214 ms |

Notes: the 1070's link reported PCIe gen1 x4 idle (it is an x4 slot); the achieved 0.66-1.1
GB/s is what *our* copy path (CubeCL `create_from_slice` + sync) gets on it, not the link
maximum. c3's CPU numbers show no multi-thread gain (its LFM2.5 Lemonade seat was serving
on the CPUs); they are what that box gives today. Raw files:
`/data/lfm2-scrapes/measure-{c4-1070,c3-wx5100}.{json,table.txt}` (t440).

What the kernel rewrite bought (Phase A item "fold in the faster kernel"): 0.43 ms per
expert call versus 1.7 ms at step 2 on the same card **[measured]**, i.e. 12 GB/s of
weight traffic on a 256 GB/s card: still 20x off the bandwidth roof. The forge's own
LB-Q4GPU-V2 / Q8 / Q6K branches did not build (wrong imports, a missing function), so this
kernel is mine; those jobs are still cycling on the pool.

## 3. The simulator (`crates/lb-calibrate`)

* `timings.rs`: parses `lb-gpu-bench` output into `CardTimings` / `HostTimings`; derives
  per-layer dense costs from the measured matvecs; `hypothetical()` scales a measured card
  (compute, bus, VRAM) for what-if sweeps, flagged `measured: false`.
* `topology.rs`: cards from the drivers' counters (amdgpu sysfs `mem_info_vram_*`,
  link speed/width; `nvidia-smi` CSV), not the graphics API.
* `trace.rs`: NautiBurn scrape files -> per-token routing; deterministic held-out split
  (every 4th request); histogram; `top_k` staged sets; `coverage`.
* `sim.rs`: decode, one stream. Per token, per layer: dense cost on the layer's card (or
  CPU + a handoff), resident experts on the card (`expert_ms` each + one sync per layer),
  misses on the CPU overlapped (`max`) or copied on demand, or the cheaper of the two per
  layer; handoff when the next layer is on another card; **static** residency when a
  card's `k x layers_on_card <= slots`, otherwise **rolling** staging of `k` experts per
  layer, hidden by `lookahead x (time of the other cards' intervening layers)`, the rest
  a stall (a lone card hides nothing); optional dynamic admission into spare slots
  (`admit_after` demands, coldest-out, counts halve every `age_every` demands).
* `tune.rs`: enumerates cards 1..N, k in {8..64}, lookahead, miss policy, dynamic
  admission, dense placement; simulates each on the held-out trace with sets fitted on
  the train trace; writes `StagingConfig` (JSON: k per layer, staged ids, card
  assignment, predicted tok/s, coverage, notes).
* CLI `lb-calibrate topology | tune | sweep`.

Validation against reality (same box, single card, `sweep --card ... --dense cpu`):

| engine state | measured (docs/LFM2_CUBECL §8, c4, 32 threads) | simulated |
|---|---|---|
| step-2 GPU path (old kernel 1.7 ms/expert, dense on CPU, ~20 experts/layer resident) | 1.8-3.2 tok/s | 3.5 tok/s |
| CPU-only | 4.6-5.3 tok/s | 3.6 tok/s |

The model is within 1.4x in both directions: it over-charges the CPU dense path (derived
from the expert rate, not measured) and charges a per-layer sync the CPU-only engine does
not pay. Good enough to rank configurations; the absolute tok/s carry that error bar.
Trace used: 177 scrape files, 13,868 expert activations (the step-1/2 runs); held-out
= 44 requests. Only a handful of prompts: more scrapes sharpen the sets, per-domain sets
(SAR vs code) probably help; all numbers are top-4 routing on LFM2.5 QAD-Q4_0.

## 4. Tuned results **[simulated on measured inputs]**

Held-out coverage of the staged sets by k (per layer, of 64): k=8 27.1%, 12 37.8%,
16 46.1%, 24 62.5%, 32 74.9%, 48 92.3%, 64 99.9% (4.8 GB for k=24 over 38 layers,
6.5 GB for k=32, 12.9 GB for all).

**Single card, c4 GTX 1070, 4000 MB pool (753 slots):** best = k=12 static + dynamic
admission into the spare slots, misses on the CPU: **7.9 tok/s**, 58% hits, 0 stall
(k=16: 7.9; k=8: 7.8). Rolling with k=24 on the same card: 0.22 tok/s, 97% stall. So the
single-card rule is: never roll; stage what fits statically and let the spare admit.

**Three cards (the 1070 plus two hypothetical cards of the same compute on PCIe 3.0 x16
= 14 GB/s with 7 GB pools):** best = 2-3 cards, k=32 static, dynamic admission: **8.5
tok/s**, 85% hits; k=48 (92% hits): 8.45 tok/s. Adding cards moved the hit rate from 58%
to 95% but the token time only from 126 to 118 ms, because the layer time is
max(resident experts on the card, misses on the CPU) and the card side, 4 x 0.43 ms +
0.3 ms sync per layer, is now the floor: **the kernel is the ceiling, not the bus and
not the hit rate.**

**Rolling (pools too small for the sets), the regime the staging idea is about:**

| cards | k | per-card bus | stall at measured bus | at 2x | 4x | 8x |
|---|---|---|---|---|---|---|
| 3 | 16 | 1070 x4 + 2 x16 (14 GB/s) | 86% (0.95 tok/s) | 74% | 54% | 22% (5.4 tok/s) |
| 3 | 24 | same | 92% | 84% | 70% | 47% |
| 2 | 16 | 1070 + 1 x16 | 91% | 82% | 68% | 46% |
| 3 | 16 | all three on the 1070's 1.1 GB/s | 85% | 72% | 50% | 20% |

(the stall in these rows is dominated by whichever card has the slow bus; "8x" means
8.8 GB/s on the 1070 and 112 GB/s on the others.)

**Fast-kernel what-if (expert call 4x faster, 0.11 ms):** single 1070 + one x16 card:
10.5 tok/s; four fast x16 cards with 7 GB each, k=64 (everything resident): **13.3 tok/s**
(99.9% hits, CPU idle). That is the shape of the product on a real multi-card node.

## 5. Feasibility math (the bandwidth-vs-lead-time check)

Per MoE layer the pipeline must have `k` experts in place on the card that will compute
it: `S = k x 5.31 MB` (k=16: 85 MB, k=24: 127 MB). With N cards in round robin the card
has `lead = lookahead x (N-1) x T_layer` to load it, where `T_layer` is one layer's time on
the other cards: today **about 3.3 ms** (126 ms per token / 38 layers, with the current
kernel and 58-85% hits).

* Hidden iff `S / BW <= lead`, i.e. `BW >= k x 5.31 MB / ((N-1) x lookahead x T_layer)`.
  N=3, lookahead 1, k=16: BW >= 85 MB / 6.6 ms = **12.9 GB/s per card**; k=24: 19.3 GB/s;
  N=2: 25.8 / 38.5 GB/s. Lookahead 2 halves these but needs twice the pool (two sets in
  flight) and the prediction two layers out.
* What we have: 0.5-1.1 GB/s achieved on the x4 slots (both test cards), so rolling
  stalls 85-97% **[simulated, consistent with the formula]**. PCIe 3.0 x16 at a realistic
  10-12 GB/s (the c4 P100 slots, Avalon's box) is *just under* the k=16, N=3 requirement:
  the sim shows 22% stall at ~9-14 GB/s. A faster kernel makes this **worse**, not better:
  `T_layer` shrinks, the lead shrinks with it (0.11 ms experts -> ~1.2 ms layers -> 50 GB/s
  needed). Rolling staging therefore only pays on NVLink-class links or with k <= 8, where
  the hit rate (27%) throws most of the work back on the CPU.
* The misses (~35-40% of demand at k=24) do not need the bus at all in the overlapped
  design: they run on the CPU during the card's resident work. On c4 (1.09 ms per expert,
  80 threads) ~1.5 misses per layer cost ~1.6 ms, about the card's 1.3 ms: balanced, which
  is why `Best` (on-demand copies of misses) never won at measured bandwidth and only ties
  at 4x. On c3 (12.7 ms per CPU expert) misses dominate everything: the WX 5100 single-card
  tune gives 1.3 tok/s and wants k=24-32 resident (76% hits) just to get there.

**Verdict:** on our hardware the pipeline hides staging only in the static regime, which
is the one to build: enough VRAM across the cards to hold `k=24..32` sets per layer
(4.8-6.5 GB total; 2 x 4 GB cards or one 8 GB card with the dense weights elsewhere),
sets chosen from the scrapes, spare slots adapting by admission, misses overlapped on the
CPU. Rolling prestaging stays a documented option for x16/NVLink nodes with the pool
sized for two layer windows. The next real speed-up is the expert kernel (vectorized
loads, several rows per cube), which the simulator says lifts the ceiling from 8.5 to
10-13 tok/s on the same cards; after that, prefill/multi-stream pipelining is where
multiple cards compute in parallel.

## 6. Self-calibration flow (shippable; how it is meant to run)

1. **Probe:** `lb-calibrate topology` (driver counters: cards, free VRAM, links).
2. **Measure:** for each card `LB_WGPU_DEVICE=<spec> lb-gpu-bench > card.json` (about
   30 s per card, ~90 MB of VRAM, no model needed); the CPU numbers come with it.
3. **Traces:** the engine's own `NAUTIBURN_SCRAPE_DIR` (it writes them from day one) or
   a bundled default trace (`/data/lfm2-scrapes` on t440 today; a small default trace is
   to be shipped with the model).
4. **Tune:** `lb-calibrate tune --bench card1.json [--bench card2.json ...] --budget-mb
   ... --scrapes DIR --out staging.json` writes the config: k per layer, staged expert
   ids per layer, card assignment, miss policy, dynamic admission, predicted tok/s and
   hit rate, with notes on what was hypothetical.
5. **Load:** the engine reads `staging.json` (today: `lb-server --placement-from-scrapes
   DIR --gpu-budget-mb N --gpu` still uses the greedy byte-budget planner; wiring the
   staged-ids file into `Placement` is the next engine change) and keeps scraping.
6. **Adapt:** re-run 2-4 when a card is added (c2's Turing cards), a kernel lands, or
   the scrape histogram drifts (per-domain sets); the dynamic admission covers the gaps
   between runs. Single-card is the degenerate case of the same flow (N=1), which is what
   the first tune above produced.

What is still missing for this to be one command: `lb-server calibrate` orchestrating
2-4 (the pieces exist as two binaries), the staged-ids loader in the engine, and the
periodic re-tune trigger.

## 7. Phase A status (what the single-card engine still needs)

Done this round: workgroup-per-output kernel (4x per expert call), 2-D dispatch (a
44-token prompt had exceeded the 65,535-cube dimension cap; regression test added),
`Gpu::sync` fence, measurement harness, VRAM from driver counters (`topology.rs`).

End to end with the new kernel **[measured 2026-11-01, c4 GTX 1070, 4 GB pool, 753
resident experts, same 3 prompts as step 2, 32 threads, lb-q4-gpu 7/7 tests on the card]**:
GPU-path top-1 equals the CPU path at all 86 positions (75/86 vs the llama.cpp oracle,
unchanged); resident expert calls 5868 in 5.6 s (was 10.1 s at step 2); prefill now
faster than CPU-only (29 tokens 3.4 s vs 4.1 s; 44 tokens 4.7 s vs 6.0 s; 13 tokens
1.7 s vs 1.9 s); decode still slower (3.4 / 3.8 / 3.6 tok/s vs 5.2 / 4.8 / 5.1 CPU-only)
because the per-expert activation upload, the per-layer sync and the CPU dense path
remain: exactly the Phase A items below, which the simulator prices at the difference
between today's 3.6 and its 7.9 tok/s for this card. Not done: activations
kept on the device across layers (dense path on Burn's wgpu backend), persistent
per-layer buffers, plan/launch split with one sync per layer, pool buffer + slot-index
batched kernel. These remain the step-3 list in `LFM2_CUBECL.md` §10; the simulator now
says what they are worth: the per-layer sync and dense path are ~20% of the token on the
GPU path, the kernel is the rest.

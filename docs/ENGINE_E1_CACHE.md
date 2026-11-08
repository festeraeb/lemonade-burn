# Engine step E1: dynamic expert cache

Branch `fable/engine-e1-cache-2026-11-08` (off `fable/stage-ring-2026-11-08`). Plan and gates:
`~/agentnotes/engine-right-first-plan-2026-11-08.md` (step E1). Background: `docs/STAGE_RING.md` (MoE
exp. #2: llama.cpp beat us 1.5-2.2x at equal VRAM because its dynamic cache held 72-73 % of decode
routings against our static heat plan's 48 %).

## What changed

* **`crates/lb-lfm2/src/cache.rs`**: `Slru`, the segmented LRU llama.cpp's `--moe-cache-mib` uses
  (`src/llama-moe-cache.cpp`): slots in a *probation* and a *protected* LRU list; a demand miss takes
  the least recently used probation slot (then protected), never one holding an expert the current
  layer pass needs, and goes in on probation; a hit moves its slot to protected; protected is capped
  at half the slots. Global over layers (one expert shape for every LFM2.5 MoE layer). A dense
  `[layer][expert] -> slot` table (`slot_table`) is kept for engine step E2's device-side router.
  Unit tests: promotion, scan resistance, the half cap, "never evict what the pass needs", 500 random
  multi-layer passes with the lists and maps checked after each, seeding.
* **`crates/lb-q4-gpu` `ExpertSlots`**: the device side. One bank of slots, allocated **once** in
  CubeCL's persistent pool (exact-fit slices the pool never sweeps), 32 slots a chunk, plus 8 fixed
  staging buffers. A fill is a DMA of the expert's three raw slices from the page-locked GGUF mapping
  into a staging buffer (`client.write` into an offset view) and one repack launch into the slot,
  all on the main stream in order. **Nothing is allocated per miss**, so the budget is real device
  bytes. This is the fix for exp. #2's kept-miss bug (each kept copy took a 7 MiB pool slice plus a
  transient raw buffer while counted as 5.31 MB, and device memory grew to OOM).
* **`moe::moe_ffn_with`**: when the cache is on, every expert of the layer runs from its slot (hits
  and fills alike); the outputs are still added in ascending expert id, so the text does not depend on
  the budget or on what the cache holds.
* **`lb-server`**: the cache is **on by default** when the expert budget (`LB_EXPERT_VRAM_MIB`, else
  `--gpu-budget-mb`) is smaller than the model's experts; `LB_EXPERT_CACHE=0` keeps the static plan,
  `LB_EXPERT_CACHE=1` forces it. Slots = (budget - staging) / 5,308,416 bytes. Seeded at load from the
  scrape heat order (`--placement-from-scrapes`; `LB_EXPERT_CACHE_SEED=0` starts cold). The device
  decode's host-MoE branch is switched on with it (`LB_DD_HOST_MOE`). The staging ring stays behind
  `LB_STAGE` and is ignored when the cache is on (no prefetch copies in E1).

## Results (2026-11-08, cesarops2, LFM2.5-24B-A2B Q4_0, 5 prompts / 288 decode tokens a session)

Arms interleaved e1 / c / a0 (reversed on even rounds), 5 rounds each, expert budget 5120 MiB in every arm.
e1 = this cache; c = llama.cpp e14a779 `-cmoe --moe-cache-mib 5120` (sm_60/61/75 build); a0 = the static
heat plan + on-demand misses (`LB_EXPERT_CACHE=0`, exp. #2's arm a). Hit rate: decode routings served
without a copy, after the warmup (llama.cpp: its own counter, whole process, batches of <= 8 tokens).

### Tesla P100 Gen3 x16 (c2 GPU2, UUID GPU-c012cc70-...)

| arm | tok/s median [min-max], 5 rounds | x llama.cpp | hit rate | misses / token | device memory at load / max after any completion |
|---|---|---|---|---|---|
| **e1** cache | **15.47** [15.38-15.76] | **0.975** | **0.854** (0.844 whole process) | 22.2 | 6333 / **6439** MiB |
| c llama.cpp | 15.87 [15.86-16.03] | 1.00 | 0.719 | 44.7 | 6237 / 6297 MiB |
| a0 static plan | 10.18 [10.07-10.20] | 0.64 | 0.482 | 78.7 | 6769 / **7661** MiB (grows) |

All [measured]. e1 token 65.2 ms = 13.6 router-input reads + 37.2 MoE reads (kernels + 22 fills at
~0.65 ms of bus each) + 6.5 issuing fills + ~8 the rest; a0 99.4 = 16.1 + 64.6 + 17.2 + ~1.5.

* **Sha across budgets:** e1 at 3072 MiB (598 slots) gives the same text on all 5 prompts (sonar
  `062392b8d2b0361a`) as 5120 MiB and as the static plan; 12 lemonade-burn sessions + the soak.
* **Soak:** one server, 50 passes of the 5 prompts (250 completions): device memory 6439 MiB after pass 0
  and after every later pass (min = max = 6439), one sha per prompt throughout.

2060_PLACEHOLDER


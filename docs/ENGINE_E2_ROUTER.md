# Engine step E2: device-resident router

Branch `fable/engine-e2-router-2026-11-09` (off `fable/engine-e1-cache-2026-11-08`). Plan and gate:
`~/agentnotes/engine-right-first-plan-2026-11-08.md`, step E2.

## What changed

* **Device decode (`crates/lb-lfm2/src/device_decode.rs`).** With the dynamic cache on, a MoE layer no
  longer goes through the host path. The router input `h2` stays on the device; `router_topk` writes the
  layer's routes and weights to the token's device buffers; the host reads back only the routes (32 bytes
  of interest), plans the cache (hits promoted, misses take SLRU victims), queues the fills, writes the
  layer's slot numbers and launches the routed expert kernels on the slot bank; `weighted_sum` adds the
  outputs on the device in ascending expert id. Per layer that is **one** sync where E1 had two, and no
  activation or output crosses the bus. `LB_E2=0` restores E1's host-routed layer.
* **Slot bank (`lb-q4-gpu` `ExpertSlots`).** Now one `GpuExpertLayer` with `n_experts = n_slots`: one
  contiguous `qs` and `sc` buffer per projection, still allocated once in the persistent pool. The
  existing routed kernels address a cache slot exactly as they address an expert of a stacked layer. The
  per-slot `GpuExpert` views still serve the host-routed prefill path from the same memory. A fill is now
  three repack launches (one per projection).
* **`routes` vs `slot_routes`.** `routes` keeps expert ids (the sum's order and the scrape); a second
  buffer holds the slot numbers the expert kernels index with. The host writes it after planning.
* **Reads inside a token** use `Gpu::read_raw_keep`: the router input and the last FFN output are scratch
  buffers still needed after the read, so the pool is recycled only by the token's last read.
* **Guess on the device** (`LB_DEVICE_GUESS=1`, off by default: nothing consumes it until E3): the next
  layer's FFN pre-norm and router applied to this layer's residual, read back with the routes and scored
  against the next layer's real set.
* **Counters** (`/v1/placement` `stage`): `host_syncs_per_token`, `miss_layers_per_token`,
  `moe_layers_per_token`, `guess_recall`.
* **Test** `tests/expert_slots_match_upload.rs`: bank slots are bit-identical to a plain upload through
  the views, through device routes at a non-zero base, and after a refill.

Not blocked for the gemma port: the cache and bank know nothing of the activation or the gate (`Slru` is
keyed by layer and expert; `ExpertSlots` takes shapes). The activation still lives in the kernels, the
store is still lb-lfm2's, and `router_topk` is still the sigmoid gate; a softmax gate is a sibling of that
one kernel, and E2's flow only needs "routes and weights in the token buffers".

## Results (2026-11-09, LFM2.5-24B-A2B Q4_0, 5 prompts / 288 decode tokens a session, 5120 MiB experts)

Interleaved rounds, 5 per arm. e1 = the E1 binary (commit 1969bb7), e2 = this step, e2g = e2 + the device
guess, c = llama.cpp e14a779 `-cmoe --moe-cache-mib 5120` (AVX2 build). Hit rate and misses are decode-only
after the warmup; llama.cpp's are its own counters (whole process, batches of <= 8 tokens).

**Card, probed** (`probe/card.txt`, `probe/stage_bench.jsonl`): cesarops4om Tesla P100, UUID
GPU-2d66f961-cba7-717f-1bd7-a04fff1fc8ab, nvidia-smi index 4, D8:00.0, Gen3 x16 at both ends, NUMA node 1, Xeon
Gold 6148, measured H2D 8.4-9.7 GB/s (0.55-0.63 ms per 5.31 MB expert). Page cache and process bound to
node 1: `numa_maps` shows the whole GGUF on N1 in every session (`logs/numa-*.txt`, `logs/numastat-*.txt`).
The c4 host is not the c2 host: c2's E1 numbers are not comparable and are not used here.

| arm (c4 P100 x16) | tok/s median [min-max] | spread | x E1 | token ms (read wait) | syncs / token | hit | misses / token | layers with a miss / token | device MiB at load / max |
|---|---|---|---|---|---|---|---|---|---|
| e1 | 16.50 [16.32-16.61] | 1.8 % | 1.00 | 61.2 (36.8) | 77 [by construction] | 0.854 | 22.2 | - | 6329 / 6435 |
| **e2** | **30.49** [30.15-31.53] | 4.5 % | **1.85** (paired 1.83-1.91) | 32.7 (25.0) | **39.0** | 0.854 | 22.2 | **15.18** | 6337 / 6443 |
| e2g (+ guess on device) | 26.57 [26.42-26.77] | 1.3 % | 1.61 | 37.6 (28.5) | 39.0 | 0.854 | 22.2 | 15.18 | 6337 / 6443 |
| c llama.cpp | 17.04 [16.86-17.15] | 1.7 % | 1.03 | - | - | 0.72 | 44.7 | - | 6233 / 6293 |

All [measured]. c4 was quiet enough (load 1-3, spread 1.3-4.5 %, the largest from one high e2 round): the
gate is resolved by a wide margin. E2 is 1.79x llama.cpp on this card. All-resident reference on this card
(`baseline/`, one session, E1 binary, stacked layers, 1 sync a token): 81.85 tok/s, 13,775 MiB.

| arm (c2 2060 SUPER, Gen2 x4, 1.42-1.45 GB/s) | tok/s median [min-max] | x E1 | token ms | hit | device MiB load / max |
|---|---|---|---|---|---|
| e1 | 7.41 [7.34-7.50] | 1.00 | 136.9 | 0.854 | 6164 / 6270 |
| **e2** | **9.48** [9.47-9.52] | **1.28** | 106.9 | 0.854 | 6172 / 6278 |
| c llama.cpp | 6.79 [6.78-6.79] | 0.92 | - | 0.74 | 6062 / 6100 |

All [measured]; on this link the token is the bus (22 misses x 3.7 ms = 82 ms of 107), so E2's gain is smaller.

## Gate

| gate | result |
|---|---|
| host round trips per token <= 4 | **FAIL: 39.0** (38 MoE layers + the token's last read; E1 was 77 [by construction]) |
| tok/s >= 1.25x over re-baselined E1 on the x16 P100 | **PASS: 1.85x** (30.49 vs 16.50, paired 1.83-1.91) |
| greedy sha unchanged | **PASS**: sonar `062392b8d2b0361a`; same text on all 5 prompts in every session on the c4 P100 (e1, e2, e2g, all-resident; 3072, 5120, 8192, 11008, 12400 MiB) and on the 2060 |
| device memory flat over a soak | **PASS**: 50 passes, 250 completions, 6443 MiB after pass 0 and after every later pass, one sha per prompt |

## Why the round-trip gate fails, and what it would take

A miss is served by a DMA the host issues, so the host has to learn the routes of any layer that misses
before that layer's experts can run. At 5120 MiB, **15.2 of the 38 MoE layers per token have at least one
miss** [measured], so the floor for any host-served design is about 16 syncs a token even if syncs happened
only at layers that miss (which this step does not do: it syncs every layer, because the host cannot know a
layer is all-hit without its routes). <= 4 needs either hit rates that leave 3 or fewer miss layers
(11 GB of a 12.9 GB model here, below) or misses the device serves itself.

Where the 32.7 ms token goes, from the arms above [derived from measured runs]:

| part | ms | how |
|---|---|---|
| compute of the whole token on the device | 12.2 | all-resident path, 81.85 tok/s |
| the per-layer sync (38 extra) and host launch work | 6.6 | E2 at 100 % hit (18.8 ms) minus all-resident |
| serving 22.2 misses | 13.9 | the rest = 0.63 ms a miss; the measured bus is 0.55-0.63 |

So the misses are the largest part; the syncs cost 6.6 ms. Removing every per-layer sync would be worth at most
~1.25x more at this budget [estimate from the table], and the prediction-guided keep of E3 goes after the
14 ms directly.

Budget sweep (e2, one session each, `c4-p100-budgets/`, 3072 in `c4-p100/`, [measured]):

| expert budget MiB | tok/s | hit | misses / token | layers with a miss / token |
|---|---|---|---|---|
| 3072 | 20.66 | 0.68 | 48.2 | 26.47 |
| 5120 | 30.49 | 0.854 | 22.2 | 15.18 |
| 8192 | 40.02 | 0.94 | 8.43 | 7.07 |
| 11008 | 54.16 | 0.99 | 1.62 | 1.52 |
| 12400 (all 2432 experts) | 52.22 | 1.00 | 0.25 | 0.24 |

**Device guess:** costs 4.7-4.9 ms a token (e2g vs e2) with nothing consuming it, recall@4 0.78 on decode
(exp. #1's host-side mid1 was 0.82, on a different prompt set). Left off by default until E3 uses it.


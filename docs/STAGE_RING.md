# Expert staging ring (MoE experiment #2)

Branch `fable/stage-ring-2026-11-08` (from `fable/expert-predict-2026-11-08`, which carries the
device-resident decode D1-D3, the zero-copy miss path `eec4500` and the exp. #1 capture hook).
Background: `~/agentnotes/fable-research-review-moe-prefill-2026-11-08.md` experiment table row 2,
`docs/EXPERT_PREDICT.md` (exp. #1), `docs/LESSONS_FROM_AVALON.md` 2.5 (PREGATE, off by default).

**Question.** Under real VRAM pressure (LFM2.5-24B-A2B Q4_0, 13.58 GB, on an 8 GB card), does a
ring that prefetches the next MoE layer's experts from the `mid1` guess recover the miss cost,
>= 1.3x over today's heat-planned residency + on-demand misses, and >= llama.cpp's
`--moe-cache-mib` at the same VRAM, with the same greedy text?

## Results (2026-11-08)

Same GGUF (`LFM2.5-24B-A2B-QAD-Q4_0`), same 5 prompts (the `c2_e2e` sonar prompt, 32 tokens, plus
`code-03`, `prose-02`, `qa-05`, `forge-04` from `prompts-v1`, 64 tokens each: 288 decode tokens per
session), greedy, one warmup per engine start. Expert budget **5120 MiB** in every arm. Dense weights +
head resident (751 MB), device decode with the host-MoE branch (`LB_DD_HOST_MOE=1`), misses
`on-demand` (DMA from the page-locked mapping), 8 CPU threads. Static residents planned from the two
placement scrapes in `scrapes/` (`Placement::plan`): 1011 experts (41.6 %), the same set in every
lemonade-burn arm except that the ring arms give 256 MiB of the budget to the ring (961 residents).
Arms interleaved a-b4-b-c / c-b-b4-a, **5 rounds each**. tok/s = session tokens / session decode time.

| arm | what | tok/s median [min-max], 5 rounds | x (a) | hit rate | stalls / token | prefetch MB / token | wasted MB / token | VRAM used |
|---|---|---|---|---|---|---|---|---|
| **2060 SUPER (c2, Gen2 x4: 1.42-1.45 GB/s)** | | | | | | | | |
| (a) | heat residency + on-demand misses | **3.125** [3.048-3.139] | 1.00 | 0.482 | 78.7 | 0 | 0 | 6749 MiB |
| (b4) | + ring, 4 guesses (top-k) a layer | **2.670** [2.649-2.672] | **0.85** | 0.872 | 19.4 | 412 | 93 | 6449 MiB |
| (b) | + ring, 8 guesses (k+4) a layer | **1.599** [1.577-1.603] | **0.51** | 0.950 | 7.7 | 849 | 455 | 6449 MiB |
| (c) | llama.cpp e14a779 `-cmoe --moe-cache-mib 5120` | **6.752** [6.735-6.760] | **2.16** | 0.73 (SLRU, decode-size batches) | ~42 | - | - | 6211 MiB |
| **GTX 1070 (c1, Gen3 x8: 4.1-4.7 GB/s; host at load 35 from another job)** | | | | | | | | |
| (a) | as above | 7.71 [4.62-8.14] | 1.00 | 0.482 | 78.7 | 0 | 0 | 6770 MiB |
| (b4) | as above | 7.86 [4.61-8.54]; paired per round 1.70 / 0.81 / 1.05 / 0.60 / 1.18 | ~1.05 (median of pairs) | 0.872 | 19.4 | 412 | 93 | 6470 MiB |
| (b) | as above | 5.60 [3.77-5.64] | ~0.73 | 0.950 | 7.7 | 849 | 455 | 6470 MiB |
| (c) | as above | 11.96 in rounds 1-3; 1.72 and 0.89 in rounds 4-5 | ~1.5 | 0.73 | ~43 | - | - | 6222 MiB |

All [measured]. "stalls / token" = routings served by a demand fetch on the critical path (on-demand
copy); for llama.cpp it is its cache misses per token from its own counters (`-lv 4`, whole session,
batches of <= 8 tokens). Hit rate / stalls / prefetch counters are identical on both cards because the
routing and the plan are (the text is the same).

**Gates.** >= 1.3x over (a): **FAIL** on the 2060 (best ring 0.85x) and **FAIL** on the 1070 (best
ring ~1.05x, noisy). >= llama.cpp at the same VRAM: **FAIL** on both (llama.cpp 2.2x / ~1.5x our
(a); our best ring 0.40x / ~0.7x of it). **Greedy sha: PASS** - every one of the 30 lemonade-burn
sessions on the two cards (a, b, b4; ring on and off) gives the same text per prompt, and the sonar
prompt gives `062392b8d2b0361a`, the all-resident P100 sha, under 41.6 % residency.

**Why (one number explains every arm on the 2060).** The token is the bus. Per token, arm (a) moves
78.7 misses x 5.31 MB = 418 MB; (b4) 412 MB of prefetch + 19.4 x 5.31 = 515 MB; (b) 849 + 41 = 890 MB;
llama.cpp ~42 x 5.31 = 222 MB. At 1.45 GB/s plus ~25-35 ms of compute and host work that predicts
322 / 385 / 640 / 180 ms against the measured 320 / 375 / 625 / 148 ms [derived from measured
counters]. Prefetch can only hide transfers behind compute, and a decode token here has ~30 ms of
compute against ~290 ms of copies, so the ceiling of perfect overlap is ~1.1x; every wrong guess is
bytes on the one resource that is saturated. Host time inside the MoE reads (`moe_read_ms_per_token`)
is 300 (a) / 345 (b4) / 585 (b) ms of 324 / 381 / 635 ms tokens. On the 1070 the bus is ~3x faster,
so overlap has room and b4 roughly breaks even, but the token is still copy-bound (a: ~119 ms of
reads in a ~125 ms token).

**Where llama.cpp wins: hit rate, not overlap.** Its cache is dynamic (segmented LRU over the whole
5 GB, filled by demand) and holds 73 % of decode routings; our static plan from two scrape files holds
48 %. That is 42 vs 79 copies a token, which is the whole 2.2x on a bus-bound card.

**The guess itself held up** (counters from the ring arms, decode): with 4 guesses, 82 % of the
experts staged for a conv-target layer were used (precision), 65 % for an attention-target layer;
with 8 guesses 49 % / 40 %. Hit rate with the ring: conv targets 0.895 (b4) / 0.971 (b), attention
targets 0.808 / 0.890 (from 0.48 for both without it). So the attention-layer weakness of exp. #1
(`mid1` recall 0.74 vs 0.85) shows up 1:1 as wasted bytes. The ring's stall reduction is real
(79 -> 19 -> 8 per token); it is paid for in bytes the bus cannot carry.

**Host DRAM** (uncore CAS counters, `imc_bw`, mean over each session's completions): c2 1.6 (a) /
1.7 (b4) / 2.1 (b) / 3.4-3.9 (c) GB/s read. Nowhere near the host's limit; Avalon's PREGATE concern
(prefetch stealing RAM bandwidth from CPU expert rows) does not arise here because no expert runs on
the CPU, and the copies are capped by the link. The host cost that does show is the guess itself on
the CPU: 6.4-7.5 ms a token (`issue_ms_per_token`; 38 router dots + the side-stream submits).

### Side finding: NUMA placement of the page-locked GGUF is worth 4x on a two-socket host

c1 (2 sockets, 1070 on node 0) first ran arm (a) at **1.59 tok/s** with a miss waiting
**7.7 ms** (bus time 1.1 ms by `stage_bench`). Ruled out: CPU timeslicing (`nice -10`: no change),
VRAM pressure (4096 MiB budget: no change), IOMMU (off on both hosts), DRAM bandwidth (1.2 GB/s
read). The GGUF's page cache was split ~43/57 across the two nodes; DMA reads of node-1 pages by a
node-0 GPU cross QPI. Evicting the file (`posix_fadvise DONTNEED`) and reading it back under
`numactl --membind=0` gave **6.6 tok/s, 1.5 ms a miss** (same binary, same budget, same text) [measured;
`c1-probe/`, `c1-1070-contended/`, `c1-1070-numa-split-niced-partial/` hold the before data]. Every
two-socket box that DMAs experts out of a page-locked mapping (t440, c4: Le Chonk's helper and RDMA
window) should check where those pages live; the engine does not today.

### Exploratory (not a gate measurement): the ring as a miss cache

To separate caching from prediction, `LB_STAGE_SLOTS=0` + `LB_STAGE_KEEP_MISSES=1` (arm `k`) stages
nothing and keeps demand misses in a Least-Stale ring; `b4k` adds the top-4 guess. Runs in
`c2-2060-x/` (binary `lemonade-burn-server-stage2`, commit 9fea30d), total expert budget still 5120 MiB:

* ring 2048 MiB (static 3072) and 1024 MiB (static 4096): **out of device memory** on the first prompts
  (`allocating 7340032 bytes`). The ring is counted at 5.31 MB an expert, but a kept or staged copy
  holds CubeCL pool slices rounded to 7 MiB plus a transient raw buffer per copy, and kept misses come
  from the main stream's pool while prefetches come from the side stream's.
* ring 512 MiB (static 4609): `k` ran 4 of the 5 prompts at **5.49 / 3.51 / 3.94 / 3.17 tok/s** against
  arm (a)'s 5.01 / 3.23 / 3.37 / 3.00 the same hour (1.05-1.17x, one session, [measured]), then ran out
  of device memory on the 5th: device memory grows while kept misses churn. `b4k` failed the same way.
  Not a result yet. It points in the same direction as llama.cpp's numbers (a dynamic cache is the
  lever), and it is a bug to fix (count and pre-allocate ring slots at their real size, one stream's
  pool) before caching can be measured.


### Exp. #2b: Tesla P100 at Gen3 x16 (c2 GPU2), the link where prefetch could pay

Same model, prompts, 5120 MiB budget, binaries as above; card UUID-pinned
(<card UUID>). llama.cpp rebuilt with sm_60 (`/mnt/fast/llama-build-ivb60`).
Arms a / b4 / b4m / b interleaved 5 rounds; llama.cpp (c) then 5 rounds interleaved with 5 fresh (a)
sessions (`m5120c-*`), because its build finished after the first batch had started. Data:
`/mnt/fast/lb-stage/2026-11-08/c2-p100-x16/` (`RESUME.md` beside it). `b4m` = b4 with
`LB_STAGE_MIN_MARGIN=0.0054`: the 25th percentile of `mid1` decode margins in exp. #1 (that bottom
quartile has recall@4 0.753 vs 0.914 for the top quartile), so the gate skips the least confident
quarter of guesses.

**Bus** (`stage_bench`, this card): 4 experts 2.72 ms on demand = 7.8 GB/s, side-stream copy 8.1 GB/s;
8 experts 8.3 / 8.4 GB/s; **0.63-0.68 ms an expert** (5.5x the 2060's). A main-stream read with 4
copies in flight on the side stream: 1.79 ms vs 0.96 alone [measured].

| arm | tok/s median [min-max], 5 rounds | x (a) | hit rate | stalls / token | prefetch MB / token | wasted MB / token | token ms (MoE read / guess on CPU) |
|---|---|---|---|---|---|---|---|
| (a) | **10.18** [10.10-10.27]; paired batch 10.20 [10.07-10.32] | 1.00 | 0.482 | 78.7 | 0 | 0 | 99.4 (64.6 / 0) |
| (b4) | **11.80** [11.67-12.00] | **1.16** | 0.872 | 19.4 | 412 | 93 | 85.8 (48.8 / 9.0) |
| (b4m) margin >= 0.0054 | **9.88** [9.86-10.01] | **0.97** | 0.765 | 35.7 | 259 | 51 | 102.5 (56.0 / 11.2) |
| (b) | **9.77** [9.76-9.78] | **0.96** | 0.950 | 7.7 | 849 | 455 | 103.8 (73.0 / 8.0) |
| (c) llama.cpp | **15.72** [15.64-15.90] | **1.54** (vs the paired a) | 0.72 | ~45 | - | - | ~63.6 |

All [measured]; token ms from the engine's own counters (device-decode ms per token, host time blocked in
the MoE reads, and the guess: 38 router dots + side-stream submits on the CPU).

**Gates on x16: >= 1.3x over (a) FAIL** (best 1.16x, b4); **>= llama.cpp FAIL** (b4 = 0.75x of it).
**Sha: PASS**, 25 more lemonade-burn sessions on the P100, same text per prompt as on the 2060 and the 1070
(sonar `062392b8d2b0361a`), 55 sessions over three cards in all.

**Margin gate (b4m):** it does cut waste (prefetch 412 -> 259 MB, wasted 93 -> 51 MB a token, precision
82 -> 85 % on conv targets) but it skips whole token-layers, so stalls go back up (19 -> 36 a token) and
the guess is still computed. Net 0.97x: worse than b4. A per-expert confidence (score gap to the k-th)
would be the next thing to try, not a per-token-layer gate.

**Does the bytes model hold on x16? Only half.** On the 2060 the token was bytes / bus. Here (a)'s 418 MB
is 52 ms at 8.1 GB/s of a 99 ms token [derived]; the rest is the 76 host round trips a token (router input
read 16 ms, launches, the guess). And (b) moves 890 MB a token inside a 104 ms token, which needs >= 8.6
GB/s: two streams together get more than `stage_bench`'s single-stream 8.1-8.4 (the 11.2 GB/s registered
slice rate measured on this card on 11-03 is the likely ceiling) [estimate]. So the P100 token is about
half copies, half host work: that is why overlap pays at all here (1.16x) and why it cannot reach 1.3x
while the guess costs 9 ms on the CPU and the MoE layers still sync twice each.

**What would it take [estimate]:** the guess on the device (router_topk exists) removes ~9 ms -> ~77 ms,
~1.29x; folding the guess into the existing h2 read instead of a separate pass is the same lever. llama.cpp
is at 63.6 ms with a 0.72 hit rate and no per-layer host round trips; matching it needs the dynamic cache
(hit rate) *and* the device-side routing, not more prefetch.

### What this says to do next

1. **Not a prefetch problem on these links; a residency problem.** Replace (or back) the static heat
   plan with a dynamic cache over the budget, as llama.cpp does: the measured gap to it (2.2x) is its
   hit rate. Prefetch from `mid1` is worth adding on top only where per-token compute is comparable
   to per-token copy time (P100 on Gen3 x16 at 9-11 GB/s, or after the hit rate is up), behind
   `LB_STAGE` and gated per link (lb-calibrate's measured bandwidth), exactly as the memo and
   Avalon's 2.5 suggested. Top-k guesses only; k+4 doubles the bytes for +8 points of hit rate.
   Measured on the x16 P100 (exp. #2b): top-k ring 1.16x, the only card where it paid at all.
2. **Pin the GGUF page cache to the GPU's NUMA node** before page-locking it (or `mbind` the
   mapping), and print the node split at load.
3. Move the guess to the device (`router_topk` already exists in the device decode) to drop the
   6-7 ms of host work a token, and skip attention-target layers or gate them by margin
   (`LB_STAGE_MIN_MARGIN`, built, not measured).
4. Count the ring at its real VRAM cost (pool rounding + the transient raw buffer of each copy): a
   2048 MiB ring did not fit beside 3072 MiB of residents on 8 GB.


## What was built (all behind flags, default off)

| knob | meaning |
|---|---|
| `LB_EXPERT_VRAM_MIB` | expert VRAM budget (MiB), overrides `--gpu-budget-mb`; the ring is carved out of it |
| `LB_STAGE=1` | turn the ring on (needs `--gpu`, cuda runtime) |
| `LB_STAGE_SLOTS` | guesses staged per layer (default 8 = k+4; 4 = top-k only) |
| `LB_STAGE_RING_MIB` | ring size (default 256 MiB = 50 LFM2.5 experts) |
| `LB_STAGE_MIN_MARGIN` | do not stage a guess whose router margin `s[k-1]-s[k]` is below this (default 0 = off) |
| `LB_STAGE_KEEP_MISSES=1` | demand misses also stay in the ring (ring = small Least-Stale cache too) |

* `crates/lb-lfm2/src/stage.rs`: the `mid1` guess (`predict`/`rank`: layer L+1's FFN pre-norm and
  router on layer L's `x_mid`, the router's ranking rule, margin), the `Ring` (expert copies on a
  second CubeCL stream, Least-Stale eviction: entries not touched this token first, oldest token then
  lowest margin; then layers this token has passed; never the layer being staged), and
  `StageStats` (decode-only, per layer: routings served by resident / ring / demand fetch,
  prefetched / useful / wasted, bytes, host wait in the two MoE syncs) reported in `/v1/placement`
  under `"stage"`, split by attention-target and conv-target layers. Counted in both arms.
* `crates/lb-lfm2/src/moe.rs` `moe_ffn_with`: ring hits launch from the ring; the next layer's
  guess is staged after this layer's launches are queued and before the read that waits for them.
  **Every expert output is now added in ascending expert id** whichever path served it. Before, the
  resident group was summed first, so the greedy text depended on the VRAM budget; all-resident
  and all-CPU layers already summed in ascending id and are unchanged. Consequence: partial
  residency on the 2060 SUPER and the 1070 now gives the all-resident P100 sha `062392b8d2b0361a`.
* `crates/lb-lfm2/src/device_decode.rs`: the host-MoE branch (`LB_DD_HOST_MOE=1`, how an 8 GB card
  runs the device decode) reads `x_mid` with `h2` in the same sync when the ring is on.
* `crates/lb-q4-gpu`: `Gpu::side_stream` / `on_stream` (CubeCL 0.11 per-thread streams; a kernel
  on the main stream that binds a side-stream buffer waits for it on the device, a main-stream read
  does not wait for the side stream). `examples/stage_bench.rs` checks exactly that, plus parity of
  a side-stream upload; `examples/imc_bw.rs` reads the uncore DRAM CAS counters (no `perf` needed).
* `crates/lb-server`: `LB_EXPERT_VRAM_MIB`, ring enable, `"stage"` in `/v1/placement`.
* `bench/stage-ring/`: the c2/c1 harness copies (`e2e.sh`, `drive.py`, `run_all.sh`, `env*.sh`)
  and `score.py`.

## Run layout and resume

Results (append-only JSONL) on t440 `/mnt/fast/lb-stage/2026-11-08/` (c1/c2 see it as
`/mnt/t440/fast/lb-stage/2026-11-08/`):

```
c2-2060/  runs.jsonl (one row per completion, cumulative /v1/placement), sessions.jsonl (one per
          engine start; "session_done": true marks it finished), logs/server-*.log, logs/imc-*.jsonl,
          run_all.pid
c1-1070/  the same for c1's GTX 1070
smoke/    the first single sessions (a, b, b4, c) on the 2060
run_all-c2-2060.log, run_all-c1-1070.log
```

Working copies: c2 `/mnt/ssd/lb-stage/` (bin/, lib/ = nvrtc 12.8, cuda-path/ = libdevice,
models/ = hard links to the exp. #1 GGUF + tokenizer, scrapes/ = the two placement scrapes,
llama/ = llama.cpp e14a779 built for Ivy Bridge + sm_61/75 with CUDA 12.8 libs); c1
`/mnt/models/lb-stage/` (same layout). Binaries built on t440:
`/data/cargo-target-lbstage/build.sh` (lemonade-burn, cuda) and `build-llama-ivb.sh`.

Resume (each session is skipped when `sessions.jsonl` already marks it done):

```sh
# c2, RTX 2060 SUPER (UUID-pinned in env.sh)
ssh -n c2 'cd /mnt/ssd/lb-stage && OUT=/mnt/t440/fast/lb-stage/2026-11-08/c2-2060 setsid nohup ./run_all.sh >> /mnt/t440/fast/lb-stage/2026-11-08/run_all-c2-2060.log 2>&1 < /dev/null & disown'
# c1, GTX 1070
ssh -n c1 'cd /mnt/models/lb-stage && OUT=/mnt/t440/fast/lb-stage/2026-11-08/c1-1070 setsid nohup ./run_all.sh >> /mnt/t440/fast/lb-stage/2026-11-08/run_all-c1-1070.log 2>&1 < /dev/null & disown'
# score (t440)
python3 bench/stage-ring/score.py /mnt/fast/lb-stage/2026-11-08/c2-2060
```

`ARMS` (default `a b4 b c`), `ROUNDS` (5), `MIB` (5120), `RING` (256) are environment overrides;
`TAG` changes the session label prefix (default `m$MIB`) for a new budget.

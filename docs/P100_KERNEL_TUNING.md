# P100 kernel tuning (lemonade-burn qwen35 path), 2026-11-21

Branch `fable/engine-p100-kernels-2026-11-21` (from `fable/engine-fleet-2026-11-21` 81abd2c). Cards: c4 Tesla P100 3B
(GPU-c012cc70) + P100 86 (GPU-0c4b4f5d); RTX 2060 3C (GPU-560cc467) for the RTX checks. Model: Qwen3.6-35B-A3B-Q5_K_M (the
qwen36-lb-c4 seat model). Builds, scripts and raw output: `/mnt/raid0/engines/lemonade-burn/p100-tune-20261121/` (c4:
`/mnt/t440/raid0/...`); `out/*.out` are the JSON lines, `out/*.smi` the nvidia-smi samples (0.5 s), `out/tests-summary.txt` the GPU suites.

## Result

Two kernel changes, both bit-identical to the kernels they replace, on by card profile for the P100 and the sm_75 family (RTX 2060):

| change | 2xP100 decode tok/s short (code / prose) | 2xP100 decode ~4.2k ctx (code / prose) | prefill tok/s (4132 / 4362 tok) | 3-card 2060+3B+86 short / 4k | token-identical |
|---|---|---|---|---|---|
| baseline (81abd2c kernels) | 69.9 / 69.7 | 44.6 / 43.7 | 417 / 434 | 67.2 / 43.7 | reference |
| `LB_QW_ATTN_SPLIT=1` | 69.8 / 69.9 | **61.7 / 61.8** | 417 / 434 | - | yes (4/4 shas, bitwise test) |
| + `LB_QW_Q6_LG=1` | **71.4 / 71.4** | 63.6 / 63.2 | 417 / 432 | 68.4 / 61.3 | yes |
| + `LB_QW_PREFILL_CHUNK` auto instead of 256 | same | same | **589 / 599** | see below | yes (same shas at every chunk) |

Numbers are rounds 1-2 of `bench2` (round 0 holds the kernel compiles). Shas of the 256 greedy ids: a013dec26ae1622a,
f007ae0eeb638964, 411132592a81c6ef, da0129fad7ee9905 in every arm.

## Method

* `qwen35-dev-check bench2 <model> <ids> 256 <chunk> 3`: the server's path per prompt (prefill for 32+ tokens, then 256 greedy tokens with
  the argmax on the device), 3 rounds in one process. Prompts `prompts/all4.ids`: code 14 tok, prose 13 tok, code 4132 tok (a Rust file to
  review), prose 4362 tok (docs/LESSONS_FROM_AVALON.md to summarise). `QWEN_CAP=8192`, split 21,19.
* `qwen35-dev-check parts`: device time of each part of a layer, 300 calls recorded as one CUDA graph and replayed; `bench_head` the same
  for the head; `sweep` the same per configuration of the launch-time knobs in one process; `stage_graph_ms` (end of bench2) replays each
  stage's recorded decode graph alone.
* Bitwise kernel test `crates/lb-q4-gpu/tests/p100_variants_bitwise.rs`: each variant against the kernel it replaces, every output bit
  (random shapes, real head / attn_q / shared-expert rows, attention at n = 1..4352). Passes on the P100 and the RTX 2060.
* GPU suites (lb-q4-gpu + lb-qwen35, `--features cuda`) before (81abd2c on P100 3B) and after (e39dfd2 on P100 86): identical results.
  All pass except `prefill_kernels_match_decode::qk_rope_and_causal_attention_equal_decode_steps_with_gqa_and_a_prior_chunk`, which
  fails the same way in both trees (prefill vs per-token attention differ by 1 ulp, 6e-8; pre-existing). `device_layers` (needs
  `QWEN_IDS`) passes in both, same output with the knobs on. Summary: `docs/qwen-results/p100-gpu-tests-2026-11-21.txt`.

## Where a token went (baseline, device time from `parts`, us)

| part | us | effective GB/s |
|---|---|---|
| delta-net layer total | 257 | |
| - qkv Q8_0 8192x2048 | 41.6 | 428 |
| - z Q5_K 4096x2048 | 29.7 | 194 |
| - out Q8_0 2048x4096 | 23.9 | 372 |
| - delta-net core | 27.8 | |
| - routed experts (8 x gate+up Q5_K, down Q6_K) | 86.9 | 212 |
| - route top-k | 10.0 | |
| - shared gate+up / down (Q8_0) | 13.9 / 5.7 | |
| attention layer total (without the attention core) | 229 | |
| - q Q6_K 8192x2048 | 64.9 | 212 |
| - attention core at n = 4096 (P100) | 818 | ~22 |
| output head Q6_K 248320x2048 | 1675 | 249 |

The P100 reads ~605 GB/s; the big matrices reached 30-70 % of it. Two things stood out: the attention core (one cube per query head =
16 of 56 SMs, each head re-reading its KV head's whole K and V, a serial output loop) and Q6_K rows of 2048 columns (16 half blocks a
row, so half of each plane's lanes did nothing: head and attn_q).

The "3.4 ms a token" gap between the parts sum and the measured token is not host time: the recorded stage graphs replayed alone take
6.40 + 7.04 = 13.44 ms at short context against a 13.95 ms token (host hand-over ~0.5 ms, 3.6 %); the rest is in the graphs: the glue
kernels the parts table does not time (`qw_add_rms`, `qw_silu_mul`, `qw_moe_combine`) and the router / shared-expert matrices being
L2-cold in a real token but warm across 300 repeated calls.

## The changes

### `LB_QW_ATTN_SPLIT` (lb-q4-gpu `qwen_ops.rs`)

Three launches instead of `qw_attn_decode_wide_kernel`: scores (a plane per position of a KV head, serving its 8 query heads, K read once,
112 cubes, `LB_QW_ATTN_CUBES`), softmax (one cube per head, the wide kernel's code verbatim), output (cube per head x 64 dims, a thread per
dim x output chain: the wide kernel's four accumulators o0..o3, split by position mod 4 with the tail in o0, are independent serial
sums, combined as ((o0 + o1) + o2) + o3). Same operations, same order: bit-identical.

| card | attention core n=64 | n=4096 |
|---|---|---|
| P100 (layer 7) | 19.5 -> 21.6 us | 818 -> 194 us |
| RTX 2060 (layer 3, stage 0 of 6,17,17) | 14.4 -> 11.5 us | 532 -> 98 us |

### `LB_QW_Q6_LG` (lb-q4-gpu `kquant.rs`)

`q6_k_native_rows_lg_kernel`: a plane = 32/ipr lane groups each with its own 4 rows when a row has ipr < 32 half blocks; each lane does
the arithmetic it did before and the group sum takes plane_sum's xor offsets 1..ipr/2 in order (the remaining offsets only added zeros):
bit-identical. P100 head 1675 -> 1093 us, attn_q 64.9 -> 49.5 us; RTX 2060 attn_q 96.6 -> 94.2 us (noise level, not slower).

### Prefill chunk (`LB_QW_PREFILL_CHUNK`)

2xP100, Qwen3.6, long prompts (rounds 1-2; shas identical at every chunk):

| chunk | 4132 tok | 4362 tok | nvidia-smi peak (pool, not need) |
|---|---|---|---|
| 256 (profile until now) | 417 tok/s | 434 | 12.8 GB |
| 512 | 526 | 550 | 16.2 GB |
| 1024 (= auto at these lengths) | 589 | 599 | 14.9-16.3 GB |
| 2048 | 535 | 545 | 13.5 GB |

The 256 was set on 2026-11-16 because auto chunk OOMed on a second long prompt with Tiel-Coder Q5_K_XL (26.6 GB) on two P100s, before
the Q8 dedup (b96fa22). Re-run of that case here (Tiel, P100 3B+86, split 19,21, ctx 8192, the two long prompts x 3 rounds): auto chunk
**no OOM**, peak 15.7 / 15.5 GB (256: 15.3 / 15.2), prefill 595 / 603 tok/s against 402 / 418 at 256, same shas. So the P100 profile no
longer sets the chunk (auto: 64..1024 by prompt length). A seat whose weights fill its cards more than Tiel on two P100s can still set
`LB_QW_PREFILL_CHUNK=256` in its manifest command.

## Measured and not adopted

| variant | parts / microbench | end to end (2xP100) | why not |
|---|---|---|---|
| `LB_QW_DN_R=4` (routed down, 4 rows a lane group) | experts 87.0 -> 73.3 us (2: 77.6, 8: 76.4) | **slower**: 71.2 -> 68.4 short, 63.4 -> 60.9 at 4k (shas equal) | the microbench repeats the same 8 experts; real tokens do not |
| `LB_QW_GU_R` 1 / 4 (default 2) | experts 92.7 / 90.1 vs 87.0 | not run | microbench already worse |
| `LB_QW_R` 2 / 8 (default 4) | z 32.2 / 27.7 vs 29.9; q 58.8 / 53.3 vs 49.4; o 36.8 / 32.4 vs 29.2 | not run | worse on q and o |
| `LB_QW_SH_FUSE=1` (shared gate+up+SiLU one launch, bit-identical) | - | 71.4 -> 71.2 short, 63.4 -> 63.0 at 4k | no gain: 80 fewer tiny launches a token did not show |

FP16 arithmetic was not tried on the K-quant kernels: the X7a measurement on this card (docs/WORK_TO_WEIGHTS.md, Q4_0 rows kernels) gave
1.03x and failed its accuracy gate, and the K-quant kernels here are bound by memory access and occupancy (194-428 GB/s of 605), not by
FMA rate. No DP4A on sm_60.

## Values for the card profiles (in the JSON now)

| knob | nvidia-p100.json | nvidia-turing-sm75.json | code default |
|---|---|---|---|
| `LB_QW_ATTN_SPLIT` | 1 | 1 (measured on the RTX 2060 only) | 0 |
| `LB_QW_Q6_LG` | 1 | 1 (RTX 2060 only) | 0 |
| `LB_QW_PREFILL_CHUNK` | not set (was 256) | not set | 0 (auto 64..1024) |
| `LB_QW_DN_R`, `LB_QW_GU_R`, `LB_QW_SH_FUSE`, `LB_QW_ATTN_CUBES` | not set (defaults 1 / 2 / 0 / 112) | not set | as listed |

## Not measured

* One P100 alone with the qwen35 path: no qwen35moe file the engine loads fits 16 GB (smallest: Qwen3.6 Q5_K_M, 25.9 GB). Per-kernel
  numbers above are per card. The LFM2 single-card path (Q4_0 kernels) was not touched or re-measured.
* RTX 2060 SUPER and T10 (share the sm_75 profile): not measured.
* MTP verify path with the new knobs: not separately timed.

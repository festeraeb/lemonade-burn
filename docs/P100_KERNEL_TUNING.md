# P100 kernel tuning (lemonade-burn qwen35 path), 2026-11-21

Branch `fable/engine-p100-kernels-2026-11-21` (from `fable/engine-fleet-2026-11-21` 81abd2c). Cards: c4 Tesla P100 3B
(GPU-c012cc70) + P100 86 (GPU-0c4b4f5d), RTX 2060 3C for the RTX checks. Model: Qwen3.6-35B-A3B-Q5_K_M (the qwen36-lb-c4 seat model).
Builds and raw output: `/mnt/raid0/engines/lemonade-burn/p100-tune-20261121/` (c4: `/mnt/t440/raid0/...`), `out/*.out` are the JSON lines.

Status: in progress; tables below are filled as each step is measured.

## Method

* `qwen35-dev-check bench2 <model> <ids> 256 <chunk> 3`: the server's path for each prompt (prefill for 32+ tokens, then 256 greedy tokens
  with the argmax on the device), 3 rounds in one process, round 0 dropped (kernel compiles). Prompts (`prompts/all4.ids`): code 14 tok,
  prose 13 tok, code 4132 tok, prose 4362 tok. `QWEN_CAP=8192`, split 21,19, chunk 256 (the P100 profile's prefill chunk at the start).
  Token identity: FNV-1a sha of the 256 generated ids per prompt.
* `qwen35-dev-check parts`: device time of each part of a layer, 300 calls recorded as one CUDA graph and replayed (in-graph launch gaps
  included, host launch cost not). `bench_head`: the output head and argmax the same way. `sweep`: the same per configuration of the
  launch-time knobs in one process.
* Bitwise kernel test: `crates/lb-q4-gpu/tests/p100_variants_bitwise.rs` (each variant against the kernel it replaces, every output bit).

## Baseline (81abd2c kernels), 2 x P100, split 21,19

| prompt | prefill tok/s | decode tok/s | sha |
|---|---|---|---|
| code 14 | - | 69.9 | a013dec26ae1622a |
| prose 13 | - | 69.7 | f007ae0eeb638964 |
| code 4132 | 417 | 44.6 | 411132592a81c6ef |
| prose 4362 | 434 | 43.7 | da0129fad7ee9905 |

Where a token goes (device time, `parts`, us): delta-net layer 257 (qkv Q8_0 41.6 = 428 GB/s, z Q5_K 29.7 = 194 GB/s, out 23.9,
delta-net core 27.8, routed experts 86.9 = 212 GB/s, route top-k 10.0, shared gate+up 13.9, shared down 5.7, rms / small f32 matvecs
5-7 each); attention layer 229 (q Q6_K 64.9 = 212 GB/s, o 29.2); output head Q6_K 248320 x 2048 **1675** (249 GB/s), argmax 27.

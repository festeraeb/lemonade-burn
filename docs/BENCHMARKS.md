# Measured results (2026-10-22)

All numbers below were run as shown. Nothing here is estimated.

## Correctness vs HF transformers (t440, 4 fixed prompts, 146 positions, teacher-forced)

`lb-run compare --model <lbt dir> --ref <ref dir>`; `python -I scripts/compare_refs.py A B`

| pair | top-1 agree | where ref margin > 0.5 | max abs dlogit |
|---|---|---|---|
| lemonade-burn vs HF float32 | 138/146 | 113/113 | 1.556 |
| HF float32 vs HF float64 | 140/146 | 113/113 | 1.603 |
| lemonade-burn vs HF float64 | 142/146 | 113/113 | 1.228 |

HF's own float32 run differs from float64 by as much as lemonade-burn does. The logit
deltas around 1 are therefore intrinsic to the model's int8 activation rounding (tiny
float differences flip `round(x*127/max)`); they are not an engine error. The tokenizer
matches HF on all 4 prompts.

## Speed on cesarops3 (Xeon E3-1265L v3, AVX2, idle, 64-token prompt, 32 generated, 3 reps)

| runtime | threads | prefill tok/s | decode tok/s | peak RSS |
|---|---|---|---|---|
| lemonade-burn (avx2) | 4 | 37.9 / 40.5 / 39.9 | 7.75 / 7.22 / 7.75 | 3522 MB peak, 1859 MB steady |
| lemonade-burn (avx2) | 8 | 48.8 / 43.5 / 47.5 | 7.66 / 7.70 / 7.64 | 3523 MB peak, 1859 MB steady |
| bitnet.cpp 0b341e5, i2_s GGUF (**output incorrect, see below**) | 4 | 78.69 +- 0.59 | 8.85 +- 0.37 | 1221 MB |
| bitnet.cpp, same | 8 | 84.18 +- 3.81 | 7.91 +- 1.01 | 1221 MB |

Commands (peak RSS from `resource.getrusage(RUSAGE_CHILDREN)` in a python wrapper):

```
RAYON_NUM_THREADS=$t lb-run bench --model bitnet-b1.58-2B-4T-lbt --prompt-len 64 --gen 32 --reps 3
llama-bench -m ggml-model-i2_s.gguf -p 64 -n 32 -t $t -r 3        # microsoft/BitNet build/bin
```

### Why the bitnet.cpp row is not a valid comparison

* bitnet.cpp was built per its `setup_env.py` x86 path (`codegen_tl2.py`, then cmake with
  `-DBITNET_X86_TL2=OFF`, clang 22).
* Greedy `llama-completion -p "The capital of France is" -n 24 --temp 0 -no-cnv` printed
  "a small city, and the capital of France is a small city, ...". HF and lemonade-burn
  print " Paris. Paris is a city in the north of France ...".
* `llama-bench` labels the model `Q1_0` although the GGUF stores type 36 (`I2_S` in
  bitnet.cpp's gguf-py).
* Hypothesis (unverified): the rebased ggml in this checkout maps type 36 to a different
  type. Its speed numbers would then be for misdecoded weights.

### Why upstream llama.cpp was not used

Upstream llama.cpp b10615 `BITNET` arch targets the older 1bitLLM 3B layout: it uses a
SiLU FFN and maps `inner_attn_ln` / `ffn_layernorm`. 2B4T uses ReLU^2 and
`attn_sub_norm` / `ffn_sub_norm`, and conversion fails with
`Can not map tensor 'model.layers.0.mlp.ffn_sub_norm.weight'`.

### Notes

* **Peak vs steady RSS:** lemonade-burn copies packed weights and the head from the mmap
  into Burn tensors at load. Peak RSS is therefore about 2x steady; a zero-copy load is
  open work.
* **Decode cost split** (`lb-run bench` op timers, c3, 8 threads): about 60 ms/token for
  the 656 MB bf16 LM head and about 65 ms for all ternary layers.
* **Kernel speed:** the ternary kernel streams at 10.8 GB/s against an 11.3 GB/s read
  ceiling (`KB_STREAM=1 kbench`).

# CPU scaling work (2026-10-22, branch perf/scaling-2026-10-22)

Before = master binary (same bytes as `/data/lemonade-burn/bin/lb-run` on c4); after =
commit cdf306d. Both measured back to back in the same script, 2 rounds alternating, so
they saw the same machine load. Logits are **bit-identical** to master (see Correctness).

## What was slow (evidence)

`lb-run bench` op timers on c4, 20 threads, 64-token prefill + 32 decode, before:
ternary ops 2.9 s, of which **1.37 s was the activation quantizer**, LM head 0.70 s,
other 0.46 s. The quantizer was scalar (`round_ties_even` per element, 35 us per
2560-wide row on c4), single-threaded, and ran once per linear (7 times per layer, 64
rows each in prefill). That is why AVX-512 and AVX2 measured the same and why prefill did
not scale. Thread wake-up was *not* the limit: a persistent spinning pool was tried and
measured slightly slower (commit c5f9c48, reverted).

## Changes

1. AVX2 activation quantizer, bit-identical to the scalar `ActQuant` definition (tested on
   ties, NaN, +-inf, denormals, tails); rows quantized in parallel for prefill.
2. q/k/v and gate/up run as one fused op each (one quantization, one parallel region);
   parallel prefill transpose.
3. AVX-512 prefill kernel: activations re-laid out so one 64-byte load feeds a lane group,
   2 weight rows x 4 tokens per tile, token tile outer (activations stay in L1).
4. Optional two-socket mode: `LB_NUMA="0-19;20-39" LB_NUMA_SHARE=0.72` splits every
   ternary matrix and the LM head by rows, places each part on its node by first touch
   and runs both parts concurrently on per-node pools. Default is off.

## c4 (2 x Xeon Gold 6148, AVX-512 BW, no VNNI), 64-token prompt, 32 generated, 3 reps x 2 rounds

```
LB_KERNEL=avx512 RAYON_NUM_THREADS=$t taskset -c 0-19 numactl --membind=0 lb-run bench \
  --model bitnet-b1.58-2B-4T-lbt --prompt-len 64 --gen 32 --reps 3
# both sockets, after:
LB_NUMA="0-19;20-39" LB_NUMA_SHARE=0.72 LB_KERNEL=avx512 taskset -c 0-39 lb-run bench ...
# both sockets, before (naive): RAYON_NUM_THREADS=40 taskset -c 0-39 numactl --interleave=0,1
```

| config | prefill tok/s before | after | decode tok/s before | after |
|---|---|---|---|---|
| socket 0, 8 threads | 20.8-24.4 | 79-99 | 8.3-9.6 | 15.8-16.6 |
| socket 0, 16 threads | 31.0-37.8 | 100-117 | 8.8-10.9 | 16.1-17.4 |
| socket 0, 20 threads | 42.1-44.2 | 173-179 (first rep 136-139) | 12.0-12.3 | 18.2-19.5 |
| both sockets, 40 threads | 39.0-45.4 | 180-185 (first rep 129-131) | 8.8-9.9 | 21.0-22.2 |
| socket 0, 20 threads, 256-token prompt + 64 | 42.6-43.0 | 145-157 | 10.9 | 15.2-15.3 |

The first rep of each "after" run is slower in prefill (first touch of the larger
buffers); the ranges list it separately where it falls outside.

Load: c4's GPU seats and other agents were running (load average 2.5-3.2 before the run;
it rose to 15-17 during it, which includes this benchmark's own 20-40 threads). The
read-only DRAM sweep on socket 0 (`KB_STREAM=1 KB_THREADS=20 kbench`) gave 26.2 and
23.7 GB/s at the start of the two rounds; at other times today it gave 31-43 GB/s, so
decode numbers on this box move with other tenants' memory traffic.

## Roofline (decode)

Weights read per token: 0.52 GB ternary + 0.66 GB bf16 LM head = 1.18 GB. At the 24-26
GB/s measured during the final run that is a ceiling of 20-22 tok/s on one socket; the
measured 18.2-19.5 tok/s is about 90% of it, with ~8 ms/token of non-GEMM Burn ops
(attention, RoPE, norms) on top of the weight streaming. The model-like kernel probe
(`KB_MODEL=1 kbench`, 120 parallel regions per token, no Burn ops) reaches the same
bandwidth as one 520 MB GEMV. Socket 1 streams only ~16 GB/s on this box (vs 31-43 on
socket 0), which is why the two-socket split puts 72% of rows on node 0 (kbench split
GEMV: 56-60 GB/s at share 0.7 vs 32-34 at 0.5).

## Prefill

20-thread prefill now spends ~0.2 s of 0.36 s (64 tokens) in the ternary GEMMs; the AVX-512
tile reaches 1.5-2.1 int8 TOPS at m=64 on 20 cores (`kbench`, ~64 GMAC/s per core).
At longer prompts Burn's attention dominates: `cargo run --release -p lb-model --example
attn_probe 520` shows Flex taking ~650 us for one decode-step attention at 520 tokens of
context (q@K^T ~300-420 us, softmax ~270 us for 20 x 520 values). That also makes decode
slow at long context (12.6 tok/s at 512 tokens, 20 threads). A custom attention op would
fix it but changes float summation order, so it was left for a separate, signed-off change.

## cesarops3 (Xeon E3-1265L v3, 4 cores / 8 threads, AVX2), same bench command

Lemonade on :13305 had Qwen3-14B loaded on its GPU, idle (load average 0.12 before).

| threads | prefill before | after | decode before | after |
|---|---|---|---|---|
| 4 | 37.8-38.8 | 61.4-64.8 | 6.8-7.5 | 7.8-8.2 |
| 8 | 46.8-48.6 | 66.6-67.7 | 6.9-7.3 | 7.4-8.1 |

c4 one socket now decodes 2.4x and prefills 2.7x faster than c3 (20 vs 4 threads);
before the change it was 1.6x and 1.1x.

## Correctness

* `cargo test --release --workspace` (t440, AVX-512 VNNI): all pass, including the new
  bit-exact tests (SIMD vs scalar quantizer, fused/multi GEMM vs the integer definition
  under 1/3/8 threads, long-row overflow test, two-socket stitching vs single op).
* `lb-run compare --model ... --ref ref-f32 --dump DIR` on c4, then `cmp` of the dumped
  logits against master's: bit-identical on all 4 prompts for 20 and 7 threads, for the
  two-socket mode at shares 0.72 and 0.3, and for each intermediate commit.
  TOTAL top1 agree 138/146, max|dlogit| 1.5560 (unchanged).

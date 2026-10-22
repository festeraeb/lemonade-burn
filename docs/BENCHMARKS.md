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

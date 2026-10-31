# LFM2.5-24B-A2B on our own engine (Burn + CubeCL)

Status: **step 1 (correct forward pass) — see "Verification" for what was run and the numbers.**
Operator decision 2026-10-31: stop patching other people's engines; build our own MoE
engine in Rust on Burn 0.22.1 / CubeCL 0.11.2 (one kernel source for Vulkan, CUDA, ROCm,
Metal, DX12), and use the NautiBurn routing scrapes to place experts.

Clean room: the architecture below was learned from the GGUF metadata, LiquidAI's model
card / `config.json`, and reading llama.cpp's LFM2 graph to learn shapes and op order.
No runtime code was ported or transliterated; the quant block layouts are the file
format (documented in `crates/lb-gguf/src/types.rs`).

Figures are labelled **[measured]** (run here, command given) or **[estimate]**.

## 1. The model as we understand it

GGUF `general.architecture = lfm2moe`, file `LFM2.5-24B-A2B-QAD-Q4_0.gguf` (13.58 GB,
LiquidAI's quantization-aware Q4_0; every matrix Q4_0 except the tied embedding, Q6_K).

| hyperparameter | value | GGUF key |
|---|---|---|
| layers | 40 | `lfm2moe.block_count` |
| hidden | 2048 | `embedding_length` |
| heads / kv heads / head dim | 32 / 8 / 64 | `attention.head_count`, `head_count_kv[]` |
| attention layers | 2, 6, 10, …, 38 (every 4th from 2; 10 of 40) | `head_count_kv[l] > 0` |
| short-conv layers | the other 30 | `head_count_kv[l] == 0` |
| conv kernel length | 3 (`l_cache`) | `shortconv.l_cache` |
| dense FFN (layers 0, 1) | SwiGLU, width 11776 | `feed_forward_length`, `leading_dense_block_count = 2` |
| MoE FFN (layers 2..39) | 64 experts, top-4, width 1536, sigmoid gate + selection bias | `expert_count`, `expert_used_count`, `expert_feed_forward_length`, `expert_gating_func = 2` |
| norms | RMSNorm, eps 1e-5, per-head q/k RMSNorm | |
| RoPE | NEOX half-rotation, theta 1e7 | `rope.freq_base` |
| vocab / context | 128000 / 131072 | |
| head | tied to `token_embd` ; final norm is `token_embd_norm` | |

Per layer (pre-norm residual; `h = rms(x, attn_norm)`):

```
conv layer:  [B, C, X] = split(in_proj(h), 3)          in_proj: 2048 -> 6144
             y = C * conv3(B * X)                        causal depthwise conv, 3 taps per channel,
             x = x + out_proj(y)                         state = last 2 rows of B*X
attn layer:  q, k, v = projections; q, k = rms_head(q), rms_head(k); RoPE(q, k)
             x = x + o_proj(softmax(q k^T / 8) v)        GQA 32/8
ffn:         h2 = rms(x, ffn_norm)
             dense:  x = x + down(silu(gate h2) * up h2)
             moe:    p = sigmoid(h2 W_r^T); select top-4 by (p + bias); w_i = p_i / sum(p_sel)
                     x = x + sum_i w_i * down_i(silu(gate_i h2) * up_i h2)
logits = rms(x, token_embd_norm) @ token_embd^T
```

Gotcha found while verifying: `shortconv.conv.weight` has GGUF dims `[3, 2048]`, i.e.
**each channel's three taps are contiguous** (`raw[c*3 + k]`), not three rows of 2048.
The first run with the taps read as rows gave 1/13 agreement; with the right layout see §5.

## 2. Crates (branch `lfm2-cubecl`)

| crate | what | tests |
|---|---|---|
| `lb-gguf` | GGUF v2/v3 reader (mmap, zero copy), block dequant for F32/F16/Q8_0/Q4_0/Q6_K, CPU matvec `matvec_f32` (rayon over rows; fused Q4_0/Q8_0 dot for single tokens), `quantize_q4_0/q8_0` for tests | unit: nibble placement, round trips, matvec vs dequantized; integration: parses the real LFM2.5 file and matches **gguf-py's dequantize** on Q6_K/F32 rows (`tests/oracle_iq3xxs.txt`) |
| `lb-lfm2` | `Config` from metadata, `Weights` tensor map (quantized matrices stay in the mmap as `QuantMat` views; expert stacks as `ExpertMats`), the Burn forward pass (`model.rs`), router / expert FFN / placement / scrape writer (`moe.rs`), `lfm2-check` teacher-forced harness | unit: router selection-vs-weight rule, placement budget, scrape→histogram |
| `lb-q4-gpu` | CubeCL Q4_0 matvec kernel (own workspace, wgpu runtime; device from `LB_WGPU_DEVICE` via `lb-ternary-gpu::pick_device`) | `tests/gpu_matches_scalar.rs`: synthetic shapes incl. the LFM2 expert shapes (1536x2048, 2048x1536) and a real expert matrix from the file, vs `lb_gguf::matvec_f32` |

The forward pass is Burn tensor code on the Flex (CPU) backend for norms, the conv,
attention and residuals; quantized matmuls go through `Lfm2::linear`, which calls the
kernel on the tensor's data (same split as `lb-model`: a backend-extension op would
hide the copy, it is not needed for correctness and can come with the GPU path).

### Kernel plan (CubeCL)

* `q4_0_matvec_kernel` (done, this branch): one thread per output element, decodes the
  f16 block scale from bits (exact), 16 nibble bytes per block, f32 accumulate.
  Good enough to prove the path; it is not tuned (no vectorized loads, no shared memory,
  one token per thread).
* Next kernels, each with the CPU reference in `lb-gguf` and a `gpu_matches_scalar`
  test, specced for forge lanes (see §6): Q6_K matvec (LM head / embedding), a
  multi-token Q4_0 GEMM tile for prefill, and a fused `silu(gate)*up` expert kernel that
  reads the three expert matrices once per token batch.

## 3. Expert placement from the scrapes

Why this is where we can beat llama.cpp on small cards: per token only 4 of 64 experts
per layer run (152 expert FFNs of 2432), and the routing is far from uniform. A WX 5100
(8 GB) cannot hold the 12.1 GB of expert weights, but it can hold the hot ones; the cold
ones stay in the mmap (CPU RAM / disk page cache) and run on the CPU path.

Design (`lb_lfm2::moe`):

* `Histogram { counts[layer][expert] }` — from NautiBurn `histogram_counts` (the
  sidechannel DB fed by every scraping engine) or from this engine's own scrapes
  (`Histogram::from_scrapes`).
* `Placement::plan(hist, bytes_per_expert, budget)` — greedy: experts ranked by their
  share of their layer's routing mass, GPU-resident until the byte budget is spent;
  layers with no data keep everything on the CPU. `Home::{Gpu, Cpu}` per expert.
* `moe_ffn` groups the batch's tokens by expert, runs each expert once, and consults the
  placement per expert call. **Step 1: every call executes on the CPU path**; the
  placement is a shadow that counts would-be GPU hits (`PlacementStats`), so the
  correctness run does not depend on the residency code.
* Prefetch: `nautiburn-shadow`'s `prepredict` names the next layer's expert set from the
  kept ensemble (frequency / previous token / cross-layer bigram) before the gate runs.
  The engine hook for it is the per-layer route list `Scraper::record` already produces;
  the next layer (step 2) feeds it to the predictor and issues the copy of a predicted
  non-resident expert while the current layer computes (13 ms planning lead per the
  NautiBurn note).
* Scraper: `NAUTIBURN_SCRAPE_DIR` makes the engine write NautiBurn `schema_version: 1`
  files (`run-<epoch>-<seq>.json`, one per request, phase prefill/decode) with
  `(layer, token_pos, expert_ids)` — the same format the llama.cpp hook writes, so the
  existing consumer ingests them. The moe-scraper rule is met by the engine itself.

Expert bytes at Q4_0 **[arithmetic]**: gate+up 2 x 1536 x 2048 + down 2048 x 1536 =
9.44 M values x 18/32 B = 5.3 MB per expert, 340 MB per layer, 12.9 GB for 38 layers
(the file's experts are 12.1 GB; the rest is dense layers, attention, embedding).
A 6 GB budget on the WX 5100 holds about 1130 of 2432 experts.

## 4. Reference oracle and harness

`lfm2-check` (crate `lb-lfm2`): the same GGUF served by a throwaway llama.cpp server
(`llama-server --device none`, CPU only, port 18080 on t440, `CUDA_VISIBLE_DEVICES=`
so t440's GPUs stay untouched). The harness tokenizes through the oracle's `/tokenize`
(our BPE tokenizer is step 2 work; token ids are the input to the comparison), runs OUR
forward once over the whole prompt (all positions' logits), then asks the oracle for its
greedy next token at every prefix (`/completion`, `n_predict 1`, `top_k 1`,
`cache_prompt`). It prints per position: our top-1, the oracle's top-1, the actual next
token, agree yes/no; then the aggregate. `--generate N` additionally runs our decode
path (conv state + KV cache) greedily and prints the text.

## 5. Verification results

See the per-position tables in `docs/lfm2-check-2026-10-31.txt` (raw harness output).

All **[measured]** 2026-10-31 on t440, CPU path (48 threads, f32 activations), model
`LFM2.5-24B-A2B-QAD-Q4_0.gguf`, oracle `llama-server --device none` on the same file.

* Loader: `lb-gguf` tests pass, including the real-file test that matches gguf-py's
  dequantize on Q6_K and F32 rows (7 tests).
* Q4_0 GPU kernel: `lb-q4-gpu` 5/5 on **cesarops3's Radeon Pro WX 5100** (RADV Vulkan,
  `LB_WGPU_DEVICE=discrete:1`), including a real expert matrix (`blk.2.ffn_gate_exps`,
  expert 0, 1536x2048, two tokens) against the CPU reference.
* Teacher-forced top-1 agreement with the oracle (per position tables in
  `docs/lfm2-check-2026-10-31.txt`):

| run | prompt | positions | ours == oracle | ours == true next | oracle == true next |
|---|---|---|---|---|---|
| A | "The capital of France is Paris. The capital of Germany is" | 13 | 12 | 8/12 | 9/12 |
| B1 | SAR / side-scan sonar sentence | 29 | 25 | | |
| B2 | a Rust `fn main` snippet (with literal `\n`) | 49 | 46 | | |
| B3 | chat-template question (boiling point) | 23 | 23 | | |
| B4 | shipwreck survey notes | 44 | 39 | | |
| B total | | 145 | **133 (91.7%)** | 59/141 | 60/141 |
| C | "The sonar operator marked the target at 31 m and the dive team" (placement run) | 16 | 13 | 5/15 | 4/15 |

Our engine predicts the real next token as often as llama.cpp does (59 vs 60 of 141);
the disagreements are near-ties that flip between our f32-activation matmul and
llama.cpp's int8-activation Q4_0 dot (e.g. run B1 pos 7: ours " Michigan" = truth, oracle
" found"). Greedy continuations from our decode path (conv state + KV cache): run A
" Berlin. The capital of Italy"; B1 " the kayak was found at a depth"; B3
"<think>\nWe need to answer: \""; B4 " seabed was mostly sand with some gravel".

Speed on this path: 2-4 s per prompt token (about 56 s for 29 tokens, 69 s for 49),
nearly all of it in the scalar dequant matmuls; it is the reference, not the product.

* Placement shadow (run C): histogram from 177 scrape files (55,472 expert activations:
  this engine's own scrapes plus the oracle's NautiBurn hook output from runs A/B), budget
  6000 MB: 1130 of 2432 experts resident (5999 MB); on a fresh prompt **730 of 1097 expert
  calls (66.5%) would hit a GPU-resident expert**, 367 would take the CPU path. With more
  scrape data the hit rate should rise; this is the number to beat with prefetch.
* Scraper: every run wrote `run-<epoch>-<seq>.json` (schema_version 1) with
  `NAUTIBURN_SCRAPE_DIR` set; the files were read back by `Histogram::from_scrapes`.

## 6. Forge jobs queued

Seeds proven first: the seeded tests compile, and on c3 they fail only on the stub
(`q6_k_matvec_gpu: not implemented yet`) while the Q4_0 tests pass
(`forge-pool/checks/lb_gpu_kernel.sh q6k_matches_scalar`, 2026-10-31).

| job | spec | files | check |
|---|---|---|---|
| LB-Q8GPU | `forge-pool/specs/LB-Q8GPU.json` | `crates/lb-q4-gpu/src/lib.rs` (`q8_0_matvec_gpu`) | `lb_gpu_kernel.sh q8_0_matches_scalar` on c3 |
| LB-Q6KGPU | `forge-pool/specs/LB-Q6KGPU.json` | `crates/lb-q4-gpu/src/lib.rs` (`q6_k_matvec_gpu`) | `lb_gpu_kernel.sh q6k_matches_scalar` on c3 |

The check script runs the crate's tests on cesarops3's WX 5100 over the NFS view of the
verify worktree; t440's GPUs are never touched.

## 7. What is left for step 2

* GPU execution of resident experts (`Home::Gpu`) with the CubeCL kernels; device
  buffers per expert, uploaded at load from the placement; CPU fallback stays.
* Prefetch loop with `nautiburn-shadow` predictions; measure hit rate and time hidden.
* Q6_K kernel for the head; multi-token GEMM for prefill; attention on the GPU.
* Own BPE tokenizer from the GGUF vocab (`tokenizer.ggml.*`, pre-tokenizer `lfm2`).
* Serving: `lb-server` model trait so it can host `Lfm2` as well as BitNet; Lemonade
  adapter entry.
* Speed: the step-1 CPU path is a correctness reference (f32 activations, scalar
  dequant), not a fast path.

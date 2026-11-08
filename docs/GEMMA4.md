# lb-gemma4: gemma-4 MoE in lemonade-burn (engine step E1b, 2026-11-08)

Crate `crates/lb-gemma4`, branch `fable/engine-e1b-gemma4-2026-11-08`. CPU forward of
`google gemma-4-26B-A4B-it` from GGUF, checked against llama.cpp e14a779 (`src/models/gemma4.cpp`).
Model file used for every number here: `/mnt/fast/models-work/gemma-4-26B-A4B-it-Q4_0.gguf`
(llama-quantize e14a779 `Q4_0` from the unsloth Q8_0; token_embd/LM head Q6_K by the default mix;
sha256 beside the file). Run data: `/mnt/fast/lb-gemma4/2026-11-08/` on t440.

## Architecture as implemented (all from the GGUF + gemma4.cpp)
- 30 layers, n_embd 2816, 16 query heads. Sliding layers (25): head dim 256, 8 KV heads, RoPE base 1e4,
  window 1024. Global layers 5/11/17/23/29: head dim 512, 2 KV heads, RoPE base 1e6 with `rope_freqs`
  divisors (64 of 256 pairs rotate, the rest 1e30 = off), **no attn_v: V = K projection**.
- q/k RMS-normed per head with weights, V RMS-normed without weight; attention scale 1.0.
- Every layer: shared GEGLU FFN (2112) + MoE (128 experts, top-8). Router input is
  `rms(x_mid) / sqrt(n_embd) * ffn_gate_inp.scale` (not the experts' pre-norm), softmax over 128,
  top-8, renormalised; fused `gate_up_exps` (gate rows first); per-expert `ffn_down_exps.scale`.
  Post-norms on attention, shared FFN, MoE and the sum; `layer_output_scale` scales the residual.
- Embedding x sqrt(n_embd); tied LM head; final logit softcap 30.
- Refused at load (not in the 26B): per-layer embeddings (E2B/E4B), KV-shared layers, fused QKV.

## Interfaces reused from lb-lfm2 (so E1-E4 reach gemma)
`Route`, `Placement`, `PlacementStats` (+ `stage` per-layer decode counters), `ExpertStore` (knobs,
miss policy), `Scraper` (NautiBurn scrapes, class `lb-gemma4`), `CapturedLayer` + `set_capture`/
`take_capture`, `stage::rank` for the predictor (`Gemma4::predict`). Softmax and sigmoid are both
monotone in the logit, so `rank` with `norm = scale / sqrt(n)` and zero bias gives gemma's ranking.

**Gaps (findings, not failures):**
1. lb-q4-gpu's expert and dense kernels hard-code SwiGLU (`silu`); gemma is GEGLU (tanh GELU). No
   gemma expert can be device-resident until the kernels take an activation parameter
   (`moe::check_store` refuses residents rather than run them wrong). This is the one change E1-E4 need
   before they apply to gemma on a GPU.
2. `ExpertStore::load`, `DenseStore`, `stage::predict` and `moe_ffn_with` take `lb_lfm2::Weights`;
   gemma's tensor layout (fused gate_up, per-expert down scale, router scale vector) does not fit
   `ExpertMats`/`Ffn::Moe`. A shared `lb-moe` crate with a small expert-source trait (expert bytes,
   gate/up/down `MatRef`, activation, output scale) would let both families share one store.
3. The predictor's `margin` is a sigmoid-space margin; for softmax routers a softmax margin would be
   the comparable confidence.

## Numerics: why `numerics.rs` exists [measured]
llama.cpp quantizes activations before every quantized matmul (CPU: Q8_0 for Q4_0, Q8_K for K-quants;
CUDA: q8_1) and keeps K/V in f16. Layer-0 `Qcur` of llama.cpp (-4.1107 ...) is reproduced to every
printed digit by an f64 matmul over Q8_0-quantized activations; the exact f32 product is -3.9429.
`Numerics::LlamaCpp` (default) fake-quantizes activations the same way and rounds Q/K/V to f16;
`LB_GEMMA4_NUMERICS=exact` turns it off. With it, layer-0 attention matches to 6e-4.

## Parity [measured, t440 CPU, 2026-11-08], teacher-forced, 16 steps (France 12)
Prompts: raw "The capital of France is"; chat (template via llama-server, thinking off) sonar question
(59 tokens); chat 241-token survey-plan prompt. Gate: >= 8 identical, top-5 sets match, max|dp| <= 0.03.

| comparison | France raw | chat 59 | chat 241 |
|---|---|---|---|
| ours vs llama.cpp default (repack) | 0/12, dp 0.378 | **16/16, top-5 all, dp 0.052** | **16/16, 1 tail top-5 miss, dp 0.049** |
| llama.cpp default vs llama.cpp `--no-repack` | 0/12, dp 0.393 | 16/16, top-5 miss, dp 0.071 | 16/16, top-5 miss, dp 0.046 |
| ours vs llama.cpp `--no-repack` | 2/16, dp 0.823 | 16/16, top-5 miss, dp 0.138 | 16/16, top-5 miss, dp 0.025 |
| llama.cpp default, `-fa off -ctk f32 -ctv f32`, step 0 only | top-1 1623 (0.177), 45518 0.031 vs 0.429 | | |

Reading: the France prompt's step 0 is ill-conditioned. Two llama.cpp builds of the same file differ
there by |dp| 0.39 and pick different top-1s, and our top-1 equals llama.cpp `--no-repack`'s.
Traces (`gemma4-check --trace-dir` vs `llama-eval-callback`) show no single op diverging. The
difference grows smoothly from ~0.1% at layer 0 to large at layers 23-29, with the same experts
selected through layer 8. On the chat prompts our engine matches every greedy token and sits inside
llama.cpp's own kernel-variant spread. **The gate as written (top-5 at every step, |dp| <= 0.03) is
tighter than llama.cpp's agreement with itself across one kernel switch**, so no independent
implementation can pass it on these prompts. Suggest gating on: identical greedy steps, plus |dp| and
top-5 no worse than the llama.cpp default vs `--no-repack` floor measured on the same prompts.

Predictor on gemma (capture hook, chat 59 prefill, 1711 token-layers): mid1 recall@8 0.643, all-8 in
top-12 0.232 [measured], against 0.88 / 0.60 on LFM2.5 (exp #1). Gemma's 128-way router is harder to
guess one layer ahead.

## Not done in this step
- GPU (GTX 1070 on c1): blocked by gap 1 (no GEGLU kernel). No tok/s on a GPU.
- IQ4_XS kernels (out of scope). Speed: CPU decode about 1-1.5 tok/s on a shared t440, indicative only.
- Held-out-v1: see `/mnt/fast/lb-gemma4/2026-11-08/RESUME.md` (run in progress at the 11-08 shutdown).

## Tools
`gemma4-check --model M --oracle URL --prompt P --chat U --n 16 [--trace-dir D] [--predict-check]`;
`gemma4-serve --model M --listen 127.0.0.1:18197` (token ids in, greedy ids out; driver
`/mnt/fast/lb-gemma4/2026-11-08/bin/heldout_drive.py` renders/tokenizes through llama-server).

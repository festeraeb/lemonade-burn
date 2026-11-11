# Expert prediction (experiment #1) and the expert stand-in finder

Branch `fable/expert-predict-2026-11-08` (from `fable/device-decode-2026-11-07`). Background:
`~/agentnotes/fable-research-review-moe-prefill-2026-11-08.md` sections B-D (experiment #1).

## What is measured

**Goal 1, pre-gating hit rate (no output change).** Layer T's router (with T's FFN pre-norm)
applied to an *earlier* layer's residual stream names T's experts how often? Training-free idea
of Eliseev & Mazur (arXiv:2312.17238). Predictors, all scored against the set the engine chose:

| name | input to T's router | when it is available |
|---|---|---|
| `mid0` | `x_mid` of T itself (T's real router input) | sanity row, must be 1.000 |
| `mid1` | `x_mid` of T-1: residual after T-1's conv/attention | before T-1's experts run (a whole FFN + T's operator of lead time) |
| `out1` | `x_out` of T-1 = T's input | after T-1's FFN (only T's operator of lead time) |
| `mid2`, `out2` | the same, two layers back | a layer earlier |
| `mid1+2`, `mid1+4`, `mid2+4` | the `mid` guess widened to k+2 / k+4 experts | a staging ring that fetches more than k |
| `prev` | the previous token's set at T (NautiBurn's previous-token copy) | baseline |
| `freq` | T's k most-used experts in all other chunks (held out by chunk) | baseline |

Scores: recall@k (mean hits/k, NautiBurn's "4.21/8 = 53%" metric), Jaccard, all-k hit (every
actual expert named). Per token-layer the gate margin is kept (`sel[k-th] - sel[(k+1)-th]`,
selection score = sigmoid prob + bias) for the guess and the actual gate, in
`analysis/rows/<chunk>.jsonl`; `report.md` has the "keep only the most confident X%" curve.

**Goal 2, expert stand-in finder** (`lb-expert-standin`). For each MoE layer, the router inputs
routed to expert i (up to `--per-expert`, spread evenly over the run) are run through every
expert j of the layer (f32 dequant of the Q4_0 weights, `gemm`; checked against the engine
kernel, `kernel check` column ~1e-6). j is scored as a stand-in for i by cosine and relative L2
of the outputs; plus the router-weighted blend of the two best j ("almost expert"), the
router's own next pick when i is masked, and the effect on the whole MoE layer output when i
is dropped (renormalized), replaced by the router's next pick, or replaced by the best j.
Also routing concentration (share of routings in the top 8/16/24/32/48/64 experts per layer).

## Code

* `crates/lb-lfm2/src/model.rs`: `Lfm2::set_capture(true)` / `take_capture()` copy
  `x_mid`, `x_out` and the routes of every layer out of the host forward pass. It changes no
  arithmetic (`lb-expert-capture --check-sha` decodes with and without and compares tokens);
  the device-resident decode declines while capturing (CPU builds never have it).
* `crates/lb-expert-predict`: `metrics` (router maths identical to `route_sigmoid`, set
  scores), `capture` (chunk format), `standin` (scoring, gemm expert eval), binaries
  `lb-expert-capture`, `lb-expert-report`, `lb-expert-standin`. Prompts: `prompts/v1.jsonl`
  (32: 8 code, 8 prose, 8 Q&A, 8 forge-style task specs).
* Unit tests: set scores and accumulators on hand-made sets, router tie/margin vs the engine's
  `route_sigmoid`, chunk round trip + "done" rule, cosine/rel-err, gemm vs naive, SwiGLU expert
  vs naive, stand-in ranking of a scaled copy. `cargo test -p lb-expert-predict`.

## Run layout (resumable)

Run directory (on t440's /mnt/fast; c2 sees it as `/mnt/t440/fast/...`):
`/mnt/fast/lb-expert-predict/LFM2.5-24B-A2B-QAD-Q4_0/`

```
prompts-v1.jsonl        the prompts used
states/<id>.f32         per token, per layer: x_mid, x_out (f32), ~655 KB/token
chunks/<id>.json        tokens, prefill/decode split, engine routes + weights (written last)
progress.jsonl          one appended line per finished chunk
capture.log, capture.pid, run-capture.sh
analysis/report.{md,json}, analysis/rows/<id>.jsonl
standin/layer-LL.jsonl  every (i, j, cos, rel_err, n_samples) + per-expert rows
standin/layer-LL.summary.json, standin/standin-summary.md
analysis.log, analysis.pid, run-analysis.sh
```

A chunk (one prompt) is done when its meta exists and its states file has the size the meta
names; the runner skips done chunks. A stand-in layer is done when its summary exists.

### Resume commands

Binaries and the model live on c2 under `/mnt/ssd/lb-expert-predict/` (bin/, models/;
the GGUF is a copy of t440 `/mnt/raid0/models/lfm25/LFM2.5-24B-A2B-QAD-Q4_0.gguf`).

```sh
# capture (skips finished chunks)
ssh c2 'setsid nohup /mnt/t440/fast/lb-expert-predict/LFM2.5-24B-A2B-QAD-Q4_0/run-capture.sh >/dev/null 2>&1 </dev/null &'
# report + stand-in finder (stand-in skips finished layers)
ssh c2 'setsid nohup /mnt/t440/fast/lb-expert-predict/LFM2.5-24B-A2B-QAD-Q4_0/run-analysis.sh >/dev/null 2>&1 </dev/null &'
```

On another node: build with
`CARGO_TARGET_DIR=<own dir> cargo build --release -p lb-expert-predict` (CPU only, no GPU
needed), then run either script with `BIN=`/`BINDIR=`, `MODEL=`, `TOK=` and `RUN=` pointing at
that node's binaries, a copy of the same GGUF + `tokenizer.json`, and the run directory
(`/mnt/fast/...` on t440). `THREADS=` caps rayon threads (default 16).

Hardware: c2 CPU (2x E5-2450 v2, AVX only, 16 threads), `nice 5`; ~2.9 decode tok/s
with capture, ~1 min per prompt. No GPU is used (the free P100 was offered; the CPU path is
the one with the hooks and was fast enough). Hit rates and stand-in scores do not depend on
the hardware.

## Results, LFM2.5-24B-A2B Q4_0 (2026-11-08, c2 CPU)

Capture: 32 prompts, **7,045 tokens (5,120 decode, 1,925 prefill)**, 38 MoE layers, 64 experts
top-4: 267,710 token-layer pairs per predictor. Sanity: `mid0` reproduces the engine's set on
267,710/267,710. `--check-sha` passed (same greedy tokens with and without capture). Full
tables: `analysis/report.md` in the run directory.

### Goal 1: pre-gating hit rate, pooled over layers

| predictor | decode recall@k | decode Jaccard | decode all-k | prefill recall@k | prefill all-k |
|---|---|---|---|---|---|
| `out1` (T-1 output = T's input) | **0.881** | 0.821 | **0.602** | 0.859 | 0.553 |
| `mid1` (T-1 before its FFN) | 0.824 | 0.737 | 0.428 | 0.791 | 0.367 |
| `out2` | 0.787 | 0.687 | 0.345 | 0.752 | 0.296 |
| `mid2` | 0.754 | 0.648 | 0.293 | 0.710 | 0.242 |
| `mid1+2` (guess 6 for 4) | 0.918 | - | 0.728 | 0.888 | 0.650 |
| `mid1+4` (guess 8 for 4) | 0.949 | - | **0.828** | 0.926 | 0.764 |
| `mid2+4` (guess 8 for 4, 2 ahead) | 0.901 | - | 0.701 | 0.861 | 0.602 |
| `prev` copy, same model | 0.283 | 0.194 | 0.011 | 0.270 | 0.011 |
| `freq` held-out top-4, same model | 0.138 | 0.083 | 0.000 | 0.129 | 0.000 |
| NautiBurn previous-token copy (other models, top-8) | 0.56 | | | | |
| NautiBurn id-only ensemble (other models, top-8) | ~0.53 | | | | |

**Attention layers are the weak spot.** LFM2 has attention at layers 2, 6, ..., 38 and short
conv elsewhere. Decode, split by the *target* layer's type:

| predictor | conv targets recall / all-k | attention targets recall / all-k |
|---|---|---|
| `out1` | 0.925 / 0.718 | 0.758 / 0.277 |
| `mid1` | 0.853 / 0.488 | 0.743 / 0.257 |
| `mid1+4` | 0.968 / 0.882 | 0.897 / 0.677 |

(An attention block moves the residual stream far more than a short-conv block, so a guess
made before it is worse.) Per prompt kind the decode numbers barely move: `out1` recall
0.860 (forge) to 0.897 (Q&A), all-k 0.545 to 0.646.

**Confidence.** Keeping only the guesses with the largest gate margin: `out1` top 25% by
margin has recall 0.942, all-k 0.800; top 50%: 0.925 / 0.749. `mid1` top 10%: 0.945 / 0.805.

**Gate verdict.** The brief's gate (all-k hit >= 0.80 at L+1) **fails**: 0.602 (`out1`)
and 0.428 (`mid1`) on decode. The memo's gate on recall@k (>= 0.80 and +15 points over the
~53% ensemble) **passes on LFM2.5**: 0.881 / 0.824, +35 / +29 points over the cross-model
ensemble figure, and +60 / +54 over the same-model previous-token copy (0.283). The memo also
asks for a second model; lemonade-burn loads only LFM2 / LFM2-MoE (lb-gguf has no Q4_K/Q5_K
and there is no qwen35moe or gemma4 forward), so Qwen3.6-35B-A3B and gemma-4 were not run.
What it means for staging: a ring that fetches 8 guesses for 4 slots from `mid1` (a whole
FFN plus the next operator of lead time) holds all 4 experts on 83% of decode token-layers
and 95% of the experts needed; for a 1-in-8-misses budget that is enough, for "never miss"
it is not, and the attention-target layers need the wider guess or a fallback.

**Concentration** (share of routings in a layer's top-N experts, all 7,045 tokens): top-16
0.34-0.52 (median ~0.43), top-32 0.61-0.77 (median ~0.70), top-48 0.84-0.93. LFM2.5 has 64
experts, so "top-64" is all of them. Moderate skew, matching the 11-01 scrape measurement
(top-24 held out 64.8%).

### Goal 2: expert stand-ins (all 38 MoE layers, 64 samples per expert, 7,045 tokens)

Copies of the full outputs: `docs/expert-predict-2026-11-08/lfm25-hit-rate-report.md` and
`lfm25-standin-summary.md` (best stand-in per expert, per layer). Raw pairs:
`standin/layer-LL.jsonl` in the run directory (64 x 63 pairs per layer).

* **No expert in LFM2.5 has a stand-in.** 0 of 2,432 experts reach cos 0.90 with any other
  single expert, or with the router-weighted top-2 blend; 0 at 0.95. Best single cosine: median
  0.069, p99 0.52, max 0.66; top-2 blend median 0.084, max 0.71; 31 experts reach 0.5 and none
  reach 0.7. Experts' outputs on each other's tokens are close to orthogonal.
* Some layers have a weak "jack of all trades": layer 37 expert 56 is the nearest neighbour of
  26 of the 64 experts, layer 25 expert 4 of 25, layer 29 expert 37 of 21. They sit in the
  layers with the highest cosines (5, 9, 13, ..., 37, 38, 39: median best cos 0.17-0.45, the
  rest 0.03-0.09), but at cos ~0.3-0.5 they are a direction hint, not a substitute.
* **What dropping costs at the layer.** On the tokens routed to expert i, the MoE layer's
  output moves by a median relative L2 of **~0.44** (cos ~0.88) when i is dropped and the
  other three renormalized; letting the router's next pick fill in is about the same (beats
  plain dropping for 1,503 of 2,432 experts, by a hair); putting the best stand-in at i's
  weight is *worse* than dropping (beats it for 188). This does not depend on how hot i is
  (cold to hot quartile: 0.437, 0.425, 0.442, 0.450): a cold expert is not cheaper per token,
  it is only needed less often.
* This is the error of the MoE FFN output, not of the next-token distribution; the residual
  stream also carries the operator and earlier layers, so the effect on text must be measured
  (memo experiment #3: KL + task score on a frozen held-out set).
* Concentration: see goal 1 (median top-16 0.43, top-32 0.70, top-48 0.90 of routings).
  Moving the cold half (32 of 64) to slow storage would leave ~30% of routings to be served
  from it, every one of which has no substitute.

Reading for the plan ("drop or demote cold experts, let an almost-expert fill in"): on
LFM2.5 there is no almost-expert to fill in; the cheapest fallback for a missing expert is
renormalizing over the other three (or the router's next pick), at ~0.44 relative error of
the layer output on that token. Whether that is tolerable is a quality question for
experiment #3, and the answer should be expected to differ on gemma-4 (128 experts, 31% carry
80%) and the 256-expert qwen3moe models, which this engine cannot load yet.

### Not done, and why

* Qwen3.6-35B-A3B and gemma-4 MoE: lemonade-burn has no forward pass for qwen35moe/gemma4 and
  lb-gguf cannot read Q4_K/Q5_K (see `crates/lb-addmodel/src/lib.rs`). Running them needs the
  same capture hook in PolyStrata (`moe_rows`/decode), as the memo suggests.
* No GPU was used: the hooks are in the host path, and c2's CPU did the capture in ~35 min;
  the hit rates and stand-in scores do not depend on the hardware.
* Held out by date (memo): all prompts are from one session; held out by chunk for `freq`.

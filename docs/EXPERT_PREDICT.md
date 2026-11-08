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

Run directory (on t440's /mnt/fast; cesarops2 sees it as `/mnt/t440/fast/...`):
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

Binaries and the model live on cesarops2 under `/mnt/ssd/lb-expert-predict/` (bin/, models/;
the GGUF is a copy of t440 `/mnt/raid0/models/lfm25/LFM2.5-24B-A2B-QAD-Q4_0.gguf`).

```sh
# capture (skips finished chunks)
ssh cesarops2 'setsid nohup /mnt/t440/fast/lb-expert-predict/LFM2.5-24B-A2B-QAD-Q4_0/run-capture.sh >/dev/null 2>&1 </dev/null &'
# report + stand-in finder (stand-in skips finished layers)
ssh cesarops2 'setsid nohup /mnt/t440/fast/lb-expert-predict/LFM2.5-24B-A2B-QAD-Q4_0/run-analysis.sh >/dev/null 2>&1 </dev/null &'
```

On another node: build with
`CARGO_TARGET_DIR=<own dir> cargo build --release -p lb-expert-predict` (CPU only, no GPU
needed), then run either script with `BIN=`/`BINDIR=`, `MODEL=`, `TOK=` and `RUN=` pointing at
that node's binaries, a copy of the same GGUF + `tokenizer.json`, and the run directory
(`/mnt/fast/...` on t440). `THREADS=` caps rayon threads (default 16).

Hardware: cesarops2 CPU (2x E5-2450 v2, AVX only, 16 threads), `nice 5`; ~2.9 decode tok/s
with capture, ~1 min per prompt. No GPU is used (the free P100 was offered; the CPU path is
the one with the hooks and was fast enough). Hit rates and stand-in scores do not depend on
the hardware.

## Results

See the bottom of this file (filled in when the run finished).

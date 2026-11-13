# Device prefill and the prefill/decode split (2026-11-13)

Operator, 2026-11-13: get our CubeCL engine ready to test prefill on the RTX (tensor cores) and
decode on the P100s. Branch `opus/prefill-split-2026-11-13` (off `opus/x7-format-dispatch-2026-11-11`).

## Why

X6 (`WORK_TO_WEIGHTS.md`) measured the split with llama.cpp because our own prefill was not a real
path: lemonade-burn ran a prompt through `forward_host`, where attention, norms, RoPE, the conv,
routing and the expert sum ran on the host (Burn), attention built `[heads, m, total]` scores and a
host-side mask, and every quantized matmul round-tripped its activation through host memory.
Measured on a P100: 63-69 prompt tok/s, out of device memory at a 2k prompt; llama.cpp on the same
card 600-1085 tok/s, a T10 3.7-4.1x that.

## What is built

| Piece | Where | Checked |
|---|---|---|
| Batched glue kernels: rows RMS norm, q/k norm + RoPE + KV append, causal attention, short conv, router top-k, expert weighted sum, residual add | `lb-q4-gpu/src/prefill.rs` | `tests/prefill_kernels_match_decode.rs`: bit-identical to running the decode kernels token by token (outputs, KV rows, conv state; router selection exact incl. ties). P1000 (sm_61). |
| Device prefill pass (`LB_DEVICE_PREFILL=1`), chunks of `LB_PREFILL_UBATCH` (512) | `lb-lfm2/src/device_prefill.rs` | Builds. **Not yet run on a full model** (needs a free 16 GB card). |
| Tiled Q4_0 GEMM, f32 (64x64 tiles, block staged in shared memory), MoE slots grouped by expert on the host, fused gate/up/SiLU | `lb-q4-gpu/src/gemm.rs` | `tests/gemm_matches_rows.rs` vs the rows kernels to rounding. P1000, LFM2.5 shapes: dense 2.1x the rows kernels, MoE layer 1.3x at m=512 and 2.1x at m=2048. |
| Tensor-core GEMM (cmma f16 16x16x16, f32 accumulate), chosen per card (`Gpu::tensor_cores`; `LB_NO_TC=1` forces f32 tiles) | `gemm.rs` `q4_0_gemm_tc_kernel` | Compiles. **Never run**: no Turing card was free. |
| Prefill card: layers `0..K` of every chunk on a second card of the process, residual hops, KV/conv state copied back after the prompt (`LB_PREFILL_CARD`, `LB_PREFILL_LAYERS` or `LB_PREFILL_CARD_MB` 10500) | `device_prefill.rs` `PrefillCard` | Builds. **Never run.** |
| `lfm2-check --ab-prefill`: host vs device prompt pass per position (top-1, max abs logit diff), greedy continuation from each arm's state (device decode in both), prefill tok/s at `--prefill-lens` | `bin/lfm2-check.rs` | Builds. |

## Gates (written before any full-model run)

Model LFM2.5-24B-A2B QAD Q4_0, all experts resident as stacked layers on the decode P100 (budget
14500 MB + 900 MB dense), the five e2 prompts, greedy 32 tokens. Runner: `run_prefill.sh`.

* **G1, correctness on one P100** (f32 GEMM, no prefill card): teacher-forced top-1 host == device at
  >= 99 % of prompt positions; the greedy continuation after a device prefill is token-for-token the
  one after a host prefill, or the first difference is at a step whose top-2 logits are within 1e-2.
* **G2, speed on one P100**: device prefill >= 5x the host path at 512 tokens; an 8k prompt runs
  (no out-of-memory). Reported next to llama.cpp's X6 numbers on the same card class (P100 ub512:
  615 tok/s at 512, 607 at 8k; best ubatch 1035-1085).
* **G3, tensor cores on the Turing card**: `tests/gemm_matches_rows.rs` passes on it (f16 bound);
  `prefill_gemm_bench` reports the tensor-core GEMM's GFLOP/s and the ratio over the f32 tiles.
* **G4, the split** (P100 decodes, Turing prefill card holds leading layers): top-1 host == device at
  >= 97 % of positions (the Turing layers compute in f16), coherent continuation (match length
  reported); and at 8k, prefill time <= 1/1.2 of G2's one-P100 device prefill.

Expectations (not results): G2 lands at a few hundred to ~1000 tok/s on a P100 with f32 tiles; the
split is worth it only if the tensor-core GEMM is several times the f32 tiles (llama.cpp's T10 is
~4x its P100).

## Known limits, not built

* Attention is one cube per (head, query row) with no K/V reuse across queries: fine at these
  lengths, a flash-tiled kernel is the next step if it shows in the profile at 8k+.
* The prefill card runs its layers, then the decode card runs the rest: no overlap between the two
  cards yet (chunk c+1 on the prefill card while chunk c finishes on the decode card).
* A 12 GB card holds about 75 % of LFM2.5's layers; the rest prefill on the P100 in f32. Streaming
  the remaining layers' weights to the prefill card is not built.
* In-box expert cards and remote tiers decline the device prefill (host path runs).

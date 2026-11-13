**G3, tensor cores: correct, and 3.2-3.9x the f32 tiles.** `gemm_matches_rows` passes on the 2060 to the f16
bound (worst 3e-4 of scale). **CORRECTION:** the first speed rows here (2060 2131 vs 2129 GFLOP/s, T10 2714
vs 2684, "no faster") were a **stale example binary**: after adding the tensor-core kernel only the library
was rebuilt (`cargo build -p lb-q4-gpu` does not rebuild `examples/`), so both arms ran the f32 tiles; the
Nsight profile showed only `q4_0_gemm_kernel` running in the "tensor-core" arm. Rebuilt, T10, m=2048:
**dense 8598 GFLOP/s (f32 tiles 2684), MoE layer 10313 (2639)**. P100 f32 tiles: 4092 dense / 3735 MoE.
The G4 run below did use the tensor cores (its log: "tensor cores yes").

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

## Results (2026-11-13, c4 after the card swap, then c2's free P100)

Raw: `/mnt/fast/prefill/results/prefill-*.json`, logs `/mnt/fast/prefill/logs/`.

**G1, correctness on one P100 (c4 AF): PASS.** Teacher-forced top-1 host == device 215/215, max |logit diff|
6.4e-5; greedy continuation after a device prefill token-for-token the host-prefill one (sha 062392b8d2b0361a).

**G2, speed on one P100: PASS** (7.0x the host path at 512; 8k runs). Per-query attention, ub 512:

| prompt | device | host | llama.cpp P100 (X6, ub512 / best) |
|---|---|---|---|
| 512 | 497 tok/s | 71 | 615 / 615 |
| 2048 | 409 | OOM before | 626 / 1035 |
| 8192 | 196 | OOM before | 607 / 1085 |

**G3, tensor cores (RTX 2060, then the T10): correct, no faster.** `gemm_matches_rows` passes on the 2060 to
the f16 bound (worst 3e-4 of scale). Speed equals the f32 tiles on both cards (2060 2131 vs 2129 GFLOP/s;
T10 2714 vs 2684 dense, 2639 vs 2639 MoE): the kernel is bound by its staging (one Q4_0 block of K per two
barriers, f32 activations converted per tile), not by the multiply. P100 f32 tiles: 4092 dense / 3735 MoE.

**G4, the split (P100 decodes, 2060 holds layers 0..30, f16 tensor cores): correctness PASS, speed FAIL.**
Top-1 214/215, continuation coherent (same for 24 of 32 tokens). 512: 707 tok/s (1.42x one P100); 2048:
437 (1.07x); 8192: 160 (0.82x). The hop costs 1 ms a chunk, the state copies 50 ms a prompt; the loss at 8k
was attention on the slower-memory card (2060: ~60 % of a P100's bandwidth, 8 of the 10 attention layers).

**Tiled attention (bb0f1cf).** One cube per (KV head, 32 query rows), K/V staged 32 positions at a time:
17-20x the per-query kernel (P100, one layer over the prompt: 8192 tokens 2094 -> 103 ms). Full model on
c2's P100 (Gen3 x4 slot, weights resident; ub 512): **512: 552, 2048: 594, 8192: 588 tok/s**, flat with
length; 215/215 top-1, continuation identical. Within a few % of llama.cpp's default ubatch.
With `LB_PREFILL_UBATCH=2048` (A2, same card): **512: 557, 2048: 780, 8192: 770 tok/s** (215/215, identical
continuation); llama.cpp on a P100 at ub 2048: 1025 / 990: about 77 %.

**G4 re-run, the operator's layout (B1, c2): all 40 layers of the prompt on the T10 (tensor cores), decode on
the P100, ub 2048: PASS.** 512: **1352**, 2048: **1736**, 8192: **1598 tok/s** (5.13 s), against one P100 557 /
780 / 770 (A2): 2.1-2.4x, and 1.5-1.7x llama.cpp's best P100 prefill. llama.cpp on a T10 (X6) is still
~2.4x ours (3998 / 4222). Top-1 214/215, max |logit diff| 4.1 (f16 tensor cores); continuation identical
for 24 tokens, then a near-tie goes the other way ("bottom" vs "seabed mapping"). Hops 1 ms in all, state
copies 0.11 s. The T10 was borrowed from gemma4-c2 for 20 min (held in the manifest, restored).

**fp16 pipe on the P100 (half2 GEMM, the Kmic-68 fork's idea, own kernel): measured, NOT adopted.** Folding
a 32-column fp16 partial to f32: kernel 4860 dense / 4592 MoE GFLOP/s (f32 tiles 4096 / 3734), full model
930 / 904 tok/s at 2k / 8k (f32 780 / 770), but top-1 211/215, max |logit diff| 9.5, continuation differs
at step 4: **fails G1** (>= 99 %). Fold 16: 4658 / 4099; fold 8: 4369 / 3805 with half the error, i.e. the
f32 tiles' speed. The fork gets its accuracy from integer (q8) activations, whose products are exact in
fp16; ours are f16-rounded floats. Kept behind `LB_H2=1` (`LB_H2_FOLD`); the P100 default stays f32.
Tensor-core stage width (T10): one block per stage 8584 / 10168, two blocks 7338 / 8775 (occupancy).

## Next, in order
1. Larger chunks (`LB_PREFILL_UBATCH=2048`): fills the expert tiles (grouped GEMM 3.0x the rows kernels at
   m=2048 on a P100, 2.2x at 512). Measured as A2.
2. Tensor-core GEMM: already 3.2-3.9x the f32 tiles (see the G3 correction); next gains from a K tile of
   several blocks per barrier pair and activations converted to f16 once per chunk.
3. Re-run G4 with 1 and 2; overlap the two cards (chunk c+1 on the prefill card while c finishes on the
   decode card).

## Ideas from the operator (video, 2026-11-13: Qwen 27B at 67 tok/s on 2x P100), not started
* **Tensor parallelism across a P2P pair** (c4 AF+D8 measured 10.2 GB/s P2P): split each matrix's rows over
  both cards so decode reads both cards' HBM at once; the cost is one all-reduce per layer. The way past one
  card's bandwidth for dense models. 67 tok/s on a dense 27B Q4 is ~1 TB/s effective, ~70 % of the pair's peak.
* **MTP** (multi-token prediction heads, verified next pass): ~1.5-2x decode where the model ships MTP heads. The video ran with thinking low and temperature 0: greedy verification is MTP's best case (a draft is kept whenever it is the top token), and short answers keep the measured tok/s close to pure decode
  (LFM2.5 does not).
* **6-bit weights** (Q6_K): near-Q8 quality at 6.56 bpw; the Q6_K kernel exists (the LFM2 head); costs ~45 %
  more bytes per token than Q4_0 in bandwidth-bound decode.

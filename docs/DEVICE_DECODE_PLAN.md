# Device-resident decode: the plan to match llama.cpp on the same card (Fable, 2026-11-07)

**Bar.** LFM2.5-24B-A2B QAD-Q4_0 on one Tesla P100 (16 GB), everything resident: llama.cpp 66.2-66.7 tok/s,
lemonade-burn 18-19 (WORK_LOG, 2026-11-04). Per token the card reads ~1.56 GB of weights (dense 423 MB, 38 MoE
layers x 4 experts x 5.3 MB = 806 MB, Q6_K head 328 MB). llama.cpp's 15 ms a token is therefore ~104 GB/s
effective, 14% of HBM2; ours is 48-54 ms. Two things cost that, and both have to go:

1. **The token leaves the card ~145 times.** The forward pass is Burn tensors for the glue and `Vec<f32>` round
   trips around every quantized matvec: upload x, launch, read back (a sync), rebuild a Burn tensor, Burn ops,
   repeat. The MoE layer routes on the host from a read-back activation and sums expert outputs on the host.
   WORK_LOG measured the sync alone at ~0.24 ms each (1.46x if all were gone) but that figure excludes the
   uploads, the Burn hops and the host-side router/sum that go with them. Keeping the whole token on the device
   removes the lot, not just the sync.
2. **The matvec kernel runs at ~7% of the P100's bandwidth** (52.8 GB/s on distinct experts; WORK_LOG). One
   plane per output row, two 16-byte loads a lane, then a plane reduce: too few bytes in flight per SM. At that
   rate 1.56 GB is 30 ms on its own, so a device-resident loop alone lands near 26 tok/s. Parity needs ~15%+.

Neither lever reaches the bar alone. Measured facts to keep: cube size does not matter (32-256 all equal);
gate/up/silu fusion was 1.13x; sync elimination alone 1.46x at best; t440 cannot measure under 1.5x; the P100
has no dp4a, so the dot stays fp32 FMA (or fp16x2) after a nibble unpack.

## Phase D1: one token, one sync (`LB_DEVICE_DECODE=1`, default off)

New module `crates/lb-lfm2/src/device_decode.rs`, taken by `Lfm2::forward` when `m == 1`, the knob is on, and
every expert and dense matrix the token needs is resident (else the existing path runs unchanged; the knob is
also a run-time tunable in lb-server like `batch_miss`). Only `lb_q4_gpu` handles inside the token; no Burn
tensor until the logits.

Per token, in launch order (counts for the 40-layer LFM2.5: 30 conv layers, 10 attention, 2 dense FFN, 38 MoE):
- host: dequantize the token's Q6_K embedding row (lb_gguf), upload `x` (8 KB). One upload a token.
- per layer:
  - `add_rms_norm_kernel(x, y, w, eps) -> (x', h)`: `x' = x + y`, `h = rms(x') * w` (one launch; the first
    layer's `y` is zeros or a flag). Norm weights uploaded once at load, per layer.
  - conv layer: `GpuConv::launch(h)` (exists: in_proj, conv_decode, out_proj; state stays on the device).
  - attention layer: `q`, `k`, `v` matvecs (exist); `qk_norm_rope_append_kernel`: per-head rms over hd=64 with
    `q_norm`/`k_norm`, RoPE at position `pos` with cos/sin computed in-kernel from `rope_theta` (no host tables),
    and `k`,`v` written into the device KV cache `[kvh, cap, hd]` at `pos` (one launch); `attn_decode_kernel`:
    one cube per head, lanes stride over positions `0..total` with an online softmax (running max, sum, acc[hd]
    per lane, then a shared-memory combine), GQA by `head / group` (one launch); `o` matvec. Cache capacity from
    `Cache::cap`, grown on the host before the token as `ensure_cap` does today.
  - `add_rms_norm_kernel` with `ffn_norm`.
  - dense FFN: `q4_0_gate_up_silu_kernel` + down (exist).
  - MoE: `router_topk_kernel`: 64 sigmoid logits of `h` against the f32 router (uploaded once a layer), selection
    by `prob + bias` descending with lower expert id first on ties, weights = the top-k probs normalised by
    `max(sum, 6.103515625e-5)`, exactly `route_sigmoid` in moe.rs; writes `routes[2k]` as `[e, 0]` pairs and
    `weights[k]` into one per-token device buffer at the layer's slot (one launch). `GpuExpertLayer::launch`
    gains a variant taking the routes as a device handle plus offset (two launches, exist). `weighted_sum_kernel`:
    `y = sum_k weights[k] * out[k]` in slot order (one launch; deterministic, no atomics).
- final `add_rms_norm` with `token_embd_norm`; `GpuQ6KMat::launch` for the logits (exists).
- **one `read_all`**: the logits (512 KB) and the per-token routes/weights buffer; the scraper records the
  routes on the host afterwards, as it does today, same file format.
About 430 launches a token and one sync. Scratch comes from `Gpu::scratch`, recycled at that read.

Correctness gates (all must pass before any speed claim):
- Kernel unit tests against host references (rms norm, rope, attention over 1-257 positions with GQA, router
  top-k with ties and the bias rule, weighted sum), small shapes, on wgpu (c1's GTX 1070 is the test card: no
  cargo there, build on t440 and copy the binary; `LB_WGPU_DEVICE`).
- `lfm2-check` teacher-forced top-1: the device path equals the current GPU path at every position of the three
  prompts (86 positions), and the greedy 32-token text of `c4_e2e.sh` matches the current path's token for
  token (sha may differ by summation order; if it does, show top-1 agreement per step and the max |logit| diff).
- The scrape file written for a run is identical in content to the old path's for the same prompt.

## Phase D2: the matvec kernel's bytes in flight

Same kernels, same repacked layout, two changes, each A/B'd with `distinct24_m1` medians (3 runs) on the
test card against its own roofline (1070: 256 GB/s; P100: 732):
- each plane computes R rows at once (R = 4 or 8, a `#[comptime]` knob mirrored by `LB_Q4_ROWS`), issuing all
  loads before any FMA, so R x 2 x 16-byte loads are in flight a lane;
- the plane reduce once per R rows (R partials reduced together), not once a row.
Then `x` in shared memory once per cube (it is 8 KB and every row reads it). Report GB/s per kernel and the
e2e tok/s; keep the old path behind the knob until the new one wins on both cards.

## Phase D3: measure like for like

`c4_e2e.sh` / `c4_llama.sh` on a P100 (c4 GPU 3 once Le Chonk's A/B ends; ask on Buzz #human-questions
before taking a card), 2 interleaved rounds x 5 completions, load printed, 3-run medians, both engines on the
same GGUF and prompt. The number to quote is the quieter round's median. Anything under 1.5x on t440 is noise.

## Not in this plan
Prefill GEMM (0.86 s for 20 tokens today), misses (the on-demand path stays as it is), multi-card, and the
Movidius helper (0.5% of a P100's fp16 behind a USB link 1400x slower than HBM: see
RESEARCH_ENGINE_GAPS_2026-11-07.md).

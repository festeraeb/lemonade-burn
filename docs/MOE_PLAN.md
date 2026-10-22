# MoE phase plan: Mixtral 8x7B with expert look-ahead on lemonade-burn

Status: **plan only.** No Mixtral code exists in this repo yet. Figures are labelled:
**[measured]** = run in this project (with the source); **[published]** = from a cited
external page; **[estimate]** = arithmetic or hypothesis, not measured.

## 1. Target and fleet precedent

Mixtral 8x7B: 32 layers, 8 experts per layer, top-2 routing, hidden 4096, expert FFN
14336 (SwiGLU: w1 gate, w3 up, w2 down), about 46.7B total and 12.9B active parameters
per token (model config / model card).

The fleet has served it with llama.cpp (read-only look at cesarops4om
`~/nauti-seats/start_mixtral_cpumoe.sh` and `start_mixtral_rpc.sh`): Q4_K_M GGUF split
across the Tesla T10 (16 GB) and RTX 2060 (12 GB), `--n-cpu-moe 5` (experts of 5 layers on
CPU), `-fa on`, q8_0 KV, 16k context, NautiBurn MoE scrape enabled
(`NAUTIBURN_SCRAPE_DIR=/mnt/models/nautiburn-scrape`); the RPC variant adds a t440 P100 over
ggml-rpc. Those scripts record no throughput, so this plan cites none.

## 2. Reality check: "ternarized Mixtral under 10 GB"

* 46.7B weights x 1.58 bit = about 9.2 GB **[estimate, arithmetic only]**.
* Mixtral was not trained ternary. BitNet b1.58 quality comes from training with the
  quantizer in the loop (see `crates/lb-ternary/src/pack.rs`: the model applies absmean
  ternarization in every forward pass during training). A search on 2026-10-22 found **no
  ternary Mixtral checkpoint with published quality numbers**.
* Published low-bit Mixtral points that do exist:
  * AQLM 2-bit (1x16): WikiText2 perplexity 3.35, 12.6 GB **[published]**
    (huggingface.co/ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf; the card gives no fp16
    baseline).
  * HQQ with 4-bit attention and 2-bit experts: about 18 GB, wikitext perplexity about 4.98
    on a non-standard split **[published, not comparable]** (github.com/mobiusml/hqq issue 2).
* Decision: plan Mixtral around existing 4-bit and 2-bit group-quantized formats. The
  ternary kernel applies directly only to models trained ternary (BitNet b1.58, and
  possibly Ternary-Bonsai, see section 7). Ternarizing Mixtral would need QAT or
  distillation, which is out of scope until a quality number exists.

## 3. Where the low-bit linear kernel applies

Each expert is three matrices: `w1`/`w3` [14336 x 4096] and `w2` [4096 x 14336], about
176M parameters. Bytes per expert, weights only **[estimate]**:

| format | bits/weight | per expert | all 256 experts |
|---|---|---|---|
| bf16 | 16 | 352 MB | 90 GB |
| Q4_K-like | ~4.5 | ~99 MB | ~25 GB |
| 2-bit group | ~2.1-2.6 | ~46-57 MB | ~12-15 GB |

A decode token reads 64 expert FFNs (32 layers x top-2), about 6.3 GB at Q4
**[estimate]**. On c3, the measured DRAM read ceiling is about 11 GB/s ([measured]:
`kbench` with `KB_STREAM=1` on cesarops3, 11.27 GB/s read sweep, 10.8 GB/s ternary GEMV).
That bounds CPU-only Q4 Mixtral on c3 at roughly 0.6 s per token **[estimate]**, and in
practice c3 cannot hold Q4 Mixtral at all (15 GB RAM).

Kernel reuse: `lb-ternary` decodes 2-bit codes with shift and mask and multiplies against
int8 activations with `maddubs`/`dpbusd`. An expert kernel keeps that structure:

* Decode int4 or int2 group codes (group 32 or 64, f16 scale and min per group).
* Use the same int8 per-token activation quantization (`quant.rs`).
* Use the same row-chunk rayon split.

Golden tests carry over unchanged in style: exact integer agreement before scaling.
Mixtral's FFN is SwiGLU (`silu(w1 x) * w3 x`), unlike BitNet's ReLU^2.

## 4. Expert look-ahead (prediction)

Prior in-house result (reported by the operator; I could not locate the source file in
`cesarops-inference` or `forge_lemon-cake`):

* Pre-gate look-ahead reached recall@16 = 0.784.
* Lookup tables (static co-occurrence) did not work.

It came from a different MoE (many more experts per layer), so it does **not** transfer
to Mixtral's 8-experts/top-2 routing and must be re-measured.

Predictor for Mixtral:

* At layer `l`, after attention, apply the router weights of layers `l+1 .. l+d` to the
  current residual stream (pre-gate look-ahead) and take the top-`B` experts per target
  layer.
* Cost: one [8 x 4096] GEMV per look-ahead layer, negligible.
* Baseline for honesty: with 8 experts and top-2, a random guess of `B` experts gives
  recall `B/8` (recall@4 = 0.5). A predictor is only useful if it clearly beats that at
  `B = 2..4` and `d = 1..2`.

Data source: NautiBurn already scrapes live llama.cpp Mixtral routing. Evaluate offline on
those traces before writing runtime code: recall@B per layer and distance, plus
prefetched-but-unused bytes.

## 5. Residency and prefetch (disk, RAM, GPU)

* **Tiers:**
  * GPU VRAM holds attention, router and norms, plus a hot set of experts.
  * RAM holds all experts in compressed form.
  * Disk is an mmap of the converted file, for hosts like c3 that cannot hold the model
    in RAM.
* **Per layer:** while layer `l` computes, issue async copies for the predicted experts of
  `l+1` (and `l+2` if bandwidth allows). On a miss, either wait for a synchronous copy, or
  compute that expert on the CPU where it lives. The CPU route is what `--n-cpu-moe` does
  statically; here it would be per token.
* **Eviction:** LRU, with pins for currently predicted experts.
* **Telemetry:** report hit rate, bytes moved and stall time through the bridge's
  existing capability fields (`moe_router_visibility`, `expert_residency_visibility`,
  `expert_transfer_visibility`). These fields are already in the bridge schema, currently
  `unknown`.
* **Integration in this repo:**
  * A Burn backend-extension op `moe_expert_ffn(x, expert_ids, gate_weights)`, alongside
    `ternary_linear` in `crates/lb-model/src/ops.rs`.
  * A new crate `lb-experts` for the residency manager (tier table, async copy queue,
    predictor hook).
  * The router and top-k selection stay ordinary Burn ops.

## 6. Phases and gates

| phase | work | gate (must be measured) |
|---|---|---|
| M0 | offline predictor eval on NautiBurn Mixtral traces | recall@B for B=2..4, d=1..2 vs the B/8 baseline |
| M1 | CPU Mixtral in lemonade-burn with a Q4 group kernel (t440/c4 RAM) | top-1 agreement and max logit diff vs llama.cpp on fixed prompts (same method as `lb-run compare`) |
| M2 | residency manager + prefetch to a GPU via Burn wgpu (c3 RX 6700 XT 12 GB) | hit rate, stall time, tok/s vs no-prefetch and vs llama.cpp `--n-cpu-moe` |
| M3 | 2-bit experts | perplexity vs Q4 on a fixed text set; adopt only if the loss is within an agreed bound |

## 7. Second ternary target: Ternary-Bonsai-2 (Gated-DeltaNet hybrid)

The fleet's `bonsai` seat ran `Ternary-Bonsai-2-27B-PQ2_0.gguf` through a llama.cpp build.
The kindi project (AGPL, read for ideas only, no code copied) runs Ternary-Bonsai-2-2B and
describes it as mixing attention layers with recurrent Gated-DeltaNet layers. Support
would need:

1. **A PQ2_0 / PTQ1_0 GGUF reader.** The layout is not documented here; it must be read
   from the llama.cpp fork that serves it.
2. **The gated delta rule step** per head: `S = g*S; delta = beta*(v - k^T S);
   S += k delta^T; o = S^T q`. This is forge lane LB5 (`crates/lb-model/src/deltanet.rs`).
   It also needs a short causal conv1d and L2-normalised q/k, with exact conventions to
   verify against the reference implementation.
3. **A per-layer state cache** (S matrices) for the recurrent layers, alongside the KV
   cache for the attention layers. Prompt-cache checkpoints (lane LB2) then also snapshot
   S.
4. **A correctness gate:** the same `compare` method against the llama.cpp fork on fixed
   prompts.

Bonsai is described as post-training ternary, so its quality claims also need checking.

## 8. Output head

For a 2B model the LM head dominates decode bytes ([measured] on c3: about 60 ms/token
for the 656 MB bf16 head vs about 65 ms for all ternary layers). Two opt-in options, with
the exact bf16 head staying the default:

* **int8 per-row head** (lane LB1): about 328 MB **[estimate]**.
* **Clustered or "fast" head**: kindi reports 99.4% same-token agreement over 310 steps
  **[published by kindi, not reproduced]**.

Either must report top-1 agreement on the fixed prompts of `scripts/hf_reference.py`
before use.

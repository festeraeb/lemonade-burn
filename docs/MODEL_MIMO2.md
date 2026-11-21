# MiMo-V2.6-Flash (`general.architecture = mimo2`): facts for lemonade-burn

File: `/mnt/raid0/models/mimo-v26-flash/MiMo-V2.6-Flash-RL-Q2_K-0000{1,2}-of-00002.gguf` (ggml-org, 126 GB on disk;
shard 1 holds only metadata, shard 2 all 472 tensors). Extras in the same folder: `mtp-MiMo-V2.6-Flash-RL-Q8_0.gguf`
(3 MTP / nextn blocks, 2.4 GB) and `mmproj-...-Q8_0.gguf` (vision/audio encoder, 1.6 GB).

Raw dumps (made with `cargo run -p lb-gguf --example gguf_facts -- <file> --tensors`):
`docs/models-facts/mimo2-q2k.txt`, `mimo2-mtp-q8.txt`, `mimo2-chat-template.jinja`.

Reference implementation read: llama.cpp `src/models/mimo2.cpp`, in the tree the fleet build was made from
(`/mnt/fast/scrape-v2/llama`, upstream 276378515 + NautiBurn hook, commit 787085f36; build
`/mnt/raid0/engines/llama.cpp/20261112-nautiburn-scrape-v7-*`, which is the `mimo26-flash-c4` manifest entry). The
newer tree `/mnt/fast/scrape-v2/llama-v8` (dadb89f180) differs in `mimo2.cpp` only by refactors (`build_rope(...)` in
place of explicit `ggml_rope_ext` calls with the per-layer base; `diff` checked 2026-11-21).

## Hyperparameters (all from metadata)

| Key | Value | Use |
|---|---|---|
| block_count | 48 | trunk layers (MTP layers 48-50 live in the separate mtp file, `nextn_predict_layers = 3`) |
| embedding_length | 4096 | hidden |
| feed_forward_length | 16384 | dense FFN of layer 0 |
| expert_feed_forward_length | 2048 | per-expert FFN |
| attention.head_count | 64 | query heads (all layers) |
| attention.head_count_kv | per layer: 4 on full layers, 8 on sliding layers | GQA groups 16 / 8 |
| attention.key_length / value_length | 192 / 128 | Q and K heads are 192 wide, V heads 128 |
| rope.dimension_count | 64 | partial rope: first 64 of the 192 dims of every Q/K head |
| rope.freq_base / rope.freq_base_swa | 1e7 / 1e4 | full layers / sliding layers (`llama_model::get_rope_freq_base`, llama-model.cpp:2776) |
| (rope type) | NEOX | `llama_model_rope_type` lists LLM_ARCH_MIMO2 under NEOX (llama-model.cpp:3652); pairs (i, i+32) |
| (rope scaling) | none | no yarn keys; freq scale 1 |
| attention.sliding_window | 128 | `LLAMA_SWA_TYPE_STANDARD`: key masked when `pos_q - pos_k >= 128` (llama-hparams.h:519) |
| attention.sliding_window_pattern | 1 = sliding, 0 = full; full at layers 0, 5, 11, 17, 23, 29, 35, 41, 47 | 9 full, 39 sliding |
| attention sinks | `blk.N.attn_sinks.weight` [64] on the 39 sliding layers only | extra softmax-denominator logit per head |
| attention.value_scale | 0.707 | attention output (after `attn_output`) times 0.707 (`f_attn_value_scale`) |
| attention.layer_norm_rms_epsilon | 1e-6 | all RMS norms |
| expert_count / expert_used_count | 256 / 8 | |
| expert_group_count / group_used | 1 / 1 | no group-limited routing |
| expert_gating_func | 2 (sigmoid) | mimo2.cpp passes `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID` explicitly |
| exp_probs_b.bias | per MoE layer [256] | added to the sigmoid probs for selection only |
| expert weights | normalised (`norm_w = true`), no scale (expert_weights_scale absent = 0) | |
| shared experts | none | |
| dense layers | layer 0 (has `ffn_gate/up/down`, no `ffn_gate_inp`) | others MoE |
| context_length | 1048576 | |
| vocab | 152576 (`token_embd` and `output` Q8_0, untied) | |
| tokenizer | `gpt2` BPE, pre `qwen2`, NFC normaliser flag, no BOS added; eos 151645 (`<|im_end|>`), pad 151643 | |
| chat template | ChatML (`<|im_start|>role ... <|im_end|>`), `<think>...</think>` reasoning, tool and multimodal markers (`<|vision_start|>`, `<|image_pad|>`, `<|mimo_audio_start|>` ...) | full text in models-facts |

## Forward pass (llama.cpp `llama_model_mimo2::graph::graph`, mimo2.cpp)

Per layer `il`, hidden `h` [n_tokens, 4096]:

1. `x = rms_norm(h) * attn_norm` (`build_norm`, LLM_NORM_RMS; ggml `rms_norm` sums squares in double, ops.cpp:4103).
2. `qkv = attn_qkv @ x` (one fused Q8_0 matrix, rows `64*192 + nkv*192 + nkv*128` = 13568 on full layers, 14848 on
   sliding layers). Views: Q = first 64x192, K = next nkv x192, V = last nkv x128 (mimo2.cpp, the `wqkv` branch).
3. NEOX rope on Q and K (`n_rot = 64`, base per layer type, `ext_factor = 0`, `attn_factor = 1`).
4. `build_attn(..., wo, sinks, kq_scale = 1/sqrt(192))`: causal; sliding layers also mask `pos_q - pos_k >= 128`;
   softmax with sink (ops.cpp `ggml_compute_forward_soft_max_f32`: `max = max(max_j s_j, sink)`,
   `sum = sum_j exp(s_j - max) + exp(sink - max)`); `out = sum_j p_j v_j`; heads concatenated (64 x 128 = 8192);
   `attn_output` (Q8_0 [8192 -> 4096]).
5. `attn *= 0.707` (`attn_out_scaled`), `h = h + attn` (`ffn_inp`).
6. `x = rms_norm(h) * ffn_norm`.
7. Layer 0: `down(silu(gate x) * up x)` (`build_ffn`, LLM_FFN_SILU, LLM_FFN_PAR).
   Layers 1-47 (`build_moe_ffn`, llama-graph.cpp:2200): `logits = ffn_gate_inp @ x` (F32 [4096 -> 256]);
   `p = sigmoid(logits)`; `sel = top8(p + exp_probs_b)`; `w = p[sel] / max(sum p[sel], 6.103515625e-5)`;
   `out = sum_k w_k * down_k(silu(gate_k x) * up_k x)`.
8. `h = h + out` (`l_out`).

Head: `rms_norm(h) * output_norm`, `output` (Q8_0 [4096 -> 152576]). The last layer crops to the output token(s) after
attention (`crop_before_nextn`), which only changes which rows are computed.

MTP (separate file, `graph_mtp`): `eh_proj(concat(rms(embed)*enorm, rms(h)*hnorm))`, one decoder block (sliding
attention with sinks, kv 8, dense FFN 16384), shared head norm (`layer_output_norm` fallback) and `output`. Not ported.

## Quant types in the Q2_K file

| Type | Tensors | Bytes | GiB | bits/weight | Where |
|---|---|---|---|---|---|
| Q2_K | 94 | 66,236,448,768 | 61.69 | 2.625 | `ffn_gate_exps`, `ffn_up_exps` (47 MoE layers) |
| MXFP4 | 47 | 53,619,982,336 | 49.94 | 4.25 | `ffn_down_exps` |
| Q8_0 | 101 | 6,304,759,808 | 5.87 | 8.5 | `attn_qkv`, `attn_output`, dense FFN of layer 0, `token_embd`, `output` |
| F32 | 230 | 198,779,648 | 0.19 | 32 | norms, sinks, router `ffn_gate_inp`, `exp_probs_b` |
| total | 472 | 126,359,970,560 | 117.68 | | |

MTP file: Q8_0 20 tensors (2,377,515,008 bytes), F32 19.

## What lemonade-burn had no code for (before this branch) and what this branch adds

| Piece | Before | Now (branch `fable/engine-mimo-deepseek-2026-11-21`) |
|---|---|---|
| Split GGUF (shards) | no | `lb_gguf::GgufSet` |
| MXFP4 dequant / dot | no (type id 39 unknown) | `GgmlType::MXFP4`, `dequant_mxfp4_block`, `dot_mxfp4`; bit-exact vs ggml |
| Q2_K dequant (CPU) | yes | re-checked bit-exact on this file |
| Per-layer KV head count, per-layer rope base, sliding window, sinks | no (qwen35 has one kv count, no SWA, no sinks) | `lb-mimo2` CPU reference |
| Fused QKV with K/V of different head widths (192/192/128) | no | `lb-mimo2` |
| Partial NEOX rope (64 of 192) | qwen35 had partial (M-RoPE sections) | `lb-mimo2` |
| Sigmoid router with selection bias, no shared expert | lfm2 has sigmoid+bias; qwen35 softmax | `lb-mimo2` |
| `value_scale` after `attn_output` | no | `lb-mimo2` |
| qwen2 pre-tokenizer | qwen35 regex only | `lb-mimo2::tokenizer` |
| MTP (3 nextn blocks) | qwen35 MTP only | not ported |
| Vision/audio mmproj | no | not ported (plan step S5) |
| Device kernels: Q2_K and MXFP4 matvec, SWA+sinks attention, 192/128 head widths | Q2_K device kernel exists in lb-q4-gpu `kquant.rs`? see FLEET_MODELS_NEXT.md | not done (S1 device side) |
| NFC normalisation | no | not done (ASCII/NFC text unaffected) |

## Parity (CPU reference vs llama.cpp)

See the "Parity" section in `docs/WORK_LOG.md` (2026-11-21 entry) and `docs/models-facts/` for the commands and numbers.

# DeepSeek V4.1 Flash Coder (`general.architecture = deepseek41`): facts for lemonade-burn

File: `/mnt/raid0/models/deepseek41-coder/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-0000{1..6}-of-00006.gguf`
(ISTA DASLab GSQ-RCO quant, MIT). It is ONE model in six shards: shard 1 = metadata only, shards 2-5 = 1042 tensors
(72 GB), shard 6 is a symlink to `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` = the single Engram table
`per_layer_token_embd.weight` (IQ4_NL [256, 768022850], 110.6 GB). So "the smaller of the two files" is not a separate
model: the IQ2_XXS shards need the Engram shard (layers 1 and 14 read ~24 rows of it per token).

Raw dumps: `docs/models-facts/deepseek41-iq2xxs.txt` (metadata, every tensor), `deepseek41-chat-template.jinja`.

Reference implementation: llama.cpp `src/models/deepseek41.cpp` (933 lines) on top of `src/models/deepseek4.cpp`, in
`/mnt/fast/scrape-v2/llama-v8` (upstream master + NautiBurn hook, commit dadb89f180). The fleet's serving builds do not
run this file (the c4 seat ran it on Nauti-Strata's own port, `/codebase/repos/Nauti-Strata-wt/deepseek41`); the fleet
llama.cpp tree `/mnt/fast/scrape-v2/llama` (787085f36) also has `deepseek41.cpp`. The detailed pass below was written
from the v8 tree by a read-only sub-agent and spot-checked by hand (lagged hyper-connection pre-mix at
deepseek41.cpp:849-869, derope at :751-755, grouped `wo_a` at :757-763).

## Summary table

| Item | Value |
|---|---|
| layers / hidden | 40 / 5120 |
| attention | MQA: 64 query heads x 512, ONE shared K=V head of 512; q LoRA (q_a 5120->1280, norm, q_b 1280->32768); kv 5120->512 + norm; no per-head q norm |
| rope | NORM (adjacent pairs) on the last 64 of 512 channels; layers 0-1 base 1e4 plain; layers 2-39 base 160000 with yarn (factor 16, orig 65536, corr dims [15,25], net mscale 1); output de-roped at the query position |
| attention pattern | every layer: sliding window 128 over its own raw K/V + sink per head; layers 2-19 add compressed rows (2 tokens -> 1, softmax-gated), layers 20-39 add compressed rows (1 token -> 1); compressed rows computed only at layers 2, 8, 14, 20 and shared by the layers after them |
| indexer | 32 heads x 128, top-512 compressed rows, computed at layers 2,8,14,20,24,28,32,36; candidate-block narrowing from layer 20 (2048 blocks of 8); a no-op below 512 compressed rows (selects all) |
| output projection | grouped: 8 groups of 8 heads, each 4096 -> 1024 (`attn_output_a`), then 8192 -> 5120 (`attn_output_b`) |
| residual stream | hyper-connections (mHC): 4 streams, per sublayer pre/post/comb mixes from `hc_*_fn` (20480x24), Sinkhorn 20 iterations eps 1e-6; LAGGED pre-mix; streams start as 4 copies of the embedding, collapsed at the head with layer 39's FFN pre-mix |
| Engram | layers 1 and 14: 4-gram hashed lookups (token_map normalisation, 3 orders x 8 heads = 24 rows of 256 from the 110 GB IQ4_NL table), `engram_kv` 6144 -> 25600 (4 keys + 1 value), sigmoid-sqrt gate per stream, added to the streams before the layer |
| MoE | 192 routed experts, top-6, 1 shared expert; gating sqrt(softplus); selection with `exp_probs_b` (text) / `exp_probs_b_vl` (image tokens); weights normalised (+1e-20) then x1.5; SwiGLU with clamp 10 (gate min 10, up clamp +-10) on routed and shared |
| expert FFN | 2304 |
| norms | RMS, eps 1e-20 |
| vocab / tokenizer | 129280, gpt2 BPE with pre `joyai-llm`; bos 0, eos 1; template: DeepSeek V4.1 (`<｜User｜>`, `<｜Assistant｜>`, `<think>`, tool markers) |
| MTP | none in this file (compress_ratios has 3 trailing entries for layers 40-42 that have no tensors) |

## Quant types (whole six-shard set)

| Type | Tensors | Bytes | GiB | bits/weight | Where |
|---|---|---|---|---|---|
| IQ4_NL | 1 | 110,595,290,400 | 103.00 | 4.5 | Engram table `per_layer_token_embd` |
| IQ1_M | 60 | 29,727,129,600 | 27.69 | 1.75 | routed experts (gate/up/down, per layer choice) |
| IQ2_XXS | 42 | 24,524,881,920 | 22.84 | 2.06 | routed experts |
| IQ2_XS | 18 | 11,784,683,520 | 10.98 | 2.31 | routed experts |
| Q5_K | 244 | 5,329,223,680 | 4.96 | 5.5 | q_b, attn_output_a/b, shared expert, engram_kv, token_embd, output |
| Q8_0 | 80 | 389,939,200 | 0.36 | 8.5 | attn_q_a, attn_kv |
| BF16 | 67 | 202,375,168 | 0.19 | 16 | router, compressor, indexer |
| F32 | 531 | 159,486,400 | 0.15 | 32 | norms, sinks, hc_*, biases, engram_gate |
| total | 1043 | 182,713,009,888 | 170.17 | | (72.1 GB without the Engram table) |

All eight types dequantise bit-exactly in lb-gguf against llama.cpp's own `to_float` (see "Quant parity" in WORK_LOG).

## What lemonade-burn has no code for (as of this branch)

Everything model-specific: hyper-connections + Sinkhorn, q/kv LoRA MQA with 512-wide heads, NORM rope with channel
offset and yarn, de-rope, KV compression (ratio 1 and 2 with softmax gate), cross-layer KV/top-k sharing, the lightning
indexer and candidate blocks (needed past 512 tokens), grouped output LoRA, Engram hashing and lazy row reads from a
110 GB table, sqrt-softplus routing, SwiGLU clamp, the `joyai-llm` pre-tokenizer. The quant formats are the only part
that already exists (all eight types, CPU). No CPU reference forward was written in this branch (see WORK_LOG: step 4).

Shared crate or not: MiMo and DeepSeek V4.1 share only the generic pieces (RMS norm, top-k MoE with SiLU experts, sinks
in a softmax). Attention (GQA 192/128 + SWA vs MQA 512 + compression + indexer), the residual stream (plain vs 4-stream
mHC), routing (sigmoid vs sqrt-softplus + clamp + scale + shared expert) and the extra Engram path all differ, so a
separate crate (`lb-deepseek4`) is the right shape; the generic helpers can move to a small shared module later.

---

The rest of this file is the step-by-step pass with file:line citations (paths relative to the llama-v8 tree).


Reference build: /mnt/fast/scrape-v2/llama-v8 (commit dadb89f180). Authority is llama.cpp. Paths below are relative to that tree.
Abbreviations: `D41` = src/models/deepseek41.cpp, `D4` = src/models/deepseek4.cpp, `GR` = src/llama-graph.cpp,
`OPS` = ggml/src/ggml-cpu/ops.cpp, `CTX` = src/llama-context.cpp, `MDL` = src/llama-model.cpp.
Tensor shapes use GGUF/ggml order: `[ne0, ne1, ...]`, ne0 contiguous. A weight `W[in, out]` computes `y = W^T x` (y[o] = sum_i W[i,o] x[i], i.e. row o of the file is contiguous with `in` elements).
n_embd = 5120, n_head = 64, head dim hd = 512, rope dims = 64 (the LAST 64 of 512), hc = 4 streams, nt = tokens.

Cross-check done against /codebase/repos/Nauti-Strata-wt/deepseek41/src/poly/deepseek41.cpp (branch deepseek41): the port agrees on the lagged pre-mix, no per-head q norm, derope, Engram, and the 16384 limit (explained in 2.5/5).

---------------------------------------------------------------------------------------------------

## 1. Hparam loading

### 1.1 Parent loader `llama_model_deepseek4::load_arch_hparams` (D4:11-72), called first by D41:23-24
Keys (all prefixed `deepseek41.`; the arch name string is in src/llama-arch.cpp:92):
| key | hparam | req? | file value |
|---|---|---|---|
| attention.layer_norm_rms_epsilon | f_norm_rms_eps | yes (D4:20) | 1e-20 |
| attention.q_lora_rank | n_lora_q | yes | 1280 |
| attention.sliding_window | n_swa | yes | 128 |
| expert_feed_forward_length (scalar or array) | n_ff_exp | yes (D4:24) | 2304 |
| expert_shared_count | n_expert_shared | yes | 1 |
| expert_weights_scale | expert_weights_scale | yes | 1.5 |
| expert_weights_norm | expert_weights_norm | yes | true |
| swiglu_clamp_exp (scalar or array[n_layer_all]) | swiglu_clamp_exp[il] | yes (D4:28) | 10.0 x40 |
| swiglu_clamp_shexp | swiglu_clamp_shexp[il] | optional; if missing copies the exp array (D4:29-31) | 10.0 x40 |
| attention.indexer.head_count / key_length / top_k | indexer_n_head / indexer_head_size / indexer_top_k | yes (D4:33-35) | 32 / 128 / 512 |
| attention.output_group_count / output_lora_rank | dsv4_o_group_count / dsv4_o_lora_rank | yes | 8 / 1024 |
| attention.compress_rope_freq_base | dsv4_compress_rope_base | yes | 160000 |
| hyper_connection.count / sinkhorn_iterations / epsilon | dsv4_hc_mult / dsv4_hc_sinkhorn_iters / dsv4_hc_eps | yes (D4:40-42) | 4 / 20 / 1e-6 |
| hash_layer_count | dsv4_hash_layer_count | yes | 0 (so no `ffn_gate_tid2eid`, every layer has `exp_probs_b`) |
| attention.compress_ratios (array) | dsv4_compress_ratios[] | yes; length must be >= n_layer_all else throw (D4:47-53) | len 43 |
| expert_gating_func | expert_gating_func | yes; must equal SQRT_SOFTPLUS (=4, src/llama-hparams.h:21) else throw (D4:55-58) | 4 |

Then D4:59-66: `swa_type = STANDARD`, `set_swa_pattern(0)` which makes EVERY trunk layer SWA (src/llama-hparams.cpp:8-22); n_swa = 128 means a query at position p sees raw keys with `p - p_k < 128`, i.e. positions p-127..p including itself (src/llama-hparams.h:551-556). `n_embd_out_impl = hc*n_embd` is then overwritten to `n_embd` by D41:26 (no hc_head).

Generic loader values (src/llama-model.cpp):
- block_count = 40 trunk layers (`n_layer()`), `context_length` 1048576, `embedding_length` 5120, `head_count` 64, `head_count_kv` 1 (defaults to head_count if missing, MDL:1575-1578), `key_length`/`value_length` 512 (MDL:1622-1632 area), `rope.dimension_count` 64 (n_rot), `expert_count` 192, `expert_used_count` 6.
- RoPE: `rope.scaling.type=yarn`, `rope.scaling.factor=16` -> `rope_freq_scale_train = 1/16` (MDL:1597-1611), `original_context_length=65536` -> `n_ctx_orig_yarn` (MDL:1585-1586), `freq_base=10000` -> `rope_freq_base_train` (MDL:1589-1590).
- IMPORTANT: `rope.scaling.yarn_beta_fast/slow` and `yarn_ext_factor`, `yarn_log_multiplier` are NOT read from the GGUF in this tree (only the key names exist in src/llama-arch.cpp:385-389 and the saver). The values come from hparam defaults: beta_fast 32, beta_slow 1 (src/llama-hparams.h:177-178), which equal the file's values (so no numerical consequence). ext_factor defaults to -1 and `CTX:265-267` sets it to 1.0 because scaling type is YARN. `rope_yarn_log_mul` stays at its default, so `cparams.yarn_attn_factor = 1` (CTX:269-306); but deepseek41 does NOT use `cparams.attn_factor`, see 2.3.
- Rope TYPE is `LLAMA_ROPE_TYPE_NORM` for deepseek4/deepseek41 (MDL:3700-3701 in the case list that returns NORM at MDL:3730): rotates ADJACENT pairs (x[2i], x[2i+1]), not NeoX halves.
- `hparams.rope_type` -> graph member `rope_type` (GR:1585).

### 1.2 deepseek41-specific (D41:23-128)
- `indexer_top_k` must be > 0 (D41:27).
- Arrays `attention.kv_source_layers` = [2,8,14,20] and `attention.indexer.source_layers` = [2,8,14,20,24,28,32,36] (both required, D41:34-35); every entry must be < n_layer and must have compress ratio != 0 (D41:41-54).
- Derived per layer (D41:44-66), walking il = 0..39 keeping `last_kv`, `last_idx`:
  - `dsv41_kv_src[il]` = latest kv source layer <= il; `dsv41_topk_src[il]` = latest index source layer <= il; both -1 for ratio 0 layers.
  - Validation: ratio must be <= 2, a kv source must exist, `last_idx >= last_kv`, and both sources must have the same ratio as il.
  - Result for this file: layers 0,1 -> none. Layers 2-7: kv_src 2, topk_src 2. 8-13: kv_src 8, topk_src 8. 14-19: kv_src 14, topk_src 14. 20-23: kv_src 20, topk_src 20. 24-27: kv_src 20, topk_src 24. 28-31: kv_src 20, topk_src 28. 32-35: 20 / 32. 36-39: 20 / 36.
  - A layer owns the compressor tensors (`attn_compressor_kv/gate/norm`, `indexer.attn_k`, `indexer.k_norm`) iff `kv_src[il]==il` (D41:186-194); it owns the indexer scorer tensors (`indexer.attn_q_b`, `indexer.proj`) iff `topk_src[il]==il` (D41:195-198). Matches the tensor table: attn_compressor_kv x4 (L2,8,14,20), gate x3 (ratio>1 only, so not L20), indexer.attn_k/k_norm x4, indexer.attn_q_b/proj x8.
- Candidate selection (optional, D41:68-82): `attention.indexer.candidate_source_layer` (default -1 = off; file 20), then `candidate_top_k_blocks` 2048 and `candidate_block_size` 8 are required. Source must itself be an index source; later index-source layers must share its ratio.
- Engram (D41:84-127), only if `ple.layers` present: reads `ple.ngram_size` 4, `ple.heads_per_ngram` 8, `embedding_length_per_layer_input` 256 (= `ple_head_dim`), `ple.layer_multipliers` (8 u64 = 2 modules x 4 n-gram positions), `ple.head_offsets` (48 u64), `ple.head_vocab_sizes` (48 u64), `engram.pad_token_id` 2, `engram.token_map` (129280 i32).
  - `n_cols = (ngram-1)*heads_per_ngram = 24` hash heads per module; sizes must be mult = n_modules*4, primes = offsets = n_modules*24.
  - Module e belongs to the e-th entry of `ple.layers` ([1,14]); `is_ple_impl` set for those layers. Each bucket count >= 2 and offset+prime <= INT32_MAX.
  - FACT checked in the GGUF (parsed shard 1 read-only): `engram.token_map` is NOT identity. It is a many-to-one normalisation: 129213 of 129280 entries differ from their index, 99092 distinct canonical ids, max 99091 (e.g. token 67 -> 35, 68 -> 36, ...). `pad_token_id` = 2 and `token_map[2] = 2`. You must read the array from the GGUF.
- Tensors (D41:130-218): `per_layer_token_embd.weight` is ONE table `[256, 768022850]` (IQ4_NL), rows = max(offset+prime) = 752021961+16000889 = 768022850; module 0 (layer 1) uses head columns 0..23 of the offset/prime arrays, module 1 (layer 14) columns 24..47. There are no `hc_head_*` tensors in v4.1 and no `nextn.*` tensors.

### 1.3 compress_ratios values and the 43 entries
- File: [0,0, 2 x18 (layers 2..19), 1 x20 (layers 20..39), 0,0,0] (len 43).
- 0 = no compression: sliding-window attention only (`ratio == 0` -> `build_raw_attention`, D41:521-523). Also selects the NON-yarn rope (2.3).
- 1 = "HCA slot" in V4.1: one compressed row per token (state size 1, no softmax gating: score is filled with 0 so softmax of one element is 1). Cache comment: `src/llama-kv-cache-dsv4.cpp:1275-1280` (`ratio_hca = is_v41 ? 1 : 128`).
- 2 = "CSA slot" in V4.1: non-overlapping blocks of 2 tokens, per-channel softmax gate over the 2 tokens (no overlap, no APE, no indexer compressor; `csa_state_size = ratio_csa` at kv-cache-dsv4.cpp:1398-1399). (In V4 the same slots were ratio 4 overlap / 128.)
- Why 43 vs 40: the loader only requires `len >= block_count` (D4:49) and reads the whole array, but only indexes [0, n_layer()=40) for deepseek41. The three trailing zeros correspond to layers 40..42 that exist in the checkpoint's config (draft/nextn style layers). This GGUF has no `blk.40..42.*` tensors and `deepseek41` explicitly has no MTP graph (D41:220-226: the DSpark draft is a separate DFLASH GGUF). I could not determine from the source what the 3 extra layers are in the original model beyond that; they are unused here.

---------------------------------------------------------------------------------------------------

## 2. Per-layer forward (graph constructor D41:768-933)

Notation: X is the stream state `[5120, 4, nt]` (stream s contiguous 5120 floats). t = token index. All math in F32 on CPU unless noted.

### 2.0 Input and loop skeleton
- `inp = tok_embd[token]` (Q5_K dequantised, no scale: `build_inp_embd(tok_embd)` default tok_scale 1.0, GR:1268; D41:780). `X0[:, s] = inp` for all 4 streams (`ggml_repeat_4d`, D41:826-828, cb "hc_init", il=-1).
- `pre_prev = none`. For il in 0..39 (D41:833-916), in this order:
  1. if layer is an Engram layer (1, 14): `X = build_engram(X)` (2.6) [cb "engram_out"]
  2. `residual = X`; `attn_mix = hc_mixes(X, hc_attn_fn, hc_attn_scale, hc_attn_base)` (2.1)
  3. attention input collapse: `cur = (il==0) ? X[:,0] : sum_s pre_prev[s] * X[:,s]` (D41:853). `pre_prev` is the FFN pre-mix of the PREVIOUS layer (see 2.1 note). [cb "hc_attn_pre"]
  4. `cur = rmsnorm(cur) * attn_norm` [cb "attn_norm"]
  5. `cur = attention41(cur)` (2.2-2.5)
  6. `X = hc_post(cur, residual, attn_mix.post, attn_mix.comb)` [cb "hc_attn_post"]
  7. `residual = X`; `ffn_mix = hc_mixes(X, hc_ffn_fn, hc_ffn_scale, hc_ffn_base)`; `pre_prev = ffn_mix.pre`
  8. FFN input collapse: `cur = sum_s attn_mix.pre[s] * X[:,s]` (uses the ATTENTION mix's pre computed at step 2, D41:868) [cb "hc_ffn_pre"]
  9. `cur = rmsnorm(cur) * ffn_norm` [cb "ffn_norm"]; MoE + shared expert (2.7-2.8); `X = hc_post(ffn_out, residual, ffn_mix.post, ffn_mix.comb)` [cb "l_out"]
- RMS norms everywhere: `y = x / sqrt(mean(x^2) + eps) * w` with eps = `f_norm_rms_eps` = 1e-20; mean accumulated in double (OPS:4115-4122). Weights multiply without a +1.

### 2.1 Hyper-connections (mHC): `build_hc_mixes`, `build_hc_pre`, `build_hc_post` (D4:261-448; fused ops CPU OPS:11216-11520)
Fused ops are on by default (CTX:340-342) but are bit-for-bit the same formulas as the unfused builder.

mixes (D4:357-392): for a sublayer with weights `hc_fn [20480, 24]` (24 rows of 20480), `hc_scale [3]`, `hc_base [24]`:
```
flat   = X[:, :, t] flattened stream-major: flat[s*5120 + d] = X[d, s]      (20480)
fn_in  = flat / sqrt(mean(flat^2) + 1e-20)        # NO learned weight, one RMS over all 20480
m[r]   = sum_k hc_fn[k, r] * fn_in[k]              # r = 0..23   (GGUF dim0=20480 contiguous per row)
pre[j]  = sigmoid(m[j]      * scale[0] + base[j])      + hc_eps     j = 0..3        (hc_eps = 1e-6)  [cb "hc_pre"]
post[j] = 2 * sigmoid(m[4+j] * scale[1] + base[4+j])                j = 0..3        [cb "hc_post"]
c[src][dst] = m[8 + 4*src + dst] * scale[2] + base[8 + 4*src + dst]    16 values        (flat index 8+4*src+dst)
```
Sinkhorn on `comb` (fused CPU, OPS:11250-11330; unfused equivalent D4:308-343), eps = 1e-6, n_iter = 20:
```
for each src (row):  comb[src][dst] = softmax_over_dst(c[src][:]) + eps        # max-subtracted, expf
col_norm: for each dst: comb[:,dst] /= (sum_over_src comb[src][dst] + eps)      # OPS:11218-11232
repeat (n_iter-1 = 19) times:
    row_norm: for each src: comb[src,:] /= (sum_over_dst comb[src][dst] + eps)  # OPS:11234-11248
    col_norm
```
i.e. 20 column normalisations and 19 row normalisations, column normalisation LAST. (The sum includes `+eps` in the divisor each time, the division is multiplication by `1/sum` in the fused code.) [cb "hc_comb"]
Memory layout of the result: ggml `[4(dst), 4(src), nt]`, element (dst, src) at `dst + 4*src`.

collapse `build_hc_pre(X, w)`: `out[d] = sum_s w[s] * X[d, s]` (scale 1.0, OPS:11356-11415).
post `build_hc_post(x, residual, post, comb)` (OPS:11436-11508, GGML formula):
```
Y[d, dst] = post[dst] * x[d] + sum_src comb[dst + 4*src] * residual[d, src]       dst = 0..3
```
(equivalently with comb[src][dst] as defined above: `Y[:,dst] = post[dst]*x + sum_src comb[src][dst] * residual[:,src]`.)

THE LAGGED ("single pass") PRE-MIX (D41:830-831, 849-869; cross-checked in the port's comment). This is not the textbook mHC order:
- The collapse weights `pre` used to build a sublayer's INPUT are not that sublayer's own: attention input at layer il uses `pre` of layer il-1's FFN mix (computed from that layer's mid-state X after attention); FFN input at layer il uses `pre` of layer il's attention mix (computed from the layer input X). Layer 0 attention takes stream 0 (all 4 streams are identical at that point).
- The `post` and `comb` used in `hc_post` ARE the sublayer's own mixes (computed from the sublayer's input stream state).
- Head: final collapse uses the last FFN's `pre` (`pre_prev` of layer 39), see section 3.
- Per sublayer the mix is computed from the stream state before that sublayer: attention mix from X after Engram (since `residual = X` is taken after build_engram, D41:849-850).
cb names: "hc_mixes", "hc_pre", "hc_post", "hc_comb" (emitted TWICE per layer, attention then FFN, same name suffix `-il`).

### 2.2 Q path and KV path (`build_attention41`, D41:409-455)
Input `cur` = attn_norm output `[5120, nt]`.
```
qr     = rmsnorm( attn_q_a^T cur ) * attn_q_a_norm          # attn_q_a [5120,1280], norm [1280]      cb "norm"(unweighted),"qr_norm"
q      = attn_q_b^T qr  -> reshape [512, 64, nt]             # attn_q_b [1280, 32768]
         NO per-head RMS norm on q in v4.1 (D4:1010-1014 does that for deepseek4; D41:448-449 does not)
q      = rope(q, pos, dims 448..511)                         # section 2.3, cb "q"
kv     = rmsnorm( attn_kv^T cur ) * attn_kv_a_norm          # attn_kv [5120,512], norm [512]
kv     = rope(kv, pos, dims 448..511) -> [512, 1, nt]        # cb "kv"
```
- MQA: 1 shared KV head of 512 dims; K == V == `kv` (the same tensor is stored in the KV cache as K only, and passed as both K and V, `build_attn_mha(q, k_all, k_all, ...)`, D41:716/743). V therefore contains rope-rotated channels 448..511, which is why the output is de-roped (2.4).
- Rope applies to channels `[448, 512)` of each 512-vector (n_dims=64 via `ggml_rope_ext(..., n_embd_head_rope=64)`, then `ggml_rope_set_offset(.., 448)`, D41:439-443; implementation OPS:6232-6360 with `n_offs`). Channels 0..447 are untouched.
- Attention scale `kq_scale = 1/sqrt(512)` (D41:427), applied to q.k inside softmax; sinks are NOT scaled (2.4).

### 2.3 RoPE details (per layer kind, D41:430-443; ggml OPS:6076-6112, 6188-6204, 6320-6338; ggml.c:4373-4388)
Type NORM; pair i (i = 0..31) is `(x[448+2i], x[448+2i+1])`:
```
theta_i(p)   = p * base^(-2i/64)                              (theta_scale = base^(-2/64), accumulated by repeated multiplication in float)
x0' = x0*cos(theta) - x1*sin(theta);   x1' = x0*sin(theta) + x1*cos(theta)      (mscale multiplies cos and sin)
```
Per layer kind (ratio = compress_ratios[il]):
- ratio == 0 (layers 0 and 1 only): `freq_base = cparams.rope_freq_base = 10000`, `freq_scale = 1`, `ext_factor = 0`, `attn_factor = 1` (dsv4_rope_attn_factor returns 1 when ext_factor==0), betas 0, n_ctx_orig 0. So PLAIN rope: theta_i = p * 10000^(-2i/64), mscale 1.
- ratio != 0 (layers 2..39, both ratio 1 and 2): `freq_base = 160000` (compress_rope_freq_base), `freq_scale = 1/16`, `ext_factor = 1`, `beta_fast = 32`, `beta_slow = 1`, `n_ctx_orig = 65536`. Yarn (rope_yarn, OPS:6083-6098):
  ```
  corr_dims: start = 64*ln(65536/(32*2*pi))/(2 ln 160000) = 15.4533 -> floor 15 ; end = 64*ln(65536/(1*2*pi))/(2 ln 160000) = 24.7084 -> ceil 25 ; clipped to [0, 63]  => [15, 25]
  ramp(i)  = 1 - clamp((i - 15)/max(0.001, 25-15), 0, 1)      (uses i = i0/2 = pair index)
  theta_extrap = p * 160000^(-2i/64);  theta_interp = theta_extrap / 16
  theta   = theta_interp*(1-ramp) + theta_extrap*ramp          (ext_factor = 1 so ramp_mix = ramp)
  mscale  = attn_factor_l * (1 + 0.1*ln(16))
  ```
  truncate flag default true (floor/ceil on the corr dims, ggml.c:4455-4458). Pairs i <= 15 are pure extrapolation (unscaled), i >= 25 are fully divided by 16, linear in between (ramp 0.9 at i=16, 0.5 at i=20, 0.1 at i=24).
  `attn_factor_l = 1/(1 + 0.1*ln(16)) = 0.78293` (models.h:1223-1229, comment "cancels ggml's rope mscale; the DeepSeek-V4 reference applies none"), so the product `attn_factor_l * (1+0.1 ln 16)` is 1.0 (to float rounding). NET: no mscale. `cparams.yarn_attn_factor` is not used.
- The same `rope` lambda (same base/scale/ext/betas of the LAYER) is used for q, kv, the compressed latent, the indexer query and the indexer key (D41:439-443, 496-497, 544).
- Positions: q and kv at the token position p; compressed latent and its indexer key at the FIRST token position of their block (`state_write_pos = pos+1-ratio`, src/llama-kv-cache-dsv4.cpp:538-547; comment D41:494 "a latent stands for the first token of its group").

### 2.4 Attention itself, sinks, window (D41:456-749; `build_raw_attention` D4:831-855; masks GR:893-935)
Key set per query at position p, layer il:
- Window keys: raw cache of this layer's own `kv` rows, those with `p - p_k < 128` (and causal `p_k <= p`).
- If ratio != 0 and compressed rows exist (`ci.kq_mask` non-null i.e. plan n_kv > 0): plus compressed rows `j` with `j < n_visible(p) = floor((p+1)/ratio)` AND selected by top-k (2.5). Row j covers tokens `[j*ratio, (j+1)*ratio)`; it becomes visible to the query at position `j*ratio + ratio - 1` (a query always sees the block that ends at its own position). For ratio 1 a token sees its own compressed row. A partial block is never visible.
- Compressed row key/value vector = first 512 dims of the stored 640-dim row `[comp(512) | idxK(128)]` (D41:499, 719-720).
- Else (ratio 0, or nothing visible anywhere in the ubatch): window only.
Scores and softmax with sinks (OPS:5750-5810 for the unfused path, OPS:8906-8925 for flash attention; sinks via `ggml_*_add_sinks`, GR:3127,3184):
```
s_j  = (q_h . k_j) / sqrt(512)         over all allowed keys j (window + selected comp rows), 1 shared K/V head
m    = max(max_j s_j, sink[h])
den  = sum_j exp(s_j - m) + exp(sink[h] - m)         # sink[h] = attn_sinks.weight[h], NOT scaled by 1/sqrt(512)
o_h  = sum_j exp(s_j - m)/den * v_j                  # v_j = k_j; the sink contributes to den but has no value
```
`attn_sinks.weight [64]` per layer. In the real run flash attention is used (K/V converted to F16, F32 accumulate, GR:3100-3130) so llama.cpp's own values carry f16 rounding of K/V (and the cache stores K as F16 by default); a F32 reference will differ at ~1e-3 relative, not by structure. Key ORDER differs between paths (window then top-k rows, sorted ascending when `gather`), which only changes float summation order.
Quantised-K-cache Hadamard rotation (`k_rot`) applies only if the K cache is quantised; with the default F16 cache it is null. Ignore.

Output de-rope ("derope", D41:751-755): `out [512, 64, nt]` gets `ggml_rope_ext_back` with the QUERY position p and the SAME per-layer rope parameters and offset 448: it is the inverse rotation (sin sign flipped, OPS:6296, 6389+), applied to channels 448..511 of each head. Result: relative-position output. [cb "attn_derope"]. The net mscale is 1 as in 2.3, so the inverse is exact.

### 2.5 KV compression, indexer, sharing (D41:456-633, 531-632)
Compression at a kv-source layer (2, 8, 14, 20), on `cur` = THIS source layer's attn_norm output (consumers do not recompute; they read the source's rows):
```
kv_t    = attn_compressor_kv^T cur_t                         # [5120,512] BF16 weight
score_t = ratio==2 ? attn_compressor_gate^T cur_t : 0        # [5120,512] BF16 weight; absent at ratio 1 (zeros)
block b (tokens b*ratio .. b*ratio+ratio-1), completed when (pos+1) % ratio == 0:
comp[d] = sum_{j<ratio} kv_{b*ratio+j}[d] * softmax_j(score_{b*ratio+j}[d])      # softmax over the ratio axis, per channel; ratio 1 => comp = kv
comp    = rmsnorm(comp) * attn_compressor_norm                       # [512]   cb "comp_latent"
ik      = rmsnorm( indexer_attn_k^T comp ) * indexer_k_norm          # indexer_attn_k [512,128]; norm [128]; K from the rope-free latent
ik      = rope(ik, pos = b*ratio, dims 64..127 of 128)               # offset n_idx_embd - 64 = 64
comp    = rope(comp, pos = b*ratio, dims 448..511)
stored row b = [comp(512) | ik(128)]   (640 floats, F16/F32 per cache type)
```
(D41:466-502; state positions: the ratio-2 plan has no APE in v4.1.) At ratio 2 no overlap window: a block uses exactly its 2 tokens.

Sharing: a layer il with ratio != 0 reads compressed rows from the storage of `kv_src[il]` (layers 3-7 read layer 2's rows, 9-13 read 8's, 15-19 read 14's, 21-39 read 20's) and top-k indices from `topk_src[il]` (the most recent index-source layer). Raw sliding-window K/V is always per layer. Rows are written once by the source layer for the whole stack (KV sharing = fewer rows, `src/llama-kv-cache-dsv4.cpp:1308-1340`).

Indexer (only at a layer with `topk_src[il]==il`, i.e. 2,8,14,20,24,28,32,36; D41:531-631, scoring D41:577-586, fused CPU op OPS:12320-12408):
```
iq[h]  = rope( indexer_attn_q_b^T qr  -> [128, 32 heads], dims 64..127 )     # qr = THIS layer's normed q_a output (cb "qr_norm"); attn_q_b idx [1280,4096] BF16
iw[h]  = (indexer_proj^T cur)[h] / sqrt(128*32)                              # indexer_proj [5120,32] BF16 applied to attn_norm output cur
score(t, j) = sum_h iw[h] * relu( iq[h] . ik_j )  + mask(t, j)               # mask = 0 for j < n_visible(t), -inf otherwise; cb "idx_score"
top_k  = top min(n_comp, 512) rows by score (order unspecified)              # cb "idx_top_k"
```
No softmax scale beyond the 1/sqrt(4096) folded into iw. The selected set (intersected with visibility) is what attention may read from the compressed rows: `kq_mask_comp[j] = 0 if j in top_k (and visible) else -inf` (`build_top_k_mask`, D4:679-715, cb "comp_top_k_mask" / gather path "comp_kq_mask_sel").
Layers that are not index sources reuse the latest source's top_k tensor for each query (`topk_of[topk_src[il]]`, D41:531, 632).

Candidate-block narrowing (D41:556-609), `cand_src = 20`, block size 8, 2048 blocks:
- At layer 20 (`pick_blk`): block score = max of the (masked) scores in each block of 8 compressed rows, +1e30 pin on the block containing the newest visible row, keep the top `min(2048, n_blk)` blocks (cb "cand_block_score"). The layer-20 top-512 itself is computed over ALL rows.
- At index-source layers after 20 (24, 28, 32, 36; `use_blk`): their own scores are restricted to rows inside layer 20's selected blocks (0 inside, -inf outside; cb "cand_mask") before their top-512.
- Layers 2, 8, 14 (before the source) do no narrowing.
- It is a no-op unless `n_blk > 2048`, i.e. more than 2048*8 = 16384 visible compressed rows. Layers 20..39 are ratio 1, so that is more than 16384 tokens of context. This is exactly why the port refuses `max_context > 16384`.

### 2.6 Grouped output projection (D41:757-763)
```
out [512, 64, nt] (after derope) -> group g (0..7) = heads 8g..8g+7, vector of 4096 = concat over the 8 heads in order, each head's 512 contiguous
oa_g   = attn_output_a_g (1024 x 4096) . out_g              # attn_output_a file [4096, 8192] viewed as [4096, 1024, 8]: file row r = g*1024 + j   (Q5_K)   cb "attn_wo_a"
oa     = concat_g oa_g  -> [8192]                           # group-major
attn   = attn_output_b^T oa  -> [5120]                      # attn_output_b [8192, 5120]  (Q5_K)   cb "attn_out"
```
Then step 6 of 2.0 (hc_post).

### 2.7 Engram / per-layer embedding (layers 1 and 14; D41:228-278, 372-407, 836-840)
Applied to the 4-stream state X at the START of layers 1 and 14, before the HC mixes and attention: `X <- X + gate_s * value` (the value is the same for all 4 streams).
Hash (`engram_hash(e, ...)`, D41:240-278); e = index of the layer in `ple.layers` (layer 1 -> e=0, layer 14 -> e=1); all arithmetic uint64:
```
cid(t)   = engram_token_map[t]              (t outside [0,129280) -> DEAD)
pad      = engram_token_map[engram_pad_id]  (= token_map[2])
toks[0]  = cid(token_t)                     (DEAD -> pad; "blocked")
for s = 1..3:  c = (pos_t - s < 0) ? DEAD : cid(token at position pos_t - s)       # previous tokens of the same sequence
               blocked = blocked || c == DEAD ; toks[s] = blocked ? pad : c        # once blocked, longer shifts stay pad
mult[o]  = layer_multipliers[e*4 + o], o=0..3 ;  primes/offs index  col = e*24 + (o-1)*8 + h
rolling  = toks[0]*mult[0]
for o = 1..3:  rolling ^= toks[o]*mult[o]
               for h = 0..7:  row[(o-1)*8 + h] = rolling % head_vocab_sizes[e*24 + (o-1)*8 + h] + head_offsets[e*24 + (o-1)*8 + h]
```
So 24 rows per token per Engram layer (3 n-gram orders: 2-gram, 3-gram, 4-gram, x8 heads). Products stay below 2^64 for these multipliers and token ids < 129280 (max 82619226485591*129279 = 1.07e19 < 1.84e19), so no wraparound; a python int64 reference could overflow differently, but llama.cpp uses uint64 and is the authority. Prev-token ordering: `prev[i*3 + (3 - s)]` = token at distance s (src/llama-kv-cache.cpp:2025-2070).
Rows: `rows[col] = per_layer_token_embd[row[col]]` (256 floats each, IQ4_NL dequantised) [cb "engram_rows"]; concatenated `rows[t] = [rows(col 0) | rows(col 1) | ... | rows(col 23)]` = 6144 floats, col-major as in the hash order above.
Math (`build_engram`, D41:372-407):
```
kv      = engram_kv^T rows[t]                  # engram_kv [6144, 25600] Q5_K -> 25600 = 5*5120      cb "engram_kv"
key[s]  = kv[ s*5120 : (s+1)*5120 ]            s = 0..3  (one key per stream)
value   = kv[ 20480 : 25600 ]                  (one value, shared)
dot[s]  = ( sum_d rmsnorm(X_s)[d] * rmsnorm(key_s)[d] * engram_gate[d, s] ) / sqrt(5120)     # both rmsnorm UNWEIGHTED, eps 1e-20; engram_gate [5120,4] F32, element (d,s)    cb "engram_dot"
mag     = sqrt( max(|dot[s]|, 1e-6) )
sign    = dot[s] < 0 ? -1 : +1                 # dot == 0 -> +1 (copysign)  [step(-dot) trick]
gate[s] = sigmoid(sign * mag)                  (x keep, which is 1 for token input and 0 for embedding/image tokens)   cb "engram_gate"
X_s    += gate[s] * value                      # for each stream s
```
Nothing else of the "per-layer embedding" machinery exists in v4.1: no per-layer-input projection, no gating of the residual beyond this. The layer-input `X` that then enters 2.0 step 2 already contains the Engram addition.

### 2.8 FFN (D41:875-913)
```
cur     = rmsnorm(collapsed) * ffn_norm                                       # cb "ffn_norm"
logits  = ffn_gate_inp^T cur        # [5120,192] BF16, F32 accumulate (ggml_prec_set_acc, D41:879-880)   cb "ffn_moe_logits"
probs   = sqrt(softplus(logits))    # gating func 4 = SQRT_SOFTPLUS (GR:2316-2318); softplus(x) = x>20 ? x : log1p(exp(x))  (unary-ops.cpp:80-82)   cb "ffn_moe_probs"
bias    = exp_probs_b (text tokens)   # exp_probs_b_vl only if the ubatch carries embeddings (image tokens, D41:883-887); a text-only run never uses _vl
sel     = top-6 experts of (probs + bias)        # argsort_top_k, GR:2329-2372; tie-break unspecified    cb "ffn_moe_probs_biased","ffn_moe_topk"
w_k     = probs[sel_k]                            # UNBIASED probs gathered, GR:2386          cb "ffn_moe_weights"
w_k     = w_k / (sum_k w_k + 1e-20)               # deepseek41 uses +1e-20, not the F16 clamp (GR:2397-2412)    cb "ffn_moe_weights_norm"
w_k     = w_k * 1.5                               # expert_weights_scale (GR:2417)          cb "ffn_moe_weights_scaled"
for each selected expert e (k = 0..5):
    g = ffn_gate_exps[e]^T cur ; u = ffn_up_exps[e]^T cur          # [5120 -> 2304] each (IQ1_M / IQ2_XS / IQ2_XXS)
    h = silu( min(g, 10) ) * clamp(u, -10, 10)                     # ggml_swiglu_clamp(gate, up, limit): OPS:3544-3590; clamp is BEFORE the silu, only an upper bound on gate; silu(x)=x/(1+exp(-x))
    y_e = ffn_down_exps[e]^T h                                     # [2304 -> 5120]
moe_out = sum_k w_k * y_{sel_k}                                    # summed over k in selection order   cb "ffn_moe_weighted","ffn_moe_out"
shared: g = ffn_gate_shexp^T cur ; u = ffn_up_shexp^T cur          # [5120 -> 2304], Q5_K
    h = silu(min(g,10)) * clamp(u,-10,10)                          # same swiglu_clamp (GR:2066-2078, limit swiglu_clamp_shexp[il] = 10)   cb "ffn_up","ffn_gate","ffn_swiglu_limited"
    shexp = ffn_down_shexp^T h                                     # [2304 -> 5120]   cb "ffn_shexp"
ffn_out = moe_out + shexp   (shared expert unweighted, no gate)    cb "ffn_out"
X = hc_post(ffn_out, residual, ffn_mix.post, ffn_mix.comb)          cb "l_out"
```
The clamp limit applies only when limit > 1e-6 (always here). `swiglu_clamp_exp[il]` and `..._shexp[il]` are per layer arrays (all 10.0 in this file).
Residual adds are entirely inside `hc_post` (no separate `x + residual`): the residual path is `comb^T residual` plus `post * sublayer_out`.

---------------------------------------------------------------------------------------------------

## 3. Head (D41:918-932)
```
(outputs only) X <- X[:, :, selected tokens]; pre_prev <- pre_prev[:, selected tokens]
hc_out = sum_s pre_prev[s] * X[:, s]        # pre_prev = FFN pre-mix of layer 39, computed from layer 39's MID state (after its attention)   cb "hc_out"   (il=-1)
xn     = rmsnorm(hc_out) * output_norm       # cb "norm" (unweighted) then "result_norm"
logits = output^T xn                         # output [5120, 129280] Q5_K          cb "result_output"
```
No `hc_head_fn/base/scale` (they exist only for deepseek4), the collapse uses the lagged FFN pre-mix. Streams are initialised as 4 identical copies of the token embedding (2.0) and only merged here. `res->t_embd = result_norm`.

---------------------------------------------------------------------------------------------------

## 4. Unusual ggml ops (CPU implementations)

| op | where | formula |
|---|---|---|
| `GGML_OP_RMS_NORM` | OPS:4082-4125 | `x * (1/sqrtf(mean(x^2, double acc) + eps))`, eps = 1e-20 here, no centring |
| `GGML_OP_ROPE` / `ROPE_BACK` with `n_offs` | OPS:6232-6385, 6389; yarn OPS:6076-6112; corr dims ggml.c:4373-4388; offset ggml.c:4434-4445 | rotates `n_dims=64` channels starting at channel `n_offs=448`; NORM = adjacent pairs; back = sin sign -1 (OPS:6296); yarn formulas in 2.3 |
| `GGML_OP_DSV4_HC_COMB` | OPS:11216-11335 (softmax over dst per src row + eps, then 20 col-norm / 19 row-norm, each `/ (sum+eps)`) | see 2.1 |
| `GGML_OP_DSV4_HC_PRE` | OPS:11354-11415 | `out[d] = sum_s w[s]*x[d,s]` |
| `GGML_OP_DSV4_HC_POST` | OPS:11434-11508 | `out[d,dst] = x[d]*post[dst] + sum_src residual[d,src]*comb[dst + 4*src]` |
| `GGML_GLU_OP_SWIGLU_CLAMP` | OPS:3544-3590 (f32), 3605+ (f16), ggml.c:3112-3121 | `min(gate,L)/(1+exp(-min(gate,L))) * clamp(up,-L,L)` |
| `GGML_UNARY_OP_SOFTPLUS` | unary-ops.cpp:80-82 | `x > 20 ? x : log1pf(expf(x))` |
| `ggml_scale_bias(x, s, b)` | used in D41:399, GR:2405, D4:385 | `x*s + b` |
| `GGML_OP_SOFT_MAX` with sinks / flash-attn with sinks | OPS:5750-5810 / OPS:8906-8925 | sink logit joins max and denominator only (2.4) |
| `GGML_OP_LIGHTNING_INDEXER` | OPS:12320-12408 | `sum_h relu(q_h . k) * w_h + mask` (fused indexer; unfused graph is the same math, D41:577-586) |
| `GGML_OP_TOP_K` / `argsort_top_k` | used GR:2372, D41:611, 597, 628 | indices of the k largest; order and tie-breaking unspecified (I did not read the CPU kernel's tie behaviour) |
| `GGML_OP_POOL_1D` (max, k=s=8) | D41:594 | per-block max of 8 compressed-row scores (candidate level one only) |
| `ggml_sigmoid`, `ggml_step`, `ggml_sqrt`, `ggml_clamp`, `ggml_abs` | D41:397-400 | step(x)=1 if x>0 else 0; sigmoid=1/(1+exp(-x)) |
| `ggml_rms_norm` on the engram/hc inputs | D41:393, D4:372 | unweighted RMS over 5120 (engram) or 20480 (hc) elements |

---------------------------------------------------------------------------------------------------

## 5. What a <= 64-token single-sequence CPU reference can skip or simplify

Facts: positions are 0..n-1 (n <= 64); window 128 > n.
1. Sliding window: skip. Causal mask is identical to the 128-window (`p - p_k < 128` always true, src/llama-hparams.h:553).
2. Indexer entirely (all of `indexer.attn_q_b`, `indexer.proj`, scoring, candidate narrowing): skip, results identical. Evidence: `top_k = top_k(score, min(n_comp, 512))` (D41:611) and n_comp = max visible rows in the batch <= 64 (ratio 1 layers) or <= 32 (ratio 2 layers), always <= 512, so every row is selected; the selection only ever ANDs with the visibility mask (`build_top_k_mask` D4:679-715). Candidate blocks: n_blk <= 8 << 2048 (D41:597) so also a no-op. The `ik` (indexer K) rows in the compressed cache are then also unused by attention; skip `indexer_attn_k`/`indexer_k_norm`.
   This holds for contexts up to 512 tokens at ratio 1 layers (1024 tokens at ratio 2); the candidate step stays a no-op up to 16384 tokens.
3. Compression is NOT skippable: a ratio-1 layer (20..39) sees compressed rows from token 0 (query p sees rows 0..p, including its own); a ratio-2 layer sees row floor((p+1)/2)-1 and below, so from p=1. Per layer: ratio 0 (L0, L1) = window only (plain rope base 10000, no compressor); ratio 2 (L2..L19) and ratio 1 (L20..L39) = window + all visible compressed rows (K=V, no selection); softmax includes the sink.
4. Compressor runs only at the kv-source layers 2, 8, 14, 20; layers 3-7, 9-13, 15-19, 21-39 reuse their rows (compressed rows were computed from the source layer's own attn-normed input, with the source's `attn_compressor_*` tensors). The 64-token prompt means at most 32 (ratio 2) / 64 (ratio 1) rows. Positions of rows: `b*ratio`.
5. Rope: only two flavours (plain base 10000 for layers 0-1, yarn base 160000 factor 16 with corr dims [15, 25] elsewhere). With <= 64 positions the yarn matters only through the per-pair interpolation (pairs i>=16 are scaled by up to 1/16).
6. Engram: required at layers 1 and 14. At the sequence start the n-gram history is padded: for position p, shifts s > p give the pad token (`token_map[2]`) and stay pad. No batch-dependence.
7. No hash layers (`hash_layer_count` 0), no MTP/nextn, no hc_head, no vision router bias (`exp_probs_b_vl` unused for text), no expert-group routing (the group branch at GR:2339 needs `n_expert_groups > 1`; deepseek4/41 loaders never set it and the GGUF has no such key, but I did not check the hparam default).
8. HC: computed in full. Layer 0 attention collapse = stream 0 (identical copies anyway); all 4 streams diverge after layer 0's hc_post.
9. Not skippable but shape-simple: sinks per head, derope with the query position, grouped output proj.
10. Batch invariance: llama.cpp's result for a prompt does not depend on ubatch splits except float order; a full-prompt single pass is the reference.

---------------------------------------------------------------------------------------------------

## 6. cb() names an eval callback sees (`ggml_format_name(cur, "%s-%d", name, il)` when il >= 0, else the bare name; CTX:3238-3244)
Because the tensors are named by `cb`, several distinct tensors reuse the same name within a layer; the later one in graph order overwrites the name in the graph, but an eval callback still sees each node's name at compute time. Names repeat for: "hc_mixes","hc_pre","hc_post","hc_comb" (attention then FFN), "norm" (every `build_norm` with a weight, emitted for the UNWEIGHTED output), (`ffn_moe_logits` is emitted once, by D41:881, because `probs_in` is passed to build_moe_ffn).
Per layer, in graph order (il = layer):
- (-1) `inp_tokens`, `inp_embd`, `hc_init` [5120,4,nt] (the 4 copies of the embedding)
- Engram layers only (1, 14): `engram_ids-il` (set directly), `engram_rows-il`, `engram_kv-il`, `engram_dot-il`, `engram_gate-il`, `engram_out-il` (this is the X that enters the layer)
- `layer_inp-il` (only if DSpark features requested)
- `hc_mixes-il`, `hc_pre-il`, `hc_post-il`, `hc_comb-il` (attention mixes), `hc_attn_pre-il` (collapsed attn input), `norm-il`, `attn_norm-il`
- attention: `norm-il`+`qr_norm-il`, `q-il` (after rope), `norm-il`+... for kv (the kv rms-norm gives `norm-il`, final `kv-il` is after rope), at kv-source layers: `comp_latent-il`, `norm-il` twice more (comp norm, indexer k norm); at index-source layers: `idx_score-il`, `cand_block_score-il` (layer 20), `cand_mask-il` (24,28,32,36), `idx_top_k-il`; then either `attn_raw-il` (ratio 0 / no comp rows) or `comp_k-il` / `comp_top_k_mask-il` / `comp_kq_mask-il` (dense path) or `comp_top_k_sorted-il`, `comp_k_sel-il`, `comp_kq_mask_sel-il` (gather path), then `attn_comp-il`; then `attn_derope-il`, `attn_wo_a-il`, `attn_out-il`
- `hc_attn_post-il` (X after attention), `hc_mixes-il`,`hc_pre-il`,`hc_post-il`,`hc_comb-il` (FFN mixes), `hc_ffn_pre-il`, `norm-il`, `ffn_norm-il`
- MoE: `ffn_moe_logits-il`, `ffn_moe_probs-il`, `ffn_moe_probs_biased-il`, `ffn_moe_argsort-il` (on the argsort's src), `ffn_moe_topk-il`, `ffn_moe_weights-il`, `ffn_moe_weights_sum-il`, `ffn_moe_weights_sum_clamped-il` (holds sum+1e-20 for v4.1), `ffn_moe_weights_norm-il`, `ffn_moe_weights_scaled-il`, `ffn_moe_up-il`, `ffn_moe_gate-il`, `ffn_moe_swiglu_limited-il`, `ffn_moe_down-il`, `ffn_moe_weighted-il`, `ffn_moe_out-il`
- shared expert: `ffn_up-il`, `ffn_gate-il`, `ffn_swiglu_limited-il`, `ffn_shexp-il` (no `ffn_down` name: only emitted when down_b is set)
- `ffn_out-il` (moe_out + shexp), `l_out-il` (X after the layer, [5120,4,nt])
- Head (-1): `hc_out` [5120,nt], `norm`, `result_norm`, `result_output`.
Note: with `inp_out_ids` (only the last token's logits requested), `l_out` still has all tokens but `hc_out` and later have only the output rows (D41:919-922).
Useful comparison points for a reference: `hc_init`, `engram_out-1`, `hc_attn_pre-il`, `attn_norm-il`, `qr_norm-il`, `q-il`, `kv-il`, `attn_derope-il`, `attn_out-il`, `hc_attn_post-il`, `ffn_norm-il`, `ffn_moe_topk-il`, `ffn_moe_weights_scaled-il`, `ffn_out-il`, `l_out-il`, `hc_out`, `result_norm`.
(The local scrape hook was not located in these files; it presumably hooks the eval callback and uses these names. I did not search for it.)

---------------------------------------------------------------------------------------------------

## 7. Things I could not determine, or that are ambiguous
- The Python reference (`inference/model.py`, `engram.py`) is not in the tree; the comments cite it (D41:13, 228, 878). I did not verify the llama.cpp port against it. Items where the port could differ from upstream python and I cannot say: unweighted RMS norms inside Engram, `uint64` vs int64 hash, the lagged pre-mix.
- What the 3 trailing `compress_ratios` entries (layers 40-42) are, beyond "unused here" (no tensors, no MTP graph in deepseek41).
- Tie-breaking and ordering of `top_k` / `argsort_top_k`, and the float summation order over the 6 selected experts and over attention keys (differences ~1 ulp). With MoE near-ties a router flip could change a layer output; no way to match without copying ggml's sort.
- llama.cpp stores K (window and compressed rows) in the KV cache (F16 by default) and uses flash attention with F16 K/V; a pure F32 reference will not match bitwise. The attention path used by default (`LLAMA_DSV41_SPARSE` on, `gather` path when nt<=32) vs dense path give the same key set.
- `yarn_beta_fast/slow` come from hparam defaults, not from the GGUF keys, in this tree (equal values, so no effect).

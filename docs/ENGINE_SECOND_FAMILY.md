# A second model family on lemonade-burn: Qwen3.6-35B-A3B (`qwen35moe`)

Branch `fable/engine-123-qwen-2026-11-15` (off the Vulkan branch head 5c40b23). Date 2026-11-15. Tags: [measured] = I ran it and
have the output, [read] = taken from a source file, [estimate] = arithmetic or judgement, not run.

Reference implementations read for formats and arithmetic (credited, MIT): llama.cpp (`ggml-quants.c`, `ggml-cpu/ops.cpp`,
`src/models/qwen35moe.cpp`, `delta-net-base.cpp`, `qwen3next.cpp`, `llama-vocab.cpp`). The engine's own code was written from
those facts; no llama.cpp code is copied.

## 0. Status in one table

| Step | Result |
|---|---|
| Part 0: seat 36-37 tok/s vs bench 98 | cause found and fixed, section 1 |
| 1a. Tensor types in the Q5_K_M file | Q5_K, Q6_K, Q8_0, F32, BF16 (BF16 only in the unused MTP layer) |
| 1a. CPU dequantisation vs llama.cpp | **bit-identical on all 33.7 M rows of all 753 tensors** [measured] |
| 1a. Device kernels | Q5_K built (dense, routed experts, native layout), Q6_K / Q8_0 already existed and re-verified on this file; all within f32 summation order [measured] |
| 1b. Architecture | CPU reference decoder complete (`crates/lb-qwen35`): tokenizer, chat template, embedding, delta-net layers, gated attention layers, MoE, head |
| 1c. Correctness gate | **top-1 98.42 % (gate 99.5 %): NOT met.** The same number between llama.cpp's own batch and decode paths is 97.37 %; the first three layers match llama.cpp to 3e-7 relative once its activation rounding is emulated. Section 5 |
| 1d. Device run on two P100s | **not built** (needs the E2-layer wiring for this architecture). llama.cpp on the same two cards: 54.1 tok/s [measured] |
| 1e. Manifest / demand entry | **not added**: nothing in the engine can serve this family yet, an entry would point at nothing |

Nothing here produces tokens through `lb-server`; the decoder is a check tool (`qwen35-check`, `qwen35-tok`, `qwen35-hidden`).
It does not produce wrong tokens: it produces the tokens llama.cpp produces (section 5), at about one token a second on 48 CPU
threads.

## 1. Part 0: why the seat ran at 36-37 tok/s

Card: c4 AF (free, no seat displaced), LFM2.5-24B-A2B Q4_0, all experts resident. Client: `sc.py` (streams like
`seat_demand.chat`: the real "hard" prompt, 419 prompt tokens, 300 new tokens, temperature 0.2) and `cm.py`
(the benchmark client: non-streamed, temperature 0). tok/s is the client's number after the first token. Logs in `scripts/qwen/`
(`p0.sh`, `sc.py`).

| binary | flags | request | decode tok/s | first token |
|---|---|---|---|---|
| seat binary (`20261113-prefill-split-2337068`) | seat entry as it was | stream, T=0.2, hard prompt | 34.5 / 37.0 / 36.9 (short prompt 38.5, 38.8) | 6.1-6.7 s |
| same | same | stream, **T=0**, hard prompt | 63.6 / 72.5 / 72.4 (short 78.7, 78.5) | 6.0-6.6 s |
| same | same | `cm.py` (T=0) | 80.7-81.0 (first prompt 63.4) | |
| same | bench flags (`LB_E2=1`, threads 20, NUMA node 1, `LB_WARMUP=96`, ctx 4096, scrape placement) on AF and on D8 | `cm.py` | 79.7-80.2 on both cards | |
| **new build `20261115-sampler-fix-db07b88`** | seat entry flags | stream, T=0.2, hard prompt | **88.2 / 88.1 / 88.3** (short 97.8, 96.7) | 0.67 s (first long prompt after boot 9.0 s) |
| same | same | stream, T=0 | 90.4 / 91.5 / 91.5 (short 98.9, 99.0) | 0.67 s |
| same | same | `cm.py` | 93.7 (cold) / 92.8 / 98.6 / 89.6 / 97.3 | |
| same | plus warm-up 96 / NUMA 1 + 20 threads | stream, T=0.2 | 89.3 / 89.6 / 89.6 (short 97.1, 98.8); 89.3 / 90.1 / 91.1 | same |
| new build, **the manifest `cmd` of `hard-lfm25-c4` run verbatim** | | stream, T=0.2 | 88.2 / 90.6 / 89.9 (short 97.5, 97.5) | 0.58-0.64 s |

Cause, in the order of size:

1. **The request's temperature.** `seat_demand` sends 0.2; the benchmark sends 0. With only that changed, the seat binary goes from
   37.0 to 72.4 tok/s. The server samples on the host from the full logits (`lb-model/src/sample.rs`): at temperature > 0 it
   built an index vector of all 65,536 vocabulary entries, **sorted all of them and exponentiated all of them, every token**: 1/37.0 - 1/72.4 =
   13.2 ms a token. At temperature 0 it is an argmax. No flag or env variable of the seat entry touches this (`LB_THREADS`, NUMA,
   `LB_WARMUP`, `LB_E2`, budget, context length: tried, none moves it).
   Fix (code, commit db07b88): drop logits more than 20 nats below the maximum before sorting (they carry under 2e-9 of the top
   probability), `select_nth_unstable` for top-k, exponentiate only the survivors. Equivalence test against the old full-vocabulary
   sampler with the same random draw (`matches_full_vocab_sampler_for_the_same_draw`, four parameter sets, 800 draws): passes.
2. **The binary.** The seat's build tops out at about 80 tok/s even with every benchmark flag and at T=0 (rows 3-4), and prefills 419
   tokens in 6 s. The rebuilt server (everything on the Vulkan branch head plus the sampler change) does 89-99 tok/s and
   prefills the same prompt in 0.67 s. I did not bisect which earlier commit gave the 80 -> 90+ step; the sync-cost work
   (flush + fuse) is reported neutral at all-resident (0.998x), so it is something else between 2337068 and the branch head.
3. Things that are **not** the cause: thread count, NUMA placement, warm-up, expert-cache budget (the seat entry is all-resident: 2432
   experts, 12.9 GB uploaded), context length, the streaming path itself (T=0, same binary: streamed 72, non-streamed 81; the gap was not investigated, it is a different prompt length and client). One real leftover: the first long prompt after boot costs
   8-9 s (kernel setup for a new prefill shape); `LB_WARMUP` does not cover it (it warms decode only).

Manifest change: `hard-lfm25-c4` `cmd` now runs `/mnt/t440/raid0/engines/lemonade-burn/20261115-sampler-fix-db07b88/bin/...` (library and
CUDA path symlinked to the 2337068 build's), note extended, entry still `enabled:false`. Backup:
`var/fleet-manifest.json.bak-20261115-qwen-p0`; JSON validated; a diff shows exactly the `cmd` and `_note` of that entry. `lb-lfm25-c4`
(the cache-budget entry, 5120 MiB of 12 GB on purpose, binary e3-v7) was NOT changed: its 30-ish tok/s is the E2 design point, not
a defect, and its binary predates this fix; point it at the new build when someone wants it again.

## 2. Tensor types actually present (`gguf_table.py`, `q36.tsv`)

`Qwen3.6-35B-A3B-Q5_K_M.gguf`, 753 tensors, 25.0 GB without the MTP layer. By class (GB): routed experts 22.86, dense 1.38, head
0.42, embedding 0.35.

| type | where |
|---|---|
| Q5_K | `token_embd`; all `ffn_gate_exps` and `ffn_up_exps`; `attn_gate` (z) of the 30 delta-net layers; `ffn_down_exps` of 20 layers; `ssm_out` of 16 layers; `attn_q` of 6 of the 10 attention layers |
| Q6_K | `output` (head); `ffn_down_exps` of the other 20 layers; `attn_output` (10); `attn_q` of 4 attention layers |
| Q8_0 | `attn_qkv` (30), `attn_k`, `attn_v`, all shared-expert matrices, `ssm_out` of 14 layers |
| F32 | norms, router `ffn_gate_inp` (256 x 2048), `ffn_gate_inp_shexp`, `ssm_alpha/beta/conv1d/a/dt/norm` |
| BF16 | only in the extra MTP block 40 (router, shared gate); decode does not run it |

No Q4_K, no F16. The type of a matrix varies by layer (down experts Q5_K or Q6_K, `ssm_out` Q8_0 or Q5_K, `attn_q` Q5_K or Q6_K): every
kernel and the loader must dispatch per tensor, not per name.

## 3. Kernels and their verification

**CPU dequantisation** (`lb-gguf`, K-quant block code existed; added BF16). `scripts/qwen/dequant_check.sh`: llama.cpp's own
`ggml_get_type_traits()->to_float` (libggml-base of the fleet's llama-master build) and `lb_gguf` dequantise **every row of every
tensor** and hash the f32 bit patterns per row; the two files are byte-identical (753 tensors, 33,680,751 rows, 269 MB of hashes) [measured].
Bit-exact, no tolerance.

**Device kernels** (`lb-q4-gpu`, `src/kquant.rs`, `tests/q5k_kernels.rs`; run on a P100, CUDA with and without plane ops, and wgpu):

| kernel | what | layout |
|---|---|---|
| `q5_k_matvec_kernel` | dense Q5_K matvec | repacked: 5-bit quants widened to bytes, per-16 group `d*sc` and `dmin*mn` as f32 (1.5 B/value, 2.2x the file) |
| `q5_k_native_matvec_kernel` | dense Q5_K matvec | the file's own 176-byte blocks, no repack (0.69 B/value) |
| `q5_k_routed_gate_up_silu_kernel` | routed expert gate+up+SiLU, experts concatenated | widened |
| `q5_k_routed_matvec_kernel`, `q6_k_routed_matvec_kernel` | routed expert down (the file has both) | widened |
| Q6_K and Q8_0 dense | existed (LFM2.5); re-verified here on this file's tensors | widened |

Tolerance and results [measured, P100 AF, max |gpu - cpu|]: the test bound is `1e-4 (1 + |want|) + 1e-5 cols` per element (f32
summation order); routed layers, which chain two reductions through SiLU, are compared as max error over max output.
* Real rows (3000 rows each): Q5_K `attn_gate` 2.6e-6, `token_embd` 1.5e-6, expert gate 1.6e-6; native layout the same (2.6e-6,
  1.5e-6, 1.5e-6); Q6_K `output` 4.1e-6, `attn_q` 3.1e-6, expert down 4.8e-7; Q8_0 `attn_qkv` 3.5e-6, shared gate 1.9e-6, `ssm_out` 5.0e-6.
* Routed expert FFN, real layers 20 and 21 (down Q5_K) and 4, 7, 39 (down Q6_K), 8 routed experts of the first 64: max error 7e-9 to
  2e-8 on outputs of 3e-2 to 1e-1 (ratio 1.4e-7 to 2.3e-7). Random-weight layers (both down types): ratio 6e-8 to 2.3e-7. Same numbers
  with `LB_NO_PLANE=1` and on wgpu.
* Not tested: all 256 experts of a layer in one buffer. On wgpu a first attempt with all 256 experts failed with a device read error (the likely cause is wgpu's
  per-buffer size limit: a layer's widened gate matrix is 256 MiB, the whole expert set about 1.2 GB; not diagnosed further). The test therefore uses 64 experts, which runs on both runtimes.

**Layout finding (matters for 1d).** Widened Q5_K/Q6_K puts this model at about 50 GB on device (script in the ledger): it cannot
fit two 16 GB cards. The native-layout kernel keeps 25 GB. Throughput on one P100 for a 16384 x 2048 Q5_K matvec [measured]:
widened 148 us (340 GB/s of resident bytes), native 506 us (46 GB/s): the native kernel as written is scalar byte extraction, one
lane per 32-value group, and 3.4x slower. It is correct, not tuned.

## 4. The architecture, from the GGUF keys and the reference (`crates/lb-qwen35`)

`general.architecture = qwen35moe`. Keys used (all read at load, `Config::from_gguf`): `block_count` 41 (40 trunk + `nextn_predict_layers` 1,
the MTP block is not run), `embedding_length` 2048, `attention.head_count` 16, `head_count_kv` 2, `key_length` = `value_length` 256,
`rope.dimension_count` 64, `rope.dimension_sections` [11,11,10,0], `rope.freq_base` 1e7, `layer_norm_rms_epsilon` 1e-6, `expert_count`
256, `expert_used_count` 8, `expert_feed_forward_length` 512, `expert_shared_feed_forward_length` 512, `ssm.conv_kernel` 4,
`ssm.state_size` 128, `ssm.group_count` 16, `ssm.time_step_rank` 32, `ssm.inner_size` 4096, `full_attention_interval` 4, vocabulary 248,320.

Structure [read from qwen35moe.cpp and checked numerically, section 5]:

1. Layer `i` is attention when `(i+1) % 4 == 0` (10 layers: 3, 7, ..., 39), otherwise a gated delta-net block (30 layers).
2. Every layer: `x += mixer(rms_norm(x, attn_norm))`, then `x += moe(rms_norm(x, post_attention_norm))`. RMS norm: `x / sqrt(mean(x^2)
   + eps) * w`, weights as stored (no `1 + w`).
3. **Attention layer.** `attn_q` outputs `[head][q(256) | gate(256)]` for 16 heads; K and V 2 heads of 256. Per-head RMS norm of q (`attn_q_norm`)
   and of k (`attn_k_norm`), then rotary on the first 64 of 256 dims, NeoX pairing (`i`, `i+32`), `theta_i = pos * 1e7^(-2i/64)`, accumulated in f32
   as ggml does. The M-RoPE sections only matter for image positions: for text all four position components are equal, so the four
   sections produce one angle [read from `ggml_mrope_cache_init`; confirmed by the position-0 and later agreement]. Softmax attention, scale
   1/16, GQA group 8 (head h uses kv head h/8), then `out *= sigmoid(gate)`, then `attn_output`.
4. **Gated delta-net layer** (state per v-head: a 128 x 128 matrix; plus a 3-column conv state of 8192 channels). `qkv = attn_qkv x` (8192 =
   q 2048 | k 2048 | v 4096), `z = attn_gate x` (4096), `beta = sigmoid(ssm_beta x)` (32), `g = softplus(ssm_alpha x + ssm_dt.bias) * ssm_a`
   (`ssm_a` is stored already as `-exp(A_log)`). Causal depthwise conv of width 4 over [3 stored columns | new column] with `ssm_conv1d` ([4, 8192],
   kernel index fastest), then SiLU, state keeps the last 3 columns. q and k per head L2-normalised as `x / sqrt(sum x^2 + eps)`
   (= llama.cpp's `rms_norm(eps/n) * 1/sqrt(n)`), q scaled by 1/sqrt(128). Recurrence per v-head h, using q/k head `h % 16` (ggml
   repeat tiling, not `h / 2`): `S *= exp(g); d = (v - S k) * beta; S += d k^T; o = S q`. Then per-head RMS norm of `o` with `ssm_norm`
   (128), times `silu(z)`, then `ssm_out`.
5. **MoE.** Router logits `ffn_gate_inp x` (F32, 256), softmax over all 256, top 8 by probability, weights renormalised to sum 1 (no extra
   scale: `expert_weights_scale` absent). Expert `e`: `down_e(silu(gate_e x) * up_e x)`, rows `e*512..` of the 3-D tensors. Shared expert: the
   same with the `*_shexp` matrices, multiplied by `sigmoid(ffn_gate_inp_shexp . x)` (a scalar), added to the routed sum.
6. Head: `output_norm`, then `output` (Q6_K, 248,320 x 2048). Embedding is a Q5_K row of `token_embd`.
7. **Tokenizer**: `gpt2` byte-level BPE, pre-tokenizer `qwen35` (the regex in `llama-vocab.cpp`), built at load from the GGUF token and merge
   tables with the `tokenizers` crate; control and user-defined tokens are matched in text. Identical ids to `llama_tokenize` on 10 prompts (plain
   text, Rust code, French/English/Japanese, chat markers, `<think>`) [measured].
8. **Chat template**: the GGUF's Jinja template. Text-only turns implemented (`chat.rs`), generation prompt `<|im_start|>assistant\n<think>\n\n</think>\n\n`
   for no-thinking or `<think>\n`; verified against a jinja2 render of the file's template on three conversations. Not implemented: tools, tool
   responses, images.

## 5. Correctness numbers

Method: 5 texts of 71, 141, 102, 147, 109 tokens (570 positions): plain prose, Rust code, two chat-formatted turns (one with a think block), a multilingual
paragraph. Reference: llama.cpp built from the fleet's source tree (CPU, Release, this machine), full logits at every position (`ref_logits`, `libllama`).
This decoder: `qwen35-check`, f32 activations, one token at a time. Data: `docs/qwen-results/`.

| measure | value |
|---|---|
| top-1 agreement | **561 / 570 = 98.42 %** (gate 99.5 %: **not met**) |
| max logit difference (max over vocabulary, per position) | mean 0.82, median 0.67, worst 6.76 |
| mean KL(reference \|\| ours) | 0.00485 (median 0.00064, worst 0.51) |
| the 9 disagreements | reference top-1 margin 0.004 to 0.29 logits in eight; one with margin 2.23 (a Japanese passage) |
| greedy text, 5 chat prompts x 48 tokens | 4 identical token for token; 1 identical for 17 tokens then a different token (near tie) |

Why this is not a bug, and what was checked (each [measured]):

* **llama.cpp does not agree with itself better.** Same file, same machine, its own prompt-processing path (one batch) against its own
  token-by-token decode path: top-1 **97.37 %**, mean KL **0.00636**, mean max logit diff 0.83 (570 positions, `ref_tf1.f32`). This
  decoder against the batched reference is better than that on all three. A 99.5 % gate cannot be met against one reference path
  by a decoder whose rounding differs, and I do not claim it; the numbers above are what there is.
* **The cause of the gap is the reference's 8-bit activation rounding.** llama.cpp's CPU kernels quantise the activation vector to Q8_K (K-quants) or
  Q8_0 before each dot. With `LB_QWEN_ACTQ=1` this decoder emulates that rounding (`act_quant_roundtrip`) and
  the layer-by-layer comparison (`qwen35-hidden` against llama.cpp's graph callbacks, `ref_hidden.c`) at token 0 gives, relative L2:
  layer 0 mixer 3.2e-7, layer 1 4.4e-7, layer 2 2.5e-7, MoE outputs 3.9e-7 to 5.4e-7 (f32 noise), and the first attention layer 5.3e-7 once the K/V are also
  rounded to f16 (`LB_QWEN_KV16=1`, llama.cpp's default cache type). Without the emulation the same tensors differ by 1.0e-2 to 1.8e-2 at layer 0. So the
  delta-net recurrence, the conv, gating, routing, the shared expert and the first attention layer are the reference's maths to f32 precision.
* After that, agreement degrades layer by layer (1e-2 by layer 7) at positions where K/V history exists: ggml's CPU flash attention converts the query to the K type (f16) before the dot [read, `ops.cpp`: `q_to_vec_dot`] and accumulates in its own order; I did not emulate that, and
  I did not check whether flash attention was on in the reference run (the build's default). A rounding boundary flipped in either implementation is amplified, which is the
  same discontinuity that makes llama.cpp disagree with itself. On the 180 positions of two prompts, the emulated run gives top-1 98.89 % against 97.22 % without it and KL
  0.0078 against 0.0094 (the emulation helps but does not close the gap, for the attention reason above).

Honest reading: the structure is right (exact through three layers and the first attention layer, greedy text identical on 4 of 5 prompts); the
numerical gate as specified is not met and could not be met by changing this code short of copying the reference's rounding including
its flash-attention.

CPU speed of this check decoder [measured, order of magnitude]: about 1 token a second on 48 threads of t440 once the file is in page cache
(26 GB), 0.74 tok/s on the first, cold pass.

## 6. 1d: speed against Nauti-Strata and llama.cpp

* llama.cpp (fleet build `20261111-nautiburn-scrape-v2tp`), Qwen3.6-35B-A3B Q5_K_M, cards AF + D8 (`-ngl 99 -ts 1,1 -fa on`, c4, nothing else on them),
  256 new tokens x 5 prompts, temperature 0: **54.1 tok/s decode** on every prompt, prompt processing 91-113 tok/s [measured, `llama2card.sh`].
* Nauti-Strata, same two cards, reported in the on-demand report: 53.1 tok/s.
* This engine on two cards: **not run**. The decoder above is CPU only; the device pieces are the kernels in section 3, not a layer
  wired into the E2 cache and the multi-card placement. Saying plainly what the kernels allow:
  [estimate] a token reads 2.51 GB of weights (file density: 8 of 256 experts a layer, dense 1.4 GB, head 0.4 GB). At the widened kernel's 340 GB/s that is 7.4 ms
  (135 tok/s) if the data were resident, but widened does not fit; at the native kernel's measured 46 GB/s it is 54 ms (18 tok/s). A native kernel at
  the widened one's bandwidth would give about 7 ms of weight traffic a token; with 40 layers of launches and syncs the E2 work puts the practical figure at
  roughly 10-12 ms, i.e. 80-100 tok/s on two cards, which would beat 54 but is not measured and depends on the native kernel being tuned.
  The one-card expert-cache route [estimate]: 12 GB of experts resident of 22.9 GB, misses fetched over the bus; the LFM2.5 numbers (30.6 tok/s at 5 GB of 12.9) scale
  badly here because 256 experts of 0.7 MB each give a lower hit rate per resident byte; 15-25 tok/s is the range I would expect.
* It is slower than llama.cpp today because it does not run on the GPU yet.

## 7. 1e: manifest and demand entries

None added. `scripts/seat_demand.py` starts whatever binary a manifest entry names and waits for `/health`; there is no `lemonade-burn-server` that can load
this family, so a disabled entry would only describe a command that cannot succeed. Add `hard-qwen36-own-c4` (kind `spawn`, `enabled:false`, port 8118, two cards AF + D8)
and the `demand` class candidate when the device path serves this family; the entry shape is `hard-lfm25-c4` with the model path, `--ctx`, and a card list.

## 8. Part 2: the other families, in order

Facts from the GGUF headers on disk (`gguf_table.py`).

1. **Tiel-Coder-35B-A3B UD-Q5_K_XL (26.6 GB)**, almost free. Same architecture string and identical values for every key above. Types: Q8_0
   (shared experts, attention, `ssm_out`, and the expert down matrices of 4 layers), Q6_K (down experts of 36 layers, and gate/up of one layer), Q5_K, F32;
   no MTP block. New work: routed kernels for Q8_0 down and Q6_K gate/up (the routed kernels are per type; make the type a comptime parameter): about 1 day with
   tests; loader per-tensor dispatch already exists in the CPU decoder. Unlocks the same card plan as Qwen3.6 (two P100s resident, or one with the cache) and the
   `tiel-c4` seat on our own engine. Checks run with the Qwen3.6 code unchanged [measured]: every row of every tensor dequantises bit-identically to llama.cpp (`dequant_check.sh`); the CPU decoder loads and runs it; the
   GGUF tokenizer gives the same ids; layer-by-layer against llama.cpp's graph (activation rounding emulated) layers 0 and 1 mixers and layer 0 MoE agree to 3e-7, then the layer-1 MoE output differs by 3.9e-4
   (a rounding boundary flipped somewhere in its Q6_K/Q8_0 expert path; not chased) and the difference grows (2.4e-3 at layer 2). Greedy text against llama.cpp on 2 chat prompts: first divergence at
   generated token 7 of 48 (a synonym: 30334 vs 30599) and at token 0 (two openings of the same sentence). So the structure carries over; I have no teacher-forced gate for Tiel and its greedy
   agreement is worse than Qwen3.6's 4 of 5 prompts, two prompts being too few to say how much.
2. **Qwen3-Coder-Next IQ4_XS (42.8 GB; IQ3_M 36.6 GB)**, `qwen3next`: 48 layers, 512 experts top 10, `feed_forward_length` 5120, same delta-net/attention hybrid
   (full-attention interval 4, 36 + 12 layers, same ssm keys), plain NeoX rope, vocabulary 151,936. Differences from Qwen3.6 [read from `qwen3next.cpp`]: beta and alpha come from one
   fused `ssm_ba` matrix (Q8_0, per-k-head interleaving: reshape `[2 * n_v/n_k, n_k]`, beta first), no IMRoPE, different expert counts. Types: IQ4_XS for all experts, `attn_qkv`,
   `attn_gate`, `token_embd`; Q8_0 for shared experts (half of layers IQ4_XS), `ssm_ba`, `ssm_out`; Q6_K head. New work: IQ4_XS dense + routed kernels (CPU dequant exists in
   `lb-gguf`; device needs the 16-entry table kernel, the engine has one for IQ4_NL) about 2 days; decoder deltas about 1 day; the placement: 42.8 GB over **three** P100s (48 GB less KV and
   context: tight, about 3 GB a card free), a layer pipeline across three cards is the engine's missing multi-card piece and the reason I order it after Tiel. Unlocks: a 80B-class coder resident on three c4 cards (the
   on-demand report's untried candidate); CPU-only on t440 is possible now once IQ4_XS dot products are in the CPU decoder.
3. **GLM 5.3 Flash 3.0-bit (113.6 GB)**, `glm5-next`. Correction to the brief: "GSQ-RCO" is a quantisation recipe, **not** a new tensor format. Every tensor is a stock ggml type:
   Q2_K for the routed experts of 36 layers (85.6 GB), Q3_K and Q4_K for 6 layers, Q4_K attention/dense/shared/head, Q8_0 embedding, BF16 for MLA/KDA
   projections, F32 for norms and the indexer. Q2_K/Q3_K/Q4_K CPU dequant exists in `lb-gguf` (needs the same bit-exact check); device kernels for Q2_K, Q3_K, Q4_K and BF16 matvec are
   needed: about 3 days. The architecture is the large part: 45 layers, 4096 wide, 64 heads; 34 layers are linear (KDA: kimi delta attention, per-channel gate,
   `ssm_f_a/f_b`, `ssm_g_a/g_b` low-rank projections, three separate convs for q, k, v, `kda.head_dim` 128), 11 layers are MLA (q_lora 1536, kv_lora 512, `attn_k_b`/`attn_v_b`
   BF16 per-head, a sparse **indexer** with top-k 2048 over a 4-pooled key sequence); every layer has 4 hyper-connection streams with a Sinkhorn normalisation (20 iterations);
   3 leading dense blocks then MoE with 288 experts, 8 used, sigmoid-type gating (`expert_gating_func` 2) with a routing bias (`exp_probs_b`), weights scale 2.5, 1 shared
   expert, SwiGLU clamp 10. Plan: CPU reference decoder first (about 8-10 days, the hyper-connections and the indexer have the least public description), then the
   host expert tier. **Tier**: 113.6 GB over five P100s is 80 GB of VRAM; the 85.6 GB of experts need a host-RAM tier (c4 has 117 GB RAM, so dense + hot experts on the cards, the rest
   in pinned RAM over the E2 miss path) plus the NFS stream for the first touch (the on-demand report: 18-22 minutes cold). The E2 cache and CPU-miss paths
   exist; what is missing is a miss source larger than one machine's file mapping. Estimate 4-6 days on top, and the measured Nauti-Strata 10 tok/s warm is the bar. Unlocks:
   the largest quality step on our hardware, on five c4 cards plus RAM.
4. **DeepSeek V4.1 Flash (72 GB experts IQ2_XXS in 6 shards, plus the 110.6 GB Engram table)**, `deepseek41`: 40 layers, 5120 wide, 64 heads with 1 kv head (MLA-like, key length 512), 192
   experts top 6 (gating func 4, weights scale 1.5), YaRN rope (factor 16, 65k original), sliding window 128, per-layer compression ratios (0/2/1) and indexer top-k 512
   with source-layer sharing, grouped output projection (8 groups, rank 1024), the same hyper-connections, PLE/n-gram embedding tables (`ple.*`, `engram.*`, the 110 GB table
   is looked up per token and is the host-tier problem in its purest form: random 4-byte-index reads from a 110 GB file, so NFS is out and it must sit in RAM or on local NVMe).
   IQ2_XXS/IQ1_M kernels (CPU exist, device need lattice tables: about 3 days). This is a new architecture with several mechanisms found nowhere else (compression
   ratios, candidate-block indexer, PLE); a CPU reference alone is a multi-week piece and the 16k-context limit of the Nauti-Strata branch hints at unfinished parts of the reference. Do after GLM.
5. Mistral Large 4 (1T-A52B, UD-IQ1_S, about 290 GB): needs the expert tiers across machines (RDMA helper exists for one model at 0.29 tok/s) and IQ1_S; not worth starting before GLM's tier exists.

Order of value per day: Tiel (1 day, same cards), Qwen3-Coder-Next (about 1 week incl. three-card pipeline), GLM 5.3 (about 3 weeks), DeepSeek V4.1 (more), Mistral (not planned).

## 9. What remains of Part 1 (in this order)

1. Tune the native Q5_K kernel (vector loads, one block per lane group): target the widened kernel's 340 GB/s; then the same for Q6_K/Q8_0 native layouts (Q6_K 210-byte blocks are
   2-byte aligned only: read as u16). Without this the all-resident two-card run does not fit (widened) or is slow (native).
2. Device decode for the delta-net layer (conv + recurrence + gated norm as launches, state resident: 30 layers x 2 MB of state), attention layer with the
   f16 or f32 KV, the MoE layer with the routed kernels, shared expert, router top-8 over 256 (the existing `router_topk` is written for 64 and a bias).
3. Multi-card placement: the 26 GB over two P100s as a layer split (the engine's `open_cuda` opens in-box cards; the layer pipeline does not exist).
4. Wire the family into `lb-server` (architecture dispatch is LFM2-only), `/v1/chat/completions` template, then the gate again on the device path against the CPU decoder (this one), where
   f32-order agreement (not llama.cpp's rounding) is the right standard: the CPU decoder here is the oracle.
5. Then the manifest entry and demand class (section 7).

## 10. Continuation (same day): native Q5_K tuned

The native-layout Q5_K kernel was rewritten: one lane per 64-value chunk, `Vector<u32,4>` loads of the header, `qh` and `qs` (a 176-byte block is 11 aligned
16-byte vectors), four-wide x vectors, with the activation row permuted inside every 16 values by a small launch (`perm16_kernel`, folded into `launch`; callers sharing one x across
matrices can permute once and use `launch_permuted`). [measured, P100 AF, 16384 x 2048, includes the permute launch]

| layout | us per matvec | bytes on device | bandwidth of resident bytes |
|---|---|---|---|
| widened | 149.0 | 50.3 MB | 338 GB/s |
| native, before | 505.7 | 23.1 MB | 46 GB/s |
| **native, now** | **144.5** | 23.1 MB | 160 GB/s |

The native kernel is now as fast as the widened one at 46 % of the memory (the target was within 2x of the widened kernel's bandwidth: it exceeds it in time). Correctness unchanged: the same tests
(random shapes, real `attn_gate`, `token_embd`, expert rows) pass, max error 1.4e-6 to 2.4e-6 on real rows. That removes the capacity objection to Q5_K on two 16 GB cards.

Not done from the coordinator's follow-up list, stated plainly: native Q6_K and Q8_0 kernels (210- and 34-byte blocks are only 2-byte aligned: they need u16 loads, a different design from the vector
one above), the device delta-net / gated-attention / MoE layers, the N-card layer pipeline, `lb-server` dispatch, the measurement against llama.cpp's 54.1 tok/s, the manifest and demand entries, the Tiel run, and the
`LB_WARMUP` first-long-prompt cost. None of them was started; the speed gate (>= 54 tok/s) is therefore neither met nor measured. The revised gates (device against the CPU decoder at top-1 >= 99.9 %; against
llama.cpp no worse than its own 97.37 % / 0.0064) are stated here as the acceptance tests for that work; the 99.5 % gate against a single llama.cpp path stands as not met, for the reason in section 5
(its own two paths agree at 97.37 %).

## 11. Step A: native-density Q6_K and Q8_0

Kernels `q6_k_native_matvec_kernel` / `q8_0_native_matvec_kernel` (`kquant.rs`, `GpuQ6KNative`, `GpuQ8Native`, test `tests/native_q6_q8.rs`). Their blocks (210 and 34 bytes) are not 16-byte
aligned, so at upload the bytes are split into aligned arrays with no change to any stored value or the byte count (Q6_K: 192 bytes `ql`+`qh` as 12 vectors, 16 i8 scales as one vector, f16 `d` two to a word;
Q8_0: 32 quants as 2 vectors, f16 `d` two to a word). Activations are permuted like Q5_K. [measured, P100 AF, 16384 x 2048, permute launch included]

| type | existing kernel | native kernel | file size |
|---|---|---|---|
| Q6_K | 143.9 us, 41.9 MB (widened) | 163.2 us, **27.5 MB** | 27.5 MB |
| Q8_0 | 72.5 us, 35.7 MB | 140.0 us, 35.7 MB | 35.7 MB |

Q6_K: native costs 1.13x the time of the widened kernel at 66 % of the memory (within the 2x target). Q8_0: the existing kernel is already at file density and is 1.9x faster, so the native Q8_0 kernel is correct but
has no use; **keep `GpuQ8Mat` for Q8_0.** Correctness [max |gpu - cpu|]: real Qwen3.6 rows (3000 each, 2 tokens) Q6_K `output` 4.1e-6, `attn_q` 3.1e-6, expert down 4.8e-7; Q8_0 `attn_qkv` 3.7e-6, shared gate 1.9e-6, `ssm_out` 5.1e-6
(same as the existing kernels); random shapes within the f32 summation tolerance. With Q5_K native (section 10), every dense and routed type of this file now has a file-density kernel
(routed Q5_K/Q6_K experts still use the widened layout; native routed variants follow with the MoE layer in step B).

## 12. Step B: the device layers, checked against the CPU decoder

Code: `lb-q4-gpu/src/qwen_ops.rs` (norms, f32 matvec, the delta-net pieces, partial-rotary gated attention with KV append, router, MoE combine), `kquant.rs` (`GpuNativeExperts`: all 256 experts of a layer at file
density, gate+up+SiLU in Q5_K, down in Q5_K or Q6_K as the file has it, routes read from a device buffer), `lb-qwen35/src/device.rs` (`DevModel`: one layer = rms norm, delta-net or gated attention, residual, rms
norm, router + top-8 on the device, routed experts, gated shared expert, combine, residual; no host round trip inside a layer). Test `lb-qwen35/tests/device_layers.rs`, feature `cuda`, run on c4 card AF.

Method: the CPU decoder (f32 activations, section 4) runs 24 real tokens (first text of `tf_ids.txt`) and records every layer's input, post-mixer residual and output. **Isolated:** each device layer is fed the CPU's own input
for that layer, token by token, its state and KV cache evolving on the device, outputs compared. **Chain:** the device runs a block of consecutive layers from the CPU's input of the first, with its own states, compared with the CPU
at every layer. Errors are max |device - cpu| over the 2048-wide outputs and tokens, and relative L2 (worst token). [measured]

| layers (kinds) | isolated: layer output max abs / rel L2, worst | chain: after the last layer |
|---|---|---|
| 0-11 (9 delta-net, 3 attention, down experts Q5_K and Q6_K both present) | delta-net 9.5e-6 / 1.2e-6; attention 3.8e-6 / 9.7e-7 | 1.1e-5 / 1.5e-6 (12 layers) |
| 14-25 (9 delta-net, 3 attention) | delta-net 1.7e-6 / 7.6e-7; attention 9.5e-7 / 7.5e-7 | 2.2e-6 / 1.1e-6 (after layer 25) |
| 28-39 (9 delta-net, 3 attention, incl. the last layer) | delta-net 5.7e-6 / 8.7e-7; attention 1.7e-5 / 3.7e-6 (layer 31) | 9.5e-6 / 2.0e-6 (after layer 39) |

Per layer the mixer output alone (before the MoE block) is at the same level (rel L2 3e-7 to 3.8e-6). Layers 12, 13, 26, 27 were not in a block; 36 of 40 layers were run. Error does not grow along the chain (rel L2 stays
1e-6 to 3e-6 over 12 layers): it is f32 summation order and device `exp`/`sin`/`cos`, as in the dense kernels. The router's top-8 choice agreed with the CPU's at every token and layer tested (a differing expert would have shown as
an error of the size of an expert output, 1e-2 or more). The tests assert isolated rel L2 < 1e-3 and chain < 2e-2.

Not part of this step: the embedding and the LM head on the device, the layer pipeline across cards, any use of this from `lb-server`, speed (these launches run one layer at a time with scratch allocations and a host read per layer
in the test; no throughput claim is made). Step B passes; Step C (N-card pipeline, whole model on two P100s, A-B-B-A) is next.

## 13. Step C: the whole model on two P100s, served by `lb-server`

**Layer check, the four missing layers.** Layers 12, 13 (delta-net) and 26, 27 (delta-net, attention) isolated against the CPU decoder: relative L2 6.8e-7 / 5.0e-7 and 3.8e-7 / 4.5e-7 (max abs 3.8e-6 at worst). All 40 layers have now been checked.

**What was built.** `lb-qwen35/src/pipeline.rs`: a general N-stage pipeline. Stage `i` opens CUDA device `i` of `CUDA_VISIBLE_DEVICES` (the cards are chosen by UUID by listing the UUIDs there in stage order), owns a contiguous block of layers
(`QWEN_SPLIT=20,20`, default even), the last stage also the output norm and head (Q6_K widened or Q8_0, by the file). One token: the embedding row is dequantised on the host (bit-exact, one 8 KB row) and written to stage 0; a stage's activation
(`n_embd` f32, 8 KB) is read from its card and written to the next at the boundary; nothing else crosses cards and nothing is read back inside a layer (the router, top-8 and expert dispatch run on the device from a routes buffer). Each stage is recorded as one CUDA graph
on its second token and replayed (the engine's graph protocol; a failed capture falls back to plain launches), which also takes the host launch cost off the critical path (the ENGINE_SYNC_COST lesson: with graphs the host issues a stage in 0.5 ms; without graphs
launches are flushed to the device thread after every layer). Device kernels beyond step B: `GpuNativeExperts` generalised to the types of both files (gate/up Q5_K or Q6_K, down Q5_K, Q6_K or Q8_0), norm kernels that write the activation both natural and in the permuted order the native kernels read.
The server loads the model on its own worker thread (graphs are recorded on, and replayed from, one thread's stream), `lemonade-burn-server --model <qwen35moe.gguf>` dispatches on `general.architecture` (feature `qwen`), ChatML without BOS, `enable_thinking:false` closes the think block, stops `<|im_end|>`/`<|endoftext|>`.
Known limits of the served path: one sequence at a time on the device (no prompt-prefix reuse unless a request extends the kept conversation exactly), prompt tokens run one at a time (about 55 tok/s prefill), logits for the last position only.

**Speed work** (device time, one P100, CUDA-graph replay of repeated calls; decode on two cards in-process, 71-token prompt then 63 tokens). The first working pipeline ran at 33.3 tok/s; a graph made no difference (0.5 ms of host issue a stage against 14 ms of device time), so the limit was device time:

| part (per layer, delta-net layer) | first version | now |
|---|---|---|
| delta-net core (conv, L2, recurrence, gated norm) | 123 us (uncoalesced state rows, one thread a row) | 32 us (a plane a row, lanes along the row) |
| router top-8 | 77 us (one thread) | 21 us (one cube, rank counting) |
| routed experts (gate+up+down) | 205 us | 114 us (multi-row planes for the 512-wide down, gate+up writes permuted) |
| in-process decode, two cards | 33.3 tok/s | **55.8 tok/s** |

**Correctness gates** (570 positions, the five texts of section 5, device = both cards, graphs on) [measured]:

| gate | result |
|---|---|
| device vs CPU decoder, top-1 | **570 / 570 = 100 %** (gate 99.9 %) |
| device vs CPU decoder, max logit difference | 7.4e-4 (mean 2.1e-5), mean KL 2.9e-12, worst 1.5e-10 |
| device vs llama.cpp (batch path), top-1 | 98.42 % (561/570), mean KL 0.00485 |
| llama.cpp batch vs its own single-token path, same run | 97.37 % (555/570), mean KL 0.00636 |
| device greedy text, 5 prompts x 48 tokens | identical to the CPU decoder on all 5 (4 identical to llama.cpp, one diverges at token 18 as the CPU decoder does) |

The revised gates are met: against the CPU decoder 100 % and 7.4e-4; against llama.cpp the device is no worse than llama.cpp's own self-agreement on both top-1 (98.42 vs 97.37) and mean KL (0.00485 vs 0.00636). The original 99.5 % gate against a single llama.cpp path stands as not met, for the reason in section 5.

**A-B-B-A on c4 cards AF + D8** (nothing else on them; 5 prompts, 256 new tokens each, temperature 0, no-think, streamed, client-side tok/s after the first token; order ours, llama.cpp, Nauti-Strata, Nauti-Strata, llama.cpp, ours; per-run median of the five) [measured, `abba.sh`, `abba_client.py`]:

| engine | run 1 | run 2 |
|---|---|---|
| lemonade-burn (`20261115-qwen35-f73da05`, through `lb-server`) | 53.6 | 53.9 |
| llama.cpp (fleet build 20261111) | 54.0 | 54.0 |
| Nauti-Strata (`hard-qwen36-2card-c4` config) | 63.0 | 63.1 |

Where it lands: **parity with llama.cpp, 0.5 % below it (53.75 vs 54.0 through the server; 55.8 in-process on the bench prompts, which run shorter contexts), and 15 % below Nauti-Strata**, which on these prompts measured 63.0-63.1 tok/s, not the 53.1 of the earlier thinking-mode run: the 53.1 figure in the on-demand report is not comparable (different prompt, thinking text). The
speed gate (>= 54, matching llama.cpp) is missed by a hair through the server. What limits it: all of the remaining time is device time, about 18 ms a token (layers 0.4 ms each, of which the routed experts 0.11-0.12 ms, the two attention-layer projections 0.14 ms, the dense projections and shared expert 0.1 ms, the head 2.3 ms), and the kernels are bound by integer decode of 5-6-bit quants
(shifts, masks, casts per value: 4 G values a token) as much as by memory (the Q5_K native matvec reaches 160 GB/s of resident bytes on a 732 GB/s card). Nauti-Strata's llama.cpp-derived kernels quantise the activation to int8 and use integer dot products (dp4a), which does that work 4 values at a time; the next step for this engine is the same
(an int8-activation dp4a path for Q5_K/Q6_K/Q8_0, the `Q4_MATH_DP4A` idea of the Q4_0 kernels, checked against the f32 path as the existing kernels are). The server adds about 2 ms of host work a token over the in-process bench (stream, sampler, text decode; the device idles meanwhile) and a prompt-speed of 55 tok/s because prompt tokens are run one by one.

**Manifest.** Added, disabled: engine `hard-qwen36-own-2card-c4` (port 8118, AF + D8 by UUID in stage order, new build, model Qwen3.6-35B-A3B Q5_K_M, `--ctx 8192`, `expect_model qwen3.6-35b-a3b-own`) and demand class `hard-max-own` (candidate = that engine on cards AF + D8). `demand.enabled` stays false; nothing enabled, no seat touched. Backup `var/fleet-manifest.json.bak-20261115-qwen-c`.

**The Part 0 leftover (first long prompt after boot 8-9 s).** Measured on the LFM2.5 seat entry, first 419-token request: `LB_WARMUP` 0: 7.0 s, 8: 7.8 s, 96: 8.8 s, 512: 13.9 s, 1024: 24.0 s. So it is not a missing one-time kernel compile that a longer warm-up could pay: the first request costs about 7 s whatever, and each warm-up token adds ~10-20 ms to it
(the warm-up's own long run is paid again at the first request, which points at per-sequence device state or checkpoint release in the first request after a warm-up sequence). Not fixed; cause not isolated; a longer warm-up makes it worse. The same server shape on the Qwen path shows a 4.8-5.3 s first request (about 25 prompt tokens) against 0.6-0.8 s afterwards, so it is a property of the server's first request, not of the LFM2 kernels.

Repeat server runs of the same client after the A-B-B-A (build `20261115-qwen35-26dc5f2`, the build the manifest entry names): 52.8 while c4's load average was 10 (a neighbour job), 54.5 on a quiet c4 (median of five prompts, 54.1-54.7). So through the server it sits between 53 and 54.5 tok/s depending on c4's host load,
which is the llama.cpp figure (54.0) within noise, not above it with any margin; the in-process figure is 55.8. The manifest entry points at `26dc5f2`.

## 14. Tiel-Coder-35B-A3B (UD-Q5_K_XL) through the same path

Same architecture string and keys; the file's types differ (section 8): gate/up Q5_K except Q6_K in layer 1, down Q6_K except Q8_0 in layers 1, 34, 38, 39, shared experts and attention Q8_0, head and embedding Q8_0. Code added for it: routed Q6_K gate+up kernel, routed Q8_0 down kernel (`q8n_routed_down_mr_kernel`), Q8_0 head in the pipeline.
No change to the decoder or the layer code. [measured]

* Layers isolated against the CPU decoder (all four of its Q6_K/Q8_0 expert layers included, layer blocks 0-11, 14-25, 28-39): relative L2 <= 1.4e-6 for delta-net and attention layers except attention layer 31 (8.6e-6, max abs 4.0e-5), chains of 12 layers <= 1.1e-5 (after layer 34).
* Teacher-forced, same 570 positions, two cards, graphs on: **device vs the CPU decoder 570 / 570 = 100 %**, max logit difference 3.2e-4 (mean 2.4e-5), mean KL 1.2e-11. Device vs llama.cpp (batch path): 95.09 % (542/570), mean KL 0.01503; llama.cpp batch vs its own single-token path (same run): 94.56 % (539/570), mean KL 0.01783. The device is no
  worse than llama.cpp's own self-agreement on both measures. Tiel agrees less with llama.cpp than Qwen3.6 does (95.1 % vs 98.4 %), and llama.cpp agrees less with itself on it by about the same amount (94.6 % vs 97.4 %): the fine-tune has more near-tie positions, the cause is the reference's own rounding, as for Qwen.
* Speed: in-process 65.6 tok/s, through `lb-server` 63.5-64.2 (median 63.7) on AF + D8 with the same five prompts (cheaper than Qwen3.6 because its attention and shared experts are Q8_0, which has the fast kernel). llama.cpp and Nauti-Strata were not run on Tiel.
* No manifest entry was added for Tiel (the on-demand report's `tiel-c4` standing seat is the one-card spill setup; a two-card own-engine Tiel entry is the same shape as `hard-qwen36-own-2card-c4` with the Tiel file and model id, to add if wanted).

## 15. Step D: decode speed (2026-11-16): 53.7 -> 69.5 tok/s through `lb-server`, past Nauti-Strata

**Why Nauti-Strata was faster: what the source and the measurements say.** Read: `src/kernels/cuda/native_mmvq.cu`, `include/strata/kernels/dp4a.hpp`, `docs/OLDER_GPUS.md`, `docs/MMVQ_IL_TABLE.md`.
* The P100 is compute capability 6.0 and has **no `dp4a`** (that arrived with 6.1). Strata's K-quant decode kernels are llama.cpp's `vec_dot_*_q8_1`: the activation is rounded to int8 blocks (q8_1: int8 values, per-32 scale and sum), the weights stay packed and are unpacked
  four values at a time with word masks (`& 0x0f0f0f0f`), and the integer dot is `strata_dp4a`, a software dp4a made of **four PTX `vmad.s32` byte-select instructions** (bit-exact with a real dp4a, "2.2x faster than byte extraction" in their check). The "interleaved" decode matvec (`native_mmvq_il`, `MMVQ_IL_TABLE.md`) reads
  an interleaved copy of the q8_1 activations and takes **1, 2 or 4 matrix rows per warp**, so one activation read serves several rows. That path is off for sm_75 and older per that document; the plain mmvq runs on the P100.
* Strata's engine log has no timing lines in the build on disk (`STRATA_DECODE_TIMING` produced nothing in `hard-qwen36-c4-engine.log`), and Nsight Systems on c4 refuses CUDA 12.8 ("not installed with CUDA Toolkit 12.8"), so I could not time Strata's layers; what I compared is what its source does against what mine did, and I tested each difference on our kernels.
* **Tested here [measured, P100]:** (1) the arithmetic is not the limit. Word-wide SWAR quant assembly plus an exact int-to-float by `(byte | 0x4B000000) - 2^23` (two integer ops and an add instead of a quarter-rate convert) left the Q5_K matvec at 148 us (before 148 us) and Q6_K at 154 us (before 163): no gain. The earlier X7 work agrees: on this card f32, f16 and the dp4a polyfill
  run the Q4_0 matvec at 310 / 320 / 314 GB/s. An activation-rounding int8 path (what Strata and llama.cpp do) would not have helped for that reason, and I did not build it: **every path in this engine still uses f32 activations, no rounding switch was added or needed.** (2) What mattered was **activation reloads and parallelism**: one plane per row re-read 256 bytes of activations
  for every 80 bytes of Q5_K weights. Four rows a plane that load the activations once: Q5_K 148 -> 81 us (286 GB/s of resident bytes, from 156), Q6_K 154 -> 118 us, Q5_K 1.8x faster than the widened layout with 46 % of its memory (R=8 gains nothing more; in the real model it was slower). (3) in the real graph the routed-expert kernels were short of parallel work: 1024 planes
  for the 8 x 512-row gate/up pass. Two gate+up rows a plane and one down row a lane group (more planes in flight): in-process 61.0 -> 67.8 tok/s with no other change. (4) kernel count and chain length: the delta-net core as one launch instead of five (conv + L2 + scalars + delta rule + gated norm; conv state double-buffered by position parity so the cubes that share a q/k head
  never read a column another has shifted), wide attention decode (a plane a position with coalesced K reads, `p = e/sum` stored once), argmax on the device for greedy requests (4 bytes back instead of 1 MB of logits), norms that write the activation both natural and in the permuted order the native kernels read (no permute launches), router top-8 by rank counting in one cube.
  In-situ ablation (skipping a part in the CUDA graph, timing only) put the routed experts at 5.2 ms of a 16.4 ms token, the dense projections 4-5 ms, the delta-net core 1.5-1.8 ms, the shared expert 1.2-1.5 ms, the head 1.8 ms at that point. A fused shared-expert (gate+up+SiLU, down folded into the combine) built and verified but was slower (61.1 vs 62.6): kept opt-in (`LB_QW_SHF=1`), off.

**Correctness** (the path stays exact: f32 activations; only summation order changes):
* Layers against the CPU decoder, all 40 of Qwen3.6 (blocks 0-11, 12-25, 26-39) and Tiel-Coder layers 0-11 and 28-39, isolated and in 12-layer chains: relative L2 <= 1.4e-6 (attention layer 31 of Qwen 3.7e-6; Tiel attention layer 31 8.6e-6), chains <= 1.5e-6 (Qwen) / 1.0e-5 (Tiel, after layer 39); max abs <= 5.3e-5. Same level as before the speed work.
* Whole model, 570 teacher-forced positions, two cards, graphs on: see the table below.

**A-B-B-A on c4 AF + D8** (same client, five prompts, 256 tokens, T=0, no-think, streamed; order ours, llama.cpp, Nauti-Strata, Nauti-Strata, llama.cpp, ours; build `20261116-qwen35-6b421ab`) [measured]:

| engine | run 1 | run 2 |
|---|---|---|
| lemonade-burn through `lb-server` | **69.7** | **69.3** |
| llama.cpp (fleet build) | 53.9 | 54.0 |
| Nauti-Strata | 63.1 | 63.1 |

The speed gate (>= 63 through the server) is met: **69.5 tok/s, 10 % above Nauti-Strata and 29 % above llama.cpp** on these cards and prompts (in-process 67.8). The first request of a server still takes 5-6 s (step F).
Defaults in code: `LB_QW_R` 4 (dense rows a plane), `LB_QW_GU_R` 2, `LB_QW_DN_R` 1; `LB_QW_ROWS1=1` turns every row-batched kernel off, `LB_QW_NOGU`, `LB_QW_NODN` the expert ones, `LB_QW_ATTN_OLD` the attention one, `LB_QW_SKIP=experts,shared,core,qkv,z,out,aq,dense` is a timing ablation (wrong results).

Whole-model gates on the final build (570 positions, the five texts, both cards, graphs on, argmax path not used by the check) [measured, `d_tf_stepD.jsonl`]:

| gate | result |
|---|---|
| device vs CPU decoder, top-1 | **570 / 570 = 100 %** (gate 99.9 %) |
| device vs CPU decoder, max logit difference | 2.8e-4 (mean 1.9e-5), mean KL 2.7e-12, worst 1.2e-10 |
| device vs llama.cpp (batch path), top-1 / mean KL | 98.42 % (561/570) / 0.00485 |
| llama.cpp batch vs its own single-token path, same run | 97.37 % (555/570) / 0.00636 |

Identical to before the speed work to the digit on the llama.cpp comparisons. Tiel-Coder-35B through the same server: 75.3-76.3 tok/s (median 75.8, was 63.7). The manifest entry `hard-qwen36-own-2card-c4` now names build `20261116-qwen35-6b421ab` (backup `var/fleet-manifest.json.bak-20261116-qwen-d`).

# Card profiles: per-card tuning as data

Branch `fable/engine-fleet-2026-11-21`. Crate `crates/lb-card-profile`, profiles in `crates/lb-card-profile/profiles/*.json` (one file per card family, compiled in).

## How it works

- At device open (`lb_q4_gpu::Gpu::open*`) the engine probes the card: CUDA compute capability and name from the driver, or PCI vendor:device and adapter name for a Vulkan adapter (the existing `card_facts` sysfs code), and matches a profile.
- A profile is JSON: `match` (CUDA cc list, PCI vendor + inclusive device-id range, adapter-name substrings; all stated groups must hold; `match` may be an array of such objects, any of which may hold, e.g. PCI id or else adapter name), `knobs` (`NAME: {value, source}`), `notes` (measured findings that are not a value), `not_measured`.
- The most specific profile (most criteria groups) wins. A profile with no criteria matches nothing.
- **Precedence per knob: environment variable > profile > code default.** A knob is named by its env var, so existing seat commands keep working and still override.
- Unknown card: no profile, code defaults, one line `lb-card-profile: [label] name: no card profile matched, code defaults (sm_90)`.
- `LB_CARD_PROFILE_DIR=/dir` adds or replaces profiles (by `id`) from `*.json` files without a rebuild. A bad file or a knob that is not profile-capable is a warning on stderr; the built-ins stay.
- Only knobs marked `profile: true` in `lb_card_profile::KNOBS` are read through a profile (wired at the call sites below). Others still work through env.
- Show it: `lb-card-profile` (bin in lb-q4-gpu) or `lemonade-burn-server --print-card-profile` (gpu builds). For each visible card: the matched profile, every knob's value and its source `default | profile | env`, the profile's note for the value. Probes only; no device or context is created, safe next to a running seat.
- A multi-card pipeline (lb-qwen35) takes the smallest value any stage card's profile gives (the most constrained card decides); env still wins.

## Knob inventory (card-relevant)

"Read at" is the code that reads it. "Profile" means read through the card profile. Source = where the value was measured.

| Knob | Card families | Value | Read at | Profile | Source of the measurement |
|---|---|---|---|---|---|
| `LB_QW_PREFILL_CHUNK` | P100 (sm_60) | 256 (default auto 64..1024) | lb-server `qwen.rs::prefill_chunk`, via `Pipeline::knob_min` | yes | `~/agentnotes/lb-qwen35-c4-tuning-2026-11-16.md` Step 0: 2xP100 + Tiel Q5_K_XL, auto chunk OOMs on the second long prompt, 256 stable, 410 tok/s prompt. Measured before the Q8 dedup (b96fa22); the need is VRAM headroom, so it matters only when weights nearly fill the card |
| `LB_QW_GRAPH` | P100 | 1 (= default) | lb-qwen35 `pipeline.rs` | yes | c4 tuning note (76 tok/s decode with graphs); `c1-gemini-lessons-for-engine-2026-11-18.md` s1 (graphed verify 25.1 -> 21.5 ms) |
| `LB_QW_FLUSH` | none set | 1 | lb-qwen35 `pipeline.rs` | yes | not measured |
| `LB_MTP_DEPTH` | P100 | 1 (= default) | lb-server `qwen.rs::load`, via `Pipeline::probe_stage_profiles` | yes | `c1-gemini-lessons...` s1, commit 4ab5e29: depth 1 +16 %, depths 2-3 lose (verify of 2 rows 21.5 ms vs 15 ms a token). Used only when `LB_MTP=1` |
| `LB_QW_ATTN_SPLIT` | P100, RTX 2060 (sm_75 profile) | 1 (default 0) | lb-q4-gpu `Gpu::open` -> `KqTune`, `qw_attention` | yes | `docs/P100_KERNEL_TUNING.md`: one-token attention in three launches over many cubes, bit-identical; 2xP100 decode at ~4.2k ctx 44.6 -> 61.7 tok/s; 2060 attention core n=4096 532 -> 98 us |
| `LB_QW_Q6_LG` | P100, RTX 2060 (sm_75 profile) | 1 (default 0) | lb-q4-gpu `Gpu::open` -> `KqTune`, `GpuQ6KNative::launch_permuted` | yes | `docs/P100_KERNEL_TUNING.md`: Q6_K rows of 16 half blocks in lane groups, bit-identical; P100 head 1675 -> 1093 us, short decode 69.8 -> 71.4 tok/s |
| `LB_QW_ATTN_CUBES` | none set | 112 | lb-q4-gpu `KqTune` | yes | not tuned (split scores kernel grid) |
| `LB_MTP` | P100 (measured on) | off | lb-server `qwen.rs` | no | per-seat operator decision; +14/-2/+15/+16 % (code/prose/primes/4k) in the 11-18 note |
| `QWEN_SPLIT` | P100 pair 19,21; 2060+2xP100 6,17,17 | per seat | lb-server `qwen.rs`, qwen35 bins | no | c4 tuning note. Depends on model + card set, so it is seat placement (manifest), not a card property |
| `LB_PREFILL_MAX_ROWS` | Vega 10 (MI25 / WX 9100) | 128 | lb-q4-gpu `Gpu::open` -> `max_batch_rows`; lb-lfm2 `device_prefill.rs::ubatch_for_gpu` | yes (replaces the hard-coded `family_limits`) | `card_facts.rs` header; `c1-gemini-lessons...` finding F and s2 (86abbc4): chunks over 128 rows hung the compute ring under llama.cpp (operator + claude-t440-env1, 2026-11-16). Our own prefill on the MI25 not run |
| `LB_PREFILL_UBATCH` | P100 | 2048 (= default) | lb-lfm2 `device_prefill.rs::ubatch_for_gpu` | yes | `docs/PREFILL_SPLIT.md`: 780 vs 594 tok/s at 2048 vs 512 on a P100 |
| `LB_Q4_ROWS` | P100, GTX 1070 (sm_61) | 4 (= default) | lb-q4-gpu `Gpu::open` | yes | `docs/WORK_LOG.md` 2026-11-07: 1.29x end to end on a P100; 1.16x wgpu / 1.31x cuda on the GTX 1070 |
| `LB_Q4_MATH` | all | f32 (default kept) | lb-q4-gpu `Gpu::open` (`Q4Math::pick_from`) | yes | `docs/WORK_TO_WEIGHTS.md` X7a: f16 on sm_60 1.03x and fails the accuracy gate; dp4a fails; f16 on P106 (sm_61) 10 vs 90 GB/s. Never `f16` or `auto` on sm_61 |
| `LB_H2` | P100 | off | lb-q4-gpu `Gpu::open` | yes | `docs/PREFILL_SPLIT.md`: 930 vs 780 tok/s but top-1 211/215, fails G1; not adopted |
| `LB_NO_TC` | Turing (2060, 2060 SUPER, T10) | off (tensor cores are probed from the runtime's cmma list + plane 32) | lb-q4-gpu `Gpu::open` | yes | T10 tensor-core prefill 1352 / 1736 / 1598 tok/s at 512 / 2048 / 8192, `docs/PREFILL_SPLIT.md` G4 |
| `LB_TC_STAGE2` | T10 | off | lb-q4-gpu `gemm.rs::tc_stage2` | no | one block per stage 8584 / 10168 GFLOP/s vs two 7338 / 8775 (`PREFILL_SPLIT.md`) |
| `LB_Q4_CUBE` | none | 2 planes | lb-q4-gpu `Gpu::open` | yes | measured flat 32..256 on a P100 (commit message), no gain |
| `LB_ROUTER_SPLIT_ANY` | RDNA2 (6700 XT, wave32, plane 32..64) | 1 (= default) | lb-q4-gpu `decode.rs::router_split_ok` | yes | `docs/ENGINE_VULKAN.md` 10.2 (126f07b): 14.18 -> 16.25 tok/s on 2 cards, 1590/1590 identical |
| `LB_DD_FAST` | Polaris (RX 570 / WX 5100) | on | lb-lfm2 `knobs.rs` | no | `ENGINE_VULKAN.md` 10.2: router split over 64 cubes 5.09 -> 5.36 tok/s |
| `LB_NO_PINNED` | CUDA cards | off (pinned pool on) | lb-q4-gpu `Gpu::open` | no | 8.3 vs 10.8 s load, sha unchanged (fa2859e) |
| `LB_NO_REGISTER` | P100 | off (register on) | lb-lfm2 `moe.rs` | no | `cuMemHostRegister` miss path 0.28 ms vs 1.1 ms, 1.40x (memory gpu-zero-copy-miss-path, eec4500) |
| `LB_NO_PLANE` | any | off | lb-q4-gpu `Gpu::open` | no | switch only |
| `CUBECL_WGPU_RING_MB`, `_RING_THREADS`, `_POLL_CALLER` | Polaris RX 570 | off by default; 64 / 4 / 1 | vendored `vendor/cubecl-wgpu` (stream creation) | no (outside this layer) | `ENGINE_VULKAN.md` 10.2: 3.95 -> 5.38 tok/s |
| tiled prefill attention | all | always on (code) | lb-q4-gpu `prefill.rs`, lb-qwen35 | n/a | bb0f1cf (17-20x on a P100), 3771e5f (6k prompt 113 -> 16 s) |
| power cap | MI25 | logged at open, not acted on | `card_facts.rs` | n/a | LFM2-2.6B tg128 107 / 124 / 129 tok/s at 110 / 140 / 170 W (`c1-gemini-lessons...` finding E) |

Families with nothing measured beyond the above (Polaris RX 570 / WX 5100, Quadro P1000, T10 beyond tensor-core prefill, RTX 2060 qwen35 stage knobs, MI25 / 6700 XT decode on our engine) carry empty `knobs` and a `not_measured` list. No tuned value was invented.

## Other env knobs (not card-specific; unchanged)

Placement / cache (lb-server `main.rs`, lb-lfm2 `stage.rs`, `knobs.rs`): `LB_EXPERT_CACHE`, `LB_EXPERT_CACHE_SEED`, `LB_EXPERT_CACHE_STAGING`, `LB_EXPERT_VRAM_MIB`, `LB_PRED_*`, `LB_LA_*`, `LB_E2_*`, `LB_CPU_MISS_*`, `LB_HOST_*`, `LB_NO_BATCH_MISS`, `LB_HOST_REPACK`. Device selection: `LB_CUDA_DEVICE`, `LB_WGPU_DEVICE`, `CUDA_VISIBLE_DEVICES`. Prefill: `LB_PREFILL_CARD`, `LB_PREFILL_CARD_MB`, `LB_PREFILL_LAYERS`, `LB_PREFILL_GEMM_MIN`, `LB_DEVICE_PREFILL`. qwen35 kernels: `LB_QW_R`, `LB_QW_GU_R`, `LB_QW_DN_R`, `LB_QW_ROWS1`, `LB_QW_NOGU`, `LB_QW_NODN`, `LB_QW_ATTN_OLD`, `LB_QW_SKIP`, `LB_QGEMM_V4`, `LB_QWEN_ACTQ`, `LB_QWEN_KV16`, `LB_QW_PREFILL_MIN`. Server: `LB_THINKING`, `LB_WARMUP`, `LB_PROMPT_CACHE*`, `LB_TRACE_REQ`. CPU: `LB_NUMA*`, `LB_KERNEL`, `LB_TASKS_PER_THREAD`. Fabric: `LB_RDMA_GID`. Diagnostics: `LB_*_LOG`, `LB_*_TIMING`, `LB_*_TRACE`, `LB_SCRATCH_LOG`. These can be promoted into the registry one at a time: add a `KnobDef` with `profile: true` and read it with `gpu.knob(name)` instead of `std::env::var`.

## Adding a measurement

1. Measure on the card (A/B against the env-var setup, same model).
2. Add `"LB_X": {"value": ..., "source": "<note or commit>"}` to the family's JSON (or drop a file in `LB_CARD_PROFILE_DIR` to try it).
3. `lb-card-profile` shows it as `profile`; an env var of the same name still overrides it.

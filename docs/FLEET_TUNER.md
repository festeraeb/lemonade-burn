# Fleet tuner (`lb-tune`)

Operator ask, 2026-11-07: a tuner that suggests splits and layouts across a mixed fleet,
because choosing them is one of the hardest entry points for people using LLMs.

`lb-tune` (crate `lb-calibrate`, module `lb_calibrate::fleet`) takes a fleet description and a
model and prints a layout: which cards hold which layers, how many experts stay resident per
card, how big the RAM tier and each disk tier are, which node acts as expert helper over which
link, the expected tokens/s with a range, and which quant files fit at which depth. It builds
on the existing calibrate pieces (`sim.rs` pipeline simulator, `timings.rs` measured card
timings, `tune.rs` staging config); nothing there was redone. No GPU is touched: it is
arithmetic plus the simulator.

## Run

```
cargo build --release -p lb-calibrate --bin lb-tune
lb-tune --model /mnt/raid0/models/lfm25/LFM2.5-24B-A2B-QAD-Q4_0.gguf \
        --manifest /codebase/repos/cesaropsserver/var/fleet-manifest.json \
        --matrix /var/lib/nauti/fleet-matrix.json \
        --overlay crates/lb-calibrate/examples/fleet-today-overlay.json \
        [--bench NODE:GPU=bench.jsonl ...] [--scrapes DIR] [--explain] [--out plan.json]
```

* `--model` is a GGUF (first shard is enough: siblings `-0000N-of-0000M.gguf` are found).
  Only headers are read (lb-gguf), never tensor data. `--model-json` takes a description
  instead (`--dump-model` writes one).
* Fleet: `--fleet fleet.json` (hand-written), or `--manifest` (+ `--matrix`) to build it, then
  any number of `--overlay` files merged over it. `--dump-fleet` writes the result.
* `--node X`, `--helper Y|none`, `--variants`, `--top N` filter what is printed;
  `--explain` prints the reasoning, the measured-vs-estimated basis and the rejected layouts.
* `--out plan.json` writes the best layout: the existing staging-config shape
  (`tune::StagingConfig`, an old reader still parses it) plus a `layout` object.

### Knobs (fleet rule: tunable variables, logged at start)

Printed on stderr as `knobs: {...}` at every run. Defaults and their sources:

| flag | default | source |
|---|---|---|
| `--reserve-mib` | 1024 per card | the reserve `lb-calibrate tune` already uses |
| `--ctx` | 4096 | estimate: a typical agent turn (sizes the KV cache) |
| `--ram-reserve-frac` | 0.30 (min `--ram-reserve-min-gib` 8) | the 80 GiB RAM tier of 117 GiB used on c4 for Le Chonk |
| `--handoff-us` | 250 | measured 129-599 us across cards (V2_PIPELINE.md s3) |
| `--cpu-thread-frac` | 1.0 | all node threads may run CPU experts |
| `--synth-tokens` | 64 | tokens of synthetic routing replayed through the simulator |
| `--skew` / `--scrapes` | `profile` | see "Routing skew" |
| `--requant-gb-h` | unset | no measured quantize rate exists, so the hours line says unknown |

## What it reads

**Model** (GGUF): layers, leading dense layers, experts per layer, top-k, bytes per routed
expert (from the `*_exps.*` tensors), non-expert bytes per layer, head + embedding bytes,
element counts, largest tensor, hidden width, KV bytes per token for the plain layout
(`head_count_kv` x head dim x K,V x f16). For layouts without a plain KV (MLA, e.g. Le Chonk)
the KV size is `null` and the per-card reserve has to cover it. IQ types that lb-gguf cannot
dequantize are sized from ggml block geometry (a table in `model.rs`).

**Fleet** (`load.rs`): `fleet-manifest.json` gives nodes, GPU pins and compute capability;
the nauti fleet matrix gives what each box reports now (RAM, logical CPUs, GPUs with VRAM,
RDMA/RoCE ports, ICMP latencies per path, interface speeds). Where they disagree the live
matrix wins for the GPU list and the manifest supplies compute capability by PCI address.
Anything neither source knows (PCIe link, a node missing from the matrix, disk tiers,
measured link rates) stays `null`, is listed under `unknown fleet fields`, and is only filled
by an overlay file, whose notes carry the source of each number
(`examples/fleet-today-overlay.json`, `examples/fleet-shed-2026-11-08-overlay.json`).
Disk tiers are in neither source: `kind` optane|ssd|hdd|nfs, `bytes`, `read_mb_s` (null = listed
but unused), and exactly one tier with `holds_model: true` is where non-resident experts finally
come from.

Fleet JSON shape (see the examples): `nodes[{name, ram_gib, cpu_threads, gpus[{name, vram_mib,
arch, compute_cap, pcie_gen, pcie_width, has_dp4a, fp16_rate, mem_gb_s, available}], tiers[...]}]`,
`links[{a, b, kind: roce|ethernet, gb_s, latency_us, source}]`. `available: false` keeps a card
out of every layout (cards pinned to seats; the tuner does not read seat state).

## How it decides

1. **Candidates**: each node as primary, each prefix of its cards (largest VRAM first), with no
   helper or any linked node that has GPUs as expert helper.
2. **Layer split** (`split_layers`): layers are dealt in contiguous blocks in proportion to free
   VRAM (VRAM - reserve). That gives every layer the same number of resident expert slots, the
   thing that starved the last card in the 12,24,35 Le Chonk split (92 experts vs ~31 per layer
   elsewhere). A card that cannot hold its share of dense layers + KV is rejected with the
   numbers; a layout whose dense layers fit nowhere ends as "DOES NOT FIT. Options:" with the
   arithmetic (VRAM short by X GiB, dense-on-CPU if RAM could hold it, smaller quant).
3. **Fill order for experts**, hottest first per layer: local VRAM, helper VRAM, RAM tier, disk
   tiers by read speed, then the model file. The head/embedding stays in host RAM (the engine
   runs the head on the CPU).
4. **Speed** (`evaluate`): the resident sets go to `sim.rs` on a synthetic routing trace drawn from
   the skew curve; helper experts are modelled as resident plus one round trip per MoE layer
   (latency + 2 x hidden x 4 B over the link); RAM-tier misses are CPU experts (already in
   the simulator); disk-tier and model-file misses add `bytes / read rate`. Reported:
   * **expected**: chosen skew, warm tiers, reads not overlapped with compute;
   * **low**: uniform routing (no skew at all), same cost;
   * **high**: chosen skew, reads overlapped (max over devices instead of the sum);
   * **cold start**: RAM and disk tiers still empty, every non-resident expert read from the file
     (the first requests after a load).
5. `sim.rs` got two backward-compatible fields: `expert_bytes` (default: the LFM2.5 expert) and
   `layer_card` (explicit contiguous layer map; default keeps the round-robin deal).

## Measured vs estimated

| input | status | source |
|---|---|---|
| model shapes and bytes | exact | GGUF headers |
| GPU names, VRAM, RAM, threads, link latency | reported by the fleet | fleet matrix |
| P100 no dp4a, 1070 dp4a + fp16 1/64, P100 full fp16 | verified facts | fleet-gpu-inventory.md |
| PCIe: Gen3 x16 ~12 GB/s, x8 ~6 | measured anchors, linear in lanes | V2_PIPELINE.md |
| RoCE t440<->c4: 2.84 GB/s read, 8 us (4 KB) | measured | Le Chonk notes; applied to any 25G RoCE link |
| SAS SSD ~300 MB/s | measured ceiling | Le Chonk notes (overlay) |
| CPU expert 0.396 GB/s/thread, 12.3x parallel speedup | measured (c4, reference path) | V2_PIPELINE.md s3; c3 showed no speedup at 8 threads |
| GPU memory bandwidth per model | **estimate** | vendor specs, `arch.rs` (T10 and the 6700 XT/WX 5100 ids are the least certain) |
| kernel efficiency 0.22 of that bandwidth | **estimate** | derived from the two c3 per-expert measurements (0.20, 0.24); kernel-version dependent |
| per-card expert/dense ms when no bench | **estimate** | bytes / (bandwidth x 0.22); with `--bench NODE:GPU=file` the card's own measured effective rate replaces it |
| Optane 900 MB/s for the 13 GB stick | **estimate** | 3.5 GB/s RAID0 of 4 modules / 4; the stick itself is unmeasured |
| Optane ~80 GB at 3.5 GB/s (shed plan) | **estimate** | planned board, not installed |
| routing skew `profile` | measured on ONE model | Le Chonk pin plan: per-layer top-48/64/96 of 128 leave 24/11/1.8 file reads of 284 uses |
| helper start-up time, prefill, multi-stream | **not modelled** | listed as UNKNOWN in each plan |

Not built: dense layers pipelined across nodes (reported as an option when nothing fits),
tier files on the helper node, per-layer non-uniform residency, prefill. A card without a table
entry has unknown dp4a and bandwidth and is priced as infinitely slow, i.e. never wins.

### Routing skew

Without scrapes the tuner assumes Le Chonk's measured curve for every MoE model, which is
optimistic for a prompt mix that differs from that pin plan. `--scrapes DIR` replaces it with
the curve measured from NautiBurn traces; `--skew uniform` is the pessimistic bound. The low
figure always uses uniform.

## Quant fit and "remake the quant"

The model is repriced at the pure block-geometry floor of each name (UD-IQ1_S as IQ1_S 1.5625
bpw, IQ2_XXS 2.0625, Q4_0 4.5, Q4_K_M as Q4_K 4.5, Q8_0 8.5, F16 16) and placed on the best node:
`all-resident`, `resident+RAM`, `resident+RAM+tier`, `streaming (model-file reads)` or
`does not fit`. Mixed files (Unsloth UD, `_M`) are larger than the floor, so only the "as-is"
row (the file in hand) is exact. A card without int8 dot is named in the row: its Q4/Q8/IQ kernels
dequantize to fp16 (full-rate on the P100) or fp32 before the multiply; on the 1070 (dp4a yes,
fp16 1/64) the plan says to use the dp4a/fp32 path. When a smaller quant reaches a shallower level
than the file in hand, one line recommends it with the resources a remake needs (F16 source size
on disk, output size, RAM for the largest tensor, hours at the rate you pass in). It does not
quantize anything and does not estimate quality.

## Results (2026-11-07, fleet matrix of the day + overlay)

(a) LFM2.5-24B-A2B Q4_0 (12.6 GiB, 64 experts x 5.1 MiB, top-4): fits one P100 all-resident
(experts 64/layer). Expected ~15.7 tok/s on any P100 node (estimate: no bench file used; the
low/high band collapses because nothing misses). On c1's 6700 XT with a RAM tail: ~14 tok/s.
Layers 0-39 on one card; the tuner finds no reason to span cards.

(b) Le Chonk UD-IQ1_S (217 GiB, 72 layers, 128 experts x 22.5 MiB, top-4), today's fleet: dense
15.8 GiB does not fit one card, best is c4 four cards (split 20,20,18,14, 24-25 experts/layer
resident) + c2 as helper over 1 GbE + 80 GiB RAM + 13 GiB Optane: ~1.1 tok/s warm (low 0.18,
cold start 0.36); "streaming" because 217 GiB exceeds VRAM + helper + RAM + Optane.
With t440 as helper (the measured configuration: 1,349 vs 1,338 helper experts, 60 GiB RAM) the
tuner says 0.87 warm, low 0.12, cold start 0.22 against the **measured 0.143-0.154 tok/s**
(le-chonk-polystrata, 2026-11-07). The measured number sits between the low bound and the
cold-start figure (1.5x under the cold-start estimate): the run was a cold-tier probe on a prompt
the pin plan was not built for, so do not read the warm figure as a promise. After the shed move
(5 P100 at x8/x16 + 80 GB Optane at 3.5 GB/s, plan only): ~1.4 warm, low 0.56, cold 0.39.

## Tests

`cargo test -p lb-calibrate` runs the fleet tests on synthetic fleets (one card; four P100s
on one node; two nodes with a RoCE link and helper; a card that cannot hold the dense weights
-> split; a model that fits nowhere -> options; tier ordering with unknown speeds; a synthetic
GGUF read; manifest + matrix merge with unknowns; quant ordering and remake line) together with
the existing sim/timing/tune tests.

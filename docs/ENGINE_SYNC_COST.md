# What a host sync costs in the E2 path, and what removes most of it

Branch `fable/engine-123-sync-2026-11-15` (off 2aebf56). Card: Tesla P100 (UUID suffix D8) on c4, NUMA 1, 20 threads, LFM2.5 24B,
the `cm.py` prompts (5 x 288 greedy tokens), A-B-B-A, same binary for every arm (switches only). [measured] unless tagged.
Raw: `sync-results/` (`micro/`, `prof/`, `runs2/`, `abba2-s1-*.progress`).

## 1. The floor the hardware allows (standalone CUDA, `micro/rt.cu`, 10,000 rounds, same card)

One trivial kernel, then read 1.2 KiB back (us per round trip, mean):

| path | us |
|---|---|
| launch call only | 2.6 |
| launch + streamSynchronize, no copy | 7.0 |
| (a) pageable `cudaMemcpyAsync` + sync | 12.1 |
| (b) pinned + sync | 9.8 |
| (c) pinned + async + event query spin / event sync | 9.8 / 9.8 |
| (d) mapped zero-copy write + host flag spin, no sync call | 7.2 |
| blocking-sync context flag, (a)/(b) | 30-32 |

Chain of 38 layers x (router kernel, read, 8 kernels of 25 us): 215.7 us a layer with no read, 226.5 with a pinned read, 228.6 with a
pageable one. So a sync costs **about 11-13 us a layer** at the hardware (0.45 ms a token), and the choice of pageable / pinned / event
spin / zero-copy is worth 2 us. The default context already spins (80 logical CPUs); blocking-sync would cost 50 us more.
The 0.21 ms we pay is 17x the floor and none of the read-back variants explains it.

## 2. Where the 0.21 ms goes (`LB_E2_PROFILE`, now split into read_begin / read_finish / post-read)

12500 MiB (hit ~100 %, 0 misses), 777 tokens, per MoE layer, means:

| part | us | what it is |
|---|---|---|
| host busy before the read | 97 | the client building the layer's ~11 launches (CubeCL client side is ~12.6 us per launch against 2.6 raw) |
| host busy after the read | 52 | planning, slot words, expert launches |
| `read_begin` | 204 | the call blocks until CubeCL's device thread has **issued every launch queued since the last read**, then enqueues the copy |
| `read_finish` | 117 | the fence: the card finishing the tail |
| layer total | 475 | (18.0 ms of the 19.9 ms token) |

Cause (cubecl-common `device/handle/channel.rs`, read, not patched): launches are tasks in a 32-slot batch queue for the device thread. A
batch is delivered when it fills or when something flushes it. The E2 layer queues about a dozen tasks and then reads, and the read
is the flush, so **the device thread issues the whole layer only at the sync**, one launch after another (about 12 us each), after the
host has already spent its 150 us building them with the card idle. Client build, thread issue and device run in series instead of
overlapping. CubeCL's host staging for the read is already pinned.

Same effect in isolation (`examples/sync_floor.rs`, K = 8 launches after the read, per layer): no read 110 us; read 219-237 us (sync
cost 110-125 us); CubeCL single launch + read round trip 54 us (raw CUDA 10).

## 3. The change

`Gpu::flush_queue()` (`lb-q4-gpu`): hand the queued launches to the device thread now, without waiting. It calls
`Client::profile_abandon` with a window that was never opened: public, non-blocking, whose documented effect is "does not wait for the
server ... but does flush"; the server removes an unknown token from a map. No CubeCL patch, no new dependency.

* `LB_E2_FLUSH=N` (knob `e2_flush`, live): flush after every Nth launch group in the E2 layer (op kernels, ffn norm, router,
  fills, experts). 0 = off (default).
* `LB_E2_FUSE=1` (knob `e2_fuse`, live): the expert weighted sum is folded into the next pre-norm's launch
  (`sum_add_rms_norm_ord`, as the all-resident path does): one launch fewer a layer.

Both only touch when and how launches are submitted. Isolated: flush after every launch takes the chain's sync cost from 110 to 38 us a layer
(every 2: 47, every 4: 85, every 8: 113).

## 4. Results (A-B-B-A, tok/s median of sessions, base = same binary with the switches off)

| budget | arm | tok/s | x base | ms/token | text vs base |
|---|---|---|---|---|---|
| all resident (14500) | base / flush+fuse | 98.47 / 98.28 | 0.998 | 10.2 | identical 6/6 |
| 12500 (cache forced, hit ~1) | base | 48.91 | 1.000 | 20.4 | |
| 12500 | flush+fuse | 72.69 | **1.486** (paired 1.32-1.49) | 13.8 | identical 6/6 |
| 5120 | base | 24.33 | 1.000 | 41.1 | |
| 5120 | flush+fuse | 30.02 | **1.234** (paired 1.232, 1.236) | 33.3 | identical 4/4 |
| 5120 | CPU misses (base) | 28.88 | 1.187 | 34.6 | identical 4/4 |
| 5120 | CPU misses + flush+fuse | 34.45 | **1.416** (paired 1.398, 1.423) | 29.0 | identical 4/4 |
| 3072 | base | 17.89 | 1.000 | 55.9 | |
| 3072 | flush+fuse | 21.51 | 1.202 (paired 1.204, 1.196) | 46.5 | identical 4/4 |
| 3072 | CPU misses (base) | 21.11 | 1.180 | 47.4 | identical 4/4 |
| 3072 | CPU misses + flush+fuse | 25.88 | **1.447** (paired 1.444, 1.444) | 38.6 | identical 4/4 |

Single-session figures before the A-B-B-A: flush alone at 12500 66 tok/s, + fuse 73, + `LB_E2_SPEC` 75 (spec no longer adds).

Correctness: teacher-forced through the E2 cache against the all-resident decode, 1590 positions (`run_agree.sh`):

| arm | cache | top-1 | max abs logit diff |
|---|---|---|---|
| flush | 5120 | 100.00 % (1590/1590) | 0 |
| flush+fuse | 5120 | 100.00 % | 0 |
| flush+fuse | 12500 | 100.00 % | 0 |
| flush+fuse | 3072 | 100.00 % | 0 |
| flush+fuse + CPU misses | 5120 | 100.00 % | 2.5e-5 (same as the CPU-miss arm alone) |

Flush and fuse reorder nothing on the card (launch timing, and a sum kernel the all-resident path already uses): bit-identical.

## 5. Gate verdict

* 5120 MiB, >= 1.25x against the default E2 path: **copy misses 1.234x: FAIL by 0.016.** With `LB_CPU_MISS=1`: **1.416x: pass.**
* Logits bit-identical (max |logit diff| 0, top-1 100 %) for flush and fuse; CPU-miss arm keeps its own 2.5e-5.
* No regression: all-resident 0.998, 12500 MiB 1.486x. 3072: 1.202x copy, 1.447x with CPU misses.
* Because the copy arm misses the gate, the switches stay **default off**. They are bit-identical and never slower in any
  measurement, so the one-line flip (knobs `e2_flush` default 1, `e2_fuse` default on) is the next commit if the operator accepts
  1.23x as enough.

## 6. What the floor implies at 5120 and 3072

At 12500 the E2 token is 13.8 ms against 10.2 all-resident: 0.09 ms a sync left (from 0.21), of which ~0.04 [estimate] is the CubeCL
issue/fence round trip (38 us measured in isolation) and the rest the extra launches the E2 layer still makes (router and norm
separate, expert gate_up and down, ~12.6 us each on the client). At 5120 the copy path is **bus bound**: 32.7 misses a token x
5.31 MB = 174 MB at 9.7 of ~11.2 GB/s, 17.9 ms of the 33.3, in series with the layer because the copy, the repack and the experts
share one stream and the routes are known only at the layer. Even a perfect sync (floor 0.011 ms x 38) would give at most
~31 ms, ~1.3x, and only if the DMA ran at full bus rate. More than that needs the bus out of the critical path:

* fewer copies (CPU misses: the combination above, 1.42x);
* copies overlapped with compute: device-side translate with a slot table and a deferred second pass for misses
  (`LB_E2_SPEC` is its first half and, bit-identical, gives nothing at 5120 because the layer that misses repeats its experts);
  a second stream for copies, with a hits-only expert launch first. [estimate] hides at most the hit-expert compute of a
  miss layer (~60 us of ~500): not worth building; a predictor (E3 recall 0.78) is the lever that remains, as the residency
  doc says.

## 7. Candidates, as asked

| candidate | result |
|---|---|
| reused pinned read-back buffer | cubecl's read staging is already pinned; raw floor says pinned vs pageable is 2 us. Nothing to gain. |
| spin-poll an event instead of synchronize | floor: identical (9.8 vs 9.8 us); default context already spins. No. |
| dedicated stream for the routes read | not needed once the launch queue is not held back; the read is cheap (55 us a layer after flush). |
| no buffer creates on the path | there are none (fills `write` into the staging ring, `const_u32` is cached; ledger L was already moot). |
| fuse sum + next norm | **done** (`LB_E2_FUSE`): +10 % at 12500 over flush alone (66 -> 73 tok/s single sessions). |
| fuse router + translate | translate exists only on the spec path; spec adds nothing now. Not done. |
| batch two layers' routing reads | layer L+1's router needs layer L's experts, which need L's routes: dependent. No. |
| early launch flush | **done** (`LB_E2_FLUSH`), the whole effect. |

## 8. Reproduce

`cargo build --release --features cuda -p lb-server -p lb-lfm2 --bins`, then on a host with the card: `LB_E2_FLUSH=1 LB_E2_FUSE=1` on the
server (or `/v1/tunables` `e2_flush`, `e2_fuse`). Floor test: `nvcc -arch=sm_60 -o rt micro/rt.cu`; CubeCL isolation:
`examples/sync_floor` (`SF_FLUSH_EVERY`). Profile: `LB_E2_PROFILE=file` now records read_begin, read_finish and post-read host us per layer.

# Engine step E3: expert look-ahead

Branch `fable/engine-e3-lookahead-2026-11-09`, off E2 `d074372` (the operator's baseline change: look-ahead is
tested on the engine we ship, device router and slot bank, 30.49 tok/s on a P100 x16 at 5120 MiB, not on E1).
Plan and gate: `~/agentnotes/engine-right-first-plan-2026-11-08.md`, step E3. Off by default: with no
`LB_LOOKAHEAD`/`LB_LA_*` set the decode path is E2's (`off` arm below, same tok/s and same text as the E2 binary).

## What it does

**The guess.** With the routes of layer L the host now also reads the residual stream before L's experts run
(`x_mid`, the same read, no extra sync) and scores layer L+1's router on it with L+1's FFN pre-norm
(`stage::predict`, exp. #1's `mid1`). `dot` is hand-laned with an AVX2 path: a plain float sum cannot be
vectorized and cost 15 ms a token, this costs 2.5-3.1 ms [measured, c4 host with socket 0 busy]. Each guessed expert gets a confidence
`score - ref_score` (`ref_score` = the best score left out of the real top-k), positive inside the real top-k.

**Keep guard (no copies).** `Slru::plan_guarded/plan_la`: a miss's victim skips experts the guess names for the
coming layer (`LB_LA_KEEP=1`, top `LB_LA_N_GUESS`, default 8, depth `LB_LA_KEEP_DEPTH`). `LB_LA_PROMOTE=1|2` also
refreshes predicted cached experts (MRU of their segment | protected). When every candidate is guarded, the
plain victim is taken: a guess never fails a pass.

**Miss-only prefetch (`LB_LA_PREFETCH_CAP`).** At layer L, after L's expert launches are queued, the guessed
experts of L+1 that are **not cached** and whose confidence clears `LB_LA_MIN_CONF` are copied (raw bytes, up to
the cap, best first) into **landing buffers** on a side stream (the DMA overlaps the main stream's compute).
Nothing is evicted and no cache slot is touched until the layer really routes the expert; then the normal
victim is taken and the slot is filled from the landing buffer by a repack launch on the main stream. A wrong
guess costs bus bytes only. That is a change from the brief's "into cache slots": the copy lands in a transit
buffer (8 x 5.31 MB, charged to the budget), the expert ends up in a cache slot, and the cache sees no pollution.
Filling a slot straight from the side stream was ruled out for a concrete reason (below).

**Cross-stream ordering, the part that took reading CubeCL.** Its dependency tracking compares the origin
stream's cursor with a per-buffer cursor that is set when the buffer is **allocated**, not when it is written.
A long-lived landing buffer (or the slot bank) written repeatedly from the side stream would therefore never
make a main-stream reader wait: silent wrong answers. So each `land` allocates a small sentinel on the side
stream after its copies (it is also the repack kernel's block-count argument, so no kernel changed); the main
stream's repack binds it, and CubeCL makes the main stream wait on the device for those copies. Writing into the
main-stream slot bank from the side stream has no such tracking and would race with the routed kernels, which
is why the landing goes through a side-allocated buffer and the repack stays on the main stream. Landing buffers
are recycled only after the sync that proves the repack ran.

**Trace and simulator.** `LB_LA_TRACE=<file>` writes one JSON line per layer pass (decode and prefill) with the
routes and the guess; `la-sim` replays it against the real `Slru` and Belady. It reproduced the engine's prefetch
numbers (cap 2: covered 18.1 vs 17.0 per token, issued 58.0 vs 57.7) and was used to pick the sweep grid.

## Results (2026-11-09, c4 Tesla P100 AF:00.0, Gen3 x16, NUMA node 1, H2D 8.5-9.6 GB/s measured)

LFM2.5-24B-A2B Q4_0 (experts 12.02 GiB, 2.40x the 5120 MiB budget, 4.01x the 3072 MiB one), 5 prompts / 288 decode
tokens a session, greedy, 8 host threads, process and page cache on the card's node. Arms interleaved, odd rounds
in order, even reversed. **n = 3 sessions per arm.** `x e2` is the per-round paired ratio to the E2 binary (median
[min-max]). c4 socket 0 carried a 40-thread photogrammetry run (load 22-42) throughout; our process is bound to
socket 1, but host-side timings such as guess ms may be inflated.

Arm names: `pfCtT` = prefetch cap C, confidence threshold T (`t01` = 0.01, `t0` = 0, ...), `k` = keep + promote1.

arm | n | tok/s med [min-max] | x e2 (paired) | token ms | read wait | hit | misses/t | on bus/t | covered/t | prefetched/t | precision | wasted MB/t | guess ms | issue ms | spared/t | syncs/t | miss layers/t | VRAM MiB
---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---
e2 | 3 | 30.45 [30.12-31.35] | 1.000 [1.000-1.000] | 32.74 | 25.02 | 0.854 | 22.20 | 22.20 | 0.00 | 0.00 | - | 0.0 | 0.00 | 0.00 | 0.00 | 39.00 | 15.18 | 6337/6443
off | 3 | 31.24 [30.88-32.01] | 1.025 [0.996-1.051] | 32.18 | 24.96 | 0.854 | 22.20 | 22.20 | 0.00 | 0.00 | - | 0.0 | 0.00 | 0.00 | 0.00 | 39.00 | 15.18 | 6337/6443
cyc64 | 3 | 27.61 [27.28-28.49] | 0.907 [0.870-0.946] | 36.41 | 25.28 | 0.855 | 22.03 | 22.03 | 0.00 | 0.00 | - | 0.0 | 2.99 | 0.00 | 0.00 | 39.00 | 15.10 | 6337/6443
guess | 3 | 28.00 [27.71-29.09] | 0.920 [0.920-0.928] | 35.93 | 25.24 | 0.854 | 22.20 | 22.20 | 0.00 | 0.00 | - | 0.0 | 2.93 | 0.00 | 0.00 | 39.00 | 15.18 | 6337/6443
keep | 3 | 27.68 [27.45-28.25] | 0.909 [0.901-0.912] | 36.33 | 25.29 | 0.854 | 22.16 | 22.16 | 0.00 | 0.00 | - | 0.0 | 2.98 | 0.00 | 0.16 | 39.00 | 15.16 | 6337/6443
keepp1 | 3 | 28.05 [27.87-28.95] | 0.925 [0.895-0.951] | 35.86 | 25.09 | 0.857 | 21.69 | 21.69 | 0.00 | 0.00 | - | 0.0 | 2.93 | 0.00 | 0.13 | 39.00 | 14.83 | 6337/6443
kpf2t01 | 3 | 29.05 [28.79-29.09] | 0.946 [0.927-0.966] | 34.60 | 23.66 | 0.855 | 22.05 | 9.69 | 12.36 | 17.39 | 0.71 | 26.7 | 2.97 | 0.12 | 0.18 | 39.00 | 8.39 | 6343/6457
kpf2t02 | 3 | 28.90 [28.65-29.11] | 0.956 [0.914-0.959] | 34.78 | 23.72 | 0.855 | 22.05 | 11.18 | 10.88 | 13.96 | 0.78 | 16.4 | 3.00 | 0.10 | 0.18 | 39.00 | 9.49 | 6343/6457
pf1t0 | 3 | 28.25 [28.14-28.71] | 0.938 [0.898-0.943] | 35.59 | 24.49 | 0.852 | 22.49 | 10.77 | 11.72 | 23.41 | 0.50 | 62.0 | 3.02 | 0.16 | 0.00 | 39.00 | 8.60 | 6343/6457
pf1t01 | 3 | 28.83 [28.76-29.02] | 0.944 [0.920-0.964] | 34.87 | 23.94 | 0.852 | 22.49 | 12.61 | 9.88 | 14.20 | 0.70 | 22.9 | 2.94 | 0.10 | 0.00 | 39.00 | 10.15 | 6343/6457
pf1t02 | 3 | 28.73 [28.57-28.88] | 0.948 [0.917-0.949] | 34.99 | 24.03 | 0.852 | 22.49 | 13.59 | 8.90 | 11.64 | 0.76 | 14.5 | 2.93 | 0.08 | 0.00 | 39.00 | 10.90 | 6343/6457
pf2t0 | 3 | 26.82 [26.65-28.25] | 0.890 [0.850-0.928] | 37.51 | 26.00 | 0.852 | 22.49 | 6.59 | 15.90 | 33.82 | 0.47 | 95.1 | 3.04 | 0.23 | 0.00 | 39.00 | 5.81 | 6343/6457
pf2t01 | 3 | 29.13 [29.09-29.14] | 0.957 [0.930-0.966] | 34.53 | 23.99 | 0.852 | 22.49 | 9.85 | 12.64 | 18.35 | 0.69 | 30.3 | 2.88 | 0.12 | 0.00 | 39.00 | 8.54 | 6343/6457
pf2t02 | 3 | 29.46 [28.93-29.69] | 0.975 [0.923-0.978] | 34.15 | 23.98 | 0.852 | 22.49 | 11.40 | 11.09 | 14.62 | 0.76 | 18.7 | 2.79 | 0.09 | 0.00 | 39.00 | 9.68 | 6343/6457
pf2t05 | 3 | 28.40 [28.04-28.65] | 0.941 [0.894-0.943] | 35.41 | 24.40 | 0.852 | 22.49 | 14.59 | 7.90 | 9.02 | 0.88 | 5.9 | 2.96 | 0.07 | 0.00 | 39.00 | 11.74 | 6343/6457
pf3t01 | 3 | 28.59 [28.36-29.07] | 0.931 [0.927-0.949] | 35.16 | 24.11 | 0.852 | 22.49 | 9.24 | 13.25 | 19.25 | 0.69 | 31.9 | 2.98 | 0.13 | 0.00 | 39.00 | 8.13 | 6343/6457
  code-03: lemonade-burn 1 sha(s) {'c5d77b6cf1b729f2': ['cyc64', 'e2', 'guess', 'keep', 'keepp1', 'kpf2t01', 'kpf2t02', 'off', 'pf1t0', 'pf1t01', 'pf1t02', 'pf2t0', 'pf2t01', 'pf2t02', 'pf2t05', 'pf3t01']} | llama.cpp shas []
  forge-04: lemonade-burn 1 sha(s) {'d7938eeacde472db': ['cyc64', 'e2', 'guess', 'keep', 'keepp1', 'kpf2t01', 'kpf2t02', 'off', 'pf1t0', 'pf1t01', 'pf1t02', 'pf2t0', 'pf2t01', 'pf2t02', 'pf2t05', 'pf3t01']} | llama.cpp shas []
  prose-02: lemonade-burn 1 sha(s) {'b89a1212137bf00d': ['cyc64', 'e2', 'guess', 'keep', 'keepp1', 'kpf2t01', 'kpf2t02', 'off', 'pf1t0', 'pf1t01', 'pf1t02', 'pf2t0', 'pf2t01', 'pf2t02', 'pf2t05', 'pf3t01']} | llama.cpp shas []
  qa-05: lemonade-burn 1 sha(s) {'48fe35aec6ccb119': ['cyc64', 'e2', 'guess', 'keep', 'keepp1', 'kpf2t01', 'kpf2t02', 'off', 'pf1t0', 'pf1t01', 'pf1t02', 'pf2t0', 'pf2t01', 'pf2t02', 'pf2t05', 'pf3t01']} | llama.cpp shas []
  sonar-32: lemonade-burn 1 sha(s) {'062392b8d2b0361a': ['cyc64', 'e2', 'guess', 'keep', 'keepp1', 'kpf2t01', 'kpf2t02', 'off', 'pf1t0', 'pf1t01', 'pf1t02', 'pf2t0', 'pf2t01', 'pf2t02', 'pf2t05', 'pf3t01']} | llama.cpp shas []

All [measured]. One text per prompt across all 16 arms and 48 sessions (sonar `062392b8d2b0361a`).

Simulator at 1003 slots (`la-sim`, whole session including warmup, so it reads lower than the engine's decode-only
counter): SLRU 0.8425, keep8 0.8429, keep8 + promote1 0.8457, cycle window 64 0.8472, **Belady 0.9433**.

### The oracle: what overlap alone can buy (n = 3 per arm, interleaved with e2)

`LB_LA_ORACLE` replays a recorded session as the guess: every expert of the next layer named correctly, at no host
cost (no residual read, no router dots). `orcC` = cap C copies per layer. Same 995-slot cache and landing buffers.

arm | n | tok/s med [min-max] | x e2 (paired) | token ms | read wait | hit | misses/t | on bus/t | covered/t | prefetched/t | precision | wasted MB/t | guess ms | issue ms | spared/t | syncs/t | miss layers/t | VRAM MiB
---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---
e2 | 3 | 30.21 [30.14-30.63] | 1.000 [1.000-1.000] | 32.96 | 25.01 | 0.854 | 22.20 | 22.20 | 0.00 | 0.00 | - | 0.0 | 0.00 | 0.00 | 0.00 | 39.00 | 15.18 | 6337/6443
orc1 | 3 | 34.84 [34.63-35.01] | 1.156 [1.131-1.159] | 28.48 | 20.65 | 0.852 | 22.49 | 7.50 | 14.99 | 14.99 | 1.00 | 0.0 | 0.00 | 0.10 | 0.00 | 39.00 | 5.64 | 6343/6457
orc2 | 3 | 36.96 [36.64-38.54] | 1.223 [1.196-1.279] | 26.94 | 20.18 | 0.852 | 22.49 | 2.23 | 20.26 | 20.26 | 1.00 | 0.0 | 0.00 | 0.11 | 0.00 | 39.00 | 1.81 | 6343/6457
orc4 | 3 | 36.27 [36.01-36.91] | 1.203 [1.176-1.222] | 27.32 | 19.44 | 0.852 | 22.49 | 0.56 | 21.93 | 21.93 | 1.00 | 0.0 | 0.00 | 0.14 | 0.00 | 39.00 | 0.37 | 6343/6457

All [measured]. A perfect free guess gives 1.13-1.28x paired (cap 2 best, median 1.22x): read wait falls from 25.0 ms to
19.4-20.7 ms. So overlap works and the headroom is real, but it is about 5.6 ms of a 13.9 ms miss cost: a covered miss
saves roughly 0.25 ms of its ~0.6 ms of bus time [derived]. A wrong guess costs about its whole bus time: the real
arms at similar coverage save 1 ms, not 4-5, and the difference is close to wasted copies x 0.55 ms [derived from the
5120 MiB table: pf2 tau .02 covers 11.1 and wastes 3.5, which predicts 11.1 x 0.25 - 3.5 x 0.55 = 0.9 ms; measured 1.04].

### Worker-thread guess, spare-slot prefetch, and the deferred pick-up (n = 3 per arm, interleaved with e2)

Set B (`w...` = guess on a worker thread, raw landing + repack on the main stream):

arm | n | tok/s med [min-max] | x e2 (paired) | token ms | read wait | hit | misses/t | on bus/t | covered/t | prefetched/t | precision | wasted MB/t | guess ms | issue ms | spared/t | syncs/t | miss layers/t | VRAM MiB
---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---
e2 | 3 | 30.32 [30.31-30.40] | 1.000 [1.000-1.000] | 32.86 | 24.99 | 0.854 | 22.20 | 22.20 | 0.00 | 0.00 | - | 0.0 | 0.00 | 0.00 | 0.00 | 39.00 | 15.18 | 6337/6443
orc4 | 3 | 36.58 [35.88-36.80] | 1.207 [1.180-1.214] | 27.37 | 19.46 | 0.852 | 22.49 | 0.56 | 21.93 | 21.93 | 1.00 | 0.0 | 0.00 | 0.14 | 0.00 | 39.00 | 0.37 | 6343/6457
orcd2 | 3 | 36.42 [36.42-37.46] | 1.201 [1.198-1.236] | 27.49 | 19.62 | 0.850 | 22.73 | 0.57 | 22.16 | 22.16 | 1.00 | 0.0 | 0.00 | 0.15 | 0.00 | 39.00 | 0.38 | 6349/6463
orcd3 | 3 | 36.21 [35.97-37.00] | 1.191 [1.186-1.221] | 27.65 | 19.62 | 0.849 | 22.90 | 0.57 | 22.33 | 22.33 | 1.00 | 0.0 | 0.00 | 0.17 | 0.00 | 39.00 | 0.38 | 6355/6469
w0 | 3 | 30.30 [30.09-30.36] | 0.997 [0.992-1.002] | 33.15 | 25.10 | 0.854 | 22.20 | 22.20 | 0.00 | 0.00 | - | 0.0 | 0.18 | 0.00 | 0.00 | 39.00 | 15.18 | 6337/6443
w2t01 | 3 | 30.20 [29.78-30.38] | 0.996 [0.983-0.999] | 33.29 | 24.01 | 0.852 | 22.49 | 9.85 | 12.64 | 18.35 | 0.69 | 30.3 | 1.94 | 0.12 | 0.00 | 39.00 | 8.54 | 6343/6457
w2t02 | 3 | 29.43 [29.42-29.54] | 0.971 [0.970-0.972] | 34.12 | 24.02 | 0.852 | 22.49 | 11.40 | 11.09 | 14.62 | 0.76 | 18.7 | 1.82 | 0.11 | 0.00 | 39.00 | 9.68 | 6343/6457
w2t02d2 | 3 | 27.29 [27.07-27.80] | 0.898 [0.893-0.917] | 36.86 | 24.89 | 0.850 | 22.73 | 9.85 | 12.88 | 20.85 | 0.62 | 42.3 | 3.94 | 0.16 | 0.00 | 39.00 | 8.50 | 6349/6463
w2t05 | 3 | 30.20 [29.34-30.50] | 0.996 [0.968-1.003] | 33.33 | 24.22 | 0.852 | 22.49 | 14.59 | 7.90 | 9.02 | 0.88 | 5.9 | 1.92 | 0.06 | 0.00 | 39.00 | 11.74 | 6343/6457

Set D (`...s` = spare-slot prefetch: DMA and repack on a side stream into a bank slot no expert maps to, the slot number
swapped when the layer really routes the expert; `d...` = worker answer picked up just before the next layer's read):

arm | n | tok/s med [min-max] | x e2 (paired) | token ms | read wait | hit | misses/t | on bus/t | covered/t | prefetched/t | precision | wasted MB/t | guess ms | issue ms | spared/t | syncs/t | miss layers/t | VRAM MiB
---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---
e2 | 3 | 30.18 [29.80-30.67] | 1.000 [1.000-1.000] | 32.99 | 25.09 | 0.854 | 22.20 | 22.20 | 0.00 | 0.00 | - | 0.0 | 0.00 | 0.00 | 0.00 | 39.00 | 15.18 | 6337/6443
d2t01s | 3 | 31.46 [31.19-31.46] | 1.042 [1.017-1.056] | 31.64 | 22.80 | 0.851 | 22.58 | 9.89 | 12.69 | 18.39 | 0.69 | 30.3 | 0.24 | 0.69 | 0.00 | 39.00 | 8.56 | 6331/6445
d2t02s | 3 | 31.12 [31.07-33.03] | 1.030 [1.015-1.108] | 31.96 | 22.94 | 0.851 | 22.58 | 11.44 | 11.14 | 14.66 | 0.76 | 18.7 | 0.25 | 0.57 | 0.00 | 39.00 | 9.70 | 6331/6445
d2t02sd2 | 3 | 30.74 [30.35-30.89] | 1.005 [1.002-1.037] | 32.44 | 22.74 | 0.850 | 22.84 | 9.90 | 12.94 | 20.93 | 0.62 | 42.4 | 1.29 | 0.76 | 0.00 | 39.00 | 8.54 | 6337/6459
d2t05s | 3 | 31.06 [31.02-31.50] | 1.028 [1.027-1.042] | 32.03 | 23.32 | 0.851 | 22.58 | 14.64 | 7.94 | 9.06 | 0.88 | 5.9 | 0.30 | 0.37 | 0.00 | 39.00 | 11.75 | 6331/6445
orcs4 | 3 | 37.60 [37.01-38.88] | 1.246 [1.242-1.268] | 26.39 | 18.80 | 0.851 | 22.58 | 0.56 | 22.02 | 22.02 | 1.00 | 0.0 | 0.00 | 0.61 | 0.00 | 39.00 | 0.37 | 6331/6445
orcsd2 | 3 | 37.77 [37.43-38.60] | 1.256 [1.232-1.279] | 26.20 | 18.16 | 0.850 | 22.84 | 0.57 | 22.27 | 22.27 | 1.00 | 0.0 | 0.00 | 0.75 | 0.00 | 39.00 | 0.38 | 6337/6459
orcsd3 | 3 | 36.93 [36.83-38.55] | 1.236 [1.224-1.257] | 26.83 | 18.56 | 0.848 | 23.08 | 0.57 | 22.51 | 22.51 | 1.00 | 0.0 | 0.00 | 0.79 | 0.00 | 39.00 | 0.38 | 6337/6467

All [measured]. Moving the guess off the critical path took host guess cost from 2.9 ms to 0.25-0.30 ms a token, and the
spare slot took a covered miss's main-stream repack away (read wait 24.0 to 22.8 at the same coverage). `d2t01s` =
1.042x paired [1.017-1.056].

### The floor and the ballast: where the rest goes (12500 MiB, every expert cached, n = 3 per arm)

`bal2` / `bal1` issue a useless copy every 2nd layer / every layer (18 / 37 a token); nothing consumes them, so there is
no fence and no dependency, only a copy competing with the main stream.

arm | n | tok/s med [min-max] | x e2 (paired) | token ms | read wait | hit | misses/t | on bus/t | covered/t | prefetched/t | precision | wasted MB/t | guess ms | issue ms | spared/t | syncs/t | miss layers/t | VRAM MiB
---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---
e2 | 3 | 50.22 [50.05-51.91] | 1.000 [1.000-1.000] | 19.52 | 12.33 | 0.998 | 0.25 | 0.25 | 0.00 | 0.00 | - | 0.0 | 0.00 | 0.00 | 0.00 | 39.00 | 0.24 | 13711/13817
bal1 | 3 | 38.97 [38.43-39.40] | 0.776 [0.740-0.787] | 25.41 | 17.39 | 0.998 | 0.25 | 0.19 | 0.06 | 37.00 | 0.00 | 196.1 | 0.00 | 1.13 | 0.00 | 39.00 | 0.19 | 13717/13833
bal2 | 3 | 43.61 [42.51-44.31] | 0.868 [0.819-0.885] | 22.65 | 15.26 | 0.998 | 0.25 | 0.22 | 0.04 | 18.00 | 0.00 | 95.4 | 0.00 | 0.55 | 0.00 | 39.00 | 0.21 | 13717/13831

All [measured]. The floor is **12.33 ms read wait, 19.52 ms token** on this card (e2 at 0.25 misses a token). A
useless copy costs the main stream **0.14-0.16 ms of read wait** (+2.93 ms for 18, +5.06 for 37), and about 0.17 ms
of token with its host issue, with nothing waiting for it: bus and engine contention, the 1.79 vs 0.96 ms that exp. #2b
measured on a read with copies in flight.

### 3072 MiB (4.01x overfill), n = 3 per arm, interleaved with e2

arm | n | tok/s med [min-max] | x e2 (paired) | token ms | read wait | hit | misses/t | on bus/t | covered/t | prefetched/t | precision | wasted MB/t | guess ms | issue ms | spared/t | syncs/t | miss layers/t | VRAM MiB
---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---
e2 | 3 | 20.68 [20.64-21.66] | 1.000 [1.000-1.000] | 48.47 | 38.99 | 0.683 | 48.20 | 48.20 | 0.00 | 0.00 | - | 0.0 | 0.00 | 0.00 | 0.00 | 39.00 | 26.47 | 4285/4391
d2t01s | 3 | 23.08 [22.98-23.38] | 1.116 [1.061-1.132] | 43.42 | 33.77 | 0.678 | 48.88 | 21.16 | 27.72 | 36.13 | 0.77 | 44.6 | 0.49 | 1.19 | 0.00 | 39.00 | 16.54 | 4279/4393
d2t02s | 3 | 22.78 [22.77-22.81] | 1.101 [1.052-1.105] | 43.94 | 33.66 | 0.678 | 48.88 | 23.88 | 24.99 | 30.42 | 0.82 | 28.8 | 0.27 | 1.20 | 0.00 | 39.00 | 18.12 | 4279/4393
d2t05s | 3 | 22.57 [22.40-22.96] | 1.092 [1.034-1.112] | 44.40 | 34.93 | 0.678 | 48.88 | 30.10 | 18.78 | 20.59 | 0.91 | 9.6 | 0.37 | 0.79 | 0.00 | 39.00 | 21.11 | 4279/4393
orcs4 | 3 | 26.90 [26.80-26.94] | 1.298 [1.242-1.303] | 37.13 | 28.07 | 0.678 | 48.88 | 1.33 | 47.55 | 47.55 | 1.00 | 0.0 | 0.00 | 1.45 | 0.00 | 39.00 | 0.73 | 4279/4393
orcsd2 | 3 | 27.38 [27.29-27.55] | 1.324 [1.260-1.335] | 36.47 | 27.58 | 0.674 | 49.52 | 1.34 | 48.18 | 48.18 | 1.00 | 0.0 | 0.00 | 1.64 | 0.00 | 39.00 | 0.73 | 4285/4407
w2t02s | 3 | 22.35 [22.22-22.44] | 1.074 [1.032-1.087] | 44.83 | 33.54 | 0.678 | 48.88 | 23.88 | 24.99 | 30.42 | 0.82 | 28.8 | 1.67 | 1.01 | 0.00 | 39.00 | 18.12 | 4279/4393

All [measured]. The harder budget (48 misses a token) is where look-ahead pays: `d2t01s` 1.116x [1.061-1.132], oracle 1.30-1.32x.

### True overfill on a smaller card: c2 RTX 2060 (12 GB, Gen3 x4 at 2.9-3.0 GB/s measured), 9800 MiB, n = 2 per arm

10800 MiB ran the card out of device memory (`allocating 8388608 bytes`, the slot bank plus dense weights plus scratch); 9800 fits at 11065 MiB
used. Experts are 1.26x that budget, and the whole model does not fit on this card.

arm | n | tok/s med [min-max] | x e2 (paired) | token ms | read wait | hit | misses/t | on bus/t | covered/t | prefetched/t | precision | wasted MB/t | guess ms | issue ms | spared/t | syncs/t | miss layers/t | VRAM MiB
---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---
e2 | 2 | 37.08 [36.84-37.32] | 1.000 [1.000-1.000] | 26.96 | 22.36 | 0.976 | 3.71 | 3.71 | 0.00 | 0.00 | - | 0.0 | 0.00 | 0.00 | 0.00 | 39.00 | 3.33 | 10951/11057
d2t01s | 2 | 32.28 [32.26-32.30] | 0.870 [0.864-0.877] | 31.01 | 25.03 | 0.975 | 3.81 | 1.89 | 1.92 | 3.57 | 0.54 | 8.7 | 0.68 | 0.10 | 0.00 | 39.00 | 1.81 | 10951/11065
orcs4 | 2 | 41.16 [40.83-41.50] | 1.110 [1.094-1.126] | 24.39 | 20.21 | 0.975 | 3.81 | 0.11 | 3.71 | 3.71 | 1.00 | 0.0 | 0.00 | 0.06 | 0.00 | 39.00 | 0.10 | 10951/11065

All [measured]. On this slow bus at a 97.6 % hit rate, the real guess **loses** (0.87x: 3.7 misses a token, half of them covered, and the guess, the wrong copies and the x read cost more than they hide); the perfect guess gains 1.11x. Prediction prefetch is a fast-bus / heavy-miss tool.

### llama.cpp e14a779 on the same card, interleaved (`-cmoe --moe-cache-mib` at the same VRAM, n = 3 per arm)

`c` is llama.cpp. 5120 MiB, then 3072 MiB:

arm | n | tok/s med [min-max] | x e2 (paired) | token ms | read wait | hit | misses/t | on bus/t | covered/t | prefetched/t | precision | wasted MB/t | guess ms | issue ms | spared/t | syncs/t | miss layers/t | VRAM MiB
---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---
e2 | 3 | 30.40 [30.04-30.93] | 1.000 [1.000-1.000] | 32.79 | 25.04 | 0.854 | 22.20 | 22.20 | 0.00 | 0.00 | - | 0.0 | 0.00 | 0.00 | 0.00 | 39.00 | 15.18 | 6337/6443
c | 3 | 16.64 [16.53-16.71] | 0.544 [0.538-0.556] | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 6233/6293
d2t01s | 3 | 31.46 [31.07-31.59] | 1.034 [1.017-1.039] | 31.64 | 22.89 | 0.851 | 22.58 | 9.89 | 12.69 | 18.39 | 0.69 | 30.3 | 0.27 | 0.68 | 0.00 | 39.00 | 8.56 | 6331/6445

arm | n | tok/s med [min-max] | x e2 (paired) | token ms | read wait | hit | misses/t | on bus/t | covered/t | prefetched/t | precision | wasted MB/t | guess ms | issue ms | spared/t | syncs/t | miss layers/t | VRAM MiB
---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---
e2 | 3 | 20.95 [20.64-21.20] | 1.000 [1.000-1.000] | 47.87 | 38.96 | 0.683 | 48.20 | 48.20 | 0.00 | 0.00 | - | 0.0 | 0.00 | 0.00 | 0.00 | 39.00 | 26.47 | 4285/4391
c | 3 | 11.05 [11.04-11.13] | 0.527 [0.525-0.535] | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 4181/4241
d2t01s | 3 | 22.98 [22.81-23.57] | 1.089 [1.084-1.142] | 43.59 | 33.61 | 0.678 | 48.88 | 21.16 | 27.72 | 36.13 | 0.77 | 44.6 | 0.35 | 1.26 | 0.00 | 39.00 | 16.54 | 4279/4393

All [measured]. The look-ahead engine is 1.9-2.1x llama.cpp at both budgets. llama.cpp's own text is a different sha on this prompt set; only lemonade-burn's sha is compared across budgets and arms.

## Diagnosis: where a miss's time goes (5120 MiB, read wait, ms a token, all [derived] from the runs above)

| | read wait | above the no-miss floor |
|---|---|---|
| no-miss floor (every expert cached) | 12.33 | 0 |
| E2 | 25.1 | +12.8 (22.2 misses x 0.575 ms) |
| perfect free guess, cap 4, spare slots | 18.80 | +6.5 |
| ... of which contention: 22 copies x 0.15 ms (ballast, no dependency at all) | | ~3.3 |
| ... of which waits, repack and fences together (not separable with these arms) | | ~3.2 |
| best real arm `d2t01s` | 22.80 | +10.5: 9.9 unpredicted misses x 0.575 = 5.7, ~2.8 contention from 18.4 copies, ~2 the rest |

* **Copy lead time is not the limit.** Issuing copies 2 or 3 layers ahead with their own side stream each (`orcsd2`, `orcsd3`) moves read wait by under 0.7 ms.
  (A first attempt with one shared side stream looked the same but was not a valid test: a fence waits for everything queued on its stream, so
  deeper copies cancelled their own lead time. That is why each depth now has its own stream.)
* **Repacking on the main stream costs about 0.7 ms a token at full coverage**, and more at the real arms' mix (read wait 24.0 to 22.8 with the spare slot).
  Fixed by the spare slot.
* **Contention is the hard limit.** A copy that nothing waits for still takes about 0.15 ms of the main stream's read wait, roughly a quarter
  of its 0.575 ms bus time, so overlap is partial however good the guess. Half of the oracle's remaining gap to the floor is this.
* **Wrong guesses are paid in full.** The real arm wastes 5.7 copies a token at about 0.15-0.3 ms each, against 12.7 misses covered.
* **The guess cost** is handled: the worker and deferred pick-up take it from 2.9 ms to 0.25-0.5 ms (host issue of the copies adds 0.7-1.2 ms).
* **Hit rate** is where the remaining large lever sits: Belady's 0.943 vs SLRU's 0.843 is ~13 misses a token (~7.5 ms at 5120 MiB, in the simulator); prediction-guided keep and cycle-aware eviction, the cheap attempts here, gain under 0.5 points.
* **Syncs:** the host syncs every layer in every arm (39.0 a token). Skipping them was not tested; the guess cannot replace reading the real routes without device-side miss detection and rollback (exp. #1: all four guessed correctly 43 % of the time at k).

## Off by default

With no `LB_LOOKAHEAD` / `LB_LA_*` set, the final binary (`off`) against the E2 binary, n = 3 interleaved:

arm | n | tok/s med [min-max] | x e2 (paired) | token ms | read wait | hit | misses/t | on bus/t | covered/t | prefetched/t | precision | wasted MB/t | guess ms | issue ms | spared/t | syncs/t | miss layers/t | VRAM MiB
---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---
e2 | 3 | 30.27 [30.19-30.38] | 1.000 [1.000-1.000] | 32.89 | 24.99 | 0.854 | 22.20 | 22.20 | 0.00 | 0.00 | - | 0.0 | 0.00 | 0.00 | 0.00 | 39.00 | 15.18 | 6337/6443
off | 3 | 30.48 [30.41-30.51] | 1.005 [1.003-1.011] | 32.69 | 25.03 | 0.854 | 22.20 | 22.20 | 0.00 | 0.00 | - | 0.0 | 0.00 | 0.00 | 0.00 | 39.00 | 15.18 | 6337/6443

All [measured]. 1.005x [1.003-1.011]: no cost when off.

## Gate (P100 x16, all [measured])

| gate | result |
|---|---|
| tok/s >= 1.10x the interleaved E2 baseline at 5120 MiB, net of guess cost | **FAIL: best real arm `d2t01s` 1.042x** [1.017-1.056], n = 3; a second interleaved set gives 1.034x [1.017-1.039]. Perfect free guess (oracle) 1.25x |
| the same at 3072 MiB (the harder budget, not the gate's) | **d2t01s 1.116x** [1.061-1.132] in one interleaved set and **1.089x** [1.084-1.142] in a second (n = 3 each), so about the gate; oracle 1.30-1.32x |
| >= llama.cpp e14a779 at the same VRAM on this card | **PASS**: 31.46 vs 16.64 tok/s (1.89x) at 5120, 22.98 vs 11.05 (2.08x) at 3072, n = 3 |
| greedy sha unchanged across arms and budgets | **PASS**: 1248 lemonade-burn completions across every arm and set above, one text per prompt (sonar `062392b8d2b0361a`), at 3072, 5120, 9800 and 12500 MiB on two cards |
| no device memory growth | **PASS** for `d2t01s`: 250 completions (50 passes) at 3072 MiB flat at 4393 MiB after pass 0; 100 completions at 5120 MiB flat at 6445 |
| wasted bytes per token | reported: `d2t01s` 30.3 MB/token at 5120, 44.6 at 3072 (of 18.4 / 36.1 copies issued a token, precision 0.69 / 0.77) |

Not done: `LB_DEVICE_GUESS` as the guess source (the 4.9 ms device guess was not measured as a replacement; the worker took host cost below it), a skip-the-sync design, stronger eviction than the keep guard (Belady's 0.943 is the bound), and the c2 2060's numbers at more rounds.

Caveat: c4 socket 0 carried a 40-thread photogrammetry run for most of these sessions (load 22-46), with quiet stretches between; the arms are interleaved so drift cancels, but host-side timings may be inflated. Spread between rounds is 1-4 % for most arms.

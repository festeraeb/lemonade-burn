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

## Results (2026-11-09, cesarops4om Tesla P100 AF:00.0, Gen3 x16, NUMA node 1, H2D 8.5-9.6 GB/s measured)

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

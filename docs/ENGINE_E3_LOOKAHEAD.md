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

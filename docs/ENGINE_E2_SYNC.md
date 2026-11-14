# E2 per-layer router read-back (idea E): measured, not built

Branch `fable/engine-123-e2sync-2026-11-14` (off cbe7b65). Raw: `/mnt/fast/eng123/e2sync-results/prof/`.
The only code change is a diagnostic: `LB_E2_PROFILE=<file>` writes one JSON line per decode token with, for each
MoE layer, [layer, misses, sync wait us, host busy us since the last sync, bytes read]. Off by default.
Method: c4 P100 D8, NUMA 1, 20 threads, `cm.py` prompts, 160 decode tokens a prompt, 671 decode tokens analysed
(`profan.py`, `simspec.py`). All numbers [measured] unless marked [derived] or [model].

## 1. How often is a layer or token miss-free

| cache MiB | tokens with zero misses | miss-free layers / 38 | mean miss-free run (layers) | runs / token | tok/s (profiled run) |
|---|---|---|---|---|---|
| 12500 | 89.6 % | 37.9 | 33.4 | 1.1 | 44.5 |
| 11008 | 50.1 % | 37.1 | 19.4 | 1.9 | 44.2 |
| 8192 | 1.9 % | 30.1 | 3.4 | 8.9 | 36.4 |
| 5120 (copy misses) | **0 %** | 15.4 | 0.65 | 23.6 | (24.6) |
| 5120 (CPU misses) | **0 %** | 14.4 | 0.59 | 24.6 | (28.4) |
| 3072 | **0 %** | 6.9 | 0.22 | 32.1 | (18.1) |

At 5120 and 3072 MiB no token is miss-free, and the first miss is at layer 0 in the median token. A miss-free layer is
followed by a miss on average after 0.65 layers.

## 2. What the 10 ms is made of at ~100 % hit (12500 MiB, per token)

issue-to-last-sync 17.8 ms (this run) = sync wait 12.2 ms (321 us a layer) + host busy 5.5 ms (145 us a layer).
Bytes read per token: 45 KiB (1.2 KiB a layer), so the read is not bandwidth. All-resident is 9.9 ms (101 tok/s,
earlier measurement), so 7.9 ms [derived] (about 0.21 ms a layer) is paid for having a sync in every layer. The wait is
not the device finishing real work only: the device work is ~0.26 ms a layer [derived 9.9/38] and the wait is 0.32, and
the host needs 0.145 ms to launch the next layer after each sync, with the stream empty at that point. So the cost is
drain + refill of the stream (read round trip plus the first launches of the next layer), a fixed ~0.2 ms a sync.
CPU misses add host busy time: 274 us a layer at 5120 (150 without), 473 us at 3072 (201 without).

## 3. Would speculation pay (model from the recorded per-layer miss vectors)

[model] cost per token = 38 x 0.26 ms device + syncs x 0.21 ms + wasted layer re-runs x 0.26 ms; a window of w layers runs
without a sync, one check at its end, a miss at layer m discards the layers after m in the window (that layer is
then served on today's synced path), the next window starts after m. Miss service costs are unchanged and left out.

| cache MiB | best window | syncs/token | model change vs today | verdict |
|---|---|---|---|---|
| 12500 | 38 (one check a token) | 1.1 | 17.8 -> ~10.7 ms (-7 ms) | pays |
| 11008 | 8 | 5.5 | -5.9 ms of ~22.6 | pays, ~20 % |
| 8192 | 2 | 21.6 | -2.2 ms of ~27 | ~7 % |
| 5120 | none (window 1) | 23-38 | window 2: +2.4 ms, 3: +6.7, 4: +11.6 | **loses** |
| 3072 | none | 31-38 | window 2: +5.9 ms | **loses** |

## Verdict

Not built. At the budgets that matter (5120 and 3072 MiB) every token has a miss in the first layers and 60-80 % of
layers miss, so speculating past a layer wastes more device time than the syncs it saves; per-token speculation has
no miss-free tokens to win on. The idea only pays when the cache holds nearly every expert (>= 11 GB, where the card
is nearly all-resident anyway, 12500 MiB: ~-7 ms of 17.8 [model]).

What could still help at 5120/3072 (not measured): (a) cut the fixed ~0.2 ms per sync (a smaller read, pinned staging
reuse, issuing the next layer's operator kernels before the read and using a second stream); (b) cut the CPU-miss host
busy time (+125 us a layer: the h2 read, vector conversions, the expert batch call, the per-expert `write`); (c) serve
misses so that the layer's other work overlaps them. Each is a per-layer constant, not fewer syncs.

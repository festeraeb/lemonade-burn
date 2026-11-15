# Residency policy ceiling, and the per-layer constants of the E2 path

Branch `fable/engine-123-e2sync-2026-11-14`. Raw: `/mnt/fast/eng123/e2sync-results/` (`trace/`, `sim-*.txt`, `runs2/`).
[measured] = a run on the c4 P100 D8 (x16, NUMA 1, 20 threads, greedy); [sim] = offline replay of recorded routes;
[model] = arithmetic on measured costs.

## Step 1: how much can a better residency policy win (offline)

Traces: `LB_LA_TRACE` of the E2 server, five prompts x 288 decode tokens (1435 tokens, 54,530 layer passes each).
Set A = the `cm.py` prompts (sonar, Rust CSV, 1913 storm, magnetometer, MoE). Set B = five other topics (beef stew, binary
search tree in Python, French Revolution, common vs civil law, photosynthesis). Routes do not depend on the cache, so one
trace serves every budget. `residency_sim` (`crates/lb-lfm2/src/bin/residency_sim.rs`) replays them against the real
`Slru` and the other policies with the server's slot counts (5120 MiB = 1003 slots, 3072 MiB = 598); every policy starts
from the same scrape-heat seed; prefill passes go through the cache as in the server; misses are counted on decode passes.
The simulated SLRU reproduces the measured server: 32.68 misses/token on A against 32.7 [measured].
Static placement is trained on one set and tested on the other (held out); the online policies need no training.

Misses per token (hit rate), 5120 MiB:

| policy | test A (trained on B where it trains) | test B (trained on A) |
|---|---|---|
| SLRU, today | 32.68 (0.785) | 28.77 (0.811) |
| Belady, oracle | **13.01 (0.914)**, -60 % | **11.13 (0.927)**, -61 % |
| Belady with bypass | 12.97 | 11.09 |
| static top-frequency, never evict | 61.71 (0.594) | 61.65 (0.594) |
| decayed LFU, half-life 8 tokens, admit if hotter | 32.18 (-1.5 %) | 28.31 (-1.6 %) |
| decayed LFU, half-life 16 | 32.01 (-2.0 %) | **27.35 (-4.9 %)** |
| decayed LFU, half-life 32 | 33.48 (+2.4 %) | 28.35 (-1.5 %) |
| decayed LFU, half-life 256 / infinite | 42.8 / 47.3 | 38.4 / 45.1 |
| per-layer quota SLRU, equal | 34.48 (+5.5 %) | 31.31 (+8.8 %) |
| per-layer quota SLRU, train-proportional | 34.21 | 31.54 |
| topic seed from the prompt's prefill routes (top 250 / 501 keys) | 32.68 / 32.69 (0 %) | 28.76 / 28.77 (0 %) |

3072 MiB: SLRU 59.01 (A) / 55.04 (B); Belady 28.57 / 26.49 (-52 %); best realisable LFU half-life 8: 58.28 / 53.85
(-1.2 % / -2.2 %); static 91.4 / 90.8; quota 59.7-59.2 / 56.2; topic seed 58.98-59.00 / 55.00-55.02 (0 %).
Admit-all and admit-if-hotter LFU differ by less than 0.2 %. A decaying LFU with a short memory is the best realisable
policy and the best half-life is the same (8-16 tokens) on both sets, which says the cache's win is recency, not
frequency: the static plan from a held-out trace is 2x worse than today's SLRU.

Topic seeding does nothing because the prompt's own prefill already touches 40+ of the 64 experts of every layer
through the same cache, so after prefill the prompt's experts are resident; reseeding with the top 250-500 of them adds
100-265 copies a request and changes no decode miss.

**Gate (a realisable policy that cuts misses/token by >= 25 % at 5120 on the held-out set): FAIL.** Best realisable:
-4.9 % on B, -2.0 % on A (LFU half-life 16). Belady is -60 %, so policy work is not closed for good: the headroom exists
but it is foresight (which expert a layer needs next), and prediction is what E3 (1.04x) and the stage ring tested and lost.

[model] translated with the costs measured on this card: a copy miss costs ~0.62 ms in the token ((40.7 - 19.9) ms / 32.7
misses, E2 at 100 % hit being 19.9 ms; 0.60 at 3072) and a CPU miss ~0.45 ms ((35.3 - 19.9) / 34.6): LFU half-life 16 on A
would save 0.7 misses = 0.4 ms of 40.7 ms (+1 %); the Belady ceiling would save 19.7 misses = 12 ms (+43 % tok/s).

## Step 2: per-layer constants

### The CPU-miss host busy time is the CPU kernel

The +125 us a layer I listed as overhead is the expert batch itself: at 5120 the host is busy 10.4 ms a token with CPU
misses against 5.7 without, and the CPU batch costs 7.0 ms on 14.4 layers (0.49 ms a layer, 0.41 ms an expert in the
server, 0.14-0.29 ms in the microbench). There is no separate overhead to remove; the remaining lever is kernel speed
(memory bandwidth: 35 of 65 GB/s).

### `LB_E2_SPEC`: launch the experts before the routes are read

Mechanism (bit-identical): a device table (layer, expert) -> physical slot (`ExpertCache::sync_table`, rewritten only
when the SLRU changed); `slot_translate_kernel` (new, `lb-q4-gpu`, test `tests/slot_translate.rs`) turns the router's
experts into slot words on the card; the routes read is enqueued with cubecl's `read_async` right behind the router (the
copy runs before the experts in the stream) and the experts are launched from the table before the host waits for it.
A layer with a copy-miss is repaired after the read (slot words written, experts launched again); a CPU-served miss is
overwritten into the output rows as before. Needs the look-ahead off. Per layer it is used only while the layer's recent
miss frequency is under 0.3 (`LB_E2_SPEC_MISS`), because a speculated miss launches the experts twice.

Gates: teacher-forced against the all-resident decode over 1590 positions at 5120 MiB: spec 100.00 % top-1, **max |logit
diff| 0** (same as the E2 control); spec + CPU misses 100.00 %, 2.5e-5 (same as the CPU arm). Same greedy text in every
session (see below).

Speed, A-B-B-A [measured] (same `cm.py` prompts, 288 tokens x 5, text identical to the first arm in all sessions):

| budget | arm | tok/s median | x base | ms/token |
|---|---|---|---|---|
| 12500 (cache forced, hit ~1.0) | base | 50.76 | 1.000 | 19.7 |
| 12500 | spec | 54.86 | **1.081** (paired 1.059-1.122) | 18.2 |
| 8192 | base / spec | 39.66 / 39.66 | 1.000 | 25.2 |
| 5120 | base | 24.52 | 1.000 | 40.8 |
| 5120 | spec | 24.09 | 0.982 | 41.5 |
| 5120 | CPU exact | 28.96 | 1.182 | 34.5 |
| 5120 | CPU exact + spec | 27.65 | 1.128 | 36.2 |
| 3072 | base | 17.94 | 1.000 | 55.7 |
| 3072 | spec | 17.64 | 0.983 | 56.7 |
| 3072 | CPU exact | 20.57 | 1.147 | 48.6 |
| 3072 | CPU exact + spec | 20.57 | 1.147 | 48.6 |

(Before the per-layer rule, spec on every layer measured 0.943x at 5120 and 0.941x at 3072 for the copy arm: 22.94 vs 24.33
and 16.84 vs 17.90 tok/s, 6 sessions each.)

At 100 % hit the E2 token went from 19.9 to 18.2 ms; the 9.9 ms all-resident floor is not reached. The profile
(`LB_E2_PROFILE`) shows host busy per layer 154 -> 103 us and the sync wait unchanged at ~313 us a layer: the card, not the
host, is the limit once the read is no longer behind the experts, and the E2 layer still launches ~3 more kernels than
the fused all-resident layer (router, translate, sum, next-layer norm unfused).

Verdict: stays behind the switch, default off. It only pays when the cache holds nearly every expert (+8 % at 12500 MiB),
is neutral at 8192, and is 1-4 % slower at 5120 and 3072 even with the per-layer rule, so it fails the "never slower"
condition for a default.

## What is left

* Residency: the 60 % between SLRU and Belady needs a predictor of the next layers' experts better than what E3 had
  (guess recall 0.78); nothing realisable and cheap reaches 25 %.
* Per layer: fuse the E2 layer's extra launches (router + translate, expert sum + next norm) so it approaches the
  all-resident layer; CPU expert kernel bandwidth (35 -> 65 GB/s).

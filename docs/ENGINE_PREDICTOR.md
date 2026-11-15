# Next-expert predictor: how predictable is routing, and what a predictor buys at 5120 MiB

Branch `fable/engine-123-predict-2026-11-15` (off 24d4369, the E2 flush+fuse defaults). Card: Tesla P100 (UUID suffix D8) on c4, NUMA 1, 20
threads, LFM2.5 24B, greedy. [measured] = a run on the card; [sim] = offline replay of recorded routes with a bus-timing model;
[estimate] = arithmetic. Raw data: `predict-results/` next to the other engine result folders (traces, features, logs, `runs2/`).
Tools: `scripts/predict/` (set `PRED_RES` to the results folder).

Short version: routing at layer L+2 is predictable from the residual stream at layer L (held-out recall of the experts that are
not already cached: 0.72 with the 2 best guesses). A predictor passes gate A (critical-path misses -48 to -56 % at 1.3-1.5x the
bus bytes) but does **not** pass gate B: the best measured speedup at 5120 MiB is **1.084x** (gate 1.25x). Results are correct
(teacher-forced top-1 100 %, max logit diff 0) and there is no regression at 12500 MiB or all-resident, but only with a miss gate
added after the first binary lost 12.5 % at 12500 MiB. Why it stops there: the bus is about 78 % occupied once the copies overlap,
the host spends about half of what the overlap wins on issuing the copies, and every 0.1 ms between the routes read and the first
copy costs about 1 % of the token. `LB_PRED` stays off by default. Card A-B-B-A: section 8.

## 1. Data and method

Route traces and residual streams from the E2 server (`LB_PRED_DUMP`: per decode token and MoE layer, the experts as routed and the
`x_mid` residual stream that E3 already reads with the routes; a sidecar file holds every MoE layer's FFN pre-norm, router and
selection bias). Routes do not depend on the cache, so one trace serves every budget. Replaying the router on a record's own `x_mid`
reproduces the routed expert set on 7790 of 7790 sampled passes. Prefill passes come from `LB_LA_TRACE` of the same run so the cache
starts each prompt as the server's does.

| set | prompts | decode tokens | use |
|---|---|---|---|
| TRAIN | 30 new prompts (code, cooking, history, law, maths, translation, email, poetry, chat ...), 256 tokens | 7599 | fit tables and models (features cross-fitted over two prompt halves) |
| VAL | 6 new prompts, 200 tokens | 1193 | choose configurations, thresholds, knob values on the card |
| TEST_A | the `cm.py` prompts (sonar, Rust CSV, 1913 storm, magnetometer, MoE), 288 tokens | 1433 | held-out; also the final A-B-B-A prompts |
| TEST_B | five other topics (beef stew, BST in Python, French Revolution, common vs civil law, photosynthesis) | 1433 | held-out |

(The dump of the last prompt of a set lost its buffered tail: 1433 of 1435 tokens.) Nothing was tuned on TEST_A / TEST_B; the card
sweeps used VAL (`CM_SET=VAL`). Recall below = share of the 4 routed experts of a layer pass that are among the m fetched; precision =
share of the m fetched that are routed. "Uncached recall" = share of the routed experts that are not in the 5120 MiB SLRU at that
moment which are among the m best uncached guesses (that is what a prefetch has to find).

## 2. Predictability by feature (held-out, all 38 MoE layers, m experts fetched per layer pass)

Recall / precision. TEST_A then TEST_B; the two sets agree to 0.02-0.04.

| predictor (information available at issue time) | m=4 A | m=8 A | m=16 A | m=4 B | m=8 B | m=16 B |
|---|---|---|---|---|---|---|
| static frequency of the layer | 0.13 / 0.13 | 0.24 / 0.12 | 0.42 / 0.11 | 0.14 / 0.14 | 0.25 / 0.13 | 0.44 / 0.11 |
| (a) same layer, previous token | 0.31 / 0.31 | 0.36 / 0.18 | 0.45 / 0.11 | 0.34 / 0.34 | 0.39 / 0.19 | 0.47 / 0.12 |
| (a) same layer, last 4 tokens | 0.28 / 0.28 | 0.42 / 0.21 | 0.56 / 0.14 | 0.31 / 0.31 | 0.45 / 0.22 | 0.58 / 0.14 |
| (a) same layer, last 16 tokens, decay 0.8 | 0.31 / 0.31 | 0.46 / 0.23 | 0.62 / 0.16 | 0.34 / 0.34 | 0.48 / 0.24 | 0.64 / 0.16 |
| (c) Markov: previous token's experts in this layer -> this token's | 0.31 / 0.31 | 0.46 / 0.23 | 0.65 / 0.16 | 0.35 / 0.35 | 0.51 / 0.25 | 0.69 / 0.17 |
| (d) token id (per-layer table with backoff) | 0.33 / 0.33 | 0.46 / 0.23 | 0.61 / 0.15 | 0.33 / 0.33 | 0.47 / 0.23 | 0.63 / 0.16 |
| (b) cross-layer table, experts of layer L -> L+1 | 0.37 / 0.37 | 0.54 / 0.27 | 0.73 / 0.18 | 0.40 / 0.40 | 0.58 / 0.29 | 0.76 / 0.19 |
| (b) cross-layer table, L -> L+2 | 0.35 / 0.35 | 0.52 / 0.26 | 0.70 / 0.18 | 0.38 / 0.38 | 0.55 / 0.28 | 0.73 / 0.18 |
| (b) cross-layer table, L -> L+4 | 0.35 / 0.35 | 0.51 / 0.25 | 0.69 / 0.17 | 0.37 / 0.37 | 0.54 / 0.27 | 0.71 / 0.18 |
| (e) early routing, router of L+1 on layer L's residual | **0.81 / 0.81** | **0.93 / 0.47** | 0.97 / 0.24 | **0.81 / 0.81** | **0.93 / 0.47** | 0.97 / 0.24 |
| (e) early routing, L+2 | 0.73 / 0.73 | 0.87 / 0.44 | 0.93 / 0.23 | 0.74 / 0.74 | 0.88 / 0.44 | 0.94 / 0.23 |
| (e) early routing, L+4 | 0.64 / 0.64 | 0.80 / 0.40 | 0.88 / 0.22 | 0.65 / 0.65 | 0.80 / 0.40 | 0.88 / 0.22 |
| (e) early routing, L+8 | 0.50 / 0.50 | 0.66 / 0.33 | 0.78 / 0.19 | 0.52 / 0.52 | 0.67 / 0.34 | 0.79 / 0.20 |
| (e) router on the previous token's last residual, every layer | 0.17 / 0.17 | 0.28 / 0.14 | 0.44 / 0.11 | 0.18 / 0.18 | 0.30 / 0.15 | 0.47 / 0.12 |

(f) runner-ups are the m > 4 columns of (e): fetching the top 8 of the later router names 93 % of the experts one layer ahead, 80 % four
layers ahead. Per layer (TEST_A, m=4): early routing L+1 is 0.69-0.90 from layer 3 on (layers 2 and 3 have no earlier MoE state: 0.08),
L+4 0.61-0.82, L+8 0.56-0.72; the token-id table is the best predictor of the first MoE layers (0.37-0.45 at layers 2-5, where every
state-based predictor is blind) and weak in the middle layers (0.23-0.38).

(g) combination: a model on the early-routing score, its gap to the 4th best score, its rank, and (for the larger models) the
cross-layer / Markov / token / recency / age / frequency features, fitted per lead on TRAIN. Recall (precision) of the **not-cached**
routed experts at m uncached fetches per layer, held-out:

| lead | model | m=1 A | m=2 A | m=4 A | m=1 B | m=2 B | m=4 B |
|---|---|---|---|---|---|---|---|
| 1 | early routing, uncached top-m | 0.57 (0.49) | 0.80 (0.34) | 0.92 (0.20) | 0.59 (0.45) | 0.81 (0.31) | 0.92 (0.17) |
| 1 | gradient-boosted trees, 14 features | 0.59 (0.50) | 0.82 (0.35) | 0.94 (0.20) | 0.60 (0.46) | 0.83 (0.32) | 0.95 (0.18) |
| 2 | early routing | 0.51 (0.44) | 0.72 (0.31) | 0.85 (0.18) | 0.53 (0.40) | 0.73 (0.28) | 0.85 (0.16) |
| 2 | **logistic, 4 features (shipped)** | 0.51 (0.44) | 0.72 (0.31) | 0.85 (0.18) | 0.53 (0.40) | 0.73 (0.28) | 0.86 (0.16) |
| 2 | gradient-boosted trees | 0.54 (0.47) | 0.76 (0.33) | 0.90 (0.19) | 0.56 (0.42) | 0.78 (0.29) | 0.90 (0.17) |
| 4 | early routing | 0.43 (0.37) | 0.61 (0.26) | 0.74 (0.16) | 0.44 (0.33) | 0.62 (0.23) | 0.74 (0.14) |
| 4 | gradient-boosted trees | 0.49 (0.42) | 0.69 (0.30) | 0.83 (0.18) | 0.50 (0.38) | 0.70 (0.27) | 0.84 (0.16) |
| 8 | early routing | 0.32 (0.28) | 0.46 (0.20) | 0.59 (0.13) | 0.33 (0.25) | 0.47 (0.18) | 0.59 (0.11) |
| 8 | gradient-boosted trees | 0.42 (0.36) | 0.60 (0.26) | 0.74 (0.16) | 0.43 (0.33) | 0.62 (0.23) | 0.76 (0.14) |

The trees add 3-5 points at lead 2 and 8-14 points at lead 8 (they combine the early score with recency and the cross-layer table); a logistic
model on the early-routing numbers alone is as good as the raw ranking but gives a calibrated probability, which is what the
threshold in the prefetch rule needs. A learned correction of the early router (ridge from the residual to the true logits,
tried on layers 10/20/30, leads 2/4/8) changed recall by -0.03..+0.02: closed.

## 3. Why routing is or is not predictable (what the traces say)

* **Marginally routing is nearly flat.** Over all layers the routed-expert distribution has 5.75 bits of entropy (6.00 = uniform), an effective 54 of
  64 experts a layer; the top 4 experts of a layer hold 15 % of the mass (uniform 6.25 %), the top 16 hold 44 %. Per layer the
  entropy is 5.50-5.96 bits; layers 2-5 and 12 are the flattest (5.9+, top-4 mass 9-11 %, driven by the token), layers 18-19 the most skewed (5.50, top-4 19-23 %).
  The 5120 MiB cache misses are on experts that are routed in 6.7 % of tokens on average (uniform: 6.25 %); 90 % of misses are on experts used
  in 5-10 % of tokens, 4 % in under 2 %. There is no hot set to keep: a frequency or history predictor cannot name them,
  which is why static frequency, recency, Markov and token tables all sit at 0.3-0.5 recall for 8 guesses.
* **Routing depends on the topic, but weakly per layer.** One prompt concentrates routing: the entropy of a prompt's own distribution is 5.05 bits against
  5.75 for all prompts (about 33 effective experts instead of 54). Jensen-Shannon divergence of a prompt's distribution from the rest: 0.18 bits on
  average, 0.27-0.37 in layers 14-16, 18-19 and 22-23 (topic-sensitive), 0.04-0.06 in layers 3-5 (token-driven, topic-blind). The top-8 experts of a
  prompt overlap those of the other prompts 34 % (random: 12.5 %). That is the mass the SLRU already exploits, and why topic seeding found nothing: the prompt's own prefill
  puts its experts in the cache.
* **Consecutive tokens share little.** Same layer, previous token: 34 % of the 4 experts repeat on average (51-59 % in layers 14-19, 14-15 % in
  layers 3-5). Recency beyond one token adds recall only through the larger m.
* **What does predict it is the residual stream.** The router of layer L+k applied to layer L's residual (the stream changes slowly) names 81 / 73 / 64 / 50 % of the experts
  for k = 1 / 2 / 4 / 8 with 4 guesses and 93 / 87 / 80 / 66 % with 8. This is state, not statistics: the information is in the activations of this token, which is why the
  tables over past routes (b, c, d) lose by 0.3-0.4 recall at every k.
* **Why a better predictor does not make the token proportionally faster.** Only the uncached experts matter (about 31 a token at 5120 MiB, 0.8 a layer) and each is a 5.31 MB copy, 0.55 ms
  of a bus that is also carrying the wrong guesses. The compute of a layer is 0.36 ms, so a copy issued k layers early overlaps k x 0.36 ms
  of compute and costs 0.55 ms of bus: the lead must be 2 or more to hide a copy, and at lead 2 the recall of the uncached experts is 0.72 with 2 guesses
  (0.51 with 1). The remaining 25-30 % of misses arrive unannounced and are copied on the critical path, so the demand copies queue behind the prefetches in flight.

## 4. Timed replay (`scripts/predict/sim.py`)

Model: layer j runs 0.18 ms to its routes read, waits for every needed expert that is not resident (a demand copy queues FIFO on one bus behind any prefetch in flight, 5.31 MB at
9.7 GB/s = 0.547 ms), repacks 0.05 ms per filled miss, runs 0.18 ms. Compute 13.8 ms a token with everything resident, the E2 flush+fuse number. A prefetch issued when a layer's
routes are known (after `pred_delay` ms) copies into a bounded staging area of S buffers charged to the cache budget (the cache has S fewer slots), is repacked into a slot only when a layer
really routes it (so a wrong guess costs bus time, never a cache slot), and leaves the staging when its layer has run. No prefetch is issued when the staging or the in-flight
queue is full. The SLRU is a port of `cache::Slru`: it reproduces the measured server (32.74 misses a token against 32.68 for the Rust replay and 32.7 on the card on TEST_A; 28.79 against 28.77 on TEST_B) and the
no-prefetch token time (32.5 ms against 33.0 measured on VAL). Critical-path misses = experts that were not resident when the routes were known (demand copies plus prefetches that were late).

The simulator matches the engine's prefetch counts: for the VAL config measured on the card (lead 2, threshold 0.4, 2 guesses, 8 staging buffers) the engine reports 15.3 demand misses, 21.3 prefetches, 5.0 wasted a token;
the replay predicts 15.6 / 21.0 / 4.9. Its token time depends on the prediction-to-issue delay: 27.3 ms at 0 delay, 28.0 at 0.2, 28.8 at 0.36, 30.1 at 0.6 (x1.19, 1.16, 1.13, 1.08 over 32.5): the card measured 30.2 ms
(x1.08), so the engine's effective delay is about 0.5-0.6 ms (worker thread, picking the answer up, the issue itself).

### Predictor families in the replay (lead 2, S=8, queue 8, delay 0.2 ms, TEST_A | TEST_B, nothing tuned on TEST)

See `simtable.txt` (copied below). "+thr" = the probability or score threshold at which the VAL precision of the top-2 uncached fetches is 0.4; without it a predictor fetches its top m every layer.
Columns: critical-path misses a token (change against SLRU alone), bus copies against SLRU alone, token-time speedup.

| predictor | m | A: misses/token (change) | A: copies | A: speed | B: misses/token (change) | B: copies | B: speed |
|---|---|---|---|---|---|---|---|
| static frequency | 1 | 32.1 (-2 %) | 2.08x | 0.868x | 28.3 (-2 %) | 2.23x | 0.853x |
| static frequency | 2 | 31.3 (-4 %) | 3.14x | 0.587x | 27.6 (-4 %) | 3.44x | 0.567x |
| static frequency +thr | 1 | 33.1 (+1 %) | 1.01x | 0.993x | 29.2 (+1 %) | 1.01x | 0.992x |
| static frequency +thr | 2 | 33.1 (+1 %) | 1.01x | 0.993x | 29.2 (+1 %) | 1.01x | 0.992x |
| a recency N=1 | 1 | 32.4 (-1 %) | 2.09x | 0.864x | 28.5 (-1 %) | 2.23x | 0.850x |
| a recency N=1 | 2 | 31.7 (-3 %) | 3.16x | 0.583x | 28.0 (-3 %) | 3.46x | 0.563x |
| a recency N=1 +thr | 1 | 33.1 (+1 %) | 1.01x | 0.993x | 29.2 (+1 %) | 1.01x | 0.992x |
| a recency N=1 +thr | 2 | 33.1 (+1 %) | 1.01x | 0.993x | 29.2 (+1 %) | 1.01x | 0.992x |
| a recency (16, decay .8) | 1 | 32.0 (-2 %) | 2.07x | 0.870x | 28.3 (-2 %) | 2.22x | 0.855x |
| a recency (16, decay .8) | 2 | 31.3 (-4 %) | 3.14x | 0.588x | 27.7 (-4 %) | 3.44x | 0.566x |
| a recency (16, decay .8) +thr | 1 | 33.1 (+1 %) | 1.01x | 0.993x | 29.2 (+1 %) | 1.01x | 0.992x |
| a recency (16, decay .8) +thr | 2 | 33.1 (+1 %) | 1.01x | 0.993x | 29.2 (+1 %) | 1.01x | 0.992x |
| b cross-layer | 1 | 27.5 (-16 %) | 1.91x | 0.938x | 24.1 (-16 %) | 2.05x | 0.924x |
| b cross-layer | 2 | 26.1 (-20 %) | 2.89x | 0.638x | 23.1 (-20 %) | 3.17x | 0.615x |
| b cross-layer +thr | 1 | 30.8 (-6 %) | 1.14x | 1.012x | 27.2 (-6 %) | 1.14x | 1.012x |
| b cross-layer +thr | 2 | 30.5 (-7 %) | 1.17x | 1.002x | 26.9 (-7 %) | 1.16x | 1.005x |
| c markov same-layer | 1 | 30.4 (-7 %) | 2.02x | 0.893x | 26.7 (-7 %) | 2.16x | 0.880x |
| c markov same-layer | 2 | 29.0 (-11 %) | 3.04x | 0.607x | 25.5 (-12 %) | 3.32x | 0.587x |
| c markov same-layer +thr | 1 | 33.1 (+1 %) | 1.01x | 0.993x | 29.2 (+1 %) | 1.01x | 0.992x |
| c markov same-layer +thr | 2 | 33.1 (+1 %) | 1.01x | 0.993x | 29.2 (+1 %) | 1.01x | 0.992x |
| d token id | 1 | 27.5 (-16 %) | 1.90x | 0.945x | 24.2 (-16 %) | 2.04x | 0.929x |
| d token id | 2 | 27.0 (-18 %) | 2.89x | 0.638x | 23.9 (-17 %) | 3.18x | 0.613x |
| d token id +thr | 1 | 28.1 (-14 %) | 1.27x | 1.035x | 24.7 (-14 %) | 1.31x | 1.032x |
| d token id +thr | 2 | 27.5 (-16 %) | 1.39x | 0.977x | 24.3 (-15 %) | 1.44x | 0.975x |
| e early-routing raw | 1 | 22.0 (-33 %) | 1.60x | 1.116x | 19.7 (-32 %) | 1.74x | 1.085x |
| e early-routing raw | 2 | 24.4 (-25 %) | 2.50x | 0.739x | 22.1 (-23 %) | 2.79x | 0.700x |
| e early-routing raw +thr | 1 | 21.9 (-33 %) | 1.39x | 1.167x | 19.5 (-32 %) | 1.48x | 1.151x |
| e early-routing raw +thr | 2 | 23.0 (-30 %) | 1.85x | 0.942x | 20.8 (-28 %) | 2.01x | 0.915x |
| (g) logistic on early-routing numbers (shipped) | 1 | 22.1 (-32 %) | 1.13x | 1.184x | 19.7 (-31 %) | 1.14x | 1.172x |
| (g) logistic on early-routing numbers (shipped) | 2 | 21.2 (-35 %) | 1.17x | 1.165x | 19.1 (-34 %) | 1.18x | 1.157x |
| (g) gradient-boosted trees | 1 | 21.8 (-33 %) | 1.12x | 1.202x | 19.4 (-33 %) | 1.14x | 1.191x |
| (g) gradient-boosted trees | 2 | 21.4 (-34 %) | 1.17x | 1.186x | 19.3 (-33 %) | 1.19x | 1.176x |

E3 as it was (early routing one layer ahead, 2 guesses, confidence cut): lead 1 cannot arrive in time; the replay gives -0..+1 % critical-path misses with 1.0-1.7x the copies, token time 0.95-1.03x
(`e3row.txt`), consistent with the 1.04x measured in E3 (the default build has the look-ahead off, so the default is SLRU alone).

### Ceiling: a perfect predictor (VAL, oracle routes, delay 0, 64 staging buffers, queue 8)

| lead | critical-path misses | ms/token | x |
|---|---|---|---|
| 2 | 23.0 of 31.3 | 23.3 | 1.40 |
| 4 | 15.2 | 23.1 | 1.41 |
| 8 | 13.2 | 25.0 | 1.30 |
| 16 | 17.6 | 27.9 | 1.17 |

A single-lead perfect predictor reaches 1.41x; its limit is the bus (34.6 copies a token = 18.9 ms of the 23). Gate B (1.25x) needs a predictor that keeps most of that: recall 0.72 at precision 0.31 reaches 1.16-1.19x with free issue.

## 5. Gate A (offline): PASS

Gate: critical-path misses a token at 5120 MiB at least 40 % below the default (SLRU; E3 is off by default), bus bytes at most 1.5x, held-out. Tuned on VAL (random search over lead sets, threshold, guesses per layer, staging,
queue; `final_sim.py`), scored on TEST_A / TEST_B:

| predictor | delay | config (leads, thr, m, S, Q) | TEST_A misses | TEST_A copies | TEST_B misses | TEST_B copies |
|---|---|---|---|---|---|---|
| trees, objective = misses | 0 | (2,3,6) 0.3/0.5 m3 S16 Q12 | 14.4 vs 32.7 (-56 %) | 1.35x | 12.8 vs 28.8 (-55 %) | 1.37x |
| logistic, objective = misses | 0 | (2,3,6) 0.4 m2 S16 Q8 | 16.3 (-50 %) | 1.31x | 14.5 (-50 %) | 1.32x |
| trees, objective = misses | 0.2 ms | (2,4,6,8) 0.6/0.3 m2 S16 Q12 | 14.7 (-55 %) | 1.45x | 13.0 (-55 %) | 1.48x |
| logistic, objective = misses | 0.2 ms | (2,4,6,8) 0.6/0.3 m2 S16 Q12 | 16.8 (-49 %) | 1.38x | 14.9 (-48 %) | 1.40x |
| logistic, objective = token time | 0.2 ms | (1,2,3,4) 0.5/0.6 m2 S8 Q6 | 22.2 (-32 %) | 1.16x | 19.9 (-31 %) | 1.17x |

Verdict: gate A passes on the stated metric (-48 to -56 % at 1.31-1.48x the bus bytes). It is the wrong objective: the configurations that cut misses most are not the fastest (x1.05-1.14 in the replay) because the bus is
the limit; the fastest cut misses by a third. What limits it is the bus, not recall.

## 6. What was built

`LB_PRED=1` (default off) on the E3 prefetch path, no new daemon, nothing in the manifest.

* `LB_PRED_MIN_MISS=6` / tunable **`pred_min_miss`** (live): the residual read, guesses and prefetches run only while the moving average of demand misses a token is at least this (0 = always); needed so a cache that holds nearly everything pays nothing (12500 MiB: 0.875x without it, 1.006x with it).
* `LB_PRED_LEADS=2` (list of layers ahead; default 1..pf_depth), `LB_PRED_MODEL=<json>` (another table), `LB_PRED_THETA=0.4` / tunable **`pred_theta`** (live, thousandths, 0-1000).
* The table is compiled in (`crates/lb-lfm2/src/pred_model.json`): per lead a logistic model on `[score, gap to the 4th best score, ln(1 + rank), rank < 4]` of the guessed expert, with the training-trace hash
  (`acb49dfed5cdeafc`, sha-256 prefix of the first 64 MiB of TRAIN's residual dump) and the held-out recall in the file. Fitted on TRAIN only.
* The guess is E3's (the router and FFN pre-norm of layer L+k on layer L's residual, on the `lb-guess` worker thread); the prefetch rule replaces `min_conf` with `p >= pred_theta`.
  Landing buffers (`LB_LA_LANDINGS`, the bounded staging area, 8 by default, charged to the budget) take the copies on a side stream; with `LB_LA_SPARE=1` the side stream repacks them into spare bank
  slots so the main stream only fences; nothing is evicted for a guess and an unconsumed landing is released when its layer has run, so a wrong guess cannot evict an expert needed in the same token.
  Copies run on a side stream, so they do not block the layer's launches. Use with `LB_LA_WORKER=1 LB_LA_SPARE=1 LB_LA_PREFETCH_CAP=2`.
* `LB_PRED_DUMP=<file>`: the trace writer used for this study.
* Cost of the machinery measured on the card (VAL prompts, 5120 MiB, single sessions): the residual read plus the worker with the threshold at 1.0 (no copies issued) is 30.6 tok/s against 31.4 (-2.5 %);
  with `LB_E2_PROFILE`, per MoE layer: sync wait 388 -> 276 us, host busy 398 -> 455 us (+57 us = 2.2 ms a token: 26 us the residual read and scoring, 31 us issuing the copies, ~21 a token, ~0.06 ms each), token (issue to last sync) 30.2 -> 28.0 ms.
* Early routing as a device launch (norm + router per predicted layer) was not built: the worker does the same dots off the critical path, and the measured loss is host issue time, not the dots.

## 7. Card sweep on VAL prompts (5120 MiB, 6 prompts x 160 tokens, one session per row, no A-B-B-A; used only to pick the arm)

Default 30.9-31.7 tok/s across the sweeps. Lead 2, threshold 0.4, 2 guesses, 8 landings, worker, spare slots: 34.0 (no defer) / 33.7 (defer); one guess: 33.9; 16 landings: 33.9; lead 3: 33.3-33.7;
lead 4: 32.9; leads 1+2: 33.0; threshold 0.3 / 0.55 / 0.7: 33.8 / 33.8 / 32.9; no spare slots (repack on the main stream): 31.7. Every variant that covers 10-20 misses a token lands within 0.3 tok/s: the
prefetch issue cost cancels what extra coverage wins. Chosen arm: lead 2, theta 0.4, cap 2, 8 landings, worker, spare, no defer.

## 8. Gate B (on the card): FAIL (1.084x at 5120 MiB against the 1.25x gate)

A-B-B-A, same binary for both arms (the arm differs by switches only), `cm.py` prompts (TEST_A), 5 prompts x 288 tokens, median of every prompt of every session, paired ratio per round in brackets. Arm = lead 2, threshold 0.4,
2 guesses, 8 landings, worker, spare slots (section 6); the default arm has `LB_PRED` unset. Raw: `predict-results/runs2/g2-*` (final binary, with the miss gate), `g1-*` (first binary, before the gate).

| budget | default tok/s | predictor tok/s | x | text vs default |
|---|---|---|---|---|
| 5120 MiB | 29.85 | 32.34 | **1.084** (paired 1.083, 1.093) | identical 4/4 |
| 3072 MiB | 21.36 | 23.38 | 1.095 (one round, 2+2 sessions) | identical |
| 12500 MiB (cache forced, hit ~1) | 71.77 | 72.17 | 1.006 (paired 0.999, 1.008) | identical 4/4 |
| all resident (14500) | 98.05 | 98.28 | 1.002 | identical |
| 5120, `LB_CPU_MISS=1` | 35.70 (cpu only) | 32.45 (cpu + predictor) | **0.909** (paired 0.92, 0.892) | identical to the CPU arm |
| 3072, `LB_CPU_MISS=1` | 27.94 | 24.74 | 0.886 | identical to the CPU arm |

Gate verdicts: speed at 5120 **1.084x < 1.25x: FAIL**. No regression above 1 % at all-resident (1.002) and 12500 MiB (1.006): pass, but only with the miss gate (`pred_min_miss`): the first binary, without it, measured **0.875x** at 12500 MiB
(63.2 against 72.2 tok/s, paired 0.884, 0.870) because the residual read and the worker run every layer of a cache that never misses. The default token at 5120 was 29.5-29.9 here (30.0 in the sync doc). With CPU-served misses the predictor makes things worse
(0.91x, 0.89x): the CPU already removes the copies the prefetch hides, and prefetched copies then compete with the CPU's admitted copies for the bus and the host.

Correctness (`run_agree.sh`: teacher-forced through the E2 cache against the all-resident reference, 1590 positions; the predictor only chooses what to copy when, it cannot change what a layer computes):

| arm | cache | top-1 | max abs logit diff | KLD |
|---|---|---|---|---|
| default (final binary) | 5120 | 1590/1590 (100.00 %) | 0 | 0 |
| predictor | 5120 (991 slots + 8 spare/landing) | 1590/1590 (100.00 %) | 0 | 0 |
| predictor | 3072 | 1590/1590 (100.00 %) | 0 | 0 |

Why 1.084 and not the replay's 1.16-1.19: [measured, `LB_E2_PROFILE`, VAL prompts] the predictor's own cost is 2.5 % (the residual read and the worker with nothing issued: 30.6 against 31.4 tok/s) and the issue of ~21 copies a token is 1.2 ms of host
time in series; the per-layer sync wait falls 388 -> 276 us but the host busy time rises 398 -> 455 us. Replay at the engine's measured delay (about 0.5-0.6 ms) gives 1.08. Defaults: **`LB_PRED` off**; if on, `pred_theta` 0.4, `pred_min_miss` 6.


## 9. Closed by this study, and what is next

Closed: more history (recency, Markov, token tables) as a predictor of uncached experts (0.3-0.5 recall at 8 guesses, below the early-routing recall at 4); a learned linear correction of the early router; staging larger
than 16 (replay: no gain); several leads at once on one bus (waste crowds the demand copies); lead 1 (arrives late); a threshold as low as 0.3 (bus saturates); prefetching on top of CPU-computed misses (below).

Open, in order of measured value:
1. **Issue the copies from the worker thread (or one batched copy per expert).** The host spends +57 us a layer (2.2 ms a token, [measured]: 1.0 ms reading and scoring the residual, 1.2 ms issuing ~21 prefetches) and the main thread waits for the
   worker's answer: the replay at zero delay gives x1.19 against the engine's 1.08; the difference is issue latency and host time. [estimate] x1.15 if both go away.
2. **Early routing on the card** (norm + router launch on the residual, 64 scores read with the routes): removes the 2.5 % fixed cost of reading and scoring the residual on the host and about 0.2 ms of the delay.
3. **Token-start prefetch for the first MoE layers from the token-id table** (the only predictor that works there: 0.37-0.45 recall at layers 2-5, where early routing has no state); lead of 0-3 layers, so small.
4. Fewer bytes per miss: the replay's floor is the bus (34 copies x 0.55 ms); anything that makes a copy smaller or a miss rarer (the Belady headroom of 60 %) is worth more than recall now.

## 10. Reproduce

`cargo build --release --features cuda -p lb-server -p lb-lfm2 --bins`. Trace: `LB_E2=1 LB_LA_TRACE=<set>.jsonl LB_PRED_DUMP=<set>.dump` on the server, then `scripts/predict/run_set.py SET out.json N` per set (prompts in
`prompts.py`). Offline: `mkfeat.py VAL TEST_A TEST_B` (features, cross-fitted on TRAIN), `ana1.py` (section 2), `train.py` / `train_lr.py lrmin e,gap,log1p_rank,rank_lt4` (models), `final_sim.py DELAY KIND [crit]`
(section 5), `simtable.py`, `simcal.py`, `ana_entropy.py` (section 3). On the card: `LB_PRED=1 LB_LA_WORKER=1 LB_LA_SPARE=1 LB_LA_LANDINGS=8 LB_LA_PREFETCH_CAP=2 LB_PRED_LEADS=2 LB_PRED_THETA=0.4`.

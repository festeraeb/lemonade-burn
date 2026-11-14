# E2 CPU misses (`LB_CPU_MISS`): compute cache misses on the CPU

Branch `fable/engine-123-cpumiss-2026-11-14`. Item D of the 2026-11-14 idea audit
(`~/agentnotes/engine-technique-ledger.md` section 6). Raw runs: `/mnt/fast/eng123/cpumiss-results/`.

## What it is

* **Exact CPU kernel** `crates/lb-gguf/src/kernels_f32x.rs`: f32 activations, Q4_0 weights dequantized in
  registers (AVX2, AVX-512, scalar fallback, runtime dispatch). Same arithmetic as `kernels::dot_q4_0`; only the
  summation order differs. Test `every_level_agrees_with_the_scalar_reference` (worst row 9e-6 relative) and a row in
  `examples/i8_bench.rs`. Clean-room.
* **Spinning pool** `crates/lb-expert-host/src/spin.rs`: the workers spin on one generation counter for 20 ms after the
  last job, then block. `swiglu_batch` runs a layer's missed experts in two barriers (gate+up+SiLU, down).
  Rayon cost a futex wake-up per parallel region (6 per expert): 0.44 ms per expert before, 0.27 after (same run,
  `LB_CPU_MISS_ADMIT=0`, 69 misses a token, 18 -> 27 tok/s).
* **E2 path** `device_decode.rs`: with `LB_CPU_MISS=1` the router sync also returns the router input `h2` (same sync).
  A missed expert is copied in only on its Nth miss (`LB_CPU_MISS_ADMIT`, default 2, 0 = never, 1 = always); the
  others are computed on the CPU from the host GGUF bytes while the card runs the hits (the card's output row for that
  route is overwritten, stream-ordered, before `weighted_sum`, so the ascending-expert-id order is unchanged).
  Kernel: exact unless `LB_CPU_I8=1`. Both are live knobs (`/v1/tunables`: `cpu_miss`, `cpu_i8`). Default off.
* Counters in `/v1/placement` `stage`: `cpu_misses_per_token`, `cpu_miss_layers_per_token`, `cpu_miss_ms_per_token`,
  `cpu_kernel_ms_per_token`, `cpu_admitted_per_token`.
* `lfm2-check --cache-mb N --tf-ref FILE`: E2 cache in the checker; writes teacher-forced logits (all-resident run) or
  compares against them (top-1 %, max |logit diff|, mean KLD).

## Results (c4 P100 D8 x16, NUMA 1, 20 threads, greedy, 5 prompts x 288 decode tokens, NAUTIBURN unset)

Prompts are long-form (`cm.py`), not the E2 doc's set, so the baseline hit rate is 0.785 (E2 doc: 0.854) and the
baseline 24.6 tok/s (E2 doc: 30.5). tok/s = total tokens / total decode time of the session. A-B-B-A x 3 rounds,
6 sessions an arm. Misses = cache fills + CPU-served. All measured.

| budget | arm | tok/s median [min-max] | x base | ms/token | misses/token (copied + CPU) | hit | CPU ms/token | read wait ms/token | text vs base |
|---|---|---|---|---|---|---|---|---|---|
| 5120 | E2 copy misses (base) | 24.55 [24.40-25.69] | 1.000 | 40.7 | 32.7 + 0 | 0.785 | 0 | 31.8 | - |
| 5120 | CPU misses exact | 28.36 [27.18-29.04] | **1.155** (paired 1.115-1.175) | 35.3 | 17.1 + 17.5 | 0.772 | 7.4 | 23.7 | 6/6 identical |
| 5120 | CPU misses int8 | 29.18 [28.22-29.72] | 1.189 (paired 1.158-1.198) | 34.3 | 17.7 + 18.1 | 0.765 | 6.1 | 23.7 | 0/6 identical |
| 3072 | base | 18.05 [17.82-18.28] | 1.000 | 55.4 | 58.9 + 0 | 0.612 | 0 | 45.2 | - |
| 3072 | CPU misses exact | 20.50 [20.12-21.24] | 1.136 (1.132-1.146) | 48.8 | 30.5 + 31.0 | 0.596 | 12.0 | 31.4 | 6/6 identical |
| 3072 | CPU misses int8 | 20.86 [19.62-21.28] | 1.156 (1.129-1.174) | 47.9 | 31.2 + 31.8 | 0.586 | 11.2 | 31.6 | 0/6 identical |

**Gate 1.5x at 5120 MiB: FAIL, 1.155x (exact), 1.189x (int8).**

Agreement, teacher-forced through the E2 cache at 5120 MiB over 5 x ~320 positions (prompt + the baseline's own answer,
1590 positions) against the all-resident device decode:

| arm | top-1 | max abs logit diff | mean KLD |
|---|---|---|---|
| E2 cache, copy misses (control) | 100.00 % | 0 | 0 |
| CPU misses exact | **100.00 %** (1590/1590) | 2.5e-5 | 4.9e-12 |
| CPU misses int8 | 97.99 % (1558/1590) | 5.85 | 3.7e-3 |

int8 misses the 99 % bar: not recommended.

Kernel, one thread, 5.31 MB expert (t440, `i8_bench`): f32 reference 13.6 ms, int8 AVX-512 1.03-1.13, **exact AVX-512
0.84-0.87**, exact AVX2 1.39-1.56. On this CPU the exact kernel is faster than the int8 one. Pool microbench
(`spin_bench`, c4 node 1, 20 threads, cold weights): 0.29 ms (1 expert), 0.34 (2), 0.56 (4), i.e. 0.14-0.29 ms per
expert, 35 GB/s of a 65 GB/s socket.

## Where the time goes (5120 MiB, exact, per token)

| part | ms | source |
|---|---|---|
| all-resident device decode | 9.9 | measured, 101 tok/s |
| E2 path at ~100 % hit (38 router syncs + launches), cache 12500 MiB | 19.9 | measured, 50.2 tok/s, 0.22 misses a token |
| CPU compute of the misses, serial after each router sync | 7.4 (7.0 in the expert batch) | measured |
| admitted copies (17 a token x ~0.45 ms) and the rest | ~8 | remainder |
| token | 35.3 | measured |

* The ledger's estimate (~15 ms, ~65 tok/s) subtracted the whole 25 ms read wait. That wait is mostly the device's own
  work behind 38 syncs (the E2 path costs 10 ms over all-resident even with no miss), not copy time. The part CPU misses
  can remove is the copy share only (13.9 ms in the E2 doc); the CPU work is added back in series.
* Pure CPU (admit 0, cache frozen at its seed): 26.3 tok/s, 69 misses a token, 18.7 ms CPU, 38.1 ms token = 19.9 + 18.7.
* admit 4: 30.0 tok/s (single run, 160 tokens): 9.1 copied + 29.7 CPU, 10.9 ms CPU. admit 2 28.9, admit 0 26.3 (same
  length, single runs): flat within noise above admit 2.
* Threads: 12 threads 25.6, 20 threads 28.9 (single runs).
* The CPU kernel is serial with the card: layer L's misses cannot start before the routes are read and layer L+1 cannot
  start before they are summed. Per layer with a miss the CPU costs ~0.5 ms (0.27-0.41 ms an expert in the server,
  0.14-0.29 in the microbench).

## What would have to change to reach 1.5x

1. The 10 ms of syncs and launch work the E2 path pays at 100 % hit (item E: skip the routing read-back where the whole
   layer is resident, or route a layer ahead). With that removed the same CPU misses would give ~25 ms (40 tok/s),
   [estimate].
2. CPU expert cost per miss toward the 0.08 ms bandwidth floor (software prefetch, rows split by cache line, both
   sockets are slower: node-local only).
3. Overlap CPU misses with the previous layer's device work (needs the router input a layer early: the device guess of
   `LB_DEVICE_GUESS` is 0.78 recall).

## Verdict

Behind the switch, default off. The exact arm is a safe 1.14-1.16x (bit-compatible to 2.5e-5 in logits, same text 12/12);
the int8 arm is 0.03x faster and changes text: do not use. Not a 1.5x lever on its own.

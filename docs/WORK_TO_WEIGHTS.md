# Work goes to the weights (W-series)

Branch `fable/work-to-weights-2026-11-11` (off `fable/engine-e3-lookahead-2026-11-09`). Plan and gates:
`~/agentnotes/work-to-weights-plan-2026-11-11.md`. Data: `/mnt/fast/w2w/results/*.jsonl` (append-only),
`/mnt/fast/w2w/RESUME.md`.

An expert's weights never move after load. An **expert host** holds a set of `(layer, expert)` and
computes them; the **trunk** sends each MoE layer's activation (2048 f32 = 8 KB on LFM2.5) to the hosts
that hold the routed experts, gets the per-expert outputs back **unweighted**, and does the weighted
ascending-expert-id sum itself, so the float order is the one-process path's.

Figures are **[measured]** (run here, command given) or **[estimate]**.

## W0: the hop budget

Gate, written before the run (plan): **RoCE round trip p50 <= 60 us; 25G TCP p50 <= 250 us.**

What was measured: one real `Run` request (layer, 4 expert ids, one row of 2048 f32: **8,260 bytes**)
and its real reply (4 x 2048 f32: **32,800 bytes**) through the real encode, transport and decode, against
an echo host that does no arithmetic (it copies the activation into each output; its own time is
0.6-2.5 us p50). 10,000 round trips after 1,000 warm-up, single connection, wall clock around
encode + send + receive + decode on the client. `block` = both sides sleep in the kernel (TCP: blocking
read; RDMA: completion channel); `busy` = both sides spin; `spin:200` = spin 200 us, then sleep.
Load is the 1-minute load average read just before the run (t440 has 48 threads, c4 80).

Commands: `lb-hop-bench local --out ...` and `bench/w2w/w0-remote.sh <ssh-host> <ip> <binary> <label> <tcp|rdma>
<server-wait> <client-wait>`. All 27 rows: `/mnt/fast/w2w/results/w0-hop.jsonl`. **[measured]** 2026-11-11.

| path | wait | p50 us | p99 us | p99.9 us | load t440 / far side | gate |
|---|---|---|---|---|---|---|
| same process, channel | block | 15.7 | 25.9 | 36.7 | 6.0 / - | |
| same process, channel | busy | 9.8 | 14.3 | 27.1 | 6.0 / - | |
| same host, loopback TCP_NODELAY | block | 39.3 | 58.5 | 70.2 | 6.0 / - | |
| same host, loopback TCP_NODELAY | busy | 29.2 | 46.7 | 57.0 | 6.0 / - | |
| same host, unix socket | block | 35.5 | 56.5 | 73.4 | 6.0 / - | |
| same host, unix socket | busy | 28.8 | 46.8 | 60.0 | 6.0 / - | |
| t440 <-> c4, 25G TCP_NODELAY | block | 113.6-129.1 (4 runs) | 146-215 | 186-286 | 4.3-5.8 / 1.5-6.6 | **pass** (<= 250) |
| t440 <-> c4, 25G TCP_NODELAY | busy | 104.5 | 157.9 | 212.8 | 5.6 / 1.5 | pass |
| t440 <-> c4, 25G TCP_NODELAY | spin:200 | 106.8 | 141.5 | 218.5 | 6.0 / 1.0 | pass |
| t440 <-> c4, RoCE v2 send/recv | block | 57.7, 69.2, 74.3, 73.4, 69.4 | 94-119 | 121-210 | 4.6-6.1 / 1.0-6.6 | **FAIL** (4 of 5 runs > 60) |
| t440 <-> c4, RoCE v2 send/recv | busy | 33.8 | 44.2 | 51.7 | 6.0 / 1.0 | **pass** (<= 60) |
| t440 <-> c4, RoCE v2 send/recv | spin:200 | 34.1, 34.0, 33.9, 34.4 | 41-50 | 52-66 | 4.3-6.0 / 1.0-6.2 | **pass** |
| t440 <-> c4, RoCE v2 | host block, trunk busy | 51.9 | 86.8 | 169.9 | 6.0 / 1.0 | pass |
| t440 <-> c2, 1 GbE TCP_NODELAY | block | 533.9 | 617.5 | 816.0 | 5.5 / 0.5 | |
| t440 <-> c2, 1 GbE TCP_NODELAY | busy | 511.7 | 586.0 | 777.2 | 5.3 / 0.5 | |
| t440 <-> c3, 1 GbE TCP_NODELAY (*) | block | 611.0 | 968.9 | 2310.1 | 5.5 / 6.8 | |
| t440 <-> c3, 1 GbE TCP_NODELAY (*) | busy | 619.2 | 722.0 | 913.1 | 5.3 / 6.8 | |

(*) c3's firewall drops inbound connections on unlisted ports and was not changed; the echo host ran on
t440 and the client on c3, so the 8 KB travels c3 -> t440 and the 32 KB back. Same link, same bytes, roles
swapped.

**Gate result.** 25G TCP **passes** (p50 114-129 us against 250). RoCE **passes with a spinning wait
(34 us) and fails with a sleeping one (58-74 us, over 60 in four of five runs)**: waking from the
completion channel costs about 35 us a round trip. So an RDMA host is run with `--wait spin:200` (or
`busy`) and the trunk's placement gives that host `"wait": "spin:200"`; `spin:200` costs nothing over
`busy` here and stops burning a core 200 us after the last request. c4 was lightly loaded during these
runs (load 1-7, not the ~16 expected), so the c4 numbers are for a mostly idle far side; row 9 of the
JSONL (57.7 us) was taken with the build before the peer-close watch was added to the RDMA wait.

About 10 us of every round trip is not the wire: it is encoding 8 KB, copying 33 KB out of the reply and
the channel itself (the same-process row). RoCE spends about 24 us on the link, TCP about 95-120 us.

### Per-token budget

Budget = (MoE layers) x (hosts touched per layer) x round trip. LFM2.5-24B has 38 MoE layers. The trunk
sends to every host of a layer before it reads any reply, so with more than one host the round trips
overlap and the cost per layer is nearer the slowest host's round trip than the sum; the table gives one
host per layer (also the lower bound for several) and the sum for two (the upper bound).

| transport (p50) | 1 host/layer | 2 hosts/layer, summed | against 12.5 ms/token (80 tok/s, D3 one-card) |
|---|---|---|---|
| same process, channel, busy (9.8 us) | 0.37 ms | 0.74 ms | +3 % -> 0.97x |
| same host, unix socket, busy (28.8 us) | 1.09 ms | 2.19 ms | +9 % -> 0.92x |
| RoCE, spin (34 us) | 1.29 ms | 2.58 ms | +10 % -> 0.91x |
| RoCE, block (70 us) | 2.66 ms | 5.3 ms | +21 % -> 0.82x |
| 25G TCP, block (125 us) | 4.75 ms | 9.5 ms | +38 % -> 0.72x |
| 1 GbE TCP (530-610 us) | 20-23 ms | 40-46 ms | 0.35-0.38x |

**[estimate]** The last column is arithmetic on the measured round trip only: it adds the hop to a
12.5 ms token and nothing else. It leaves out what a hop also costs the trunk (one device read of the
activation and one upload of the summed output per MoE layer, where the all-resident path syncs once a
token) and what it saves (experts computed while the trunk is idle are free). W3 measures that.

What the budget says about who can be a host:
* **Same box, second card**: 0.4-1.1 ms a token. Fine for every W3 case.
* **RoCE (t440 <-> c4)**: 1.3 ms a token with a spinning wait: inside W3a's 0.80x gate by budget.
* **25G TCP**: 4.75 ms a token: by budget alone 0.72x, **under W3a's 0.80x**. TCP hosts are for the
  spill case (W3d), where the alternative is a 0.28-0.9 ms miss per expert, not for all-remote.
* **1 GbE ("other people's gear")**: 20+ ms a token, a 44-50 tok/s ceiling before any compute. A 1 GbE
  host only makes sense for experts that are rarely routed, or for a model that fits nowhere else;
  batching several sequences per hop (plan, "Later") is what would change this.

## W1: the expert host

Crate `crates/lb-expert-host` (lib + two binaries). It depends on `lb-gguf` only (and on `lb-q4-gpu`
behind a feature), never on the trunk; the trunk depends on it.

| file | what |
|---|---|
| `src/wire.rs` | the frame format. `Run {layer, ids[], row groups, x[m][d]}` -> `RunOk {host_ns, y}` with one **unweighted** output block per expert, or `Refused {ids}`, or `Error`. `Hello` -> model fingerprint, shapes, held bitmap, backend, CPU arithmetic fingerprint |
| `src/transport.rs` | `Conn` / `Listener` traits; in-process channel, TCP (`TCP_NODELAY`), unix socket; wait modes `block`, `busy`, `spin:<us>` |
| `src/rdma.rs`, `csrc/w2w_rdma.c` (feature `rdma`) | RoCE / InfiniBand reliable-connected queue pair, SEND/RECV over a ring of registered receive buffers, frames larger than 1 MiB fragmented; the queue pair numbers travel over a TCP rendezvous (`rdma://host:port`), which also tells each side when the other has gone; device and RoCE v2 GID index are found from the rendezvous socket's own address, never configured by index. The shim follows our Nauti-Virtualizer `nauti-fabric` one (same author) and adds send/recv and the three waits |
| `src/placement.rs` | placement as data: one JSON file read by trunk and hosts (below) |
| `src/model.rs` | the expert stacks of a GGUF found by name, nothing else of the model loaded; the model fingerprint |
| `src/expert.rs` | the CPU arithmetic of one expert (`swiglu`). **One source**: `lb_lfm2::moe::expert_ffn` now calls it, so a local expert and a remote one are the same instructions |
| `src/host.rs` | `Host` (held set, refusal, stats), `Backend` trait, `CpuBackend`, `GpuBackend` (features `gpu` / `cuda`) |
| `src/client.rs` | the trunk's end of a connection: hello, `send_run`, `recv_run(deadline)` |
| `src/testgguf.rs` | a tiny synthetic `lfm2moe` GGUF (5 layers, 8 experts, d = 64) written in milliseconds, so the whole path runs under `cargo test` without the 13 GB file |
| `src/bin/lb-expert-host.rs` | the host process |
| `src/bin/lb-hop-bench.rs` | W0's bench; `arith` prints the CPU arithmetic fingerprint |

**Why the request carries row groups.** A decode token is one activation row and every routed expert runs
on it: 8 KB out, k x 8 KB back, the plan's shape. A prefill pass is m rows and each expert gets the subset
of rows routed to it. The CPU kernel takes a fused path for a one-row batch and a dequantize-then-dot path
for a many-row one, and the two differ in the last bit, so an expert must be given exactly the rows the
one-process path would give it, as one batch. The request therefore lists, per expert, its rows; the host
runs each expert once on its batch. (A mutation that ran the rows one at a time on the host is caught by
the tests.)

**A host refuses what it does not hold.** `Host::handle` checks every id of a `Run` against the held set
before any arithmetic; if one is missing the whole request is answered `Refused` with the missing ids:
no partial answer, no zeros, and nothing loaded on the side. A layer without experts, a layer or expert id
past the model and a wrong activation width are refused or errored the same way. The hello carries a
fingerprint of the model file (length, first MiB, 4 KiB of every expert stack), and a trunk never uses a
host whose fingerprint is not its own.

**Backends.** `CpuBackend` computes from the file mapping with `swiglu`; "held" is only the set it answers
for. `GpuBackend` uploads the held experts through `lb-q4-gpu` exactly as the trunk's own resident store
does (a layer held whole as one stacked buffer per projection, otherwise per expert) and runs them through
the same `GpuExpert::launch` / `read_all`; on a placement reload it releases what it lost and uploads what
it gained. `Gpu::open()` picks the runtime, so the same backend is CUDA (`cuda` feature,
`LB_CUDA_DEVICE`) or wgpu (`LB_WGPU_DEVICE`: Polaris / Intel / other people's cards). That is the wgpu
seam: no separate code, a different device behind `Gpu`.
**The device backend is written and compiles (`cargo check` and a release build with `--features
cuda,rdma`); it has not been run: no GPU memory was used anywhere in this phase.**

**CPU arithmetic across machines.** Everything in an expert is plain IEEE f32 in a fixed order except
`exp`, which is the C library's. `expert::arith_fingerprint` hashes `silu` over a 2^20-point sweep; a CPU
host says it in the hello and the trunk warns when it differs from its own. **[measured]** t440 (Xeon Gold
6226, glibc 2.39), c4 (Gold 6148, glibc 2.44), c2 (E5-2450 v2, no FMA, glibc 2.44) and c3 (E3-1265L v3,
glibc 2.44) all give `3485a00c79570e6d`, so a CPU host on any of them is expected to be bit-identical to a
t440 trunk. A cross-node CPU-host parity run was not done (it would have pulled 12 GB of the model through
c4's page cache beside its serving seats).

## W2: the trunk side

`crates/lb-lfm2/src/remote.rs` (`RemoteTier`), hooked into `moe::moe_ffn_with`, attached with
`Lfm2::attach_remote(placement_file)`.

* **Placement is data.** A JSON map layer -> expert -> host (`lb_expert_host::placement`):

  ```json
  { "version": 1,
    "hosts": { "a": { "addr": "rdma://c4-25g:7411", "wait": "spin:200", "deadline_ms": 50 },
               "b": { "addr": "unix:///run/w2w-b.sock" } },
    "default": "local",
    "layers": { "*": "a", "2-9": "local", "12": { "0-31": "a", "32-63": "b", "7": "local" } } }
  ```
  The most specific entry wins (expert over layer, single layer over range, range over `*`, `*` over
  `default`); `local` is the trunk; layers without experts are ignored. Host fields: `addr`,
  `deadline_ms` (default 1000), `deadline_row_ms` (added per extra activation row of a prefill batch,
  default 250), `wait` (`block` | `busy` | `spin:<us>`), `retry_s` (0 = a host that was marked down is
  tried again only when the file is reloaded). **Reloadable at run time**: the trunk looks at the file's
  modification time between forward passes (at most every 250 ms) and a host process re-reads it every
  500 ms; a file that does not parse leaves the old map in force. This is the hook for the
  load-by-prompt-type goal: rewrite the file, the experts move.
* **One message per host per layer.** `serve_layer` groups the layer's routed experts by host, sends every
  message before it reads any reply (the hosts work side by side), then collects.
* **Deadline, mark-down, fall back.** A reply must arrive within `deadline_ms` (+ `deadline_row_ms` per
  extra row) of the send. A host that misses it, drops the connection, answers `Error` or answers with the
  wrong shape is **marked down**: its connection is closed and its experts are not sent again. A host that
  answers `Refused` is believed, not marked down: those experts are local for 2 s, then asked again (so a
  host that reloads its placement a moment after the trunk is picked up). Whatever a host did not answer
  stays in the layer's expert list and runs on **the local path, unchanged**: CPU, resident, on-demand
  miss or the E1 cache, whichever the build and flags give. The outputs of both sides are then weighted
  and summed in ascending expert id by one closure, so where an expert ran is not visible in the result.
  The cost of a dead host is one deadline, once.
* **Hop count and hop ms per token.** Every forward pass closes with a `PassRecord` (`pos0`, `m`, `hops`,
  `experts_remote`, `experts_fallback`, `hop_wall_ms`, `hop_net_ms` = round trips minus the hosts' own
  compute, `host_ms`, hosts down). `LB_W2W_LOG=<file>` appends one JSON line per pass;
  `RemoteTier::status()` (counters and host states) is in the server's `GET /v1/placement` under `w2w`.
* **Where it is wired.** `lemonade-burn-server --w2w-placement <file>` (`LB_W2W_PLACEMENT`),
  `lfm2-check --w2w-placement <file>`, and the harness `w2w-check`. On a device trunk (`--gpu`) a remote
  tier sends every MoE layer through the host side of the device decode (`dd_host_moe` is forced on, the
  E2 device router and stacked layers are not used: the activation has to leave the card to be sent
  anyway), and with the E1 cache on, local experts come from the cache and are summed with the remote ones
  in ascending id. **That device-trunk wiring compiles (`cargo check --features cuda`) and has not been
  run.**

Known limit, for W3: the trunk asks its hosts and waits for them **before** it runs its own local experts
of the same layer; the two are not overlapped yet. It costs nothing when a layer is all-remote or
all-local and one round trip of latency when a layer is mixed.

## W1 and W2: gates and results

All on t440, CPU reference path only, hosts as separate processes with the CPU backend, **no GPU memory
used and no service touched**. Harness: `crates/lb-lfm2/src/bin/w2w-check.rs`, driver
`bench/w2w/w1w2-parity.sh`. One process loads the model once, runs the reference arm with no remote tier,
then attaches each placement in turn and repeats; every logit of every pass is compared **as bits**.
An arm also fails if its hosts were not all up at the start, if no expert was answered remotely, or if
anything fell back without a kill (an arm that silently ran everything locally would otherwise "pass").

* **prefill**: the three lfm2-check harness prompts (29 + 44 + 13 = **86 positions** with BOS; the texts
  are the ones in `/data/lfm2-scrapes/c4-out/cpu2.txt`), each as one pass: the batched wire shape.
* **decode**: the same 86 positions token by token, teacher-forced: the one-row wire shape.
* **greedy**: 32 tokens from the sonar prompt, text sha as the e2e driver computes it.

The reference arm's own greedy sha on the CPU path is **`062392b8d2b0361a`**, the one-card sha, so the
full sonar-prompt gate was run as written, not a shortened stand-in.

Run 2 (the committed code), **[measured]** 2026-11-11 02:18-02:31Z, t440 load 14-41 during the arms, host
processes capped at 8 threads and the trunk at 16. Rows: `/mnt/fast/w2w/results/w2w-parity.jsonl`, label
`w1w2-parity-run2` (run 1, label `w1w2-parity`, is the same seven first arms on the code just before the
last harness changes: all pass, same sha). Per-pass hop records:
`/mnt/fast/w2w/results/w2w-passes-*.jsonl`.

| arm | placement | prefill 86 | decode 86 | greedy 32 | sha | hops / token | fell back |
|---|---|---|---|---|---|---|---|
| a-unix | all 2432 experts on one host, unix socket | 86/86 bits | 86/86 bits | 32/32 bits | 062392b8d2b0361a | 38 | 0 |
| a-tcp | the same host over loopback TCP | 86/86 | 86/86 | 32/32 | 062392b8d2b0361a | 38 | 0 |
| a-rdma | the same host over RoCE (t440's own port, `spin:200`) | 86/86 | 86/86 | 32/32 | 062392b8d2b0361a | 38 | 0 |
| b-split | experts 0-31 on host a (TCP), 32-63 on host b (unix) | 86/86 | 86/86 | 32/32 | 062392b8d2b0361a | 66-76 (mean 71.5) | 0 |
| b-mixed | as b-split, layers 2-9 and experts 60-63 on the trunk | 86/86 | 86/86 | 32/32 | 062392b8d2b0361a | 50-60 (mean 55.2) | 0 (6,219 expert calls local by placement) |
| c-kill-decode | split; host b SIGKILLed before decode position 40 | 86/86 | 86/86 | - | - | up to 74 before, 32-38 after | 3,545 expert calls, 46 tokens with b down |
| c-kill-greedy | split; a fresh host b SIGKILLed before greedy step 12 | - | - | 32/32 | 062392b8d2b0361a | up to 73 before, 31-38 after | 1,644 expert calls, 20 tokens with b down |
| c-hang-greedy | split; a fresh host b SIGSTOPped before greedy step 12, deadline 3000 ms | - | - | 32/32 | 062392b8d2b0361a | as above | 1,644; 1 timeout, found once |

`max |logit diff|` is 0.0 in every cell. In the kill arms the trunk logged `host b marked down, its experts
run locally from here: connection lost on layer 2: Connection reset by peer`; in the hang arm `no reply for
layer 2 inside 3000 ms`; one mark-down each, and the dead host was not asked again.

**W1 gate: teacher-forced parity with the one-process CPU path at all 86 positions: PASS** (arms a-*:
86/86 bit-identical in both the batched and the one-row shape, over three transports).
**W1 gate: a host refuses an expert it does not hold: PASS** (`tests/host.rs`
`a_host_refuses_an_expert_it_does_not_hold`).
**W2 gate (a) all experts on one remote host: PASS. (b) split over two hosts: PASS. (c) one host killed
mid-decode: PASS**, each with greedy sha `062392b8d2b0361a` and bit-identical logits.

Tests (`cargo test -p lb-expert-host -p lb-lfm2 -p lb-server`, all pass): lb-expert-host 9 unit (wire,
placement, transports incl. timeouts and a dead peer, rdma rendezvous line) + 3 integration (`tests/host.rs`:
model view, every transport bit-for-bit against the shared kernel in both wire shapes, refusal and held-set
reload); lb-lfm2 27 unit (unchanged) + 5 integration on the synthetic model (`tests/w2w.rs`: one host, two
hosts + trunk, a host that dies and one that hangs mid-decode, a refusing host / a host on another model /
an absent host, the placement file rewritten while the model runs and a broken file ignored); lb-server 7.
Two mutations were tried and are caught: summing remote parts unsorted, and running a host's batch one row
at a time.

What the CPU runs say about speed: nothing. The hosts and the trunk shared one loaded machine, and a CPU
host spent 8-10 ms a layer on its four experts (8 threads). `hop_net_ms` per token was 5.7 ms over the unix socket for 38 hops (0.15 ms a hop,
blocking wait, load 16) and 7.5-7.9 ms over loopback TCP and RoCE loopback in the same conditions; these
are sanity readings of the accounting, not W0 numbers.

## Not run, not verified

* **No card was used.** `GpuBackend` (host on CUDA / wgpu) and the device-trunk wiring (remote tier inside
  the device-resident decode, remote + E1 cache merge in `moe_cached`'s caller) compile with
  `--features cuda` and are **unrun**. The claim that a GPU host returns what a GPU trunk computes for the
  same expert rests on both calling `GpuExpert::launch` on the same uploaded words; it has not been
  observed. On a mixed CPU-host / GPU-trunk layout the outputs would agree to rounding only.
* **No cross-node host run.** Trunk and hosts were all processes on t440. RoCE was exercised cross-node
  only by the W0 echo bench (t440 <-> c4, 9 runs of 10,000 trips) and by a real host only over t440's own
  port (arm a-rdma). The 25G TCP path with a real host was not run.
* **The llama.cpp oracle of lfm2-check was not run.** The gate is parity with the one-process path, and
  that is what was compared.
* **c3 as a far side for an inbound connection**: its firewall drops the port; measured with the roles
  swapped (W0 table note).
* c4 was at load 1-7 during W0, not the ~16 expected, so the c4 round trips are for a mostly idle host.
* Run 1 of the parity harness used every core for three processes and pushed t440's load average to 56
  for about five minutes (nice 10); run 2 is capped. The CUDA release build did the same for four minutes.

## For W3

Binaries (built on t440, glibc 2.39, from this branch): `/mnt/fast/w2w/bin/` (`/mnt/t440/fast/w2w/bin/` on
c4 and c2): `lb-expert-host-cuda-rdma`, `lemonade-burn-server-cuda-rdma`, `lfm2-check-cuda-rdma`,
`w2w-check-cuda-rdma` (need libibverbs.so.1 and the CUDA 12.8 libraries at run time), and CPU builds
`lb-expert-host-cpu-rdma`, `lemonade-burn-server-cpu`, `lfm2-check-cpu`, `w2w-check-cpu`,
`lb-hop-bench-{tcp,rdma}`.

A host on a named card (UUID pin through the environment, as the engine benches do):

```
CUDA_VISIBLE_DEVICES=GPU-<uuid> LB_CUDA_DEVICE=0 \
  lb-expert-host-cuda-rdma --model /mnt/t440/raid0/models/lfm25/LFM2.5-24B-A2B-QAD-Q4_0.gguf \
  --placement /mnt/t440/fast/w2w/run/w3a.json --name h1 --backend gpu \
  --listen unix:///tmp/w2w-h1.sock --listen rdma://c4-25g:7411 --wait spin:200
```

The trunk on a named card, no experts of its own (dense weights and head only):

```
CUDA_VISIBLE_DEVICES=GPU-<uuid> LB_CUDA_DEVICE=0 \
  lemonade-burn-server-cuda-rdma --model <same file> --gpu --gpu-budget-mb 0 --gpu-dense-mb 600 \
  --w2w-placement /mnt/t440/fast/w2w/run/w3a.json --port <p>
```

and for the spill case (W3d) `--gpu-budget-mb 5120` with a placement that leaves some experts `local`: the
local ones come from the E1 cache. On-card parity first: `lfm2-check-cuda-rdma --tokenizer ... --ab-device
--gpu --gpu-dense-mb 600 --w2w-placement ...` (greedy sha must be `062392b8d2b0361a`).

Placement for W3a (`w3a.json`): `{"hosts":{"h1":{"addr":"unix:///tmp/w2w-h1.sock","wait":"spin:200",
"deadline_ms":200}},"layers":{"*":"h1"}}`; for a split, place **whole layers** per host
(`"2-20":"h1","21-39":"h2"`): a layer held whole is uploaded as one stacked buffer.

Expected VRAM **[estimate]**, from the measured all-resident figures in `moe.rs` (13,775 MiB for all 2,432
experts as stacked layers, 15,291 MiB uploaded per expert, both on a P100): **about 360 MiB per whole MoE
layer (5.7 MiB an expert)**, about 6.3 MiB an expert when a layer is split between hosts, plus the CUDA
context and pool (a few hundred MiB). One host with everything: ~13.8 GiB (fits a 16 GiB P100, not a
2060 SUPER or a 1070); two hosts with 19 layers each: ~6.9 GiB each; four with 9-10 layers: ~3.4-3.6 GiB.
The trunk with no experts: the dense weights (424 MB) and the head, under 1 GiB with the context.

Knobs: host `--wait block|busy|spin:<us>`, `--backend cpu|gpu`, `--threads`, `--stats-s`,
`LB_CUDA_DEVICE` / `LB_WGPU_DEVICE`, `LB_RDMA_DEV` + `LB_RDMA_GID` (override the GID search); placement
`addr`, `wait`, `deadline_ms`, `deadline_row_ms`, `retry_s`; trunk `--w2w-placement` / `LB_W2W_PLACEMENT`,
`LB_W2W_LOG`. With a remote tier attached `LB_E2` has no effect (the device router is not used) and
`LB_DD_HOST_MOE` is implied. E3 look-ahead stays off (it is off by default: `LB_LOOKAHEAD` unset).

Risks to look at first in W3 **[estimate]**: (1) a device trunk with a remote tier reads the activation
off the card and uploads the sum once per MoE layer, 38 syncs a token where the all-resident path has one;
W0's budget does not include that. (2) Mixed layers are not overlapped (see W2). (3) `GpuBackend::run`
uploads the activation and reads the outputs back per request: one sync on the host per hop.

A first guess at W3 from numbers already in this repo **[estimate, nothing here was run]**: E2 measured
6.6 ms a token for 39 host round trips on c4's P100 (0.17 ms a sync, `docs/ENGINE_E2_ROUTER.md`). A device
trunk with every expert remote pays that (38 syncs), the hop (0.4-1.3 ms a token, W0) and the host's own
launch-and-read per hop, on top of a 12.5 ms all-resident token: somewhere near 20-25 ms a token, 0.5-0.6x
of one card all-resident. **W3a's 0.80x gate looks out of reach as built**; the sync per MoE layer, not the
wire, is what to attack (keep the router on the card and read only the activation, or put host and trunk
cards in one process). W3d's gate is the easier one: the one-card spilling E2 token is 32.7 ms at 5120 MiB
(13.9 ms of it misses), so a 20-25 ms work-to-weights token would be 1.3-1.6x.

## W3: measured on our gear (c4)

Gates written **before** any W3 run (2026-11-11 03:15Z, resumed run; the first W3 agent stopped before
running anything). c4 only: five P100s freed for this through the fleet manifest (glm53-strata disabled;
the manifest was not touched by this run). Cards by UUID only (`CUDA_DEVICE_ORDER=PCI_BUS_ID`,
`CUDA_VISIBLE_DEVICES=GPU-...`): trunk **D8** (x16, NUMA 1, NIC node), first host **AF** (x16, NUMA 1),
then 86 (x8, NUMA 1), 3B and 3C (x8, NUMA 0). The GGUF's page cache is moved to NUMA 1 first
(`bench/w2w/numa-load.sh`) and every process is bound to its card's node. Model: c4's local copy
`$HOME/lb-e2/models/LFM2.5-24B-A2B-QAD-Q4_0.gguf` (same 13,578,190,880 bytes as
`/mnt/t440/raid0/models/lfm25/`; the E2 runs used it), prompts `$HOME/lb-e2/prompts.jsonl`
(5 prompts, 288 decode tokens a round, the E2 set). Driver `bench/w2w/w3-session.sh`, client `w2w-drive`.

**P. On-card parity (before any timing).** GPU host on AF holding every expert, device trunk on D8 with
no experts of its own, `lfm2-check --ab-device --gpu --gpu-dense-mb 900 --w2w-placement`: the old path
and the device path both with every expert on the host. Gate: greedy sonar sha **062392b8d2b0361a** on
both arms, teacher-forced top-1 old == device at all 86 positions (the D1 parity gate), 0 fallbacks.
If it fails: stop, diagnose, no timing.

**Arms** (one server session each; every completion's text sha checked against the one-card shas
sonar-32 `062392b8d2b0361a`, code-03 `c5d77b6cf1b729f2`, prose-02 `b89a1212137bf00d`, qa-05
`48fe35aec6ccb119`, forge-04 `d7938eeacde472db`, the lemonade-burn shas of every E2 session):
* `allres`: D8 alone, every expert resident as stacked layers (`--gpu-budget-mb 14500`), 1 sync a token.
* `spill`: D8 alone, E2 cache at 5120 MiB of experts (`LB_EXPERT_VRAM_MIB=5120 LB_E2=1`), as E2's gate arm.
* `w3a`: trunk D8 with no experts, every expert on a GPU host on AF (unix socket, `spin:200`).
* `w3b2` / `w3b4`: trunk D8, experts as whole layers over AF+86 / AF+86+3B+3C.
* `w3d`: trunk D8 holds whole MoE layers 2-15 itself (14 layers, about 5,075 MiB of device bytes,
  <= 5120 MiB), layers 16-39 on a GPU host on AF.
Rounds: at least 2, each round runs every arm once, odd rounds in the order above, even rounds
reversed; 5 completions per arm per round. tok/s of a session = decode tokens / decode seconds over its
5 completions; an arm's figure is the median of its sessions (min-max given).

**Gates** (plan, unchanged): **W3a >= 0.80x allres** (same session block); **W3d >= 1.25x spill**;
**every completion of every arm gives its one-card sha**; no expert falls back (hosts up throughout).
W3b has no speed gate in the plan: reported against allres and w3a. Hop ms and hops per token are
reported against W0's budget (unix socket, busy: 28.8 us a hop, 1.09 ms a token at one host a layer).

**W3c, cross-node wire test (not a speed claim).** t440's P100 is a live seat and is not used. A
**CPU-backend** host on t440 holds layers 36-39, over RoCE (`spin:200`) and then over 25G TCP; the
rest as w3a (trunk D8, layers 2-35 on AF). Gate: sha on every completion, the t440 host answers every
expert placed on it (0 fallbacks), hop ms per token recorded. A CPU expert's arithmetic is not the
P100's, so a sha difference here would be rounding, not a wire fault: if it happens it is reported, not
hidden, and the run is repeated with the t440 host's layers checked against a CPU trunk.

**W3e (only if W3a fails and time allows): device-to-device in one process.** When host cards are in
the trunk's box, the trunk's process opens them itself and keeps the activation on device (expert
outputs copied card to card, one sync a token), sockets only for other boxes. Same gates as W3a (sha +
>= 0.80x allres).

**W3 host v2 (written before its run, 03:25Z).** Rounds 1-2 of the arms as built showed the GPU host's
own time is most of the hop (25 ms of a 29 ms hop per token in w3a). Profile (`LB_HOST_PROFILE`): the W1
host launched every expert of a message separately (2-3 dispatches and a scratch set each): 135-245 us
to queue, 600-630 us in the read, per message. v2 runs a message for a layer held whole as **one routed
launch of the stacked layer** (`GpuExpertLayer::launch` with host routes: two dispatches), one upload,
one read: 45 us to queue, 270-310 us in the read; on-card parity unchanged (both arms sha
062392b8d2b0361a, 86/86, the same max logit difference). `LB_HOST_LAUNCH=per-expert` keeps the old path.
The arms are re-run with the v2 host (labels `w3v2-*`), allres and spill in the same rounds, **same gates**.

### W3 results: P, a, b, d (c4, 2026-11-11 03:05-03:30Z) [measured]

**P, on-card parity: PASS.** GPU host on AF holding all 2,432 experts (12,909 MiB), device trunk on D8,
`lfm2-check --ab-device`: teacher-forced top-1 old == device **86/86**, max |logit diff| 2.19e-5, 0 route
differences, 0 fallbacks; greedy sonar sha **062392b8d2b0361a** on both arms, token for token identical.
CPU trunk (no card) against the same GPU host (`w2w-check`, decode + greedy): top-1 86/86, max |logit
diff| 2.8e-5, greedy sha **062392b8d2b0361a**; logits are not bit-identical (a P100 expert against the CPU
reference: rounding, as expected; the harness's bit check reports that arm "FAIL" for that reason only).
Logs `logs/w3/parity-p1-*.log`, rows `results/w3/parity-cpu-trunk.jsonl`, report `results/w3/parity-p1-ab.json`.

Arms: `results/w3/runs.jsonl` (one row per completion), `results/w3/sessions.jsonl` (VRAM per card, NUMA
map: the model's pages were on node 1 in every session), summary by `/mnt/fast/w2w/tools/w3-summary.py` (a reporting script kept out of the repo). c4 load 2-10.
Every completion of every arm gave its one-card sha (60 + 60 completions), **0 experts fell back**.

W1 host (as built; labels `w3-*`, 2 rounds x 5 completions per arm, interleaved):

| arm | cards (VRAM MiB) | tok/s median [min-max] | vs allres | vs spill | hops / token | hop wall ms / token | of it wire ms | of it host compute ms | gate |
|---|---|---|---|---|---|---|---|---|---|
| allres | D8 (13,775) | **81.36** [81.18-81.55] | 1.00 | | 0 | 0 | | | (baseline) |
| spill | D8 (6,337), E2 cache 5120 MiB | **29.81** [29.75-29.88] | 0.37 | 1.00 | 0 | 0 | | | (baseline) |
| w3a | D8 (697) + AF (12,909) | 15.19 [15.15-15.24] | **0.187** | | 38 | 29.3 | 3.65 | 25.3 | **FAIL** (>= 0.80) |
| w3b2 | D8 + AF, 86 (6,637 each) | 14.97 [14.82-15.11] | 0.184 | | 38 | 30.3 | 3.69 | 26.3 | no gate |
| w3b4 | D8 + AF, 86 (3,599), 3B, 3C (3,305) | 14.58 [14.57-14.60] | 0.179 | | 38 | 30.0 | 3.94 | 25.7 | no gate |
| w3d | D8 (5,855; layers 2-15) + AF (8,205; 16-39) | 16.27 [15.90-16.64] | | **0.546** | 24 | 18.0 | 2.34 | 15.4 | **FAIL** (>= 1.25) |

v2 host (one routed launch per message, see above; labels `w3v2-*`, 2 rounds x 5, allres and spill re-run
in the same rounds):

| arm | tok/s median [min-max] | vs allres | vs spill | hops / token | hop wall ms / token | wire ms | host compute ms | gate |
|---|---|---|---|---|---|---|---|---|
| allres | **81.82** [80.99-82.65] | 1.00 | | 0 | | | | |
| spill | **30.55** [30.22-30.88] | | 1.00 | 0 | | | | |
| w3a | 20.68 [19.88-21.48] | **0.253** | | 38 | 15.0 | 3.34 | 11.4 | **FAIL** (>= 0.80) |
| w3b2 | 22.84 [22.34-23.34] | 0.279 | | 38 | 13.8 | 3.25 | 10.3 | no gate |
| w3b4 | 21.36 [21.14-21.58] | 0.261 | | 38 | 14.2 | 3.48 | 10.4 | no gate |
| w3d | 20.54 [20.16-20.92] | | **0.672** | 24 | 9.0 | 2.24 | 6.5 | **FAIL** (>= 1.25) |

"wire ms" is round trip minus the host's own time, over a unix socket with spinning waits: 88 us a hop,
against W0's 28.8 us echo (the difference is the 8 KB / 32 KB encode, copy and decode at both ends plus
the scheduler). Hops are one a MoE layer whatever the number of host cards, because whole layers are placed
per card: w3b spreads the same 38 hops over more cards and changes nothing but VRAM per card.

**Where the w3a token goes** (v2, 48 ms against allres's 12.2 ms): about 15 ms waiting on hosts (11.4 ms of
it the host's own launch and **read**: one device sync on the host a hop, 270-310 us, `LB_HOST_PROFILE`),
and about 21 ms more than allres on the trunk side: 38 reads of the activation (one device sync each), the
CPU router, the CPU weighted sum and 38 uploads. Each hop therefore costs **two device syncs**, one on
each card, and a device sync on a P100 through CubeCL is 0.15-0.3 ms whatever is read. The wire is not the
problem (3.3 ms a token). As built, a socket host cannot reach 0.80x: 76 syncs a token is ~15-20 ms of
waiting before any arithmetic. **W3d** loses for the same reason: the trunk's own 14 layers also go through
the host-side path once a remote tier is attached (one sync a layer), so 38 syncs a token remain, while the
E2 spill token pays syncs only on the 15 layers with a miss.

What fixes it is fewer syncs, not a faster wire: (1) **in the same box**, keep the activation on the
device and move it card to card (W3e / plan X1, below); (2) **across boxes**, a host can only drop its own
sync if the reply leaves the card without a host read (GPUDirect RDMA from device memory, not built), so a
remote host stays at ~0.3 ms a hop: it is for the spill case and cold layers, as W0 already said for TCP.

**W3e / X1 arms (gates written before the run, 03:50Z).** In-box cards (`gpu://N` in the placement,
`crates/lb-lfm2/src/inbox.rs`), server binary `lemonade-burn-server-cuda-rdma-v3`, NCCL 2.21.5 (CUDA 12.4
build, has sm_60; the c4 system NCCL 2.31/2.32 are CUDA 13 builds without Pascal kernels and hang the first
copy) from `/mnt/fast/w2w/lib-nccl/`. Parity first (`bench/w2w/w3e-parity.sh`, all 38 layers on AF):
**PASS**, teacher-forced 86/86, greedy sha 062392b8d2b0361a token for token, 0 fallbacks.
* `w3e`: trunk D8, all 38 MoE layers on AF. Gate **>= 0.80x allres** (plan W3a / X1 "experts over 2 cards").
* `w3e2`: AF 2-20, 86 21-39. `w3e4`: AF 2-11, 86 12-21, 3B 22-30, 3C 31-39. Gate (X1) **>= 0.80x allres**.
* `w3ed`: D8 holds layers 2-15 itself (stacked, 5,075 MiB, the W3d cap), 16-39 on AF. Gate **>= 1.25x spill**.
Same rounds as before (2 x 5 per arm, interleaved with allres and spill), sha on every completion. Hops per
token are 38 card-to-card hand-offs (two copies each) by construction (24 for w3ed); the cost per hand-off
is reported as (token ms - allres token ms) / hand-offs.

### W3e / X1 results: in-box cards, activation card to card (c4, 03:42-03:52Z) [measured]

Labels `w3e-*` in `results/w3/runs.jsonl`, 2 rounds x 5 completions per arm, interleaved, allres and spill
re-run in the same rounds with the same (v3) server binary. Every completion gave its one-card sha.

| arm | cards (VRAM MiB) | tok/s median [min-max] | vs allres | vs spill | card-to-card hand-offs / token | token ms | added ms / token | us per hand-off | gate |
|---|---|---|---|---|---|---|---|---|---|
| allres | D8 (13,775) | **81.92** [81.77-82.07] | 1.00 | | 0 | 12.21 | | | |
| spill | D8 (6,337) | **30.16** [30.11-30.21] | | 1.00 | 0 | 33.16 | | | |
| w3e | D8 (699) + AF (12,3xx) | **70.74** [70.37-71.10] | **0.864** | | 38 | 14.14 | 1.93 | 51 | **PASS** (>= 0.80) |
| w3e2 | D8 + AF, 86 (6,663 each) | **70.14** [69.97-70.30] | **0.856** | | 38 | 14.26 | 2.05 | 54 | **PASS** (>= 0.80) |
| w3e4 | D8 + AF, 86 (3,627), 3B, 3C (3,333) | **69.11** [68.91-69.32] | **0.844** | | 38 | 14.47 | 2.26 | 59 | **PASS** (>= 0.80) |
| w3ed | D8 (5,857; layers 2-15) + AF (8,231; 16-39) | **72.50** [72.41-72.59] | 0.885 | **2.40** | 24 | 13.79 | 1.58 | 66 | **PASS** (>= 1.25) |

"us per hand-off" = (token ms - allres token ms) / hand-offs [derived]: one hand-off is two NCCL copies of
8 KB (h2 out, the summed output back) plus the router/experts/sum on the other card instead of this one;
51-66 us, i.e. near W0's same-host floor (unix socket 29 us one way-and-back for a bare echo) and well
under the 0.6-0.8 ms a socket hop costs with its two device syncs. 3B and 3C (x8, NUMA 0, crossing the
socket interconnect: `nvidia-smi topo` SYS) cost about 8 us more a hand-off than AF/86 (NODE).

So, on c4: **the trunk on one P100 with every expert on other P100s of the same box runs at 0.84-0.86x of
one card holding everything**, and **a trunk card capped at 5120 MiB of experts with the overflow on a
second card runs at 2.40x the one-card spilling config** (0.89x all-resident), all at the one-card greedy
text. The plan's W3a and W3d gates fail with socket hosts and pass with in-box cards.

How it is built (`crates/lb-lfm2/src/inbox.rs`, `crates/lb-lfm2/src/device_decode.rs`,
`crates/lb-q4-gpu/src/store.rs` `Gpu::open_cuda`, `copy_to`, `sync_recycle`): placement hosts with
`addr: "gpu://N"` are CUDA devices of the trunk's process holding whole layers; in the device decode such
a layer copies h2 to its card (cubecl `to_client`, NCCL send/recv on the communication streams, no host
sync), runs `router_topk`, the stacked experts on the device routes and `weighted_sum` there (the
one-card kernels), and copies the sum back; the token still has **one** read on the trunk, plus one fence
per in-box card after it (the card is idle by then) to recycle its scratch. Prefill and declined tokens run
an in-box layer's experts on its card through per-expert views (one upload and one read per layer pass).
Parity: device path bit-identical in routes (0 route differences) and the same greedy text; the old path
and the device path agree to 2.3e-5 as on one card.

Requirements found on the way: **NCCL must have sm_60 kernels.** c4's system NCCL (2.31.2 / 2.32.3,
CUDA 13 builds) has none (CUDA 13 dropped Pascal): its first send fails with `named symbol not found` and
cubecl's receive then waits for ever (the hang is cubecl's documented behaviour for a failed send). The
runs used NCCL 2.21.5+cuda12.4 (the nvidia-nccl-cu12 wheel's library, sm_50-sm_90), placed as
`/mnt/fast/w2w/lib-nccl/libnccl.so` (cudarc looks for `libnccl.so`, not `libnccl.so.2`). A mixed
placement (in-box cards and socket hosts together) is refused: not built.

### W3c results: cross-node wire test (c4 trunk, t440 CPU host), 04:00-04:04Z [measured, a wire test, not a speed claim]

Trunk on D8, layers 2-35 on a v2 GPU host on AF (unix socket), **layers 36-39 on a CPU-backend host on t440**
(`bench/w2w/w3c-t440-host.sh`, 8 threads, nice 5, t440 load average about 27 from other work; t440's P100 not
used), reached over the 25G link, by RoCE v2 (`spin:200`) and by TCP (`block`). Labels
`w3c2-*`, 2 rounds x 5 per arm, interleaved. Same model file on both nodes (fingerprint
8f5e6596b72284a3eeec808f0c5656c2), CPU arithmetic fingerprint 3485a00c79570e6d.

| arm | tok/s median [min-max] | hops / token | hop wall ms / token | wire ms / token (all 38 hops) | host compute ms / token | fallback | sha | gate |
|---|---|---|---|---|---|---|---|---|
| w3c-rdma | 11.58 [11.40-11.77] | 38 | 49.4 | 4.02 | 45.1 | 0 | all 10 OK | **PASS** |
| w3c-tcp | 11.40 [10.95-11.85] | 38 | 53.3 | 4.29 | 48.7 | 0 | all 10 OK | **PASS** |

Gate (sha on every completion, the t440 host answers everything placed on it, hop ms recorded): **PASS on
both transports**. The CPU experts on t440 did not change a single greedy token here (rounding against the
P100's arithmetic, as expected, but no flip in 20 completions). The token is slow because a loaded CPU
computes four layers' experts (about 11 ms a layer), not because of the wire. The wire figure is the sum
over all 38 hops; with the 34 unix hops at the 88 us measured above (3.0 ms), the four cross-node hops cost
about **250 us each over RoCE and about 320 us over TCP** including encode/decode on a loaded t440
[derived, not separately measured].

**Fault found and fixed in this step.** The first wire run (labels `w3c-*`, kept in the results) had the
RoCE host refused at connect: a bare TCP reachability probe of the rendezvous port (from this run) had made
the RDMA `accept` fail, and `host::serve` ended on the first failed accept, so the listener was gone for
good while the TCP listener kept working; the trunk then ran those layers locally (9,056 expert fallbacks,
sha still correct: the fallback path did its job). `serve` now logs a failed accept and keeps listening
(stops only after 1,000 failures in a row); test `a_bad_client_does_not_take_the_listener_down` (fails on the
old code, passes on the new). The rerun above used the fixed host.

## W3 / X0 / X1: not run, not verified, open

* **X0's target (socket host w3a >= 0.80x allres) is not met** and cannot be with a sync on each side of
  every hop: v2 host 0.253x. Removing the host's own sync needs the reply to leave its card without a host
  read (GPUDirect RDMA from device memory to the NIC): not built, not tried.
* **Mixed placements** (in-box cards and socket hosts at once) are refused: not built. So a box with both
  local cards and a far host cannot use the in-box path yet.
* **Prefill with in-box cards** runs per layer through per-expert views and a read on the card; its speed was
  not measured (decode tok/s was the gate). Long prompts (X5) not run.
* **In-box hops are not counted by the server's `/v1/placement`** (`InBox::status` exists, the server does
  not expose it); hand-offs per token are by construction and the per-hand-off cost is derived from tok/s.
* Two rounds per arm (the minimum asked); spreads were 0.2-4 %. No third round.
* **Per-card memory bandwidth was not measured** (X2 needs it). Per-hop latencies measured here are in
  RESUME.md for X2-X4.
* The W1 (per-expert) GPU host is still selectable (`LB_HOST_LAUNCH=per-expert`); the staged
  `lb-expert-host-cuda-rdma` binary is the W1 one, `-v2`/`-v3` are the new ones.
* NCCL: the run depends on a CUDA 12 NCCL with sm_60 kernels (`/mnt/fast/w2w/lib-nccl/libnccl.so`, a copy
  of the nvidia-nccl-cu12 2.21.5 wheel's library). With a CUDA 13 NCCL the first copy hangs (cubecl waits on
  a receive whose send failed); the load log says which NCCL was loaded only through `NCCL_DEBUG=INFO`.
  A start-up self-test copy with a timeout would turn that hang into a refusal: not built.

## X6: prefill on Turing, decode on P100 (and the 1G vs 25G hand-off)

Operator, 2026-11-11: "test what you can with the current setup and measure and account for the 1G vs 25G pass-off".
Gates and expectations written 2026-11-11 ~00:20Z (node clocks), **before** any X6 run. Results:
`/mnt/fast/w2w/results/x6/*.jsonl` (append-only). Every number below is tagged [measured] or [estimate].

Layouts (operator's idea, not final):
* **A, prefill on board**: c4 = 4x P100 + RTX 2060 12 GB. The 2060 prefills, a P100 decodes, the state goes card ->
  host RAM -> card inside the box (no Turing card is P2P with anything). The 2060 is not on c4 today: A is an
  **estimate** built from the T10's prefill (scaled to the 2060 by spec, stated) and c4's measured H2D/D2H.
* **B, prefill a hop away**: c1's T10 prefills, a c4 P100 decodes, the state crosses the c1-c4 25G link (TCP today,
  RoCE once c1's NIC firmware allows). c1 is off limits for this run: B is an **estimate** built from the T10's prefill
  measured on c2 and the 25G link measured t440 <-> c4 (same NIC model, same DAC, same MTU).
* **C, today**: T10 on c2 prefills, a c4 P100 decodes, the state crosses 1 GbE. Measured as far as one borrowed seat
  allows; the hand-off file is really saved on c2, moved, and restored into a P100 server on c4.
* **P100-only** (the baseline): prefill and decode on the same c4 P100.

Engine: the fleet's shared llama.cpp build (2026-10-27, sm60/61/75/86), scraper off (no NAUTIBURN_SCRAPE_DIR, no
/etc/nautiburn config on c2/c4). `llama-bench` is not in that build, so prefill and decode are timed by
`llama-server`'s own per-request timings (`prompt_ms`, `predicted_ms`) with `cache_prompt: false`, the prompt given as
token ids of exactly N tokens (real text, repeated), 3 repeats per point, median reported. Model LFM2.5-24B-A2B Q4_0
(13.6 GB), all layers resident (`-ngl 99`), one card per process, UUID pinned, NUMA-bound to the card's node.

What is measured:
1. Prefill tok/s at N = 512, 2k, 8k, 16k, 32k on the T10 (c2) and on c4-AF, flash attention on and off on each.
   Decode tok/s for 128 tokens after each prompt (so decode at depth N, not only at depth 0).
   lemonade-burn's prefill on c4-AF at the same N where its context allows (the gap to llama.cpp).
2. Hand-off bytes: the slot state file (`POST /slots/0?action=save`) after each N, f16 KV (and q8_0 KV for one N),
   checked against arithmetic from the model's metadata (attention layers x KV heads x head dim x dtype x N, plus the
   short-conv state of the conv layers, which does not grow with N).
3. Transfer time for those bytes on an open connection, sender memory -> receiver memory, 10 repeats: t440 <-> c2
   1 GbE TCP, c2 <-> c4 1 GbE TCP (layout C's real path), t440 <-> c4 25G TCP, t440 <-> c4 RoCE (ib_write_bw at the
   same message size). In-box (layout A stand-in): c4 pinned and pageable H2D/D2H on the NIC-side x16 cards, and
   P100 <-> P100 D2D for reference. Restore time (`action=restore`) into the P100 server.
4. TTFT per layout = prefill(prefill card) + save + transfer + restore; then decode on the P100. Compared with P100-only.

Gates (written before the run):
* **X6 gate (plan)**: at N = 8k, (TTFT + 128 decode tokens) of the split layout beats P100-only by **>= 1.2x** end to
  end. Judged per layout; A and B are judged on estimates and say so.
* **Bytes gate**: measured slot-file size agrees with the metadata arithmetic within 5 % (plus a fixed header).
* **Hand-off correctness**: a state saved on the T10 and restored on the P100 continues the same prompt coherently
  (the first 32 greedy tokens after restore are compared with P100-only's; an exact match is not required across
  card types, a coherent continuation is, and the match length is reported).

Expectations, written before the run (to be checked, not assumed):
* LFM2.5-24B-A2B has 40 blocks of which only a minority are attention (GQA, 8 KV heads), the rest short-conv; so the
  KV is roughly 10x smaller than a dense 24B's and the hand-off is MB to tens of MB even at 16k, not GB.
* At those sizes 25G (any transport) costs single-digit ms, so A vs B differ by little; 1 GbE costs ~8-9 ms per 1 MB,
  so C pays tens to hundreds of ms at 8k-16k.
* The T10 (TU102, 40 TU SMs, tensor cores, 320 GB/s) should prefill Q4_0 faster than a P100 (no tensor cores, no DP4A,
  732 GB/s), by maybe 1.5-3x; the P100 decodes faster (bandwidth). Whether the split wins at 8k depends on that prefill
  ratio, since the hand-off is small: the gate is about the T10's prefill speed, not the wire.

### X6 results (c4 + one borrowed c2 card, 2026-11-11 05:08-05:58Z)

Cards: **c2-T10** (Tesla T10, TU102, 56 SMs, x16, power-capped 150 W, ~1500 MHz under load), borrowed for 7.5 min;
**c4-AF** (P100, x16, NUMA node of the NIC). Engine: the fleet's shared llama.cpp build, scraper off, one card per
process, UUID pinned, NUMA-bound. Raw rows: `/mnt/fast/w2w/results/x6/` (`llama.jsonl`, `xfer.jsonl`,
`roce-raw.txt`, `bwprobe.txt`, `lb.jsonl`, `e2e.json`). One-off drivers (not shipping code): `/mnt/fast/w2w/x6/`.

**1. Prefill tok/s, median of 3 (ub 512) or 2 (other rows) [measured].** FA on, f16 KV unless stated.

| N | T10 ub512 | T10 ub2048 | T10 ub4096 | P100 ub512 | P100 ub2048 | P100 ub4096 | T10 / P100 (best each) |
|---|---|---|---|---|---|---|---|
| 512 | **2250** | 1949 | 1980 | **615** | 602 | 602 | 3.7x |
| 2048 | 2396 | 3753 | **3998** | 626 | 1025 | **1035** | 3.9x |
| 8192 | 2387 | 3926 | **4222** | 607 | 990 | **1085** | 3.9x |
| 16384 | 2283 | 3751 | **4107** | 583 | 928 | **1014** | 4.0x |
| 32768 | 2080 | 3439 | **3687** | 538 | 820 | **889** | 4.1x |

* The ubatch matters as much as the card: the default ub 512 leaves 1.7x (T10) and 1.8x (P100) on the table at 8k.
  The fleet's seat commands do not set `-ub`.
* FA off: T10 (ctx 8.7k) 1399 / 2081 / 1817 tok/s at 512 / 2k / 8k, slower than FA on everywhere; P100 the same as FA
  on at <= 2k (581, 615) and **out of device memory at 8k with a 33k context** (its KQ buffer). FA on is the setting.
* q8_0 KV at 8k (ub 512): T10 2190 (-8 % vs f16), P100 605 (same). Decode: T10 102 vs 112, P100 77 vs 83.
* **Decode, 128 tokens after the prompt, tok/s [measured]: T10 111 / 115 / 112 / 107 / 97, P100 82 / 84 / 83 / 81 / 77**
  at 512 / 2k / 8k / 16k / 32k. **The T10 decodes this model faster than the P100** in llama.cpp (Q4_0: sm_60 has no
  DP4A, the matvec is not bandwidth-bound there). Our engine's all-resident P100 decode is the same ~82 (W3).
* **lemonade-burn prefill on c4-AF [measured]**: 63-69 tok/s at ~450 tokens (all experts resident); at 2k the
  all-resident config (and a 13 GB expert budget) **runs out of device memory** in the prefill (a 94 MB buffer);
  with a 9 GB budget 55-56 tok/s at ~1970 tokens. So our prefill is **11-19x behind llama.cpp on the same card**;
  8k not run. Prefill is not our engine's job yet; for X6 llama.cpp is the hardware reference.

**2. Hand-off bytes [measured, matches arithmetic].** Slot file after an N-token prefill (identical sizes on T10 and P100):

| N | f16 KV bytes | arithmetic (10 attn layers x 8 KV heads x 64 x 2 x 2 B = 20,480 B/token) | diff | q8_0 |
|---|---|---|---|---|
| 512 | 11,478,824 | 10,485,760 + conv | +4.6 % vs 10,977,280 | |
| 2048 | 43,452,988 | 41,943,040 + conv | +2.4 % | |
| 8192 | 169,380,412 | 167,772,160 + conv | +0.7 % | 90,737,212 (10,880 B/token) |
| 16384 | 337,283,644 | 335,544,320 + conv | +0.4 % | |
| 32768 | 673,090,108 | 671,088,640 + conv | +0.2 % | |

The remainder is 16 B/token of cell metadata plus 1.0-1.5 MB fixed (the 30 short-conv layers' state, f32, and
llama.cpp's recurrent "context checkpoints", ~0.9 MiB). **Bytes gate PASS** (within 5 %). **My expectation was wrong**:
I wrote "MB to tens of MB"; it is **20 KB per token, 169 MB at 8k, 673 MB at 32k**. LFM2.5 has few attention layers,
but its GQA heads are 8 x 64 in f16, and 10 layers of that is still a dense-7B-sized KV per token.

**3. Transfer of those bytes, memory to memory on an open connection, p50 [measured].**

| path | 11.5 MB (512) | 43 MB (2k) | 169 MB (8k) | 337 MB (16k) | 673 MB (32k) | rate |
|---|---|---|---|---|---|---|
| c2 -> c4, 1 GbE TCP (layout C's path) | 98 ms | 370 | **1441** | 2870 | 5726 | 117 MB/s |
| t440 -> c2, 1 GbE TCP | 98 | 370 | 1450 | 2976 | 5894 | 114-117 MB/s |
| t440 -> c4, 25G TCP | 4.3 | 17.6 | **61.9** | 113 | 228 | 2.7-3.0 GB/s |
| t440 -> c4, RoCE v2 write (ib_write_bw, one message at a time) | 3.8 | 14.2 | **55.5** | 110 | 220 | 2912 MiB/s |
| in-box, pinned D2H (x16) + H2D (x16) [stand-in] | 1.8 | 6.8 | **26.7** | 53 | 106 | 13.1 / 12.3 GB/s |
| llama.cpp slot save (T10) + restore (P100), file in RAM | 24 | 54 | **187** | 357 | 696 | ~1.8 GB/s + 2.3 GB/s |

* **1 GbE vs 25G: 23x** on the wire (1441 vs 62 ms at 8k). **25G TCP vs RoCE: 6 ms at 8k**, nothing that matters for
  a bulk hand-off of this size (since c4's `iommu=pt` fix TCP fills the link). RoCE matters for the per-hop latency
  of W0/W3, not here.
* On 25G the wire is **smaller than llama.cpp's own save + restore** (56-62 ms vs 187 ms at 8k): the next cost to
  remove is the file round trip, not the network.
* A t440 -> c4 1 GbE run (65 MB/s) is in `xfer.jsonl` but **contaminated**: t440's 1 GbE port was serving the
  restarted c2 seat's model over NFS at the same time. Not used.
* Host links on c4 [measured, X2 input]: x16 cards (AF, D8) H2D 12.3 / D2H 13.1 GB/s pinned, x8 cards (3B, 3C, 86)
  6.2 / 6.6; pageable 4.8-5.2 everywhere; wrong-node pinned H2D on AF unchanged (12.3), pageable 4.5. P2P AF->D8
  10.2 GB/s, AF->86 / AF->3B / 3B->3C 5.2 GB/s (all report peer access).
* **Per-card memory bandwidth [measured, X2 needed it]**: all five c4 P100s copy kernel 506 GB/s, read kernel 605,
  cudaMemcpy D2D 538 (spec 732); spread across the five < 0.2 %. T10: copy 338, read 394, D2D 358.

**4. End to end at the gate point and around it.** total = prefill + save + transfer + restore + 128 decode tokens on
the P100; P100-only = best-config prefill + 128 decode on the same P100. All parts [measured] unless the row says
[estimate].

| N | P100-only total (prefill) | T10-only total (no split) | C: T10 on c2, 1 GbE (TTFT) | B: 25G TCP / RoCE (TTFT) | A': a 16 GB Turing in c4 (TTFT) | A: RTX 2060 12 GB in c4 [estimate] |
|---|---|---|---|---|---|---|
| 512 | 2402 ms (832) | 1385 | 1919 = **1.25x** (349) | 1825 = 1.32x (256) | 1821 = 1.32x | 2029-2086 = 1.15-1.18x |
| 2048 | 3494 (1978) | 1622 | 2452 = **1.42x** (937) | 2100 = 1.66x (584) | 2083 = 1.68x | 2577-2705 = 1.29-1.36x |
| **8192** | **9102 (7554)** | **3086** | **5117 = 1.78x** (3569) | **3738 / 3731 = 2.44x** (2189) | 3676 = 2.48x (2127) | **5564-6049 = 1.50-1.64x** |
| 16384 | 17739 (16151) | 5183 | 8804 = 2.01x (7216) | 6047 = 2.93x (4459) | 5934 = 2.99x | 9971-10969 = 1.62-1.78x |
| 32768 | 38516 (36851) | 10200 | 16973 = 2.27x (15309) | 11475 = 3.36x (9811) | 11247 = 3.42x | 20460-22681 = 1.70-1.88x |

**X6 gate (>= 1.2x at 8k): PASS for C [measured end to end, real path], B [estimate from parts measured on other
boxes: the T10's prefill on c2, the 25G link t440 <-> c4], A' [measured parts], A [estimate].** Against the P100's
*default* ub 512 (15.0 s at 8k), every split is 1.8-4x.

**Hand-off correctness: PASS (by reading).** The T10's saved state, moved to c4 and restored into the P100 server,
continues the prompt with coherent text at every N (prompt_n after restore = 1: only the hand-off token is
recomputed). Exact greedy match with the P100's own continuation: 1, 20, 4-33, 2-3, 9-17 of 33 tokens; the P100
does not match *itself* across ub 512 vs ub 2048 either (33, 20, 4, 2, 9 of 33), so exact match is not a property
llama.cpp has across batch shapes. Same-card save/restore (P100 -> P100) reproduces 33 of 33 at every N, which
validates the method.

**Break-even.** The split wins at every measured N, from 512 up, on every link including 1 GbE: the T10 saves
0.69-1.18 ms per prompt token over the best P100 config (1.18 at 512, 0.69 at 8k, 0.85 at 32k) while the hand-off
costs 0.20 ms per token over 1 GbE and 0.03 ms over 25G (wire + save + restore). Below 512 not measured; the fixed part
of the hand-off is ~13 ms (25G) to ~20 ms (1 GbE) at 512 (conv state, headers, slot calls, one RTT), so break-even is
**~10-30 tokens over 25G and ~20-50 over 1 GbE [estimate]**; at such lengths both cards run far from their prefill rate.

**What the numbers say for the card move (for the operator):**
* **The 1 GbE hand-off costs 1.4 s at 8k and 5.7 s at 32k; 25G costs 62 / 228 ms.** At 8k that is 1.78x vs 2.44x
  over P100-only. 25G is worth 1.4 s of TTFT at 8k and 5.5 s at 32k.
* **TCP on 25G is enough** for the hand-off: RoCE adds 6 ms at 8k. c1's RoCE is not on the critical path for X6.
* **Layout A as corrected (RTX 2060 12 GB) cannot hold this model** (13.6 GB): its prefill needs a layer split with a
  P100 (estimated 1.50-1.64x at 8k), or a model that fits 12 GB. A 16 GB Turing (the T10) in c4 would give 2.48x; in
  c1 over 25G 2.44x: **the hop costs 1 % once it is 25G**. The card's VRAM decides A vs B, not the link.
* **Not splitting beats every split for one request**: the T10 alone does prefill + 128 tokens in 3.1 s at 8k (B: 3.7 s,
  C: 5.1 s) because it also decodes faster (111 vs 83 tok/s). The split is a **capacity** play: the scarce Turing
  card is busy ~2.1 s per 8k request instead of 2.1 s + the whole answer, so one T10 can prefill for several P100
  decoders (at 500-token answers: ~3 P100s per T10 [estimate]).
* **Streaming would hide even 1 GbE at T10 speed [estimate, not built]**: the T10 produces KV at 20.5 KB x 4222 tok/s
  = 87 MB/s, under 1 GbE's 117 MB/s. Sending each ubatch's KV while the next one computes (llama.cpp finalises KV per
  ubatch) leaves only the last chunk exposed. A 1.5x faster prefill card would outrun 1 GbE again; 25G never.

### X6: not run, not verified
* Layout A and B hardware: c1 off limits (NIC flash), the 2060 not in c4. Both are compositions of measured parts.
  The 2060's prefill is **spec-scaled from the T10** (34 vs 56 SMs, 1.1-1.25x clock): 0.60-0.75x the T10.
* The c2 RTX 2060 / 2060 SUPER were not touched (seats). T10 in c1 would sit in a different slot and CPU: not measured.
* Restore was timed with the file already in the receiver's RAM; a real pipeline would also write it there
  (overlappable). No streaming hand-off was built; the "direct pinned copy" row is a D2H + H2D measurement, not an
  engine path.
* lemonade-burn prefill at 8k+ not run (OOM at 2k with experts resident; 55 tok/s otherwise).
* X4 (look-ahead per tier) and X5 (prefill batched by destination) not run in this session.

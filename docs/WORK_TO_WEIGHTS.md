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
    "hosts": { "a": { "addr": "rdma://10.0.10.204:7411", "wait": "spin:200", "deadline_ms": 50 },
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
  --listen unix:///tmp/w2w-h1.sock --listen rdma://10.0.10.204:7411 --wait spin:200
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

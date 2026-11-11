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

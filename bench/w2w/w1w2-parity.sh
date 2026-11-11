#!/usr/bin/env bash
# W1/W2 parity gates on the CPU reference path, everything on this node, no GPU touched:
# starts the expert hosts (CPU backend), runs w2w-check (one process: reference arm, then one arm
# per placement), stops the hosts. Appends one JSON line per arm to $OUT.
#   a-unix / a-tcp / a-rdma   every expert on one host, three transports            (W1 gate, W2 gate a)
#   b-split                   experts 0-31 on host a (TCP), 32-63 on host b (unix)  (W2 gate b)
#   b-mixed                   as b-split, with layers 2-9 and experts 60-63 kept on the trunk
#   c-kill-decode             split; host b is SIGKILLed before teacher-forced position 40
#   c-kill-greedy             split; a fresh host b is SIGKILLed before greedy step 12  (W2 gate c)
#   c-hang-greedy             split; a fresh host b is SIGSTOPped before greedy step 12: the deadline finds it
set -u
W=${W:-/mnt/fast/w2w}; BIN=${BIN:-$W/target/release}; RUN=$W/run; LOG=$W/logs
MODEL=${MODEL:-/mnt/raid0/models/lfm25/LFM2.5-24B-A2B-QAD-Q4_0.gguf}
TOK=${TOK:-/mnt/raid0/models/lfm25/tokenizer.json}
OUT=${OUT:-$W/results/w2w-parity.jsonl}
RDMA_IP=${RDMA_IP:?set RDMA_IP to the host address}; P=${PORT_BASE:-47410}
NICE=${NICE:-10}
mkdir -p "$RUN" "$LOG"
rm -f "$RUN"/*.sock "$RUN"/*.pid
host() { # tag name hold listen...
  local tag=$1 name=$2 hold=$3; shift 3
  local args=(); for l in "$@"; do args+=(--listen "$l"); done
  setsid nohup nice -n "$NICE" "$BIN/lb-expert-host" --model "$MODEL" --name "$name" --hold "$hold" --wait "${HOST_WAIT:-block}" --threads "${HOST_THREADS:-8}" --stats-s 0 "${args[@]}" > "$LOG/host-$tag.log" 2>&1 < /dev/null &
  echo $! > "$RUN/$tag.pid"
}
host all all "*"        "unix://$RUN/all.sock" "tcp://127.0.0.1:$P" "rdma://$RDMA_IP:$((P+1))"
host sa  a   "*:0-31"   "tcp://127.0.0.1:$((P+2))"
host sb  b   "*:32-63"  "unix://$RUN/sb.sock"
host kb1 b   "*:32-63"  "tcp://127.0.0.1:$((P+3))"
host kb2 b   "*:32-63"  "tcp://127.0.0.1:$((P+4))"
host kb3 b   "*:32-63"  "tcp://127.0.0.1:$((P+5))"
sleep 3
for t in all sa sb kb1 kb2 kb3; do kill -0 "$(cat "$RUN/$t.pid")" 2>/dev/null || { echo "host $t did not start:"; cat "$LOG/host-$t.log"; exit 1; }; head -1 "$LOG/host-$t.log"; done
D='"deadline_ms":5000,"deadline_row_ms":2000'
echo "{\"hosts\":{\"all\":{\"addr\":\"unix://$RUN/all.sock\",$D}},\"layers\":{\"*\":\"all\"}}" > "$RUN/a-unix.json"
echo "{\"hosts\":{\"all\":{\"addr\":\"tcp://127.0.0.1:$P\",$D}},\"layers\":{\"*\":\"all\"}}" > "$RUN/a-tcp.json"
echo "{\"hosts\":{\"all\":{\"addr\":\"rdma://$RDMA_IP:$((P+1))\",\"wait\":\"spin:200\",$D}},\"layers\":{\"*\":\"all\"}}" > "$RUN/a-rdma.json"
echo "{\"hosts\":{\"a\":{\"addr\":\"tcp://127.0.0.1:$((P+2))\",$D},\"b\":{\"addr\":\"unix://$RUN/sb.sock\",$D}},\"layers\":{\"*\":{\"0-31\":\"a\",\"32-63\":\"b\"}}}" > "$RUN/b-split.json"
echo "{\"hosts\":{\"a\":{\"addr\":\"tcp://127.0.0.1:$((P+2))\",$D},\"b\":{\"addr\":\"unix://$RUN/sb.sock\",$D}},\"layers\":{\"*\":{\"0-31\":\"a\",\"32-59\":\"b\",\"60-63\":\"local\"},\"2-9\":\"local\"}}" > "$RUN/b-mixed.json"
echo "{\"hosts\":{\"a\":{\"addr\":\"tcp://127.0.0.1:$((P+2))\",$D},\"b\":{\"addr\":\"tcp://127.0.0.1:$((P+3))\",$D}},\"layers\":{\"*\":{\"0-31\":\"a\",\"32-63\":\"b\"}}}" > "$RUN/c-kill-decode.json"
echo "{\"hosts\":{\"a\":{\"addr\":\"tcp://127.0.0.1:$((P+2))\",$D},\"b\":{\"addr\":\"tcp://127.0.0.1:$((P+4))\",$D}},\"layers\":{\"*\":{\"0-31\":\"a\",\"32-63\":\"b\"}}}" > "$RUN/c-kill-greedy.json"
echo "{\"hosts\":{\"a\":{\"addr\":\"tcp://127.0.0.1:$((P+2))\",$D},\"b\":{\"addr\":\"tcp://127.0.0.1:$((P+5))\",\"deadline_ms\":3000}},\"layers\":{\"*\":{\"0-31\":\"a\",\"32-63\":\"b\"}}}" > "$RUN/c-hang-greedy.json"
echo "== $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg)"
RAYON_NUM_THREADS=${TRUNK_THREADS:-16} LB_W2W_LOG="$W/results/w2w-passes-$(date -u +%Y%m%dT%H%M%S).jsonl" nice -n "$NICE" "$BIN/w2w-check" --model "$MODEL" --tokenizer "$TOK" --out "$OUT" --label "${LABEL:-w1w2-parity}" \
  --arm a-unix="$RUN/a-unix.json" --arm a-tcp="$RUN/a-tcp.json" --arm a-rdma="$RUN/a-rdma.json" \
  --arm b-split="$RUN/b-split.json" --arm b-mixed="$RUN/b-mixed.json" \
  --arm c-kill-decode="$RUN/c-kill-decode.json" --arm-phases c-kill-decode=prefill,decode --kill "c-kill-decode=decode@40:$(cat "$RUN/kb1.pid")" \
  --arm c-kill-greedy="$RUN/c-kill-greedy.json" --arm-phases c-kill-greedy=greedy --kill "c-kill-greedy=greedy@12:$(cat "$RUN/kb2.pid")" \
  --arm c-hang-greedy="$RUN/c-hang-greedy.json" --arm-phases c-hang-greedy=greedy --kill "c-hang-greedy=greedy@12:$(cat "$RUN/kb3.pid"):STOP"
RC=$?
echo "== $(date -u +%FT%TZ) w2w-check exit $RC load $(cut -d' ' -f1 /proc/loadavg)"
for t in all sa sb kb1 kb2 kb3; do p=$(cat "$RUN/$t.pid"); kill -9 "$p" 2>/dev/null && echo "stopped host $t ($p)" || echo "host $t ($p) was already gone"; done
exit $RC

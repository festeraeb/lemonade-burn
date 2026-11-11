#!/usr/bin/env bash
# W3c wire test, the far side: a CPU-backend expert host on t440 (no card used) holding MoE layers
# $LAYERS, listening on t440's 25G port over TCP and RoCE. Stops after $LIFE seconds (backstop).
# usage (on t440): w3c-t440-host.sh start|stop
set -u
W=/mnt/fast/w2w; MODEL=/mnt/raid0/models/lfm25/LFM2.5-24B-A2B-QAD-Q4_0.gguf
IP=${IP:?set IP to the host address}; LAYERS=${LAYERS:-36-39}; LIFE=${LIFE:-3600}; PIDF=$W/run/w3c-t440-host.pid
case ${1:-start} in
  start)
    [ -f "$PIDF" ] && kill -0 "$(cat $PIDF)" 2>/dev/null && { echo "already running"; exit 0; }
    setsid nohup timeout $LIFE nice -n 5 "${HOSTBIN:-$W/bin/lb-expert-host-cpu-rdma-v2}" --model "$MODEL" --name hx --hold "$LAYERS:0-63" \
      --listen tcp://$IP:47420 --listen rdma://$IP:47421 --wait spin:200 --threads 8 --stats-s 30 > "$W/logs/w3/w3c-t440-host.log" 2>&1 < /dev/null &
    echo $! > "$PIDF"
    for _ in $(seq 1 120); do grep -q "listening on rdma" "$W/logs/w3/w3c-t440-host.log" && break; sleep 1; done
    grep "listening\|held\|rror" "$W/logs/w3/w3c-t440-host.log";;
  stop)
    [ -f "$PIDF" ] && { kill -INT "$(cat $PIDF)" 2>/dev/null; sleep 2; pkill -9 -P "$(cat $PIDF)" 2>/dev/null; kill -9 "$(cat $PIDF)" 2>/dev/null; rm -f "$PIDF"; }
    pgrep -af "lb-expert-host-cpu-rd[m]a" || echo "t440 host stopped";;
esac

#!/usr/bin/env bash
# W0 hop bench against another node: starts the echo host there over ssh (one connection, then it
# exits; `timeout` is the backstop), reads that node's load, runs the client here, appends a JSONL row.
#   w0-remote.sh <ssh-host> <remote-ip> <remote-binary> <label> <scheme: tcp|rdma> <server-wait> <client-wait>
# The binary must be reachable on the remote node (NFS path of /mnt/fast, or a copy in its /tmp).
set -euo pipefail
SSH=$1; IP=$2; RBIN=$3; LABEL=$4; SCHEME=$5; SW=$6; CW=$7
OUT=${OUT:-/mnt/fast/w2w/results/w0-hop.jsonl}
BIN=${BIN:-/mnt/fast/w2w/target/release/lb-hop-bench}
PORT=${PORT:-47400}
ssh -o BatchMode=yes "$SSH" "setsid nohup timeout 180 nice -n 5 $RBIN serve --listen $SCHEME://$IP:$PORT --wait $SW --conns 1 >/tmp/w2w-hop-serve.log 2>&1 </dev/null & sleep 1; cut -d' ' -f1 /proc/loadavg" > /tmp/w2w-rload.$$
RLOAD=$(cat /tmp/w2w-rload.$$); rm -f /tmp/w2w-rload.$$
"$BIN" run --connect "$SCHEME://$IP:$PORT" --wait "$CW" --server-wait "$SW" --label "$LABEL" --remote-load "$RLOAD" --out "$OUT" ${TRIPS:+--trips $TRIPS}

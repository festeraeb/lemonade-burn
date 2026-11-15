#!/bin/bash
# p0.sh <label> <card uuid> <numa> [ENV=..]... -- <server args>   (on c4). Runs lemonade-burn-server, cm.py 3 prompts x NTOK
R=/mnt/t440/fast/eng123/qwen-results; L=$1; U=$2; NUMA=$3; shift 3
ENVS=(); while [ "$1" != "--" ]; do ENVS+=("$1"); shift; done; shift
BINDIR=${BINDIR:-/mnt/t440/raid0/engines/lemonade-burn/20261113-prefill-split-2337068}
PRE=""; [ "$NUMA" != "none" ] && PRE="numactl --cpunodebind=$NUMA --membind=$NUMA"
unset NAUTIBURN_SCRAPE_DIR
env LD_LIBRARY_PATH=$BINDIR/lib CUDA_PATH=$BINDIR/cuda-path CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$U LB_CUDA_DEVICE=0 "${ENVS[@]}" $PRE $BINDIR/bin/lemonade-burn-server --model /mnt/t440/raid0/models/lfm25/LFM2.5-24B-A2B-QAD-Q4_0.gguf --host 127.0.0.1 --port 8198 --model-id lfm "$@" > $R/$L.log 2>&1 &
SP=$!
for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:8198/v1/models >/dev/null && break; kill -0 $SP 2>/dev/null || break; ping -c 2 127.0.0.1 >/dev/null; done
python3 $R/${CLIENT:-cm.py} $R/$L.json ${NTOK:-200}
kill $SP; for w in $(seq 1 30); do kill -0 $SP 2>/dev/null || break; ping -c 2 127.0.0.1 >/dev/null; done; kill -0 $SP 2>/dev/null && kill -9 $SP
grep -E "loaded|warm|cache|threads" $R/$L.log | cut -c1-200 | head -5

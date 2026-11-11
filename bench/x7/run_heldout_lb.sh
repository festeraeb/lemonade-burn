#!/usr/bin/env bash
# held-out-v1 through lemonade-burn-server on one card with one Q4_0 math (run on c4).
# usage: run_heldout_lb.sh <UUID> <numa> <math> <label> [model]  -> results/heldout-lb.jsonl
set -u
U=$1; N=$2; MATH=$3; L=$4; M=${5:-$HOME/lb-e2/models/LFM2.5-24B-A2B-QAD-Q4_0.gguf}
X=/mnt/t440/fast/x7; BASE=$HOME/lb-e2
used=$(nvidia-smi --id=$U --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
[ "$used" -lt 64 ] || { echo "card busy: $used MiB"; exit 3; }
for p in $(seq 18400 18460); do ss -ltn | grep -q ":$p " || { PORT=$p; break; }; done
export LD_LIBRARY_PATH=$X/lib CUDA_PATH=$X/cuda-path CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$U LB_CUDA_DEVICE=0 LB_EXPERT_CACHE=0 LB_EXPERT_LAYERS=1 LB_THREADS=8 LB_Q4_MATH=$MATH
unset NAUTIBURN_SCRAPE_DIR
LOG=$X/logs/server-$L.log
numactl --cpunodebind=$N --membind=$N $X/target/release/lemonade-burn-server --model "$M" --port $PORT --ctx 4096 --gpu --gpu-dense-mb 900 \
  --miss-policy on-demand --placement-from-scrapes $BASE/scrapes --gpu-budget-mb ${BUDGET:-14500} > "$LOG" 2>&1 &
SRV=$!
trap 'kill -INT $SRV 2>/dev/null; sleep 3; kill -9 $SRV 2>/dev/null' EXIT
for _ in $(seq 1 600); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; kill -0 $SRV 2>/dev/null || { echo "server died"; tail -5 "$LOG"; exit 1; }; sleep 2; done
python3 $X/heldout_run.py http://127.0.0.1:$PORT "$L" $X/results/heldout-lb.jsonl
echo "vram $(nvidia-smi --id=$U --query-gpu=memory.used --format=csv,noheader)"

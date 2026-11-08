#!/usr/bin/env bash
# One measured session of MoE experiment #2 (STAGE ring) on cesarops2: start one engine arm,
# warm up, run every prompt of prompts.jsonl once (greedy), append one JSON row per completion to
# $OUT/runs.jsonl and one per session to $OUT/sessions.jsonl. Idempotent: a session whose label
# is already marked done in sessions.jsonl is skipped.
# usage: e2e.sh <arm> <round>
#   a  = lemonade-burn, heat-planned residency + on-demand misses (device decode, host MoE layers)
#   b  = a + staging ring (LB_STAGE=1, mid1 guess, 8 slots/layer, Least-Stale ring)
#   bk = b + demand misses kept in the ring (LB_STAGE_KEEP_MISSES=1)
#   c  = llama.cpp e14a779, -cmoe + --moe-cache-mib at the same expert budget
# env: MIB (expert VRAM MiB, default 5120), RING (ring MiB, default 256), OUT, SLOTS, MARGIN
set -u
. /mnt/ssd/lb-stage/env.sh
ARM=$1; ROUND=$2
case $ARM in b4m) SLOTS=4; MARGIN=${MARGIN_GATE:-0.0054};; b4) SLOTS=4;; b4k) SLOTS=4;; k) SLOTS=0;; esac
MIB=${MIB:-5120}; RING=${RING:-256}; SLOTS=${SLOTS:-8}; MARGIN=${MARGIN:-0}
LLAMA=${LLAMA:-$BASE/llama}
OUT=${OUT:-/mnt/t440/fast/lb-stage/2026-11-08}
TAG=${TAG:-m$MIB}
LABEL=$TAG-$ARM-r$ROUND
mkdir -p "$OUT/logs"
if grep -q "\"label\": \"$LABEL\".*\"session_done\": true" "$OUT/sessions.jsonl" 2>/dev/null; then echo "skip $LABEL (done)"; exit 0; fi
card_check || exit 3
for p in $(seq 8300 8360); do ss -ltn | grep -q ":$p " || { PORT=$p; break; }; done
LOG=$OUT/logs/server-$LABEL.log
LBARGS="--model $LB_MODEL --port $PORT --ctx 2048 --gpu --placement-from-scrapes $BASE/scrapes --gpu-dense-mb 900 --miss-policy on-demand"
COMMON="LB_DD_HOST_MOE=1 LB_EXPERT_VRAM_MIB=$MIB LB_THREADS=8"
case $ARM in
  a)  env $COMMON $BASE/bin/lemonade-burn-server-stage $LBARGS > "$LOG" 2>&1 & ;;
  b|b4|b4m)  env $COMMON LB_STAGE=1 LB_STAGE_SLOTS=$SLOTS LB_STAGE_RING_MIB=$RING LB_STAGE_MIN_MARGIN=$MARGIN $BASE/bin/lemonade-burn-server-stage $LBARGS > "$LOG" 2>&1 & ;;
  bk|b4k|k) env $COMMON LB_STAGE=1 LB_STAGE_SLOTS=$SLOTS LB_STAGE_RING_MIB=$RING LB_STAGE_MIN_MARGIN=$MARGIN LB_STAGE_KEEP_MISSES=1 $BASE/bin/lemonade-burn-server-stage $LBARGS > "$LOG" 2>&1 & ;;
  c)  LD_LIBRARY_PATH=$LLAMA/bin:$BASE/llama/cuda12-libs $LLAMA/bin/llama-server -m $LB_MODEL --port $PORT -c 2048 -ngl 99 -cmoe --moe-cache-mib $MIB -fa on -t 8 -np 1 -lv 4 > "$LOG" 2>&1 & ;;
  *) echo "unknown arm $ARM"; exit 2;;
esac
SRV=$!
echo $SRV > "$OUT/logs/server-$LABEL.pid"
sudo -n $BASE/bin/imc_bw > "$OUT/logs/imc-$LABEL.jsonl" 2>/dev/null &
IMC=$!
cleanup() { kill -INT "$SRV" 2>/dev/null; for _ in $(seq 1 20); do kill -0 "$SRV" 2>/dev/null || break; sleep 1; done; kill "$SRV" 2>/dev/null; sudo -n kill "$IMC" 2>/dev/null; sleep 3; }
trap cleanup EXIT
for _ in $(seq 1 600); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
  kill -0 "$SRV" 2>/dev/null || { echo "{\"label\": \"$LABEL\", \"error\": \"server exited during load\"}" | tee -a "$OUT/sessions.jsonl"; tail -5 "$LOG"; exit 1; }
  sleep 2
done
VRAM=$(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")
LABEL=$LABEL ARM=$ARM ROUND=$ROUND PORT=$PORT OUT=$OUT VRAM=$VRAM MIB=$MIB RING=$RING SLOTS=$SLOTS MARGIN=$MARGIN python3 $BASE/drive.py

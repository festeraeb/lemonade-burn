#!/usr/bin/env bash
# Engine step E1 (dynamic SLRU expert cache) session on cesarops2: one engine start, warmup, every
# prompt PASSES times (default 1), one JSON row per completion (with device memory after it) to
# $OUT/runs.jsonl, one per session to $OUT/sessions.jsonl. Idempotent: finished sessions are skipped.
# usage: e2e-e1.sh <arm> <round>
#   e1 = lemonade-burn with the expert cache (LB_EXPERT_VRAM_MIB=MIB, seeded from scrapes/)
#   a0 = lemonade-burn static heat plan + on-demand misses (LB_EXPERT_CACHE=0), arm (a) of exp 2
#   c  = llama.cpp e14a779 -cmoe --moe-cache-mib MIB (sm_60/61/75 build in llama60/)
# env: CARD_UUID (required), MIB (default 5120), OUT, TAG, PASSES
set -u
. /mnt/ssd/lb-stage/env.sh
ARM=$1; ROUND=$2
MIB=${MIB:-5120}; PASSES=${PASSES:-1}
OUT=${OUT:?OUT}
TAG=${TAG:-m$MIB}
LABEL=$TAG-$ARM-r$ROUND
LBBIN=${LBBIN:-$BASE/bin/lemonade-burn-server-e1}
LLAMA=${LLAMA:-$BASE/llama60}
mkdir -p "$OUT/logs"
if grep -q "\"label\": \"$LABEL\".*\"session_done\": true" "$OUT/sessions.jsonl" 2>/dev/null; then echo "skip $LABEL (done)"; exit 0; fi
card_check || exit 3
for p in $(seq 8300 8360); do ss -ltn | grep -q ":$p " || { PORT=$p; break; }; done
LOG=$OUT/logs/server-$LABEL.log
LBARGS="--model $LB_MODEL --port $PORT --ctx 2048 --gpu --placement-from-scrapes $BASE/scrapes --gpu-dense-mb 900 --miss-policy on-demand"
case $ARM in
  e1) env LB_DD_HOST_MOE=1 LB_EXPERT_VRAM_MIB=$MIB LB_THREADS=8 LB_EXPERT_CACHE=1 $LBBIN $LBARGS > "$LOG" 2>&1 & ;;
  a0) env LB_DD_HOST_MOE=1 LB_EXPERT_VRAM_MIB=$MIB LB_THREADS=8 LB_EXPERT_CACHE=0 $LBBIN $LBARGS > "$LOG" 2>&1 & ;;
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
LABEL=$LABEL ARM=$ARM ROUND=$ROUND PORT=$PORT OUT=$OUT VRAM=$VRAM MIB=$MIB RING=0 SLOTS=0 MARGIN=0 PASSES=$PASSES python3 $BASE/drive-e1.py

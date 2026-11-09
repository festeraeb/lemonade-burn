#!/usr/bin/env bash
# One measured engine session: start one arm, warm up, run every prompt PASSES times (greedy), append
# a row per completion to $OUT/runs.jsonl and one per session to $OUT/sessions.jsonl. A session already
# marked done is skipped, so a rerun resumes. usage: e2e.sh <arm> <round>
#   e1      lemonade-burn step E1 (dynamic cache, host router), binary lemonade-burn-server-e1
#   e2      step E2 (device router and expert routing), binary lemonade-burn-server-e2
#   e2g     e2 + the next-layer guess on the device (LB_DEVICE_GUESS=1)
#   c       llama.cpp e14a779 -cmoe --moe-cache-mib MIB
#   allres  E1 binary with every expert resident (LB_EXPERT_CACHE=0): the all-resident sha reference
# env: MIB (expert VRAM MiB, default 5120), OUT, TAG, PASSES
set -u
. "$(dirname "$0")/env.sh"
ARM=$1; ROUND=$2
MIB=${MIB:-5120}; PASSES=${PASSES:-1}
OUT=${OUT:?OUT}; TAG=${TAG:-m$MIB}
LABEL=$TAG-$ARM-r$ROUND
mkdir -p "$OUT/logs"
if grep -q "\"label\": \"$LABEL\".*\"session_done\": true" "$OUT/sessions.jsonl" 2>/dev/null; then echo "skip $LABEL (done)"; exit 0; fi
card_check || exit 3
for p in $(seq 8400 8460); do ss -ltn | grep -q ":$p " || { PORT=$p; break; }; done
LOG=$OUT/logs/server-$LABEL.log
LBARGS="--model $LB_MODEL --port $PORT --ctx 2048 --gpu --placement-from-scrapes $BASE/scrapes --gpu-dense-mb 900 --miss-policy on-demand"
CM="LB_DD_HOST_MOE=1 LB_EXPERT_VRAM_MIB=$MIB LB_THREADS=8 LB_EXPERT_CACHE=1"
case $ARM in
  e1)  env $CM $NUMA $BASE/bin/lemonade-burn-server-e1 $LBARGS > "$LOG" 2>&1 & ;;
  e2)  env $CM LB_E2=1 $NUMA $BASE/bin/lemonade-burn-server-e2 $LBARGS > "$LOG" 2>&1 & ;;
  e2g) env $CM LB_E2=1 LB_DEVICE_GUESS=1 $NUMA $BASE/bin/lemonade-burn-server-e2 $LBARGS > "$LOG" 2>&1 & ;;
  c)   env LD_LIBRARY_PATH=$BASE/llama/bin:$BASE/llama/cuda12-libs $NUMA $BASE/llama/bin/llama-server -m $LB_MODEL --port $PORT -c 2048 -ngl 99 -cmoe --moe-cache-mib $MIB -fa on -t 8 -np 1 -lv 4 > "$LOG" 2>&1 & ;;
  allres) env LB_THREADS=8 LB_EXPERT_CACHE=0 LB_EXPERT_LAYERS=1 $NUMA $BASE/bin/lemonade-burn-server-e1 --model $LB_MODEL --port $PORT --ctx 2048 --gpu --placement-from-scrapes $BASE/scrapes --gpu-budget-mb 14500 --gpu-dense-mb 900 --miss-policy on-demand > "$LOG" 2>&1 & ;;
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
# where the model's pages really are: the file mapping's per-node counts in this process
grep -F "$(basename "$LB_MODEL")" /proc/$SRV/numa_maps | grep -oE "N[0-9]+=[0-9]+" | tr '\n' ' ' > "$OUT/logs/numa-$LABEL.txt"
numastat -p $SRV > "$OUT/logs/numastat-$LABEL.txt" 2>&1
export NUMA_MAP="$(cat "$OUT/logs/numa-$LABEL.txt")"
LABEL=$LABEL ARM=$ARM ROUND=$ROUND PORT=$PORT OUT=$OUT VRAM=$VRAM MIB=$MIB RING=0 SLOTS=0 MARGIN=0 PASSES=$PASSES python3 "$(dirname "$0")/drive.py"

#!/usr/bin/env bash
# One measured E3 session: start one arm, warm up, run every prompt PASSES times (greedy), append a row
# per completion to $OUT/runs.jsonl and one per session to $OUT/sessions.jsonl. A session marked done is
# skipped, so a rerun resumes. usage: e2e.sh <arm> <round>
#   e2 = the E2 binary from the E2 working copy (the gate baseline)
#   c  = llama.cpp e14a779 -cmoe --moe-cache-mib MIB
#   anything else = the E3 binary with the LB_LA_* settings of that arm from arms.env
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
for p in $(seq 8500 8560); do ss -ltn | grep -q ":$p " || { PORT=$p; break; }; done
LOG=$OUT/logs/server-$LABEL.log
LBARGS="--model $LB_MODEL --port $PORT --ctx 2048 --gpu --placement-from-scrapes $BASE/scrapes --gpu-dense-mb 900 --miss-policy on-demand"
CM="LB_DD_HOST_MOE=1 LB_EXPERT_VRAM_MIB=$MIB LB_THREADS=8 LB_EXPERT_CACHE=1 LB_E2=1"
case $ARM in
  e2) env $CM $NUMA $BASE/bin/lemonade-burn-server-e2 $LBARGS > "$LOG" 2>&1 & ;;
  c)  env LD_LIBRARY_PATH=$BASE/llama/bin:$BASE/llama/cuda12-libs $NUMA $BASE/llama/bin/llama-server -m $LB_MODEL --port $PORT -c 2048 -ngl 99 -cmoe --moe-cache-mib $MIB -fa on -t 8 -np 1 -lv 4 > "$LOG" 2>&1 & ;;
  *)  S=$(grep "^$ARM " "$(dirname "$0")/arms.env" | cut -d" " -f2-)
      [ -n "$S" ] || [ "$ARM" = off ] || { echo "unknown arm $ARM"; exit 2; }
      [ -n "${TRACE:-}" ] && S="$S LB_LA_TRACE=$OUT/logs/trace-$LABEL.jsonl"
      env $CM $S $NUMA "$E3BIN" $LBARGS > "$LOG" 2>&1 & ;;
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
grep -F "$(basename "$LB_MODEL")" /proc/$SRV/numa_maps | grep -oE "N[0-9]+=[0-9]+" | tr '\n' ' ' > "$OUT/logs/numa-$LABEL.txt"
export NUMA_MAP="$(cat "$OUT/logs/numa-$LABEL.txt")"
LABEL=$LABEL ARM=$ARM ROUND=$ROUND PORT=$PORT OUT=$OUT VRAM=$VRAM MIB=$MIB RING=0 SLOTS=0 MARGIN=0 PASSES=$PASSES python3 "$(dirname "$0")/drive.py"

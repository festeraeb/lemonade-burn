#!/usr/bin/env bash
# One measured W3 session on c4 (run there, from the NFS copy of this file): start the expert hosts of
# one arm on their cards, start the trunk server on its card, warm up, run every prompt (greedy), append a
# row per completion to $OUT/runs.jsonl and one per session to $OUT/sessions.jsonl, stop everything and
# check that every card is back to 0 MiB. Cards are named by UUID only. usage: w3-session.sh <arm> <round>
#   allres    one card, every expert resident (the same-session baseline for W3a)
#   spill     one card, E2 cache with 5120 MiB of experts (the baseline for W3d)
#   w3a       trunk on $TRUNK, every expert on one host card
#   w3b2/w3b4 trunk on $TRUNK, experts as whole layers over 2 / 4 host cards
#   w3d       trunk holds layers 2-15 itself (whole layers, <= 5120 MiB of experts), layers 16-39 on a host card
#   w3c-rdma / w3c-tcp   as w3a, but layers 36-39 on a CPU host on t440 (started separately): a wire test
set -u
ARM=$1; ROUND=$2
W=${W:-/mnt/t440/fast/w2w}; BASE=${BASE:-/home/cesarops/lb-e2}; BIN=$W/bin
OUT=${OUT:-$W/results/w3}; LOGS=${LOGS:-$W/logs/w3}; PASSES=${PASSES:-1}; TW=${TW:-spin:200}
MODEL=$BASE/models/LFM2.5-24B-A2B-QAD-Q4_0.gguf
export LD_LIBRARY_PATH=$BASE/lib:${LD_LIBRARY_PATH:-} CUDA_PATH=$BASE/cuda-path CUDA_DEVICE_ORDER=PCI_BUS_ID
unset NAUTIBURN_SCRAPE_DIR
declare -A U=( [3B]=GPU-72784f3e-9536-8d2b-cc49-3081a131de54 [3C]=GPU-c012cc70-3f06-1718-84e2-073aa275f0af [86]=GPU-0c4b4f5d-5f54-46db-be25-f6ffdf75ac79 [AF]=GPU-fd9b03e5-aed9-cbca-a57b-efc90d61addd [D8]=GPU-2d66f961-cba7-717f-1bd7-a04fff1fc8ab )
TRUNK=${TRUNK:-D8}
node_of() { local b; b=$(nvidia-smi --id="${U[$1]}" --query-gpu=pci.bus_id --format=csv,noheader | tr 'A-Z' 'a-z' | sed 's/^0000\(....:\)/\1/'); cat /sys/bus/pci/devices/$b/numa_node; }
used_of() { nvidia-smi --id="${U[$1]}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }
LABEL=${TAG:-w3}-$ARM-r$ROUND
mkdir -p "$OUT" "$LOGS" "$W/run/w3"
if grep -q "\"label\":\"$LABEL\".*\"session_done\":true" "$OUT/sessions.jsonl" 2>/dev/null; then echo "skip $LABEL (done)"; exit 0; fi
for c in 3B 3C 86 AF D8; do [ "$(used_of $c)" -lt 64 ] || { echo "card $c busy ($(used_of $c) MiB): stop"; exit 3; }; done
PIDS=()
cleanup() {
  for p in "${PIDS[@]}"; do kill -INT "$p" 2>/dev/null; done
  for _ in $(seq 1 20); do local alive=0; for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done; [ $alive = 0 ] && break; sleep 1; done
  for p in "${PIDS[@]}"; do kill -9 "$p" 2>/dev/null; done
  rm -f /tmp/w2w-$LABEL-*.sock
  sleep 2
}
trap cleanup EXIT
HOSTS=()   # name:card:layers
case $ARM in
  allres|spill) ;;
  w3a) HOSTS=(h1:AF:2-39);;
  w3b2) HOSTS=(h1:AF:2-20 h2:86:21-39);;
  w3b4) HOSTS=(h1:AF:2-11 h2:86:12-21 h3:3B:22-30 h4:3C:31-39);;
  w3d) HOSTS=(h1:AF:16-39);;
  w3c-rdma|w3c-tcp) HOSTS=(h1:AF:2-35);;
  w3e|w3e2|w3e4|w3ed) ;;
  *) echo "unknown arm $ARM"; exit 2;;
esac
# in-box cards (same process as the trunk, W3e / X1): name:card:layers, CUDA index = position + 1
INBOX=()
case $ARM in
  w3e) INBOX=(g1:AF:2-39);;
  w3e2) INBOX=(g1:AF:2-20 g2:86:21-39);;
  w3e4) INBOX=(g1:AF:2-11 g2:86:12-21 g3:3B:22-30 g4:3C:31-39);;
  w3ed) INBOX=(g1:AF:16-39);;
esac
PL=$W/run/w3/$LABEL.json
HJ=""; LJ=""
for h in "${HOSTS[@]}"; do
  IFS=: read -r name card layers <<< "$h"
  sock=/tmp/w2w-$LABEL-$name.sock
  HJ="$HJ\"$name\":{\"addr\":\"unix://$sock\",\"wait\":\"$TW\",\"deadline_ms\":30000},"
  LJ="$LJ\"$layers\":\"$name\","
done
case $ARM in
  w3c-rdma) HJ="$HJ\"hx\":{\"addr\":\"rdma://10.0.10.61:47421\",\"wait\":\"spin:200\",\"deadline_ms\":30000},"; LJ="$LJ\"36-39\":\"hx\",";;
  w3c-tcp)  HJ="$HJ\"hx\":{\"addr\":\"tcp://10.0.10.61:47420\",\"wait\":\"block\",\"deadline_ms\":30000},"; LJ="$LJ\"36-39\":\"hx\",";;
esac
VIS=${U[$TRUNK]}; i=1
for h in "${INBOX[@]}"; do
  IFS=: read -r name card layers <<< "$h"
  VIS="$VIS,${U[$card]}"; HJ="$HJ\"$name\":{\"addr\":\"gpu://$i\"},"; LJ="$LJ\"$layers\":\"$name\","; i=$((i+1))
done
echo "{\"version\":1,\"hosts\":{${HJ%,}},\"default\":\"local\",\"layers\":{${LJ%,}}}" > "$PL"
for h in "${HOSTS[@]}"; do
  IFS=: read -r name card layers <<< "$h"
  n=$(node_of $card)
  CUDA_VISIBLE_DEVICES=${U[$card]} LB_CUDA_DEVICE=0 numactl --cpunodebind=$n --membind=$n "${HOSTBIN:-$BIN/lb-expert-host-cuda-rdma}" --model "$MODEL" --placement "$PL" --name "$name" --backend gpu \
    --listen "unix:///tmp/w2w-$LABEL-$name.sock" --wait "$TW" --threads 4 --stats-s 0 > "$LOGS/host-$LABEL-$name.log" 2>&1 &
  PIDS+=($!)
done
for h in "${HOSTS[@]}"; do
  IFS=: read -r name card layers <<< "$h"
  for _ in $(seq 1 300); do grep -q "listening on" "$LOGS/host-$LABEL-$name.log" && break; grep -q "^Error" "$LOGS/host-$LABEL-$name.log" && { echo "host $name failed:"; tail -3 "$LOGS/host-$LABEL-$name.log"; exit 1; }; sleep 1; done
  grep -q "listening on" "$LOGS/host-$LABEL-$name.log" || { echo "host $name did not come up"; exit 1; }
done
for p in $(seq 8400 8460); do ss -ltn | grep -q ":$p " || { PORT=$p; break; }; done
LOG=$LOGS/server-$LABEL.log
n=$(node_of $TRUNK)
COMMON="--model $MODEL --port $PORT --ctx 2048 --gpu --gpu-dense-mb 900 --miss-policy on-demand"
export CUDA_VISIBLE_DEVICES=$VIS LB_CUDA_DEVICE=0 LB_THREADS=8
[ ${#INBOX[@]} -gt 0 ] && export LD_LIBRARY_PATH=$W/lib-nccl:$LD_LIBRARY_PATH
SRVBIN=${SRVBIN:-$BIN/lemonade-burn-server-cuda-rdma}
NUMA="numactl --cpunodebind=$n --membind=$n"
case $ARM in
  allres) LB_EXPERT_CACHE=0 LB_EXPERT_LAYERS=1 $NUMA "$SRVBIN" $COMMON --placement-from-scrapes $BASE/scrapes --gpu-budget-mb 14500 > "$LOG" 2>&1 & ;;
  spill)  LB_DD_HOST_MOE=1 LB_EXPERT_VRAM_MIB=5120 LB_EXPERT_CACHE=1 LB_E2=1 $NUMA "$SRVBIN" $COMMON --placement-from-scrapes $BASE/scrapes > "$LOG" 2>&1 & ;;
  w3ed)   LB_EXPERT_CACHE=0 LB_EXPERT_LAYERS=1 $NUMA "$SRVBIN" $COMMON --whole-layers --gpu-budget-mb ${D_BUDGET_MB:-4800} --w2w-placement "$PL" > "$LOG" 2>&1 & ;;
  w3e|w3e2|w3e4) LB_EXPERT_CACHE=0 $NUMA "$SRVBIN" $COMMON --gpu-budget-mb 0 --w2w-placement "$PL" > "$LOG" 2>&1 & ;;
  w3d)    LB_EXPERT_CACHE=0 LB_EXPERT_LAYERS=1 LB_W2W_LOG=$LOGS/passes-$LABEL.jsonl $NUMA "$SRVBIN" $COMMON --whole-layers --gpu-budget-mb ${D_BUDGET_MB:-4800} --w2w-placement "$PL" > "$LOG" 2>&1 & ;;
  *)      LB_EXPERT_CACHE=0 LB_W2W_LOG=$LOGS/passes-$LABEL.jsonl $NUMA "$SRVBIN" $COMMON --gpu-budget-mb 0 --w2w-placement "$PL" > "$LOG" 2>&1 & ;;
esac
SRV=$!; PIDS+=($SRV)
for _ in $(seq 1 600); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
  kill -0 "$SRV" 2>/dev/null || { echo "{\"label\":\"$LABEL\",\"error\":\"server exited during load\"}" | tee -a "$OUT/sessions.jsonl"; tail -5 "$LOG"; exit 1; }
  sleep 2
done
VR=""; for c in 3B 3C 86 AF D8; do VR="$VR\"$c\":$(used_of $c),"; done
NMAP=$(grep -F "$(basename "$MODEL")" /proc/$SRV/numa_maps | grep -oE "N[0-9]+=[0-9]+" | tr '\n' ' ')
WANT=(); [ -f "$OUT/ref-shas.txt" ] && while read -r id sha; do WANT+=(--want "$id=$sha"); done < "$OUT/ref-shas.txt"
[ ${#WANT[@]} -gt 0 ] || WANT=(--want sonar-32=062392b8d2b0361a)
"$BIN/w2w-drive" --port $PORT --prompts $BASE/prompts.jsonl --passes $PASSES --out "$OUT/runs.jsonl" "${WANT[@]}" \
  --tag label=$LABEL --tag arm=$ARM --tag round=$ROUND --tag trunk=$TRUNK --tag "host_cards=$(IFS=,; echo "${HOSTS[*]:-}${INBOX[*]:-}")" --tag "trunk_wait=$TW" --tag "trunk_vram_mib=$(used_of $TRUNK)"
RC=$?
curl -s "http://127.0.0.1:$PORT/v1/placement" > "$LOGS/placement-$LABEL.json"
echo "{\"label\":\"$LABEL\",\"arm\":\"$ARM\",\"round\":$ROUND,\"vram_mib\":{${VR%,}},\"numa_map\":\"$NMAP\",\"drive_rc\":$RC,\"load1\":$(cut -d' ' -f1 /proc/loadavg),\"session_done\":true}" | tee -a "$OUT/sessions.jsonl"
cleanup; trap - EXIT
LEFT=""; for c in 3B 3C 86 AF D8; do LEFT="$LEFT $c=$(used_of $c)"; done; echo "cards after:$LEFT"
exit $RC

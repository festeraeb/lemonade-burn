#!/usr/bin/env bash
# parallel ranged download: pdl.sh URL OUT SIZE NPARTS
set -u
URL=$1; OUT=$2; SIZE=$3; N=$4; D=$OUT.parts; mkdir -p "$D"
CH=$(( (SIZE + N - 1) / N )); PIDS=()
for i in $(seq 0 $((N-1))); do
  s=$((i*CH)); e=$(( s+CH-1 )); [ $e -ge $SIZE ] && e=$((SIZE-1))
  want=$((e-s+1))
  ( for t in $(seq 1 20); do
      have=$(stat -c %s "$D/p$i" 2>/dev/null || echo 0)
      [ "$have" -ge "$want" ] && break
      curl -sSL --retry 3 -r $((s+have))-$e "$URL" >> "$D/p$i" || sleep 5
    done ) &
  PIDS+=($!)
done
wait "${PIDS[@]}"
tot=0; for i in $(seq 0 $((N-1))); do tot=$((tot + $(stat -c %s "$D/p$i"))); done
[ $tot -eq $SIZE ] || { echo "size mismatch $tot != $SIZE"; exit 1; }
for i in $(seq 0 $((N-1))); do cat "$D/p$i"; done > "$OUT" && rm -rf "$D" && echo "done $OUT"

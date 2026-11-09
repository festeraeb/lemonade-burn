#!/usr/bin/env bash
# Interleaved rounds (odd rounds in ARMS order, even reversed), then optional budget sweep and soak.
# Resumable: finished sessions are skipped. env: OUT, ARMS, ROUNDS, SWEEP ("3072 ..."), SOAK (passes).
cd "$(dirname "$0")"
OUT=${OUT:?OUT}; export OUT; mkdir -p "$OUT"; echo $$ > "$OUT/run.pid"
ARMS=${ARMS:-"e1 c"}; ROUNDS=${ROUNDS:-5}
for r in $(seq 1 "$ROUNDS"); do
  list=$ARMS; [ $((r % 2)) = 0 ] && list=$(echo $ARMS | tr " " "\n" | tac | tr "\n" " ")
  for a in $list; do echo "== $(date -u +%FT%TZ) round $r arm $a"; ./e2e.sh "$a" "$r"; rc=$?; [ $rc = 3 ] && exit 3; done
done
for m in ${SWEEP:-}; do echo "== $(date -u +%FT%TZ) sweep $m"; TAG=m$m MIB=$m ./e2e.sh ${SWEEP_ARM:-e2} 1; done
if [ -n "${SOAK:-}" ]; then echo "== $(date -u +%FT%TZ) soak $SOAK passes"; TAG=soak PASSES=$SOAK ./e2e.sh ${SWEEP_ARM:-e2} 1; fi
echo "== run all done $(date -u +%FT%TZ)"

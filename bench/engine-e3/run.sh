#!/usr/bin/env bash
# Interleaved rounds (odd rounds in ARMS order, even reversed), then optional extras.
# Resumable: finished sessions are skipped. env: OUT, ARMS, ROUNDS, MIB.
cd "$(dirname "$0")"
OUT=${OUT:?OUT}; export OUT; mkdir -p "$OUT"; echo $$ > "$OUT/run.pid"
ARMS=${ARMS:-"e2 off"}; ROUNDS=${ROUNDS:-5}
for r in $(seq 1 "$ROUNDS"); do
  list=$ARMS; [ $((r % 2)) = 0 ] && list=$(echo $ARMS | tr " " "\n" | tac | tr "\n" " ")
  for a in $list; do echo "== $(date -u +%FT%TZ) round $r arm $a mib ${MIB:-5120}"; ./e2e.sh "$a" "$r"; rc=$?; [ $rc = 3 ] && exit 3; done
done
echo "== run all done $(date -u +%FT%TZ)"

#!/usr/bin/env bash
# E1 gate runs on one card: ARMS x ROUNDS interleaved (odd rounds in order, even reversed), then the
# budget sweep for the sha check. Resumable: finished sessions are skipped. usage: CARD=2060|p100 run_e1.sh
cd /mnt/ssd/lb-stage
case ${CARD:?CARD} in
  2060) export CARD_UUID=${UUID_RTX2060S:?card UUID} ;;
  p100) export CARD_UUID=${UUID_P100_3C:?card UUID} ;;
esac
export OUT=/mnt/t440/fast/lb-stage/2026-11-08/e1-cache/$CARD
mkdir -p $OUT; echo $$ > $OUT/run_e1.pid
ARMS=${ARMS:-"e1 c a0"}; ROUNDS=${ROUNDS:-5}
for r in $(seq 1 $ROUNDS); do
  list=$ARMS; [ $((r % 2)) = 0 ] && list=$(echo $ARMS | tr " " "\n" | tac | tr "\n" " ")
  for a in $list; do echo "== $(date -u +%FT%TZ) round $r arm $a"; ./e2e-e1.sh $a $r; rc=$?; [ $rc = 3 ] && exit 3; done
done
for m in ${SWEEP:-3072}; do echo "== $(date -u +%FT%TZ) sweep $m"; TAG=m$m MIB=$m ./e2e-e1.sh e1 1; done
if [ -n "${SOAK:-}" ]; then echo "== $(date -u +%FT%TZ) soak $SOAK passes"; TAG=soak PASSES=$SOAK ./e2e-e1.sh e1 1; fi
echo "== e1 all done $(date -u +%FT%TZ)"

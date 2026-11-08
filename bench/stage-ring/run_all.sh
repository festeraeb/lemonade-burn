#!/usr/bin/env bash
# MoE exp #2 formal run: ROUNDS x ARMS sessions, interleaved (odd rounds in ARMS order, even rounds
# reversed). Resumable: e2e.sh skips sessions already marked done in sessions.jsonl, so rerunning
# this script continues where it stopped. Pid in $OUT/run_all.pid.
set -u
cd /mnt/ssd/lb-stage
OUT=${OUT:-/mnt/t440/fast/lb-stage/2026-11-08}; export OUT
ARMS=${ARMS:-a b4 b c}; ROUNDS=${ROUNDS:-5}
mkdir -p $OUT; echo $$ > $OUT/run_all.pid
for r in $(seq 1 $ROUNDS); do
  list=$ARMS; [ $((r % 2)) = 0 ] && list=$(echo $ARMS | tr ' ' '\n' | tac | tr '\n' ' ')
  for a in $list; do
    echo "== $(date -u +%FT%TZ) round $r arm $a"
    ./${E2E:-e2e.sh} $a $r; rc=$?; [ $rc = 0 ] || echo "session $a r$r failed ($rc)"; [ $rc = 3 ] && exit 3
  done
done
echo "== all done $(date -u +%FT%TZ)"

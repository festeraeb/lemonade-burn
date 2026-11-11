#!/usr/bin/env bash
# Interleaved W3 rounds on c4: odd rounds in ARMS order, even rounds reversed. Resumable (a finished session
# is skipped by w3-session.sh). Stops on a busy card (rc 3). env: ARMS, ROUNDS, R0 (first round, default 1).
cd "$(dirname "$0")"
ARMS=${ARMS:-"allres spill w3a w3b2 w3b4 w3d"}; ROUNDS=${ROUNDS:-2}; R0=${R0:-1}
for r in $(seq "$R0" "$ROUNDS"); do
  list=$ARMS; [ $((r % 2)) = 0 ] && list=$(echo $ARMS | tr " " "\n" | tac | tr "\n" " ")
  for a in $list; do echo "== $(date -u +%FT%TZ) round $r arm $a"; ./w3-session.sh "$a" "$r"; rc=$?; echo "== rc $rc"; [ $rc = 3 ] && exit 3; done
done
echo "== rounds all done $(date -u +%FT%TZ)"

#!/usr/bin/env bash
# exp #2b: the same arms on c2 GPU2 (Tesla P100, Gen3 x16), UUID-pinned; b4m = b4 + margin gate.
# Resumable: rerun this script; finished sessions (session_done in sessions.jsonl) are skipped.
cd /mnt/ssd/lb-stage
export CARD_UUID=GPU-c012cc70-3f06-1718-84e2-073aa275f0af LLAMA=/mnt/ssd/lb-stage/llama60 E2E=e2e-p100.sh
export OUT=${OUT:-/mnt/t440/fast/lb-stage/2026-11-08/c2-p100-x16}
ARMS=${ARMS:-"a b4 b4m b c"} ROUNDS=${ROUNDS:-5} ./run_all.sh
echo "== p100 all done $(date -u +%FT%TZ)"

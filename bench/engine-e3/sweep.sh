#!/usr/bin/env bash
# E3 sweep: e2 baseline + arms interleaved over ROUNDS at 5120 MiB, resumable.
cd "$(dirname "$0")"
export OUT=/mnt/t440/fast/lb-stage/2026-11-09/e3/${SET:-sweep-5120}
ARMS=${ARMS:-"e2 off guess keep keepp1 cyc64 pf1t0 pf1t01 pf1t02 pf2t0 pf2t01 pf2t02 pf2t05 pf3t01 kpf2t01 kpf2t02"} ROUNDS=${ROUNDS:-3} MIB=${MIB:-5120} ./run.sh

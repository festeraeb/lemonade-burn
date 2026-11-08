#!/usr/bin/env bash
# exploratory: does a Least-Stale cache (ring keeping demand misses) close the gap to llama.cpp?
cd /mnt/ssd/lb-stage
export LBBIN=/mnt/ssd/lb-stage/bin/lemonade-burn-server-stage2 E2E=e2e-x.sh OUT=/mnt/t440/fast/lb-stage/2026-11-08/c2-2060-x
TAG=m5120r2048 RING=2048 ARMS="a k b4k" ROUNDS=3 ./run_all.sh
TAG=m5120r4096 RING=4096 ARMS="k b4k" ROUNDS=3 ./run_all.sh
echo "== x all done $(date -u +%FT%TZ)"

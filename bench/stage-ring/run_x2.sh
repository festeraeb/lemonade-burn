#!/usr/bin/env bash
# exploratory, after the 2048/4096 MiB rings ran out of device memory: 512 MiB ring (1024 also ran out of memory with kept misses)
cd /mnt/ssd/lb-stage
export LBBIN=/mnt/ssd/lb-stage/bin/lemonade-burn-server-stage2 E2E=e2e-x.sh OUT=/mnt/t440/fast/lb-stage/2026-11-08/c2-2060-x
TAG=m5120r512 RING=512 ARMS="k b4k a" ROUNDS=3 ./run_all.sh
echo "== x2 all done $(date -u +%FT%TZ)"

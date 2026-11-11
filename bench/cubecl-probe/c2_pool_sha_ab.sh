#!/usr/bin/env bash
# Same-config A/B on c2 GPU2: pinned pool on vs off, greedy sha must be identical. Residency is
# cut to fit beside a live seat on the card (the all-resident reference sha 062392b8d2b0361a
# needs ~13.7 GB and a free card). `on-demand` misses make every token upload experts through
# the pool, which is the path the pool exists for.
set -u
BASE=$HOME/lb-measure; OUT=$BASE/out/pool-sha-ab; mkdir -p $OUT
export LD_LIBRARY_PATH=$BASE/lib:${LD_LIBRARY_PATH:-}; export CUDA_PATH=$BASE/cuda-path
ARGS="--gpu --placement-from-scrapes $BASE/scrapes --gpu-budget-mb ${BUDGET:-3200} --gpu-dense-mb ${DENSE:-600} --miss-policy on-demand"
cd $BASE
for label in pool-on pool-off pool-on-rep2; do
  extra=""; [ "$label" = pool-off ] && extra="LB_NO_PINNED=1"
  echo "== $label $(date -u +%H:%M:%S) load $(cut -d' ' -f1 /proc/loadavg)"
  env $extra LB_CUDA_DEVICE=2 BIN=$BASE/cuda-bin/lemonade-burn-server-pinned REPS=2 ./c2_e2e.sh sha-ab-$label $ARGS 2>&1 | tee $OUT/$label.out | grep -E "card:|text_sha|error|decode_tok_s" | cut -c1-220
  grep -E "placement:|dense:|pinned|miss" $OUT/../server-sha-ab-$label.log | head -4
done

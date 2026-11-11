#!/usr/bin/env bash
# Device repack (page-locked mapping, raw DMA, repack kernel) vs host repack (SIMD into the
# pinned pool) vs the previous build, on c2 GPU2 beside the live seat: parity tests, the miss
# breakdown bench, then the same-config on-demand e2e (greedy sha must stay 062392b8d2b0361a).
set -u
BASE=$HOME/lb-measure; OUT=$BASE/out/devrepack-b1; mkdir -p $OUT
export LD_LIBRARY_PATH=$BASE/lib:${LD_LIBRARY_PATH:-}; export CUDA_PATH=$BASE/cuda-path
export LB_MODEL=$BASE/models/LFM2.5-24B-A2B-QAD-Q4_0.gguf LB_CUDA_DEVICE=2
cd $BASE
echo "== $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg) gpu2 $(nvidia-smi -i 2 --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
echo "== parity: device repack vs host repack, cuda GPU2"
./cuda-bin/test-device_repack_matches_host --test-threads=1 2>&1 | grep -E "^test |n_blocks=|test result"
echo "== parity: gpu_matches_scalar (kernels unchanged; dims now shared)"
./cuda-bin/test-gpu_matches_scalar --test-threads=1 2>&1 | grep -E "^test |test result"
for r in 1 2; do
  echo "== miss_bench run $r (registered mapping on; both paths A/B inside)"
  LB_BENCH_REPS=30 LB_BENCH_WARM=40 LB_PINNED_POOL_MB=768 ./cuda-bin/miss_bench > $OUT/miss-r$r.jsonl 2> $OUT/miss-r$r.err; cat $OUT/miss-r$r.jsonl; grep -E "registered|pool" $OUT/miss-r$r.err
done
echo "== miss_bench, LB_NO_REGISTER=1 (host path only, as before this change)"
LB_NO_REGISTER=1 LB_BENCH_REPS=30 LB_BENCH_WARM=40 LB_PINNED_POOL_MB=768 ./cuda-bin/miss_bench > $OUT/miss-noreg.jsonl 2> $OUT/miss-noreg.err; cat $OUT/miss-noreg.jsonl
ARGS="--gpu --placement-from-scrapes $BASE/scrapes --gpu-budget-mb ${BUDGET:-3200} --gpu-dense-mb ${DENSE:-600} --miss-policy on-demand"
for round in 1 2 3; do
  for cfg in devrepack hostrepack prev; do
    case $cfg in
      devrepack) bin=$BASE/cuda-bin/lemonade-burn-server-devrepack; extra="";;
      hostrepack) bin=$BASE/cuda-bin/lemonade-burn-server-devrepack; extra="LB_NO_REGISTER=1";;
      prev) bin=$BASE/cuda-bin/lemonade-burn-server-pinned; extra="";;
    esac
    label=dr-$cfg-r$round
    echo "== e2e $label $(date -u +%H:%M:%S) load $(cut -d' ' -f1 /proc/loadavg)"
    env $extra BIN=$bin REPS=2 ./c2_e2e.sh $label $ARGS > $OUT/$label.out 2>&1
    grep -E "text_sha|error" $OUT/$label.out | cut -c1-200
    grep -ohE "loaded .* in [0-9.]+s|page-locked [^;]*|placement: .*|uploaded .*" $BASE/out/server-$label.log | head -4
    grep -oE '"(gpu_hits|cpu_calls|on_demand|on_demand_s|gpu_s|cpu_s)":[0-9.]+' $OUT/$label.out | tr '\n' ' '; echo
  done
done
echo "== done $(date -u +%FT%TZ)"

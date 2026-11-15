#!/bin/bash
# c1_trace.sh <label> <wgpu idx> <cache_mb> [env ...]: --launch-trace (launches per token, queue time, device time when the runtime has timestamps)
V=/mnt/models/vk; L=$1; D=$2; CMB=$3; shift 3
M=/mnt/models/lfm25/LFM2.5-24B-A2B-QAD-Q4_0.gguf
env LB_WGPU_DEVICE=discrete:$D LB_EXPERT_CACHE=0 LB_EXPERT_LAYERS=1 LB_THREADS=20 "$@" \
 $V/bin/lfm2-check --model "$M" --tokenizer /mnt/models/lfm25/tokenizer.json --gpu --launch-trace \
 --placement-from-scrapes $V/scrapes --gpu-dense-mb ${DENSE:-900} --cache-mb $CMB --gpu-budget-mb 0 --greedy-n 12 --report $V/results/$L.json > $V/results/$L.log 2>&1
echo rc=$? >> $V/results/$L.log

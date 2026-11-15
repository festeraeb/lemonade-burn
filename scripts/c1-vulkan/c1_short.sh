#!/bin/bash
# c1_short.sh <label> <wgpu idx> <cache_mb> [env k=v ...]: one ~70 token prompt, tok/s + E2 profile (writes its own throwaway tf file)
V=/mnt/models/vk; L=$1; D=$2; CMB=$3; shift 3
M=/mnt/models/lfm25/LFM2.5-24B-A2B-QAD-Q4_0.gguf
P="Write a short paragraph about how side-scan sonar images are used to find shipwrecks on the lake bottom, and what a searcher looks for in the data before diving on a target."
env LB_WGPU_DEVICE="${DEV:-discrete:$D}" LB_EXPERT_CACHE=0 LB_EXPERT_LAYERS=1 LB_THREADS=20 LB_E2_PROFILE=$V/results/$L.prof "$@" $PFX \
 $V/${BIN:-bin}/${CHK:-lfm2-check} --model "$M" --tokenizer /mnt/models/lfm25/tokenizer.json --gpu \
 --placement-from-scrapes $V/scrapes --gpu-dense-mb ${DENSE:-900} --cache-mb $CMB --gpu-budget-mb ${BUDGET:-0} $XARGS --prompt "$P" --tf-ref $V/results/$L.tf > $V/results/$L.log 2>&1
echo rc=$? >> $V/results/$L.log

#!/bin/bash
# on c1: c1_agree.sh <label> <wgpu discrete index> <cache_mb> [extra env k=v ...]
# Teacher-forced decode of the 24B on a wgpu card against the CUDA all-resident reference tf-ref.bin.
V=/mnt/models/vk; L=$1; D=$2; CMB=$3; shift 3
M=/mnt/models/lfm25/LFM2.5-24B-A2B-QAD-Q4_0.gguf
PR=(); while read -r t; do PR+=(--prompt "$t"); done < <(python3 -c 'import json;[print(t.replace("\n"," ")) for t in json.load(open("'$V'/agree-prompts.json"))]')
[ -n "$NP" ] && PR=("${PR[@]:0:$((2*NP))}")
env LB_WGPU_DEVICE=discrete:$D LB_EXPERT_CACHE=0 LB_EXPERT_LAYERS=1 LB_THREADS=20 "$@" \
 $V/bin/${CHK:-lfm2-check} --model "$M" --tokenizer /mnt/models/lfm25/tokenizer.json --gpu \
 --placement-from-scrapes $V/scrapes --gpu-dense-mb ${DENSE:-900} --cache-mb $CMB --gpu-budget-mb 0 "${PR[@]}" --tf-ref $V/tf-ref.bin --report $V/results/$L.json > $V/results/$L.log 2>&1
echo rc=$?; grep -E "TOTAL|prompt |expert cache|Gpu|device" $V/results/$L.log | cut -c1-220 | head -20

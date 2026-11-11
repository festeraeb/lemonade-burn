#!/usr/bin/env bash
# usage (on the GPU node): run_abmath.sh <UUID> <numa|-> <label> [model] [maths]   -> results/abmath-<label>.json, logs/abmath-<label>.log
set -u
U=$1; N=$2; L=$3; X=/mnt/t440/fast/x7; [ -d $X ] || X=/mnt/fast/x7
M=${4:-$HOME/lb-e2/models/LFM2.5-24B-A2B-QAD-Q4_0.gguf}; MATHS=${5:-f32m,f16,dp4a}
E2=$X/q/e2
used=$(nvidia-smi --id=$U --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
[ "$used" -lt 64 ] || { echo "card busy: $used MiB"; exit 3; }
export LD_LIBRARY_PATH=$X/lib CUDA_PATH=$X/cuda-path CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$U LB_CUDA_DEVICE=0 LB_EXPERT_CACHE=0 LB_EXPERT_LAYERS=1 LB_THREADS=8
unset NAUTIBURN_SCRAPE_DIR
P=(); while read -r line; do P+=(--prompt "$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["text"])' "$line")"); done < $E2/prompts.jsonl
NUMA=""; [ "$N" != "-" ] && NUMA="numactl --cpunodebind=$N --membind=$N"
$NUMA $X/target/release/lfm2-check --model "$M" --tokenizer $X/models/tokenizer.json --gpu \
  --placement-from-scrapes $E2/scrapes --gpu-budget-mb ${BUDGET:-14500} --gpu-dense-mb 900 "${P[@]}" \
  --ab-math $MATHS --greedy-n 32 --ab-rounds ${ROUNDS:-3} --report $X/results/abmath-$L.json > $X/logs/abmath-$L.log 2>&1
rc=$?; echo "rc=$rc card after: $(nvidia-smi --id=$U --query-gpu=memory.used --format=csv,noheader)"; exit $rc

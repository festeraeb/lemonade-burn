#!/usr/bin/env bash
# llama-bench (pp512 + tg128, 3 reps) per quant on one card of this node. Run ON the node.
# usage: bench_card.sh <UUID> <card-label> <numa|-> <tag>...   -> results/llama-bench.jsonl
set -u
U=$1; CL=$2; N=$3; shift 3
X=/mnt/t440/fast/x7; B=${LBIN:-$X/llama-build/bin}; MD=$X/models
export LD_LIBRARY_PATH=/mnt/t440/raid0/engines/llama.cpp/20261027-nautiburn-flint-cuda12.8-sm60_61_75_86-cpuall/cuda12-libs CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$U
unset NAUTIBURN_SCRAPE_DIR
NUMA=""; [ "$N" != "-" ] && NUMA="numactl --cpunodebind=$N --membind=$N"
for tag in "$@"; do
  used=$(nvidia-smi --id=$U --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "$used" -lt 64 ] || { echo "card busy ($used MiB): stop"; exit 3; }
  grep -q "\"tag\":\"$tag\",\"card\":\"$CL\"" $X/results/llama-bench.jsonl 2>/dev/null && continue
  echo "== $(date -u +%T) $tag on $CL"
  $NUMA $B/llama-bench -m $MD/LFM2.5-24B-A2B-$tag.gguf -ngl 99 -p 512 -n 128 -r 3 -o jsonl 2>> $X/logs/bench-$tag-$CL.err | sed "s/^{/{\"tag\":\"$tag\",\"card\":\"$CL\",/" >> $X/results/llama-bench.jsonl
done
echo "card after: $(nvidia-smi --id=$U --query-gpu=memory.used --format=csv,noheader)"

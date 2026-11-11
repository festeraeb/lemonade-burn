#!/usr/bin/env bash
# usage (on the GPU node): run_rowsbench.sh <UUID> <numa|-> <label>  -> results/rowsbench-<label>.jsonl
set -u
U=$1; N=$2; L=$3; X=/mnt/t440/fast/x7; [ -d $X ] || X=/mnt/fast/x7
used=$(nvidia-smi --id=$U --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
[ "$used" -lt 64 ] || { echo "card busy: $used MiB"; exit 3; }
export LD_LIBRARY_PATH=$X/lib CUDA_PATH=$X/cuda-path CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$U LB_CUDA_DEVICE=0
export LB_BENCH_MATHS=${MATHS:-f32,f32m,f16,dp4a} LB_BENCH_ROUNDS=${ROUNDS:-3}
NUMA=""; [ "$N" != "-" ] && NUMA="numactl --cpunodebind=$N --membind=$N"
$NUMA $X/target/release/examples/rows_bench | while read -r l; do echo "{\"label\":\"$L\",\"host\":\"$(hostname -s)\",\"gpu\":\"$(nvidia-smi --id=$U --query-gpu=name --format=csv,noheader)\",${l#\{}"; done >> $X/results/rowsbench.jsonl
echo "rc=${PIPESTATUS[0]} card after: $(nvidia-smi --id=$U --query-gpu=memory.used --format=csv,noheader)"

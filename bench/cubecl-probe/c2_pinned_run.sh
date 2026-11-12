#!/usr/bin/env bash
set -u
BASE=/home/cesarops/lb-measure; OUT=$BASE/out/pinned-b5; mkdir -p $OUT
export LD_LIBRARY_PATH=$BASE/lib:${LD_LIBRARY_PATH:-}; export CUDA_PATH=$BASE/cuda-path
BIN=$BASE/cuda-bin/copy_bench_pinned
run() { local label=$1; shift
  echo "== $label $(date -u +%H:%M:%S) load $(cut -d' ' -f1 /proc/loadavg) gpu2util $(nvidia-smi -i 2 --query-gpu=utilization.gpu --format=csv,noheader)"
  env "$@" LB_BENCH_REPS=20 LB_BENCH_WARM=40 LB_PINNED_POOL_MB=768 "$BIN" > $OUT/$label.jsonl 2> $OUT/$label.err &
  local pid=$!; sleep 2
  echo "compute-apps: $(nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv,noheader | tr '\n' '|')"
  wait $pid; echo "exit $?"; head -1 $OUT/$label.err
}
run pool-w40 LB_CUDA_DEVICE=2
run nopool-w40 LB_CUDA_DEVICE=2 LB_NO_PINNED=1
run pool-w40-rep2 LB_CUDA_DEVICE=2
echo "== parity gpu_matches_scalar, cuda GPU2, pool on"
LB_CUDA_DEVICE=2 $BASE/cuda-bin/test-gpu_matches_scalar-pinned --test-threads=1 2>&1 | grep -E "^test |test result"

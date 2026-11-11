#!/usr/bin/env bash
# copy_bench with the CubeCL write-path probe, c2 GPU2 (bus 0D:00.0), two passes.
set -u
BASE=$HOME/lb-measure; OUT=$BASE/out/probe-b2; mkdir -p $OUT
export LD_LIBRARY_PATH=$BASE/lib:${LD_LIBRARY_PATH:-}; export CUDA_PATH=$BASE/cuda-path
BIN=$BASE/cuda-bin/copy_bench_probe
run() { local label=$1; shift
  echo "== $label $(date -u +%H:%M:%S) load $(cut -d' ' -f1 /proc/loadavg) gpu2util $(nvidia-smi -i 2 --query-gpu=utilization.gpu --format=csv,noheader)"
  env "$@" LB_BENCH_REPS=20 "$BIN" > $OUT/$label.jsonl 2> $OUT/$label.err &
  local pid=$!; sleep 2
  echo "compute-apps: $(nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv,noheader | tr '\n' '|')"
  wait $pid; echo "exit $? ; probe lines $(grep -c '^probe{' $OUT/$label.err)"
}
nvidia-smi --query-gpu=index,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,utilization.gpu,memory.used --format=csv,noheader
run cuda-gpu2-probe LB_CUDA_DEVICE=2 LB_CUBECL_PROBE=1
run cuda-gpu2-probe-sync LB_CUDA_DEVICE=2 LB_CUBECL_PROBE=1 LB_CUBECL_PROBE_SYNC=1
run cuda-gpu2-noprobe LB_CUDA_DEVICE=2
echo "after: gpu2util $(nvidia-smi -i 2 --query-gpu=utilization.gpu --format=csv,noheader) load $(cut -d' ' -f1 /proc/loadavg)"

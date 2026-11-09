#!/usr/bin/env bash
# Record what the card and host actually are: identity, link, NUMA node, where the GGUF's page cache
# lives, the measured host-to-device rate. usage: probe.sh <out-dir>
set -u
. "$(dirname "$0")/env.sh"
OUT=$1; mkdir -p "$OUT"; cd "$BASE"
{
  date -u +%FT%TZ; hostname; lscpu | grep -E "Model name|Socket|NUMA node"
  nvidia-smi --id="$CARD_UUID" --query-gpu=index,uuid,name,memory.used,memory.total,pci.bus_id,pcie.link.gen.max,pcie.link.width.max,pcie.link.gen.current,pcie.link.width.current,clocks.max.sm,driver_version --format=csv
  echo "numa node of card: $CARD_NODE"
  sudo -n lspci -vv -s "${CARD_BUS_FULL#0000:}" 2>/dev/null | grep -E "LnkCap:|LnkSta:"
  numastat -m | grep -E "Per-node|MemFree|FilePages|^Token|Node"
} | tee "$OUT/card.txt"
card_check || exit 3
for N in 4 8; do
  LB_STAGE_N=$N LB_STAGE_C=16 LB_BENCH_REPS=30 LB_BENCH_WARM=10 $NUMA ./bin/stage_bench 2>"$OUT/stage_bench-N$N.err" | tee -a "$OUT/stage_bench.jsonl"
done

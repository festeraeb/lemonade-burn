# Environment for engine E2 runs. The host is told by the UUID only: index, bus and NUMA node are
# read from the driver at run time, never assumed.
export BASE=${BASE:-/home/cesarops/lb-e2}
export CARD_UUID=${CARD_UUID:-GPU-2d66f961-cba7-717f-1bd7-a04fff1fc8ab}
export CUDA_VISIBLE_DEVICES=$CARD_UUID LB_CUDA_DEVICE=0
export LD_LIBRARY_PATH=$BASE/lib:${LD_LIBRARY_PATH:-} CUDA_PATH=$BASE/cuda-path
export LB_MODEL=$BASE/models/LFM2.5-24B-A2B-QAD-Q4_0.gguf
unset NAUTIBURN_SCRAPE_DIR
CARD_BUS=$(nvidia-smi --id="$CARD_UUID" --query-gpu=pci.bus_id --format=csv,noheader | tr 'A-Z' 'a-z' | sed 's/^0000//; s/^0000://')
CARD_BUS_FULL=$(nvidia-smi --id="$CARD_UUID" --query-gpu=pci.bus_id --format=csv,noheader | tr 'A-Z' 'a-z' | sed 's/^0000\(....:\)/\1/')
export CARD_NODE=$(cat /sys/bus/pci/devices/$CARD_BUS_FULL/numa_node 2>/dev/null || echo -1)
# bind to the card's node when the host has more than one (node -1 = single node, no binding)
if [ "${CARD_NODE:--1}" -ge 0 ] && [ "$(ls -d /sys/devices/system/node/node* | wc -l)" -gt 1 ]; then
  NUMA="numactl --cpunodebind=$CARD_NODE --membind=$CARD_NODE"
else
  NUMA=""
fi
export NUMA
card_check() {
  local used; used=$(nvidia-smi --id="$CARD_UUID" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")
  echo "card $CARD_UUID bus $CARD_BUS_FULL node $CARD_NODE used ${used} MiB, load $(cut -d" " -f1 /proc/loadavg)"
  [ "${used:-99999}" -lt 1024 ] || { echo "card busy (${used} MiB): stop"; return 1; }
}

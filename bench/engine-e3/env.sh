# Environment for engine E3 (look-ahead) runs. Shares the E2 working copy (model, libs, scrapes) on c4;
# only the binary is ours. The card is named by UUID; index, bus and NUMA node are read at run time.
export BASE=${BASE:-$HOME/lb-e2}
export E3BIN=${E3BIN:-$HOME/lb-e3/bin/lemonade-burn-server-e3}
export CARD_UUID=${CARD_UUID:?set CARD_UUID to the card UUID}
export CUDA_VISIBLE_DEVICES=$CARD_UUID LB_CUDA_DEVICE=0
export LD_LIBRARY_PATH=$BASE/lib:${LD_LIBRARY_PATH:-} CUDA_PATH=$BASE/cuda-path
export LB_MODEL=$BASE/models/LFM2.5-24B-A2B-QAD-Q4_0.gguf
unset NAUTIBURN_SCRAPE_DIR
CARD_BUS_FULL=$(nvidia-smi --id="$CARD_UUID" --query-gpu=pci.bus_id --format=csv,noheader | tr 'A-Z' 'a-z' | sed 's/^0000\(....:\)/\1/')
export CARD_NODE=$(cat /sys/bus/pci/devices/$CARD_BUS_FULL/numa_node 2>/dev/null || echo -1)
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

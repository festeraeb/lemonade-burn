# common env for lb-stage runs on c1 (GTX 1070 pinned by UUID)
export BASE=/mnt/models/lb-stage
export UUID1070=${UUID_GTX1070:?card UUID}
export CUDA_VISIBLE_DEVICES=${CARD_UUID:-$UUID1070} LB_CUDA_DEVICE=0
export LD_LIBRARY_PATH=$BASE/lib:${LD_LIBRARY_PATH:-} CUDA_PATH=$BASE/cuda-path
export LB_MODEL=$BASE/models/LFM2.5-24B-A2B-QAD-Q4_0.gguf
unset NAUTIBURN_SCRAPE_DIR
card_check() {
  local used; used=$(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")
  echo "card $CUDA_VISIBLE_DEVICES used ${used} MiB, load $(cut -d" " -f1 /proc/loadavg)"
  [ "${used:-99999}" -lt 1024 ] || { echo "card busy (${used} MiB): stop"; return 1; }
}

# common env for lb-stage runs on cesarops2 (RTX 2060 SUPER pinned by UUID)
export BASE=/mnt/ssd/lb-stage
export UUID2060=GPU-de3ce5bc-1a27-10e4-be70-97993d287aed
export CUDA_VISIBLE_DEVICES=${CARD_UUID:-$UUID2060} LB_CUDA_DEVICE=0
export LD_LIBRARY_PATH=$BASE/lib:${LD_LIBRARY_PATH:-} CUDA_PATH=$BASE/cuda-path
export LB_MODEL=$BASE/models/LFM2.5-24B-A2B-QAD-Q4_0.gguf
unset NAUTIBURN_SCRAPE_DIR
card_check() { # refuse to run if the pinned card has >1 GB used by someone else
  local used; used=$(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")
  echo "card $CUDA_VISIBLE_DEVICES used ${used} MiB, load $(cut -d" " -f1 /proc/loadavg)"
  [ "${used:-99999}" -lt 1024 ] || { echo "card busy (${used} MiB): stop"; return 1; }
}

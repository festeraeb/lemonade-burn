#!/usr/bin/env bash
# X7b transcode-quality variants on c4-AF after queue3: KLD of IQ4_XS->Q4_0 (plain encoder) and IQ4_XS->Q8_0.
set -u
X=/mnt/t440/fast/x7; B=$X/llama-build/bin; MD=$X/models; AF=${UUID_AF:?card UUID}
until grep -q "^rc=" $X/logs/c4-queue3.log 2>/dev/null; do sleep 30; done
export LD_LIBRARY_PATH=/mnt/t440/raid0/engines/llama.cpp/20261027-nautiburn-flint-cuda12.8-sm60_61_75_86-cpuall/cuda12-libs CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$AF
unset NAUTIBURN_SCRAPE_DIR
for tag in IQ4_XS-imat-tcs IQ4_XS-imat-tc8; do
  until [ -s $MD/LFM2.5-24B-A2B-$tag.gguf ]; do sleep 30; done
  extra=""; [ $tag = IQ4_XS-imat-tc8 ] && extra="--n-cpu-moe 18"
  [ "$(nvidia-smi --id=$AF --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 64 ] || { echo busy; exit 3; }
  echo "== $(date -u +%T) $tag"
  numactl --cpunodebind=1 --membind=1 $B/llama-perplexity -m $MD/LFM2.5-24B-A2B-$tag.gguf -f $X/q/kltext.txt -c 512 -b 512 -ngl 99 $extra -t 16 \
    --kl-divergence-base $X/kl/base-q8.bin --kl-divergence > $X/logs/kl-$tag.log 2>&1; grep -E "Mean    KLD|Same top" $X/logs/kl-$tag.log
done
echo "queue4 done $(date -u +%T)"

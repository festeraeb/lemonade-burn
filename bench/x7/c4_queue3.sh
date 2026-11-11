#!/usr/bin/env bash
# X7a on the X7q text: lemonade-burn ab-math (f32 vs f32m/f16/dp4a) teacher-forced over the held-out KLD text, c4-AF.
set -u
X=/mnt/t440/fast/x7; AF=${UUID_AF:?card UUID}
until grep -q "queue2 done" $X/logs/c4-queue2.log 2>/dev/null; do sleep 30; done
used=$(nvidia-smi --id=$AF --query-gpu=memory.used --format=csv,noheader,nounits); [ "$used" -lt 64 ] || { echo busy; exit 3; }
export LD_LIBRARY_PATH=$X/lib CUDA_PATH=$X/cuda-path CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$AF LB_CUDA_DEVICE=0 LB_EXPERT_CACHE=0 LB_EXPERT_LAYERS=1 LB_THREADS=8
unset NAUTIBURN_SCRAPE_DIR
mapfile -d '' PIECES < <(python3 - "$X/q/kltext.txt" <<'P'
import sys
t=open(sys.argv[1]).read(); n=1700; out=[]
i=0
while i < len(t):
    j=min(len(t), i+n); k=t.rfind(" ", i+n//2, j) if j < len(t) else j
    k = k if k > i else j
    out.append(t[i:k]); i=k
sys.stdout.write("\0".join(out))
P
)
P=(); for p in "${PIECES[@]}"; do P+=(--prompt "$p"); done
echo "$(date -u +%T) ${#PIECES[@]} prompts"
numactl --cpunodebind=1 --membind=1 $X/target/release/lfm2-check --model $HOME/lb-e2/models/LFM2.5-24B-A2B-QAD-Q4_0.gguf --tokenizer $X/models/tokenizer.json --gpu \
  --placement-from-scrapes $X/q/e2/scrapes --gpu-budget-mb 14500 --gpu-dense-mb 900 "${P[@]}" --ab-math f32m,f16,dp4a --greedy-n 32 --ab-rounds 1 \
  --report $X/results/abmath-c4AF-qad-kltext.json > $X/logs/abmath-c4AF-qad-kltext.log 2>&1
echo "rc=$? $(date -u +%T) card after $(nvidia-smi --id=$AF --query-gpu=memory.used --format=csv,noheader)"

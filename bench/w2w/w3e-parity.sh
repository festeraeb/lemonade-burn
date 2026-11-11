#!/usr/bin/env bash
# W3e / X1 parity (run on c4): trunk on $TRUNK with whole MoE layers on in-box cards of the SAME process
# (placement hosts gpu://N), lfm2-check --ab-device: the old path (host-routed, experts on the in-box cards'
# views) against the device path (activation card to card, one read a token). Cards back to 0 MiB after.
set -u
W=${W:-/mnt/t440/fast/w2w}; BASE=${BASE:-$HOME/lb-e2}; BIN=$W/bin
OUT=$W/results/w3; LOGS=$W/logs/w3; RUN=$W/run/w3; mkdir -p "$OUT" "$LOGS" "$RUN"
MODEL=$BASE/models/LFM2.5-24B-A2B-QAD-Q4_0.gguf; TOK=$BASE/models/tokenizer.json
export LD_LIBRARY_PATH=$W/lib-nccl:$BASE/lib:${LD_LIBRARY_PATH:-} CUDA_PATH=$BASE/cuda-path CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_DEBUG=${NCCL_DEBUG:-WARN}
unset NAUTIBURN_SCRAPE_DIR
declare -A U=( [3B]=${UUID_C4_3B:?card UUID} [3C]=${UUID_P100_3C:?card UUID} [86]=${UUID_C4_86:?card UUID} [AF]=${UUID_C4_AF:?card UUID} [D8]=${UUID_C4_D8:?card UUID} )
TRUNK=${TRUNK:-D8}; CARDS=${CARDS:-AF}; TAG=${TAG:-e1}; LAYERS=${LAYERS:-"2-39"}
used_of() { nvidia-smi --id="${U[$1]}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }
for c in 3B 3C 86 AF D8; do [ "$(used_of $c)" -lt 64 ] || { echo "card $c busy: stop"; exit 3; }; done
VIS=${U[$TRUNK]}; HJ=""; LJ=""; i=1
read -ra CA <<< "$CARDS"; read -ra LA <<< "$LAYERS"
for c in "${CA[@]}"; do VIS="$VIS,${U[$c]}"; HJ="$HJ\"g$i\":{\"addr\":\"gpu://$i\"},"; LJ="$LJ\"${LA[$((i-1))]}\":\"g$i\","; i=$((i+1)); done
PL=$RUN/inbox-$TAG.json
echo "{\"version\":1,\"hosts\":{${HJ%,}},\"default\":\"local\",\"layers\":{${LJ%,}}}" > "$PL"; cat "$PL"
P1="A search and rescue team on Lake Erie found the missing kayaker after three hours. The coordinator said the side-scan sonar image showed"
P2="Great Lakes shipwreck survey notes: the schooner lies in 45 m of water, bow to the north, with the mainmast collapsed across the deck. Visibility was 3 m and the"
P3="The capital of France is Paris. The capital of Germany is"
EXTRA=${EXTRA:-}
CUDA_VISIBLE_DEVICES=$VIS LB_CUDA_DEVICE=0 LB_THREADS=8 RAYON_NUM_THREADS=8 numactl --cpunodebind=1 --membind=1 "${CHECKBIN:-$BIN/lfm2-check-cuda-rdma-v3}" --model "$MODEL" --tokenizer "$TOK" \
  --ab-device --gpu --gpu-dense-mb 900 $EXTRA --w2w-placement "$PL" --prompt "$P1" --prompt "$P2" --prompt "$P3" --report "$OUT/inbox-$TAG-ab.json" > "$LOGS/inbox-$TAG-ab.log" 2>&1
echo "ab-device rc=$?"; grep -E "in-box|teacher-forced|alternating:|^old |^device |greedy token|fallback|panick|rror" "$LOGS/inbox-$TAG-ab.log" | cut -c1-300 | head -20
sleep 2; L=""; for c in 3B 3C 86 AF D8; do L="$L $c=$(used_of $c)"; done; echo "cards after:$L"

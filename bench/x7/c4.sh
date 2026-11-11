#!/usr/bin/env bash
# X7q llama.cpp measurements on c4 (P100s AF / D8, NUMA 1). Models are read over NFS from t440.
# usage:
#   c4.sh imatrix                 Q8_0 on AF+D8, calib-v1 text -> models/imatrix-q8.gguf
#   c4.sh heldout <tag> <cards>   llama-server held-out-v1 (cards: AF, D8 or AF,D8) -> results/heldout-llama.jsonl
#   c4.sh klbase                  Q8_0 base logits on AF+D8 over q/kltext.txt -> /mnt/t440/fast/x7/kl/base-q8.bin
#   c4.sh kl <tag> <card>         KLD of quant <tag> vs the Q8_0 base -> logs/kl-<tag>.log
#   c4.sh bench <tag> <card>      llama-bench tg128/pp512 -> results/llama-bench.jsonl
set -u
X=/mnt/t440/fast/x7; B=$X/llama-build/bin; MD=$X/models
export LD_LIBRARY_PATH=/mnt/t440/raid0/engines/llama.cpp/20261027-nautiburn-flint-cuda12.8-sm60_61_75_86-cpuall/cuda12-libs CUDA_DEVICE_ORDER=PCI_BUS_ID
unset NAUTIBURN_SCRAPE_DIR
declare -A U=( [AF]=${UUID_AF:?card UUID} [D8]=${UUID_D8:?card UUID} )
NUMA="numactl --cpunodebind=1 --membind=1"
vis() { local v=""; for c in ${1//,/ }; do v="$v${v:+,}${U[$c]}"; [ "$(nvidia-smi --id=${U[$c]} --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 64 ] || { echo "card $c busy" >&2; exit 3; }; done; echo $v; }
model() { echo $MD/LFM2.5-24B-A2B-$1.gguf; }
# Q8_0 (25.5 GB) on one P100: the experts of the first NCMOE layers stay in host RAM; on two cards, split
q8split() { if [[ "${CARDS:-AF}" == *,* ]]; then echo "-ts 1,1"; else echo "--n-cpu-moe ${NCMOE:-18}"; fi; }
case $1 in
  imatrix)
    export CUDA_VISIBLE_DEVICES=$(vis ${CARDS:-AF})
    $NUMA $B/llama-imatrix -m $(model Q8_0) -f /mnt/t440/fast/moe-budget-eval/calib-v1/calib.txt -ngl 99 $(q8split) -c 512 -b 512 -t 16 \
      -o $MD/imatrix-q8.gguf.tmp > $X/logs/imatrix.log 2>&1 && mv $MD/imatrix-q8.gguf.tmp $MD/imatrix-q8.gguf; tail -3 $X/logs/imatrix.log;;
  heldout)
    tag=$2; export CUDA_VISIBLE_DEVICES=$(vis $3); ts=""; [[ $3 == *,* ]] && ts="-ts 1,1"; [ $tag = Q8_0 ] && ts="$(q8split)"
    for p in $(seq 18500 18560); do ss -ltn | grep -q ":$p " || { PORT=$p; break; }; done
    $NUMA $B/llama-server -m $(model $tag) -ngl 99 $ts --port $PORT --host 127.0.0.1 -np 1 --jinja --reasoning-budget 0 -t 8 -c 4096 > $X/logs/llama-server-$tag.log 2>&1 &
    SRV=$!; trap 'kill $SRV 2>/dev/null; sleep 3; kill -9 $SRV 2>/dev/null' EXIT
    for _ in $(seq 1 300); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; kill -0 $SRV 2>/dev/null || { echo "server died"; tail -5 $X/logs/llama-server-$tag.log; exit 1; }; sleep 2; done
    python3 $X/heldout_run.py http://127.0.0.1:$PORT "llama-$tag" $X/results/heldout-llama.jsonl;;
  klbase)
    export CUDA_VISIBLE_DEVICES=$(vis ${CARDS:-AF}); mkdir -p $X/kl
    $NUMA $B/llama-perplexity -m $(model Q8_0) -f $X/q/kltext.txt -c 512 -b 512 -ngl 99 $(q8split) -t 16 --kl-divergence-base $X/kl/base-q8.bin > $X/logs/kl-Q8_0-base.log 2>&1; tail -4 $X/logs/kl-Q8_0-base.log;;
  kl)
    tag=$2; export CUDA_VISIBLE_DEVICES=$(vis $3)
    $NUMA $B/llama-perplexity -m $(model $tag) -f $X/q/kltext.txt -c 512 -b 512 -ngl 99 -t 8 --kl-divergence-base $X/kl/base-q8.bin --kl-divergence > $X/logs/kl-$tag.log 2>&1; grep -E "Mean    KLD|Same top|Mean PPL\(Q\)" $X/logs/kl-$tag.log;;
  bench)
    tag=$2; export CUDA_VISIBLE_DEVICES=$(vis $3)
    $NUMA $B/llama-bench -m $(model $tag) -ngl 99 -p 512 -n 128 -r 3 -o jsonl 2>> $X/logs/bench-$tag.err | sed "s/^{/{\"tag\":\"$tag\",\"card\":\"c4-$3\",/" >> $X/results/llama-bench.jsonl; tail -2 $X/results/llama-bench.jsonl;;
esac

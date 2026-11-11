#!/usr/bin/env bash
# W3 gate P, on-card parity (run on c4 from the NFS copy): a GPU expert host on $HCARD holding every expert,
# then (1) lfm2-check --ab-device with the trunk on $TRUNK (old path vs device path, both with every expert on
# the host) and (2) w2w-check with a CPU trunk against the same GPU host (greedy + teacher-forced decode; logits
# are expected to differ to rounding, the gate is the sha). Stops the host and checks every card is back to 0 MiB.
set -u
W=${W:-/mnt/t440/fast/w2w}; BASE=${BASE:-$HOME/lb-e2}; BIN=$W/bin
OUT=$W/results/w3; LOGS=$W/logs/w3; RUN=$W/run/w3; mkdir -p "$OUT" "$LOGS" "$RUN"
MODEL=$BASE/models/LFM2.5-24B-A2B-QAD-Q4_0.gguf; TOK=$BASE/models/tokenizer.json
export LD_LIBRARY_PATH=$BASE/lib:${LD_LIBRARY_PATH:-} CUDA_PATH=$BASE/cuda-path CUDA_DEVICE_ORDER=PCI_BUS_ID
unset NAUTIBURN_SCRAPE_DIR
declare -A U=( [3B]=${UUID_C4_3B:?card UUID} [3C]=${UUID_P100_3C:?card UUID} [86]=${UUID_C4_86:?card UUID} [AF]=${UUID_C4_AF:?card UUID} [D8]=${UUID_C4_D8:?card UUID} )
TRUNK=${TRUNK:-D8}; HCARD=${HCARD:-AF}; TAG=${TAG:-p1}
used_of() { nvidia-smi --id="${U[$1]}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }
for c in 3B 3C 86 AF D8; do [ "$(used_of $c)" -lt 64 ] || { echo "card $c busy: stop"; exit 3; }; done
SOCK=/tmp/w2w-parity-$TAG.sock; PL=$RUN/parity-$TAG.json
echo "{\"version\":1,\"hosts\":{\"h1\":{\"addr\":\"unix://$SOCK\",\"wait\":\"spin:200\",\"deadline_ms\":30000,\"deadline_row_ms\":2000}},\"default\":\"local\",\"layers\":{\"*\":\"h1\"}}" > "$PL"
HP=""
cleanup() { [ -n "$HP" ] && { kill -INT $HP 2>/dev/null; sleep 3; kill -9 $HP 2>/dev/null; }; rm -f "$SOCK"; sleep 2; }
trap cleanup EXIT
CUDA_VISIBLE_DEVICES=${U[$HCARD]} LB_CUDA_DEVICE=0 numactl --cpunodebind=1 --membind=1 "${HOSTBIN:-$BIN/lb-expert-host-cuda-rdma}" --model "$MODEL" --placement "$PL" --name h1 --backend gpu \
  --listen "unix://$SOCK" --wait spin:200 --threads 4 --stats-s 0 > "$LOGS/parity-$TAG-host.log" 2>&1 &
HP=$!
for _ in $(seq 1 300); do grep -q "listening on" "$LOGS/parity-$TAG-host.log" && break; kill -0 $HP 2>/dev/null || { echo "host died:"; tail -5 "$LOGS/parity-$TAG-host.log"; exit 1; }; sleep 1; done
grep -q "listening on" "$LOGS/parity-$TAG-host.log" || { echo "host not up"; exit 1; }
echo "host up, $HCARD uses $(used_of $HCARD) MiB"
P1="A search and rescue team on Lake Erie found the missing kayaker after three hours. The coordinator said the side-scan sonar image showed"
P2="Great Lakes shipwreck survey notes: the schooner lies in 45 m of water, bow to the north, with the mainmast collapsed across the deck. Visibility was 3 m and the"
P3="The capital of France is Paris. The capital of Germany is"
if [ "${SKIP_AB:-0}" != 1 ]; then
CUDA_VISIBLE_DEVICES=${U[$TRUNK]} LB_CUDA_DEVICE=0 LB_THREADS=8 RAYON_NUM_THREADS=8 numactl --cpunodebind=1 --membind=1 "$BIN/lfm2-check-cuda-rdma" --model "$MODEL" --tokenizer "$TOK" \
  --ab-device --gpu --gpu-dense-mb 900 --w2w-placement "$PL" --prompt "$P1" --prompt "$P2" --prompt "$P3" --report "$OUT/parity-$TAG-ab.json" > "$LOGS/parity-$TAG-ab.log" 2>&1
echo "ab-device rc=$?"; grep -E "teacher-forced|alternating:|^old |^device |greedy token|w2w:" "$LOGS/parity-$TAG-ab.log" | cut -c1-400
fi
if [ "${SKIP_CPU:-0}" != 1 ]; then
RAYON_NUM_THREADS=16 numactl --cpunodebind=1 --membind=1 "$BIN/w2w-check-cpu" --model "$MODEL" --tokenizer "$TOK" --arm gpuhost="$PL" --phases decode,greedy \
  --out "$OUT/parity-cpu-trunk.jsonl" --label w3-$TAG-cpu-trunk-gpu-host > "$LOGS/parity-$TAG-cpu.log" 2>&1
echo "cpu-trunk rc=$?"; tail -6 "$LOGS/parity-$TAG-cpu.log" | cut -c1-400
fi
cleanup; trap - EXIT; HP=""
L=""; for c in 3B 3C 86 AF D8; do L="$L $c=$(used_of $c)"; done; echo "cards after:$L"

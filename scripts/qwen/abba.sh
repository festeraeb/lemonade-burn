#!/bin/bash
# on c4: ours, llama.cpp, Nauti-Strata, Nauti-Strata, llama.cpp, ours  (A B C C B A), same client and prompts, cards AF + D8 only
R=/mnt/t440/fast/eng123/qwen-bin; M=/mnt/t440/raid0/models/Qwen3.6-35B-A3B-Q5_K_M.gguf
AF=GPU-fd9b03e5-aed9-cbca-a57b-efc90d61addd; D8=GPU-2d66f961-cba7-717f-1bd7-a04fff1fc8ab; X=/mnt/t440/fast/x7
OURS=/mnt/t440/raid0/engines/lemonade-burn/20261115-qwen35-f73da05/bin/lemonade-burn-server
LL=/mnt/t440/raid0/engines/llama.cpp/20261111-nautiburn-scrape-v2tp-cuda12.8-sm60_61_75_86-cpuall
killtree() { local p=$1; for c in $(pgrep -P $p); do killtree $c; done; kill $p 2>/dev/null; }
waitup() { for i in $(seq 1 400); do curl -s -m 2 http://127.0.0.1:$1/health | grep -q ok && return 0; kill -0 $2 2>/dev/null || return 1; ping -c 2 127.0.0.1 >/dev/null; done; return 1; }
arm() {
  name=$1; case $name in
   ours) env LD_LIBRARY_PATH=/mnt/t440/raid0/engines/lemonade-burn/20261113-prefill-split-2337068/lib CUDA_PATH=/mnt/t440/raid0/engines/lemonade-burn/20261113-prefill-split-2337068/cuda-path CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$AF,$D8 $OURS --model $M --host 127.0.0.1 --port 8190 --ctx 4096 --model-id q36 > $R/abba-ours.log 2>&1 & SP=$!; PORT=8190; MODEL=q36;;
   llama) env LD_LIBRARY_PATH=$LL/bin:$LL/cuda12-libs CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$AF,$D8 $LL/bin/llama-server -m $M -ngl 99 -ts 1,1 -fa on -c 8192 -np 1 -t 16 --host 127.0.0.1 --port 8199 --alias q36 > $R/abba-llama.log 2>&1 & SP=$!; PORT=8199; MODEL=q36;;
   strata) env CUDA_DEVICE_ORDER=PCI_BUS_ID STRATA_ENGINE_READY_S=1800 ~/Strata/.venv/bin/python /codebase/repos/Nauti-Strata/serve/server.py --engine strata --config ~/PolyStrata/strata-qwen36-c4-af-d8.json --host 127.0.0.1 --port 8117 > $R/abba-strata.log 2>&1 & SP=$!; PORT=8117; MODEL=qwen3.6-35b-a3b;;
  esac
  t0=$(date +%s)
  if waitup $PORT $SP; then echo "$name up in $(( $(date +%s)-t0 )) s"; echo "$name $(python3 $R/abba_client.py $PORT $MODEL | head -1)"; else echo "$name FAILED to start"; fi
  killtree $SP; for w in $(seq 1 30); do kill -0 $SP 2>/dev/null || break; ping -c 2 127.0.0.1 >/dev/null; done; ping -c 4 127.0.0.1 >/dev/null
  echo "cards after: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed -n '4,5p' | tr '\n' ' ')"
}
for a in ours llama strata strata llama ours; do arm $a; done
echo ABBA done

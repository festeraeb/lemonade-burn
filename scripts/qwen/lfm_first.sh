R=/mnt/t440/fast/eng123/qwen-bin; AF=GPU-fd9b03e5-aed9-cbca-a57b-efc90d61addd
B=/mnt/t440/raid0/engines/lemonade-burn/20261116-qwen35-a7a9dcf
env $EXTRA LD_LIBRARY_PATH=$B/lib CUDA_PATH=$B/cuda-path CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$AF LB_CUDA_DEVICE=0 LB_THREADS=8 $B/bin/lemonade-burn-server --model /mnt/t440/raid0/models/lfm25/LFM2.5-24B-A2B-QAD-Q4_0.gguf --gpu --whole-layers --gpu-budget-mb 14500 --gpu-dense-mb 900 --ctx 16384 --host 127.0.0.1 --port 8196 --model-id lfm > $R/lfm-first.log 2>&1 &
SP=$!
for i in $(seq 1 200); do curl -s -m 2 http://127.0.0.1:8196/v1/models >/dev/null && break; kill -0 $SP 2>/dev/null || break; ping -c 2 127.0.0.1 >/dev/null; done
echo "listening at $(date +%T)"; grep -E "warm|loaded" $R/lfm-first.log | head -2
python3 $R/${LENS:-lens.py} 8196
kill $SP; for w in $(seq 1 30); do kill -0 $SP 2>/dev/null || break; ping -c 2 127.0.0.1 >/dev/null; done; grep -E "warm" $R/lfm-first.log

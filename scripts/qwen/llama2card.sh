#!/bin/bash
# on c4: llama.cpp (fleet build) on AF + D8, Qwen3.6-35B-A3B Q5_K_M, tok/s on the cm prompts
R=/mnt/t440/fast/eng123/qwen-results; E=/mnt/t440/raid0/engines/llama.cpp/20261111-nautiburn-scrape-v2tp-cuda12.8-sm60_61_75_86-cpuall
AF=GPU-fd9b03e5-aed9-cbca-a57b-efc90d61addd; D8=GPU-2d66f961-cba7-717f-1bd7-a04fff1fc8ab
env LD_LIBRARY_PATH=$E/bin:$E/cuda12-libs CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$AF,$D8 $E/bin/llama-server -m /mnt/t440/raid0/models/Qwen3.6-35B-A3B-Q5_K_M.gguf -ngl 99 -ts 1,1 -fa on -c 8192 -np 1 -t 16 --host 127.0.0.1 --port 8199 --alias q36 > $R/llama2card.log 2>&1 &
SP=$!
for i in $(seq 1 240); do curl -s -m 2 http://127.0.0.1:8199/health | grep -q ok && break; kill -0 $SP 2>/dev/null || break; ping -c 2 127.0.0.1 >/dev/null; done
python3 - <<'P'
import json,urllib.request
P=["Write a long, detailed essay about the history of sonar and how side-scan sonar is used to find shipwrecks.","Write a Rust program that reads a CSV file with quoted fields, groups rows by the first column and prints per-group totals, with explanations.","Explain in detail the causes and consequences of the 1913 Great Lakes storm.","Describe step by step how a magnetometer survey is planned, run and interpreted to find a steel wreck.","Explain what a mixture-of-experts language model is, how routing works, and why it is cheaper to run than a dense model."]
out=[]
for p in P:
    b=json.dumps({"model":"q36","messages":[{"role":"user","content":p}],"max_tokens":256,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}).encode()
    r=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8199/v1/chat/completions",b,{"content-type":"application/json"}),timeout=900))
    out.append((round(r["timings"]["predicted_per_second"],1),r["usage"]["completion_tokens"],round(r["timings"]["prompt_per_second"],1)))
print("llama.cpp 2xP100 decode tok/s, n, prompt tok/s:",out)
P
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -5
kill $SP; for w in $(seq 1 30); do kill -0 $SP 2>/dev/null || break; ping -c 2 127.0.0.1 >/dev/null; done; kill -0 $SP 2>/dev/null && kill -9 $SP
grep -E "offloaded|model buffer size|KV buffer" $R/llama2card.log | cut -c1-160 | head -6

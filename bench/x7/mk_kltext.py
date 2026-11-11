#!/usr/bin/env python3
# teacher-forcing text for X7q KLD: each held-out item's prompt + the Q8_0 model's greedy answer (experiment #3's method)
import json
import os
F="/mnt/t440/fast" if os.path.isdir("/mnt/t440/fast/x7") else "/mnt/fast"
items={json.loads(l)["id"]:json.loads(l) for l in open(F+"/moe-budget-eval/heldout-v1/items.jsonl")}
parts=[]
for l in open(F+"/x7/results/heldout-llama.jsonl"):
    r=json.loads(l)
    if r["label"]=="llama-Q8_0": parts.append(items[r["id"]]["prompt"][:1200]+"\n"+r["text"]+"\n\n")
open(F+"/x7/q/kltext.txt","w").write("".join(parts)); print(len(parts),"items",sum(map(len,parts)),"chars")

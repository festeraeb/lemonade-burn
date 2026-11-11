#!/usr/bin/env python3
"""held-out-v1 against an already running OpenAI-compatible server (llama-server or lemonade-burn-server).
Same request settings as experiment #3's run_setting.py (temp 0, seed 1, reasoning off, per-kind max_tokens).
Idempotent: items already in OUT are skipped.  usage: heldout_run.py BASE_URL LABEL OUT.jsonl [ITEMS.jsonl]"""
import json, os, sys, time, urllib.request

base, label, out = sys.argv[1], sys.argv[2], sys.argv[3]
items_path = sys.argv[4] if len(sys.argv) > 4 else "/mnt/t440/fast/moe-budget-eval/heldout-v1/items.jsonl"
if not os.path.exists(items_path):
    items_path = "/mnt/fast/moe-budget-eval/heldout-v1/items.jsonl"
done = set()
if os.path.exists(out):
    for l in open(out):
        try:
            j = json.loads(l)
            if j.get("label") == label:
                done.add(j["id"])
        except Exception:
            pass
items = [json.loads(l) for l in open(items_path)]
maxtok = {"rust": 900, "qa": 40, "prose": 220, "forge": 350}
for it in items:
    if it["id"] in done:
        continue
    body = dict(model="x", messages=[{"role": "system", "content": "Answer directly. Do not write out your reasoning or analysis; give only the final answer in the requested format."},
                                     {"role": "user", "content": it["prompt"]}],
                temperature=0, seed=1, max_tokens=maxtok[it["kind"]], chat_template_kwargs={"enable_thinking": False}, cache_prompt=False)
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1200))
    wall = time.time() - t0
    tm = r.get("timings", {}) or {}
    use = r.get("usage", {}) or {}
    msg = r["choices"][0]["message"]
    rec = dict(id=it["id"], kind=it["kind"], label=label, text=msg.get("content") or "", finish=r["choices"][0].get("finish_reason"),
               n_pred=tm.get("predicted_n") or use.get("completion_tokens"), decode_tps=tm.get("predicted_per_second"),
               prompt_n=tm.get("prompt_n") or use.get("prompt_tokens"), wall_s=round(wall, 2))
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
print("done", label)

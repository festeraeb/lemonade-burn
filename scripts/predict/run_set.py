#!/usr/bin/env python3
"""run_set.py SET OUT.json [N_TOKENS]: send each prompt of a set to the local server (greedy), save texts."""
import json, sys, urllib.request, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts
P = getattr(prompts, sys.argv[1]); N = int(sys.argv[3]) if len(sys.argv) > 3 else 200
out = []
for p in P:
    b = json.dumps({"model": "lfm", "messages": [{"role": "user", "content": p}], "max_tokens": N, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8198/v1/chat/completions", b, {"content-type": "application/json"}), timeout=1800))
    out.append({"prompt": p, "n": r["usage"]["completion_tokens"], "prompt_tokens": r["usage"]["prompt_tokens"], "tps": r["timings"]["predicted_per_second"], "text": r["choices"][0]["message"]["content"]})
    print(out[-1]["n"], round(out[-1]["tps"], 1), flush=True)
json.dump(out, open(sys.argv[2], "w"))

#!/usr/bin/env python3
"""limits_bench.py PORT MODEL MODE [N] [SEED]  -- decode | prompt N (target prompt tokens) | stats.  Prints one JSON line.
Streams /v1/chat/completions at temperature 0; reports ttft (= prompt time), prompt/completion tokens, decode tok/s, the reply head and a hash of the first 48 reply tokens' text."""
import json, sys, time, urllib.request, hashlib
port, model, mode = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 256
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 0
sents = ["The survey vessel crossed the sandbank at slack water while the magnetometer logged a steady baseline of {a} nanotesla.", "Side-scan returns {b} metres to port showed a hard linear target with a short acoustic shadow behind it.", "On day {c} the second pass agreed with the first within {d} metres, which the crew noted in the margin of the log.", "A buried ferrous object produces a dipole whose amplitude falls with the cube of the distance from the sensor, here about {e} metres."]
def req(content, mt, extra=None):
    b = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": mt, "stream": True, "stream_options": {"include_usage": True}, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    if extra: b.update(extra)
    t0 = time.time(); tf = None; tl = None; usage = None; txt = []; rs = []
    with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(b).encode(), {"content-type": "application/json"}), timeout=1500) as r:
        for raw in r:
            l = raw.decode().strip()
            if not l.startswith("data:") or l[5:].strip() == "[DONE]": continue
            o = json.loads(l[5:])
            if o.get("usage"): usage = o["usage"]
            for c in o.get("choices") or []:
                d = c.get("delta") or {}
                if d.get("content"):
                    if tf is None: tf = time.time() - t0
                    tl = time.time(); txt.append(d["content"])
                if d.get("reasoning_content"): rs.append(d["reasoning_content"])
    tot = time.time() - t0
    n = (usage or {}).get("completion_tokens", len(txt))
    dec = (n - 1) / (tl - (t0 + tf)) if tf is not None and tl and tl > t0 + tf and n > 1 else 0
    return {"ttft": round(tf or -1, 2), "total": round(tot, 2), "prompt": (usage or {}).get("prompt_tokens"), "n": n, "dec_tps": round(dec, 1), "text": "".join(txt), "reasoning": "".join(rs)}
if mode == "decode":
    code = "Write a Rust function that parses a CSV line into fields, handling quoted fields with embedded commas and escaped quotes. Include unit tests."
    for i in range(3):
        r = req(code, N); r["hash"] = hashlib.md5(r["text"][:120].encode()).hexdigest()[:8]; r.pop("text"); r.pop("reasoning"); print(json.dumps(r), flush=True)
elif mode == "prompt":
    k = max(1, round(N / 27.4))
    text = " ".join(sents[i % 4].format(a=48000 + SEED + i * 7, b=20 + (i * 13 + SEED) % 90, c=i + 1 + SEED % 7, d=3 + (i * 5) % 11, e=2 + (i * 3) % 9) for i in range(k))
    r = req("Read this survey log, then answer in one sentence: which instruments are mentioned?\n\n" + text, 48)
    r["want"] = N; r["head"] = r.pop("text")[:100].replace("\n", " "); r["hash"] = hashlib.md5(r["head"].encode()).hexdigest()[:8]; r.pop("reasoning")
    print(json.dumps(r), flush=True)

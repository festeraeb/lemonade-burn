#!/usr/bin/env python3
"""think_check.py PORT [MODEL]: reasoning split check. Sends the same Rust coding prompt (a) default, (b) enable_thinking true, (c) reasoning_budget 0 / enable_thinking false,
streaming and not; prints JSON lines with content head, reasoning head, whether '</think>' or 'thinking process' leaked into content."""
import json, sys, urllib.request, time
port = sys.argv[1]; model = sys.argv[2] if len(sys.argv) > 2 else "test"
P = "Write a Rust function `fn dedup_sorted(v: &mut Vec<i32>)` that removes consecutive duplicates in place without using retain or dedup, with two unit tests. Reply with the code only."
def call(extra, stream, mt):
    b = {"model": model, "messages": [{"role": "user", "content": P}], "max_tokens": mt, "temperature": 0, "stream": stream}; b.update(extra)
    t0 = time.time()
    with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(b).encode(), {"content-type": "application/json"}), timeout=900) as r:
        if not stream:
            j = json.load(r); m = j["choices"][0]["message"]; return m.get("content", ""), m.get("reasoning_content", ""), j["usage"]["completion_tokens"], time.time() - t0
        c, rs, n = [], [], 0
        for raw in r:
            l = raw.decode().strip()
            if not l.startswith("data:") or l[5:].strip() == "[DONE]": continue
            o = json.loads(l[5:])
            for ch in o.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content"): c.append(d["content"])
                if d.get("reasoning_content"): rs.append(d["reasoning_content"])
            if o.get("usage"): n = o["usage"]["completion_tokens"]
        return "".join(c), "".join(rs), n, time.time() - t0
for name, extra, stream, mt in [("default(off) stream", {}, True, 700), ("default(off) nonstream", {}, False, 700), ("kwargs enable_thinking=false", {"chat_template_kwargs": {"enable_thinking": False}}, True, 700), ("reasoning_budget=0", {"reasoning_budget": 0}, False, 700),
                                ("enable_thinking=true stream", {"chat_template_kwargs": {"enable_thinking": True}}, True, 2500), ("enable_thinking=true nonstream", {"chat_template_kwargs": {"enable_thinking": True}}, False, 2500)]:
    c, r, n, dt = call(extra, stream, mt)
    print(json.dumps({"case": name, "completion_tokens": n, "secs": round(dt, 1), "content_len": len(c), "reasoning_len": len(r), "leak_close_tag": "</think>" in c, "leak_open_tag": "<think>" in c, "content_head": c[:160], "reasoning_head": r[:100], "content_full": c}), flush=True)

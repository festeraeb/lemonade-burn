"""Client side of one e2e.sh session: warm up, run each prompt once (greedy), append JSONL rows.

Measurement harness only (HTTP client + bookkeeping); the engine under test is the server."""
import hashlib, json, os, time, urllib.request

E = os.environ
port, label, arm, out = E["PORT"], E["LABEL"], E["ARM"], E["OUT"]
base = f"http://127.0.0.1:{port}"
is_llama = arm == "c"
prompts = [json.loads(l) for l in open("/mnt/ssd/lb-stage/prompts.jsonl")]

def post(path, body, timeout=1800):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def get(path):
    with urllib.request.urlopen(base + path, timeout=60) as r:
        return json.load(r)

def complete(p):
    t0 = time.time()
    if is_llama:
        d = post("/completion", {"prompt": p["text"], "n_predict": p["max_tokens"], "temperature": 0, "top_k": 1, "cache_prompt": False})
        t = d.get("timings") or {}
        txt, toks = d.get("content") or "", d.get("tokens_predicted")
        rate, pre = t.get("predicted_per_second"), (t.get("prompt_ms") or 0) / 1e3
    else:
        d = post("/v1/completions", {"prompt": p["text"], "max_tokens": p["max_tokens"], "temperature": 0})
        t = d.get("timings") or {}
        txt = (d.get("choices") or [{}])[0].get("text") or ""
        toks = (d.get("usage") or {}).get("completion_tokens")
        rate, pre = t.get("predicted_per_second"), t.get("prompt_s")
    return {"tokens": toks, "decode_tok_s": rate, "prefill_s": pre, "text_sha": hashlib.sha256(txt.encode()).hexdigest()[:16], "t0": t0, "t1": time.time()}

common = {"label": label, "arm": arm, "round": int(E["ROUND"]), "expert_mib": int(E["MIB"]), "ring_mib": int(E["RING"]) if arm.startswith("b") else 0,
          "slots": int(E["SLOTS"]) if arm.startswith("b") else 0, "min_margin": float(E["MARGIN"]) if arm.startswith("b") else None, "vram_used_mib": int(E["VRAM"] or 0)}
w = complete(prompts[0])
before = None if is_llama else get("/v1/placement")
rows = []
for p in prompts:
    r = complete(p)
    r.update(common)
    r["prompt"] = p["id"]
    r["load1"] = float(open("/proc/loadavg").read().split()[0])
    if not is_llama:
        r["placement"] = get("/v1/placement")  # cumulative; diff consecutive rows
    rows.append(r)
    with open(f"{out}/runs.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")
    print(json.dumps({k: r[k] for k in ("label", "prompt", "tokens", "decode_tok_s", "text_sha")}))
sess = dict(common)
sess.update({"warmup": w, "placement_before": before, "placement_after": rows[-1].get("placement"), "t_end": time.time(), "session_done": True})
with open(f"{out}/sessions.jsonl", "a") as f:
    f.write(json.dumps(sess) + "\n")

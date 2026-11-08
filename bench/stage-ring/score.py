"""Score MoE experiment #2 (STAGE ring) runs: runs.jsonl + sessions.jsonl + logs/imc-*.jsonl.

Per session (one engine start, every prompt once): decode tok/s = total tokens / total decode time;
counters = /v1/placement "stage" after the last prompt minus after the warmup. Per arm: median and
min-max over sessions. Greedy sha per prompt must agree across the lemonade-burn arms.
usage: python3 score.py <run dir> [tag]"""
import glob, json, os, statistics, sys

d = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else None
rows = [json.loads(l) for l in open(f"{d}/runs.jsonl")]
sess = [json.loads(l) for l in open(f"{d}/sessions.jsonl")]
sess = [s for s in sess if s.get("session_done") and (tag is None or s["label"].startswith(tag + "-"))]

def imc(label, t0, t1):
    p = f"{d}/logs/imc-{label}.jsonl"
    if not os.path.exists(p):
        return None
    xs = [json.loads(l) for l in open(p) if l.strip()]
    xs = [x for x in xs if t0 <= x["t"] <= t1 + 0.5]
    if not xs:
        return None
    return statistics.mean(x["read_mb_s"] for x in xs), statistics.mean(x["write_mb_s"] for x in xs)

def diff(a, b):
    """stage counters: b - a (dicts of numbers, nested one level)."""
    out = {}
    for k, v in b.items():
        if isinstance(v, dict):
            out[k] = {kk: vv - (a.get(k, {}).get(kk, 0) or 0) for kk, vv in v.items() if isinstance(vv, (int, float))}
        elif isinstance(v, (int, float)):
            out[k] = v - (a.get(k, 0) or 0)
    return out

per = []
for s in sess:
    rs = [r for r in rows if r["label"] == s["label"]]
    toks = sum(r["tokens"] or 0 for r in rs)
    secs = sum((r["tokens"] or 0) / r["decode_tok_s"] for r in rs if r["decode_tok_s"])
    o = {"label": s["label"], "arm": s["arm"], "round": s["round"], "tok_s": toks / secs if secs else None, "tokens": toks,
         "vram": s["vram_used_mib"], "shas": {r["prompt"]: r["text_sha"] for r in rs}, "load": max(r["load1"] for r in rs)}
    bw = imc(s["label"], min(r["t0"] for r in rs), max(r["t1"] for r in rs))
    if bw:
        o["dram_read_mb_s"], o["dram_write_mb_s"] = bw
    lg = f"{d}/logs/server-{s['label']}.log"
    if s["arm"] == "c" and os.path.exists(lg):
        import re
        for line in open(lg, errors="replace"):
            m = re.search(r"ubatch <= 8: hits = (\d+), misses = (\d+), hit rate = [0-9.]+%, uploaded = ([0-9.]+) MiB", line)
            if m:
                h, mi, up = int(m.group(1)), int(m.group(2)), float(m.group(3))
                # whole session incl. warmup and small prompt ubatches; 38 MoE layers x top-4 per token
                ntok = toks + (s.get("warmup") or {}).get("tokens", 0)
                o.update({"hit": h / (h + mi), "stalls_tok": mi / ntok, "llama_uploaded_mb_tok": up * 1.048576 / ntok})
    if s.get("placement_before") and s.get("placement_after"):
        st = diff(s["placement_before"]["stage"], s["placement_after"]["stage"])
        n = st["tokens"] or 1
        g = lambda grp: st[grp]
        o.update({
            "hit": g("all")["routings"] and (g("all")["resident"] + g("all")["staged"]) / g("all")["routings"],
            "staged_rate": g("all")["routings"] and g("all")["staged"] / g("all")["routings"],
            "stalls_tok": g("all")["demand"] / n,
            "prefetch_mb_tok": st["prefetch_bytes"] / n / 1e6,
            "wasted_mb_tok": st["wasted_bytes"] / n / 1e6,
        })
        for grp in ("attn_targets", "conv_targets"):
            x = g(grp)
            o[grp] = {"hit": x["routings"] and (x["resident"] + x["staged"]) / x["routings"], "staged_rate": x["routings"] and x["staged"] / x["routings"],
                      "prefetched": x["prefetched"], "useful": x["prefetch_useful"], "wasted": x["prefetch_wasted"],
                      "precision": (x["prefetch_useful"] / (x["prefetch_useful"] + x["prefetch_wasted"])) if (x["prefetch_useful"] + x["prefetch_wasted"]) else None}
    per.append(o)

for o in per:
    print(json.dumps(o))
print()
arms = sorted({o["arm"] for o in per})
summ = {}
for a in arms:
    xs = [o for o in per if o["arm"] == a]
    ts = sorted(o["tok_s"] for o in xs if o["tok_s"])
    m = lambda k: statistics.median(o[k] for o in xs if o.get(k) is not None) if any(o.get(k) is not None for o in xs) else None
    summ[a] = {"n": len(ts), "tok_s_median": statistics.median(ts), "tok_s_min": ts[0], "tok_s_max": ts[-1], "hit": m("hit"), "staged_rate": m("staged_rate"),
               "stalls_tok": m("stalls_tok"), "prefetch_mb_tok": m("prefetch_mb_tok"), "wasted_mb_tok": m("wasted_mb_tok"),
               "dram_read_mb_s": m("dram_read_mb_s"), "vram_mib": m("vram")}
    for grp in ("attn_targets", "conv_targets"):
        g = [o[grp] for o in xs if grp in o]
        if g:
            med = lambda k: statistics.median(x[k] for x in g if x.get(k) is not None) if any(x.get(k) is not None for x in g) else None
            summ[a][grp] = {"hit": med("hit"), "staged_rate": med("staged_rate"), "precision": med("precision")}
    print(a, json.dumps(summ[a]))
base = summ.get("a")
for a in arms:
    if base and a != "a":
        print(f"{a} / a = {summ[a]['tok_s_median'] / base['tok_s_median']:.3f}")
# sha: every lemonade-burn session must give the same text per prompt
lb = [o for o in per if o["arm"] != "c"]
prompts = sorted({p for o in lb for p in o["shas"]})
for p in prompts:
    vals = {o["shas"].get(p) for o in lb}
    print(f"sha {p}: {'SAME' if len(vals) == 1 else 'DIFFER'} {sorted(vals)} over {len(lb)} lemonade-burn sessions")
llama = {p: {o['shas'].get(p) for o in per if o['arm'] == 'c'} for p in prompts}
print("llama.cpp shas:", {p: sorted(v) for p, v in llama.items() if v})

"""Engine E2 table: per arm tok/s (median, min-max, per round), token time split, syncs, miss layers.
usage: python3 table.py <run dir> [tag prefix]"""
import json, re, statistics as st, sys
d = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else "m5120"
rows = [json.loads(l) for l in open(f"{d}/runs.jsonl")]
sess = [json.loads(l) for l in open(f"{d}/sessions.jsonl")]
sess = [s for s in sess if s.get("session_done") and s["label"].startswith(tag + "-")]
per = {}
for s in sess:
    rs = [r for r in rows if r["label"] == s["label"]]
    toks = sum(r["tokens"] or 0 for r in rs)
    secs = sum((r["tokens"] or 0) / r["decode_tok_s"] for r in rs if r["decode_tok_s"])
    o = {"round": s["round"], "tok_s": toks / secs, "vram": s["vram_used_mib"], "vmax": max(r.get("vram_after_mib") or 0 for r in rs), "load": max(r["load1"] for r in rs), "numa": s.get("numa_map", "")}
    a, b = s.get("placement_before"), s.get("placement_after")
    if a and b:
        ns, nb = a["stage"], b["stage"]
        n = nb["tokens"] - ns["tokens"]
        rate = lambda k: (nb.get(k, 0) * nb["tokens"] - ns.get(k, 0) * ns["tokens"]) / n
        o["token_ms"] = (b["device_decode_ms"] * b["device_decode_tokens"] - a["device_decode_ms"] * a["device_decode_tokens"]) / n
        o["read_ms"] = rate("h2_read_ms_per_token") + rate("moe_read_ms_per_token")
        o["syncs"] = rate("host_syncs_per_token")
        o["miss_layers"] = rate("miss_layers_per_token")
        x = {k: nb["all"][k] - ns["all"][k] for k in ("routings", "resident", "demand")}
        o["hit"] = x["resident"] / x["routings"]; o["misses"] = x["demand"] / n
        g_total = nb.get("guess_recall")
        o["guess"] = nb.get("guess_recall", 0.0)
        o["fallbacks"] = b["device_decode_fallbacks"]
    else:
        lg = f"{d}/logs/server-{s['label']}.log"
        for line in open(lg, errors="replace"):
            m = re.search(r"ubatch <= 8: hits = (\d+), misses = (\d+)", line)
            if m:
                h, mi = int(m.group(1)), int(m.group(2))
                o["hit"] = h / (h + mi); o["misses"] = mi / (toks + (s.get("warmup") or {}).get("tokens", 0))
    per.setdefault(s["arm"], []).append(o)
med = lambda xs: st.median(xs) if xs else None
base = None
for arm in ("e1", "e2", "e2g", "c", "allres"):
    xs = sorted(per.get(arm, []), key=lambda o: o["round"])
    if not xs:
        continue
    ts = [o["tok_s"] for o in xs]
    f = lambda k: round(med([o[k] for o in xs if o.get(k) is not None]), 2) if any(o.get(k) is not None for o in xs) else None
    print(f"{arm}: tok/s {st.median(ts):.2f} [{min(ts):.2f}-{max(ts):.2f}] spread {(max(ts)-min(ts))/st.median(ts)*100:.1f}%  per round {[round(t,2) for t in ts]}")
    print(f"    token ms {f('token_ms')} read wait {f('read_ms')} syncs/tok {f('syncs')} miss layers/tok {f('miss_layers')} misses/tok {f('misses')} hit {f('hit')} guess recall {f('guess')} fallbacks {f('fallbacks')} vram {f('vram')} vmax {max(o['vmax'] for o in xs)} max load {max(o['load'] for o in xs)} numa {xs[0]['numa']}")
    if arm == "e1":
        base = st.median(ts)
    elif base:
        print(f"    x e1 = {st.median(ts)/base:.3f}")
# paired by round, e2 / e1 and c / e1
for arm in ("e2", "e2g", "c"):
    if arm in per and "e1" in per:
        e1 = {o["round"]: o["tok_s"] for o in per["e1"]}
        print(f"{arm}/e1 per round:", [round(o["tok_s"] / e1[o["round"]], 3) for o in sorted(per[arm], key=lambda o: o["round"]) if o["round"] in e1])

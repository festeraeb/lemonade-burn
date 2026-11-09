"""Engine E3 table: per arm tok/s (median, min-max), paired ratio to the e2 arm per round, hit rate,
misses/token and how many of them were on the critical path, wasted bytes, guess and issue cost.
usage: python3 table.py <run dir> [tag prefix]"""
import json, statistics as st, sys
d = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else "m5120"
rows = [json.loads(l) for l in open(f"{d}/runs.jsonl")]
sess = [json.loads(l) for l in open(f"{d}/sessions.jsonl")]
sess = [s for s in sess if s.get("session_done") and s["label"].startswith(tag + "-")]
per = {}
shas = {}
for s in sess:
    rs = [r for r in rows if r["label"] == s["label"]]
    if not rs:
        continue
    toks = sum(r["tokens"] or 0 for r in rs)
    secs = sum((r["tokens"] or 0) / r["decode_tok_s"] for r in rs if r["decode_tok_s"])
    o = {"round": s["round"], "tok_s": toks / secs, "vram": s["vram_used_mib"], "vmax": max(r.get("vram_after_mib") or 0 for r in rs), "load": max(r["load1"] for r in rs)}
    for r in rs:
        shas.setdefault(r["prompt"], {}).setdefault(r["text_sha"], set()).add(s["arm"])
    a, b = s.get("placement_before"), s.get("placement_after")
    if a and b:
        ns, nb = a["stage"], b["stage"]
        n = nb["tokens"] - ns["tokens"]
        cum = lambda k: (nb.get(k, 0) * nb["tokens"] - ns.get(k, 0) * ns["tokens"]) / n
        o["token_ms"] = (b["device_decode_ms"] * b["device_decode_tokens"] - a["device_decode_ms"] * a["device_decode_tokens"]) / n
        o["read_ms"] = cum("h2_read_ms_per_token") + cum("moe_read_ms_per_token")
        o["syncs"] = cum("host_syncs_per_token"); o["miss_layers"] = cum("miss_layers_per_token")
        x = {k: nb["all"][k] - ns["all"][k] for k in ("routings", "resident", "staged", "demand", "prefetched", "prefetch_useful", "prefetch_wasted")}
        o["hit"] = x["resident"] / x["routings"]; o["covered"] = x["staged"] / n; o["on_bus"] = x["demand"] / n
        o["misses"] = o["covered"] + o["on_bus"]
        o["prefetched"] = x["prefetched"] / n
        o["precision"] = x["prefetch_useful"] / (x["prefetch_useful"] + x["prefetch_wasted"]) if (x["prefetch_useful"] + x["prefetch_wasted"]) else None
        o["wasted_mb"] = (nb["wasted_bytes"] - ns["wasted_bytes"]) / n / 1e6
        o["guess_ms"] = cum("la_guess_ms_per_token"); o["issue_ms"] = cum("issue_ms_per_token")
        o["spared"] = cum("la_spared_per_token")
        o["fallbacks"] = b.get("device_decode_fallbacks", 0)
    per.setdefault(s["arm"], []).append(o)
med = lambda xs: st.median(xs) if xs else None
hdr = "arm | n | tok/s med [min-max] | x e2 (paired) | token ms | read wait | hit | misses/t | on bus/t | covered/t | prefetched/t | precision | wasted MB/t | guess ms | issue ms | spared/t | syncs/t | miss layers/t | VRAM MiB"
print(hdr); print("|".join(["---"] * hdr.count("|")))
e2 = {o["round"]: o["tok_s"] for o in per.get("e2", [])}
e2m = med(list(e2.values()))
order = ["e2", "off"] + sorted(k for k in per if k not in ("e2", "off"))
for arm in order:
    xs = sorted(per.get(arm, []), key=lambda o: o["round"])
    if not xs:
        continue
    ts = [o["tok_s"] for o in xs]
    f = lambda k, nd=2: (lambda v: "-" if v is None else f"{v:.{nd}f}")(med([o[k] for o in xs if o.get(k) is not None]))
    pair = [o["tok_s"] / e2[o["round"]] for o in xs if o["round"] in e2]
    pr = f"{med(pair):.3f} [{min(pair):.3f}-{max(pair):.3f}]" if pair else "-"
    print(f"{arm} | {len(ts)} | {med(ts):.2f} [{min(ts):.2f}-{max(ts):.2f}] | {pr} | {f('token_ms')} | {f('read_ms')} | {f('hit',3)} | {f('misses')} | {f('on_bus')} | {f('covered')} | {f('prefetched')} | {f('precision',2)} | {f('wasted_mb',1)} | {f('guess_ms')} | {f('issue_ms')} | {f('spared')} | {f('syncs')} | {f('miss_layers')} | {f('vram',0)}/{max(o['vmax'] for o in xs)}")
print("\nsha per prompt (lemonade-burn sessions and llama.cpp apart):")
for p, v in sorted(shas.items()):
    lb = {h: sorted(a for a in arms if a != "c") for h, arms in v.items() if any(a != "c" for a in arms)}
    print(f"  {p}: lemonade-burn {len(lb)} sha(s) {lb} | llama.cpp shas {[h for h, arms in v.items() if 'c' in arms]}")

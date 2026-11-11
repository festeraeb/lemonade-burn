#!/usr/bin/env python3
"""X7q table: per quant tag -> size, bpw, bytes/token, KLD vs Q8_0, same-top, PPL, held-out gate, llama-bench per card.
Writes results/x7q.jsonl (one row per tag, overwritten) and prints a markdown table."""
import json, os, re, subprocess, sys
X = "/mnt/fast/x7"; MD = X + "/models"
TAGS = ["Q8_0", "QAD-Q4_0", "Q4_0", "Q4_0-imat", "Q4_K_M-imat", "IQ4_XS-imat", "IQ4_XS-imat-tc", "IQ4_XS-imat-tcs", "IQ4_XS-imat-tc8", "IQ3_M-imat", "IQ2_M-imat", "IQ1_S-imat"]
BW = {"c4-AF": 605e9, "c2-T10": 320e9, "c2-2060": 336e9, "c2-2060s": 448e9, "c3-p106": 192e9}
N_PARAMS = 23971587712
sys.path.insert(0, "/mnt/fast/moe-budget-eval/bin")

def kl(tag):
    p = f"{X}/logs/kl-{tag}.log"
    if not os.path.exists(p): return {}
    s = open(p, errors="replace").read(); o = {}
    for k, rx in [("kld_mean", r"Mean    KLD:\s+([-\d.]+)"), ("kld_99", r"99\.0%   KLD:\s+([-\d.]+)"), ("kld_max", r"Maximum KLD:\s+([-\d.]+)"),
                  ("same_top_pct", r"Same top p:\s+([\d.]+)"), ("ppl_q", r"Mean PPL\(Q\)\s+:\s+([\d.]+)"), ("ppl_base", r"Mean PPL\(base\)\s+:\s+([\d.]+)")]:
        m = re.search(rx, s)
        if m: o[k] = float(m.group(1))
    return o

def heldout(label, path):
    if not os.path.exists(path): return None
    recs = [json.loads(l) for l in open(path)]; recs = [r for r in recs if r.get("label") == label]
    if len(recs) < 60: return None
    tmp = f"/tmp/x7q-{label}.jsonl"; open(tmp, "w").write("".join(json.dumps(r) + "\n" for r in recs))
    out = subprocess.run(["python3", "/mnt/fast/moe-budget-eval/bin/score.py", tmp], capture_output=True, text=True).stdout
    try: j = json.loads(out.strip().splitlines()[-1])
    except Exception: return None
    return {"gate": j["gate_correct"], "gate_n": j["gate_n"], "by_kind": j["by_kind"]}

def bench():
    b = {}
    p = f"{X}/results/llama-bench.jsonl"
    if not os.path.exists(p): return b
    for l in open(p):
        try: j = json.loads(l)
        except Exception: continue
        key = (j["tag"], j["card"]); d = b.setdefault(key, {})
        if j["n_gen"] > 0: d["tg128"] = round(j["avg_ts"], 2)
        else: d["pp512"] = round(j["avg_ts"], 1)
    return b

def bpt(tag):
    p = f"{MD}/LFM2.5-24B-A2B-{tag}.gguf"
    if not os.path.exists(p): return None
    out = subprocess.run(["python3", f"{X}/q/bytes_per_token.py", p], capture_output=True, text=True).stdout
    return json.loads(out) if out.strip() else None

B = bench(); rows = []
for tag in TAGS:
    p = f"{MD}/LFM2.5-24B-A2B-{tag}.gguf"
    if not os.path.exists(p): continue
    size = os.path.getsize(os.path.realpath(p)); t = bpt(tag) or {}
    r = dict(tag=tag, file_bytes=size, bpw=round(size * 8 / N_PARAMS, 3), bytes_per_token=t.get("bytes_per_token"), types={k: v for k, v in (t.get("file_bytes_by_type") or {}).items()})
    r.update(kl(tag)); r["heldout_llama"] = heldout(f"llama-{tag}", f"{X}/results/heldout-llama.jsonl")
    r["bench"] = {}
    for (tg, card), d in B.items():
        if tg != tag: continue
        if "tg128" in d and r["bytes_per_token"]: d["bw_frac"] = round(d["tg128"] * r["bytes_per_token"] / BW.get(card, 1), 3)
        r["bench"][card] = d
    rows.append(r)
open(f"{X}/results/x7q.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
cards = sorted({c for r in rows for c in r["bench"]})
hdr = "| quant | GB | bpw | GB/token | KLD vs Q8_0 (mean / 99 %) | same top | PPL | held-out gate | " + " | ".join(f"{c} tg128 (BW %)" for c in cards) + " |"
print(hdr); print("|" + "---|" * (8 + len(cards)))
for r in rows:
    k = f"{r['kld_mean']:.4f} / {r['kld_99']:.3f}" if "kld_mean" in r else ("base" if r["tag"] == "Q8_0" else "-")
    st = f"{r['same_top_pct']:.1f} %" if "same_top_pct" in r else "-"
    ppl = f"{r.get('ppl_q', r.get('ppl_base', 0)):.3f}" if ("ppl_q" in r or "ppl_base" in r) else "-"
    h = r["heldout_llama"]; hs = f"{h['gate']}/{h['gate_n']}" if h else "-"
    cs = []
    for c in cards:
        d = r["bench"].get(c)
        cs.append(f"{d['tg128']:.1f} ({100 * d['bw_frac']:.0f} %)" if d and "tg128" in d and "bw_frac" in d else "-")
    gpt = f"{r['bytes_per_token'] / 1e9:.2f}" if r["bytes_per_token"] else "-"
    print(f"| {r['tag']} | {r['file_bytes'] / 1e9:.2f} | {r['bpw']} | {gpt} | {k} | {st} | {ppl} | {hs} | " + " | ".join(cs) + " |")

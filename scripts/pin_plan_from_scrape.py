#!/usr/bin/env python3
"""Static expert placement plan from the NautiBurn scrape DB (read-only).
usage: pin_plan_from_scrape.py MODEL_ID CLASS_ID OUT.json
Per layer and phase: experts ordered by heat (selection count), plus the share of all selections the first N cover.
The plan is loaded once at start and never rewritten during a run."""
import json, sqlite3, sys
DB = "file:/mnt/ssd8/nautiburn/histogram/sidechannel.sqlite?mode=ro"
model, cls, out = sys.argv[1:4]
c = sqlite3.connect(DB, uri=True)
n_exp = c.execute("select n_experts, n_layers, top_k from models where model_id=?", (model,)).fetchone()
plan = {"format": "lb-pin-plan-v1", "model_id": model, "class_id": cls, "n_experts": n_exp[0], "n_layers": n_exp[1], "top_k": n_exp[2], "phases": {}}
for phase in ("decode", "prefill"):
    rows = c.execute("select layer, expert_id, count from histogram_counts where model_id=? and class_id=? and phase=?", (model, cls, phase)).fetchall()
    per = {}
    for l, e, k in rows: per.setdefault(l, {})[e] = k
    layers = []
    for l in sorted(per):
        h = per[l]; order = sorted(range(n_exp[0]), key=lambda e: -h.get(e, 0)); tot = sum(h.values()) or 1
        cov = {str(n): round(sum(h.get(e, 0) for e in order[:n]) / tot, 4) for n in (16, 32, 64, 96, 128, 192)}
        layers.append({"layer": l, "order": order, "selections": tot, "coverage_top_n": cov})
    plan["phases"][phase] = layers
json.dump(plan, open(out, "w"))
for ph, ls in plan["phases"].items():
    for n in ("32", "64", "128"):
        v = [x["coverage_top_n"][n] for x in ls]
        print(ph, "top", n, "of", n_exp[0], "mean coverage %.3f min %.3f" % (sum(v) / len(v), min(v)), "layers", len(ls))

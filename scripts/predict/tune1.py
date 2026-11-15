import os
import sys, numpy as np, itertools, json
from load import Set
from sim import *
R = os.environ.get("PRED_RES", "predict-results")
n = sys.argv[1]
s = Set(n); tr = load_trace(f"{R}/{n}.jsonl")
P = dict(np.load(f"{R}/pred-{n}.npz")); P = {int(k[1:]): v.astype(np.float32) for k, v in P.items()}
def show(name, r): print(f"{name:52s} crit {r['crit']:5.1f} (late {r['late']:4.1f}) cov {r['cov']:5.1f} pf {r['pf']:5.1f} wasted {r['wasted']:5.1f} copies {r['copies']:5.1f} stall {r['stall']:5.2f} ms/tok {r['ms_tok']:5.1f}", flush=True)
base = run(tr, s); show("baseline SLRU", base)
grid = []
for leads in [(1,), (1, 2), (2, 4), (1, 2, 3, 4), (2, 3, 4, 6), (1, 2, 4, 6, 8), (3, 4, 6, 8)]:
    for th in (0.15, 0.3, 0.5):
        for m, stage, q in [(2, 16, 8), (3, 32, 12)]:
            r = run(tr, s, leads=leads, scorer=lambda k: P[k], m=m, theta={k: th for k in leads}, stage=stage, qmax=q)
            show(f"gbm leads={leads} th={th} m={m} S={stage} Q={q}", r)

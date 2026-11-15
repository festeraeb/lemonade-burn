import os
import numpy as np, sys
from load import Set
from sim import *
R = os.environ.get("PRED_RES", "predict-results")
n = "VAL"; s = Set(n); tr = load_trace(f"{R}/{n}.jsonl")
P = {int(k[1:]): v.astype(np.float32) for k, v in np.load(f"{R}/predlrmin-{n}.npz").items()}
b = run(tr, s)
print("baseline ms", round(b["ms_tok"], 2), "demand", round(b["demand"], 1))
for leads, th, m, st in [((2,), .4, 2, 8), ((2,), .4, 1, 8), ((1, 2), .4, 2, 8), ((3,), .4, 2, 8), ((2,), .4, 2, 16)]:
    for d in (0.0, 0.2, 0.36, 0.6):
        r = run(tr, s, leads=leads, scorer=lambda k: P[k], m=m, theta={k: th for k in leads}, stage=st, qmax=st, pred_delay=d)
        print(f"leads {leads} th {th} m {m} S {st} delay {d}: demand {r['demand']:5.1f} cov {r['cov']:5.1f} late {r['late']:5.1f} pf {r['pf']:5.1f} wasted {r['wasted']:4.1f} ms {r['ms_tok']:5.2f} x{b['ms_tok']/r['ms_tok']:.3f}")

from load import R
import sys, numpy as np
from load import Set
from feats import *
from sim import *
n = sys.argv[1]
s = Set(n); tr = load_trace(f"{R}/{n}.jsonl")
def show(name, r): print(f"{name:40s} crit {r['crit']:5.1f} (late {r['late']:4.1f}) cov {r['cov']:5.1f} pf {r['pf']:5.1f} wasted {r['wasted']:5.1f} copies {r['copies']:5.1f} stall {r['stall']:5.2f} ms/tok {r['ms_tok']:5.1f}", flush=True)
orc = s.hit.astype(np.float32) + np.random.RandomState(1).rand(*s.hit.shape).astype(np.float32) * 0.01
for k in (1, 2, 4, 8, 16):
    for q in (8, 64):
        r = run(tr, s, leads=(k,), scorer=lambda kk: orc, m=4, stage=64, qmax=q); show(f"ORACLE lead={k} m=4 Q={q}", r)

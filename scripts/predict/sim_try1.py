from load import R
import sys, numpy as np
from load import Set
from feats import *
from sim import *
n = sys.argv[1]
s = Set(n); tr = load_trace(f"{R}/{n}.jsonl")
E = {k: early(s, k) for k in (1, 2, 3, 4, 6, 8)}
def show(name, r): print(f"{name:34s} crit {r['crit']:5.1f} (late {r['late']:4.1f}) cov {r['cov']:5.1f} pf {r['pf']:5.1f} wasted {r['wasted']:5.1f} copies {r['copies']:5.1f} stall {r['stall']:5.2f} ms/tok {r['ms_tok']:5.1f}", flush=True)
base = run(tr, s); show("baseline SLRU", base)
for leads, m, stage, qmax in [((1,), 2, 8, 4), ((1,), 4, 8, 4), ((1, 2), 2, 16, 8), ((1, 2, 3), 2, 16, 8), ((1,2,3,4), 2, 32, 8), ((1,2,3,4), 4, 32, 12), ((1,2,4,8), 2, 32, 12)]:
    r = run(tr, s, leads=leads, scorer=lambda k: E[k], m=m, stage=stage, qmax=qmax)
    show(f"early leads={leads} m={m} S={stage} Q={qmax}", r)

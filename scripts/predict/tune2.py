import os
import sys, numpy as np, itertools, json, random
from multiprocessing import Pool
from load import Set
from sim import *
R = os.environ.get("PRED_RES", "predict-results")
n = "VAL"
s = Set(n); tr = load_trace(f"{R}/{n}.jsonl")
P = dict(np.load(f"{R}/pred-{n}.npz")); P = {int(k[1:]): v.astype(np.float32) for k, v in P.items()}
base = run(tr, s)
def work(cfg):
    leads, tn, tf, m, stage, q = cfg
    th = {k: (tn if k <= 2 else tf) for k in leads}
    r = run(tr, s, leads=leads, scorer=lambda k: P[k], m=m, theta=th, stage=stage, qmax=q)
    return cfg, r
if __name__ == "__main__":
    random.seed(1)
    L = (1, 2, 3, 4, 6, 8); cfgs = set()
    while len(cfgs) < 160:
        nl = random.choice([1, 2, 2, 3, 3, 4, 5]); leads = tuple(sorted(random.sample(L, nl)))
        cfgs.add((leads, random.choice([.3, .4, .5, .6, .7]), random.choice([.3, .4, .5, .6, .7]), random.choice([1, 2, 3]), random.choice([8, 16, 24]), random.choice([3, 4, 6, 8, 12])))
    with Pool(14) as p: res = p.map(work, sorted(cfgs))
    res.sort(key=lambda x: x[1]["ms_tok"])
    print("baseline", {k: round(v, 2) for k, v in base.items() if isinstance(v, float)})
    for cfg, r in res[:25]:
        print(cfg, f"crit {r['crit']:.1f} late {r['late']:.1f} pf {r['pf']:.1f} wasted {r['wasted']:.1f} copies {r['copies']:.1f} ({r['copies']/base['copies']:.2f}x) ms {r['ms_tok']:.2f} x{base['ms_tok']/r['ms_tok']:.3f}")
    json.dump([(c, {k: v for k, v in r.items() if isinstance(v, float)}) for c, r in res], open(f"{R}/tune2.json", "w"))

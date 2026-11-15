import os
"""Tune each predictor on VAL (random search, 14 workers), then score the VAL-best config on the held-out TEST_A / TEST_B."""
import sys, numpy as np, json, random
from multiprocessing import Pool
from load import Set
from sim import *
R = os.environ.get("PRED_RES", "predict-results")
DELAY = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
def loadP(kind, n, s):
    F = np.load(f"{R}/feat-{n}.npz")
    if kind == "gbm": P = dict(np.load(f"{R}/pred-{n}.npz"))
    elif kind.startswith("lr"): P = dict(np.load(f"{R}/pred{kind}-{n}.npz"))
    else:
        P = {}
        for k in (1, 2, 3, 4, 6, 8):
            q = np.load(f"{R}/predlr-{n}.npz")[f"p{k}"].astype(np.float32)
            P[f"p{k}"] = np.where(q >= 0, F[f"e{k}"], -1e9).astype(np.float32)
    return {int(k[1:]): v.astype(np.float32) for k, v in P.items()}
G = {}
def init(kind):
    for n in ("VAL", "TEST_A", "TEST_B"):
        s = Set(n); G[n] = (s, load_trace(f"{R}/{n}.jsonl"), loadP(kind, n, s))
def work(a):
    n, cfg = a; s, tr, P = G[n]
    leads, tn, tf, m, stage, q = cfg
    th = None if cfg[1] is None else {k: (tn if k <= 2 else tf) for k in leads}
    return run(tr, s, leads=leads, scorer=lambda k: P[k], m=m, theta=th, stage=stage, qmax=q, pred_delay=DELAY)
if __name__ == "__main__":
    kind = sys.argv[2]; OBJ = sys.argv[3] if len(sys.argv) > 3 else "ms_tok"
    random.seed(2); L = (1, 2, 3, 4, 6, 8); cfgs = set()
    th_grid = [None] if kind == "raw" else [.3, .4, .5, .6, .7]
    while len(cfgs) < 150:
        nl = random.choice([1, 2, 2, 3, 3, 4]); leads = tuple(sorted(random.sample(L, nl)))
        cfgs.add((leads, random.choice(th_grid), random.choice(th_grid), random.choice([1, 2, 3]), random.choice([8, 16, 24]), random.choice([3, 4, 6, 8, 12])))
    cfgs = sorted(cfgs, key=str)
    with Pool(14, initializer=init, initargs=(kind,)) as p:
        base = p.map(work, [(n, ((), None, None, 1, 8, 1)) for n in ("VAL", "TEST_A", "TEST_B")])
        res = sorted(zip(cfgs, p.map(work, [("VAL", c) for c in cfgs])), key=(lambda x: x[1]["ms_tok"]) if OBJ == "ms_tok" else (lambda x: x[1]["crit"] if x[1]["copies"] <= 1.5 * base[0]["copies"] else 1e9))
        print(f"== {kind} obj={OBJ} delay={DELAY}  baseline VAL/A/B ms {[round(b['ms_tok'],2) for b in base]} crit {[round(b['crit'],1) for b in base]}")
        top = [c for c, _ in res[:3]]
        for c, r in res[:3]:
            ra = p.apply(work, (("TEST_A", c),)); rb = p.apply(work, (("TEST_B", c),))
            line = f"{c}  VAL x{base[0]['ms_tok']/r['ms_tok']:.3f} crit {r['crit']:.1f} copies {r['copies']/base[0]['copies']:.2f}x"
            for nm, rr, b in (("A", ra, base[1]), ("B", rb, base[2])):
                line += f" | {nm}: crit {rr['crit']:.1f} ({100*(rr['crit']/b['crit']-1):+.0f}%) late {rr['late']:.1f} pf {rr['pf']:.1f} wasted {rr['wasted']:.1f} copies {rr['copies']/b['copies']:.2f}x ms {rr['ms_tok']:.2f} x{b['ms_tok']/rr['ms_tok']:.3f}"
            print(line, flush=True)
    json.dump([(c, {k: v for k, v in r.items() if isinstance(v, float)}) for c, r in res], open(f"{R}/final-{kind}-{DELAY}-{OBJ}.json", "w"))

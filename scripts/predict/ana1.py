from load import R
import sys, json, numpy as np, time
from load import Set
from feats import *
t0 = time.time()
tr = Set("TRAIN"); print("train", tr.T, "tokens, prompts", tr.prompt.max() + 1, flush=True)
tab = Tables(); tab.fit(tr)
def valid(S): return S.max(2) > -1e8
sets = {n: Set(n) for n in sys.argv[1:]}
M = [4, 6, 8, 12, 16]
out = {}
for n, s in sets.items():
    pred = {}
    pred["prior(static freq)"] = tab.p_prior(s)
    for N in (1, 2, 4, 8, 16, 32):
        pred[f"a recency N={N}"] = recency(s, N)
    pred["a recency N=16 d=.8"] = recency(s, 16, 0.8)
    pred["c markov same-layer prev token"] = tab.p_same(s)
    pred["d token id"] = tab.p_token(s)
    for k in (1, 2, 4, 8):
        pred[f"b cross-layer k={k}"] = tab.p_cross(s, k)
        pred[f"e early-routing k={k}"] = early(s, k)
    pred["e early from prev token end (all j)"] = early_prev_end(s)
    res = {}
    for name, S in pred.items():
        v = valid(S)
        # jitter ties so argpartition is not biased by index order
        S2 = S + np.random.RandomState(0).rand(*S.shape).astype(np.float32) * 1e-6
        row = {}
        for m in M:
            rec, prec = recall_at(S2, s.hit, m, v)
            row[m] = (float(np.average(rec, weights=[v[:, j].sum() for j in range(s.L)])), float(np.average(prec, weights=[v[:, j].sum() for j in range(s.L)])), rec.tolist())
        res[name] = row
    out[n] = res
    print(f"== {n}  tokens {s.T}   recall@m (precision@m)")
    print(f"{'predictor':42s}" + "".join(f"  m={m:<9d}" for m in M))
    for name, row in res.items():
        print(f"{name:42s}" + "".join(f"  {row[m][0]:.3f}({row[m][1]:.2f})" for m in M), flush=True)
json.dump(out, open(f"{R}/ana1.json", "w"))
print("secs", time.time() - t0)

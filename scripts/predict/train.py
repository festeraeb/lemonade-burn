import os
"""Fit the combined predictor (g): one gradient-boosted model per lead k on UNCACHED candidates, then report
held-out recall/precision of the uncached needed experts at fixed fetch budgets m per target layer."""
import numpy as np, sys, time, pickle, json, hashlib
from sklearn.ensemble import HistGradientBoostingClassifier
R = os.environ.get("PRED_RES", "predict-results")
KS = (1, 2, 3, 4, 6, 8)
def ld(n): return dict(np.load(f"{R}/feat-{n}.npz"))
def cols(F, k):
    nxt = [x for x in KS if x > k][:2]
    while len(nxt) < 2: nxt.append(k)
    T, L, E = F["hit"].shape
    jl = np.broadcast_to(np.arange(L, dtype=np.float32)[None, :, None], (T, L, E))
    names = ["e", "e+", "e++", "rank", "gap", "cross", "age", "rec1", "rec16d", "tokp", "prior", "same", "j", "early_flag"]
    arrs = [F[f"e{k}"], F[f"e{nxt[0]}"], F[f"e{nxt[1]}"], F[f"r{k}"].astype(np.float32), F[f"g{k}"], F[f"x{k}"], F["age"], F["rec1"], F["rec16d"], F["tokp"], F["prior"], F["same"], jl,
            np.broadcast_to((np.arange(L) < k)[None, :, None], (T, L, E)).astype(np.float32)]
    return names, arrs
def rows(F, k, sample=None, seed=0):
    names, arrs = cols(F, k)
    ok = (~F["cached"]) & (F[f"e{k}"] > -1e8)
    if sample is not None:
        rs = np.random.RandomState(seed); neg = ok & ~F["hit"]
        keep = ok & (F["hit"] | (rs.rand(*ok.shape) < sample)); w = np.where(F["hit"][keep], 1.0, 1.0 / sample)
    else:
        keep = ok; w = None
    X = np.stack([a[keep] for a in arrs], 1).astype(np.float32)
    return X, F["hit"][keep], w, keep
def fit(F, k):
    X, y, w, _ = rows(F, k, sample=0.15)
    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, early_stopping=False, random_state=0)
    m.fit(X, y, sample_weight=w); return m
def predict(models, F, k):
    names, arrs = cols(F, k)
    T, L, E = F["hit"].shape
    P = np.full((T, L, E), -1.0, np.float32)
    ok = (~F["cached"]) & (F[f"e{k}"] > -1e8)
    X = np.stack([a[ok] for a in arrs], 1).astype(np.float32)
    P[ok] = models[k].predict_proba(X)[:, 1]
    return P
def eval_unc(F, P, k, m):
    """recall of the needed-but-uncached experts when the top-m uncached by P are fetched, per target layer"""
    need = F["hit"] & ~F["cached"] & (F[f"e{k}"] > -1e8)
    idx = np.argpartition(-P, m - 1, axis=2)[:, :, :m]
    got = np.take_along_axis(need, idx, 2).sum(2)
    fetched = np.take_along_axis(P >= 0, idx, 2).sum(2)
    return got.sum() / need.sum(), got.sum() / max(fetched.sum(), 1)
if __name__ == "__main__":
    t0 = time.time(); Ftr = ld("TRAIN"); models = {}
    for k in KS:
        models[k] = fit(Ftr, k); print("fit k", k, round(time.time() - t0), "s", flush=True)
    h = hashlib.sha256(open(f"{R}/TRAIN.dump", "rb").read(1 << 26)).hexdigest()[:16]
    pickle.dump({"models": models, "train_hash": h, "ks": KS}, open(f"{R}/predictor-gbm.pkl", "wb"))
    res = {}
    for n in ("VAL", "TEST_A", "TEST_B"):
        F = ld(n); res[n] = {}
        print(f"== {n}: recall (precision) of needed-but-uncached experts at m uncached fetches per target layer")
        for k in KS:
            P = predict(models, F, k)
            base = np.where(P >= 0, F[f"e{k}"], -1e9).astype(np.float32)             # raw early routing on the same uncached candidates
            cells = []
            for m in (1, 2, 3, 4, 6):
                a = eval_unc(F, P, k, m); b = eval_unc(F, base, k, m)
                cells.append(f"m={m}: gbm {a[0]:.3f}({a[1]:.2f}) raw {b[0]:.3f}({b[1]:.2f})")
                res[n][f"k{k}m{m}"] = [a, b]
            print(f"k={k}  " + " | ".join(cells), flush=True)
    json.dump(res, open(f"{R}/train-eval.json", "w"))

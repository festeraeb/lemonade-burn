"""Deployable predictor: one logistic model per lead on a handful of cheap features (no tree ensemble, no token table)."""
import numpy as np, json, hashlib, time
from sklearn.linear_model import LogisticRegression
from train import ld, KS, R, eval_unc
NAMES = ["e", "gap", "log1p_rank", "rank_lt4", "cross", "log1p_age", "rec1", "rec16d", "prior", "same"]
def feats(F, k):
    T, L, E = F["hit"].shape
    ok = (~F["cached"]) & (F[f"e{k}"] > -1e8)
    r = F[f"r{k}"].astype(np.float32)
    cols = [F[f"e{k}"], F[f"g{k}"], np.log1p(r), (r < 4).astype(np.float32), F[f"x{k}"], np.log1p(F["age"]), F["rec1"], F["rec16d"], F["prior"], F["same"]]
    cols = [c for n_, c in zip(NAMES, cols) if n_ in SUB]
    return ok, cols
def design(F, k, sample=None, seed=0):
    ok, cols = feats(F, k)
    if sample is not None:
        rs = np.random.RandomState(seed); keep = ok & (F["hit"] | (rs.rand(*ok.shape) < sample)); w = np.where(F["hit"][keep], 1.0, 1.0 / sample)
    else: keep, w = ok, None
    return np.stack([c[keep] for c in cols], 1).astype(np.float32), F["hit"][keep], w, ok
import sys
TAG = sys.argv[1] if len(sys.argv) > 1 else "lr"
SUB = sys.argv[2].split(",") if len(sys.argv) > 2 else NAMES
if __name__ == "__main__":
    t0 = time.time(); Ftr = ld("TRAIN"); M = {}
    for k in KS:
        X, y, w, _ = design(Ftr, k, 0.15); mu = X.mean(0); sd = X.std(0) + 1e-6
        lr = LogisticRegression(C=1.0, max_iter=400); lr.fit((X - mu) / sd, y, sample_weight=w)
        M[k] = {"mu": mu.tolist(), "sd": sd.tolist(), "w": lr.coef_[0].tolist(), "b": float(lr.intercept_[0])}
        print("fit", k, round(time.time() - t0), flush=True)
    h = hashlib.sha256(open(f"{R}/TRAIN.dump", "rb").read(1 << 26)).hexdigest()[:16]
    json.dump({"features": [n_ for n_ in NAMES if n_ in SUB], "train_hash": h, "models": M}, open(f"{R}/predictor-%s.json" % TAG + "", "w"))
    for n in ("VAL", "TEST_A", "TEST_B"):
        F = ld(n); out = {}; print("==", n)
        for k in KS:
            ok, cols = feats(F, k); m = M[k]
            X = (np.stack([c[ok] for c in cols], 1) - np.array(m["mu"])) / np.array(m["sd"])
            z = X @ np.array(m["w"]) + m["b"]; P = np.full(F["hit"].shape, -1.0, np.float32); P[ok] = 1 / (1 + np.exp(-z))
            out[f"p{k}"] = P.astype(np.float16)
            print(k, " ".join(f"m={mm}: {eval_unc(F, P, k, mm)[0]:.3f}({eval_unc(F, P, k, mm)[1]:.2f})" for mm in (1, 2, 3, 4)), flush=True)
        np.savez(f"{R}/pred{TAG}-{n}.npz", **out)

"""Feature tensors per set, cached to disk as float16/float32 npz (small enough to keep)."""
import numpy as np, os, sys, time
from load import Set
from feats import *
from cachemask import cached_mask
from sim import load_trace
R = os.environ.get("PRED_RES", "predict-results")
def rank_of(S):
    return np.argsort(np.argsort(-S, 2), 2).astype(np.int8)
def kth_gap(S, k=4):
    top = -np.partition(-S, k - 1, axis=2)[:, :, k - 1:k]            # k-th best value
    return (S - top).astype(np.float32)
def age_feat(s, cap=64):
    """tokens since the expert was last used in this layer (within the prompt), capped"""
    A = np.full(s.hit.shape, cap, np.float32)
    last = np.full(s.hit.shape[1:], -10**6, np.int64)
    for t in range(s.T):
        if t > 0 and s.prompt[t] != s.prompt[t - 1]: last[:] = -10**6
        A[t] = np.minimum(t - last, cap)
        last[s.hit[t]] = t
    return A
def build(name, tab, ks=(1, 2, 3, 4, 6, 8), s=None):
    s = s or Set(name); tr = load_trace(f"{R}/{name}.jsonl")
    F = {}
    F["cached"] = cached_mask(tr, s.T)
    F["hit"] = s.hit
    F["age"] = age_feat(s)
    F["rec1"] = recency(s, 1); F["rec16d"] = recency(s, 16, 0.8)
    F["tokp"] = tab.p_token(s); F["prior"] = tab.p_prior(s); F["same"] = tab.p_same(s)
    for k in ks:
        F[f"e{k}"] = early(s, k); F[f"x{k}"] = tab.p_cross(s, k)
    for k in ks:
        F[f"r{k}"] = rank_of(F[f"e{k}"]); F[f"g{k}"] = kth_gap(F[f"e{k}"])
    F["prompt"] = s.prompt; F["tok"] = s.tok
    return F
if __name__ == "__main__":
    tr = Set("TRAIN")
    # cross-fitted TRAIN features: tables from the other half of the prompts
    fold = (tr.prompt % 2).astype(int)
    out = None
    for f in (0, 1):
        tab = Tables(); tab.fit(tr, sel=(fold != f))
        F = build("TRAIN", tab, s=tr)
        if out is None: out = {k: v.copy() for k, v in F.items()}
        else:
            for k, v in F.items():
                if v.ndim >= 1 and v.shape[0] == tr.T and k not in ("prompt", "tok"): out[k][fold == f] = v[fold == f]
        if f == 0:
            for k, v in F.items():
                if v.ndim >= 1 and v.shape[0] == tr.T and k not in ("prompt", "tok"): out[k][fold == 0] = v[fold == 0]
    np.savez(f"{R}/feat-TRAIN.npz", **out); print("TRAIN features done", flush=True)
    tab = Tables(); tab.fit(tr)
    import pickle; pickle.dump(tab, open(f"{R}/tables-TRAIN.pkl", "wb"))
    for n in sys.argv[1:]:
        t0 = time.time(); F = build(n, tab)
        np.savez(f"{R}/feat-{n}.npz", **F); print(n, "features", round(time.time() - t0), "s", flush=True)

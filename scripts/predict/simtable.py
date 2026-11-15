import os
"""Fixed-grid comparison of every predictor family in the timed simulator (nothing tuned on TEST): lead 2, S=8, Q=8."""
import numpy as np, sys, json
from multiprocessing import Pool
from load import Set
from sim import *
R = os.environ.get("PRED_RES", "predict-results")
G = {}
def init():
    for n in ("TEST_A", "TEST_B"):
        s = Set(n); F = np.load(f"{R}/feat-{n}.npz"); G[n] = (s, load_trace(f"{R}/{n}.jsonl"), {k: F[k] for k in ("cached", "e2", "e1", "e4", "x1", "x2", "same", "tokp", "rec16d", "rec1", "prior")},
                                                   {"gbm": np.load(f"{R}/pred-{n}.npz"), "lr": np.load(f"{R}/predlrmin-{n}.npz")})
def arr(n, name, k):
    s, tr, F, P = G[n]; unc = ~F["cached"]
    if name in ("gbm", "lr"): return P[name][f"p{k}"].astype(np.float32)
    src = {"a recency (16, decay .8)": "rec16d", "a recency N=1": "rec1", "b cross-layer": f"x{k}", "c markov same-layer": "same", "d token id": "tokp", "e early-routing raw": f"e{k}", "static frequency": "prior"}[name]
    return np.where(unc, F[src], -1e9).astype(np.float32)
def work(a):
    n, name, k, m, th, d = a; s, tr, F, P = G[n]; A = arr(n, name.replace(" +thr", ""), k)
    if name in ("a recency (16, decay .8)", "c markov same-layer", "d token id", "static frequency", "a recency N=1") : A = np.where(A < -1e8, -1e9, A)
    r = run(tr, s, leads=(k,) if m else (), scorer=lambda kk: A, m=m, theta=None if th is None else {k: th}, stage=8, qmax=8, pred_delay=d)
    return a, r
if __name__ == "__main__":
    # operating point for the uncalibrated scores: the threshold whose top-2 uncached fetches have precision >= 0.4 on VAL
    Fv = np.load(f"{R}/feat-VAL.npz")
    def theta_for(src):
        S = np.where(~Fv["cached"], Fv[src], -1e9).astype(np.float32)
        idx = np.argpartition(-S, 1, axis=2)[:, :, :2]
        sc = np.take_along_axis(S, idx, 2).ravel(); hit = np.take_along_axis(Fv["hit"], idx, 2).ravel(); ok = sc > -1e8
        sc, hit = sc[ok], hit[ok]; order = np.argsort(-sc); cum = np.cumsum(hit[order]) / np.arange(1, len(order) + 1)
        good = np.flatnonzero(cum >= 0.4)
        return float(sc[order][good.max()]) if len(good) else float(sc.max() + 1)
    TH = {"a recency (16, decay .8)": theta_for("rec16d"), "b cross-layer": theta_for("x2"), "c markov same-layer": theta_for("same"), "d token id": theta_for("tokp"), "e early-routing raw": theta_for("e2"), "static frequency": theta_for("prior"), "a recency N=1": theta_for("rec1")}
    print("VAL thresholds for precision 0.4:", {k: round(v, 3) for k, v in TH.items()})
    names = ["static frequency", "a recency N=1", "a recency (16, decay .8)", "b cross-layer", "c markov same-layer", "d token id", "e early-routing raw", "lr", "gbm"]
    names2 = []
    for nm in names: names2 += [nm] + ([nm + " +thr"] if nm not in ("lr", "gbm") else [])
    jobs = []
    for n in ("TEST_A", "TEST_B"):
        jobs.append((n, "static frequency", 2, 0, None, 0.0))
        for name in names:
            for m in (1, 2):
                jobs.append((n, name, 2, m, 0.4 if name in ("lr", "gbm") else None, 0.2))
                if name not in ("lr", "gbm"): jobs.append((n, name + " +thr", 2, m, TH[name], 0.2))
    with Pool(12, initializer=init) as p: res = p.map(work, jobs)
    base = {a[0]: r for a, r in res if a[3] == 0}
    out = []
    print("lead 2, staging 8, queue 8, prediction delay 0.2 ms; crit = demand + late misses per token; copies and ms relative to SLRU alone")
    print(f"{'predictor':28s} {'m':>2s} | " + " | ".join(f"{n}: crit  d%  copies  x-time" for n in ("TEST_A", "TEST_B")))
    for name in names2:
        for m in (1, 2):
            row = []
            for n in ("TEST_A", "TEST_B"):
                r = [r for a, r in res if a[0] == n and a[1] == name and a[3] == m][0]; b = base[n]
                row.append(f"{r['crit']:5.1f} {100*(r['crit']/b['crit']-1):+4.0f}% {r['copies']/b['copies']:5.2f}x {b['ms_tok']/r['ms_tok']:5.3f}")
                out.append((name, m, n, r["crit"], r["copies"], r["ms_tok"], b["crit"], b["copies"], b["ms_tok"]))
            print(f"{name:28s} {m:>2d} | " + " | ".join(row))
    print({n: (round(b['crit'], 1), round(b['ms_tok'], 2)) for n, b in base.items()})
    json.dump(out, open(f"{R}/simtable.json", "w"))

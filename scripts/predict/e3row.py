import os, numpy as np, sys
from load import Set, R
from sim import *
Fv = np.load(f"{R}/feat-VAL.npz")
def theta_for(src, prec):
    S = np.where(~Fv["cached"], Fv[src], -1e9).astype(np.float32)
    idx = np.argpartition(-S, 1, axis=2)[:, :, :2]
    sc = np.take_along_axis(S, idx, 2).ravel(); hit = np.take_along_axis(Fv["hit"], idx, 2).ravel(); ok = sc > -1e8
    sc, hit = sc[ok], hit[ok]; order = np.argsort(-sc); cum = np.cumsum(hit[order]) / np.arange(1, len(order) + 1)
    good = np.flatnonzero(cum >= prec); return float(sc[order][good.max()])
for n in ("TEST_A", "TEST_B"):
    s = Set(n); tr = load_trace(f"{R}/{n}.jsonl"); F = np.load(f"{R}/feat-{n}.npz")
    A = np.where(~F["cached"], F["e1"], -1e9).astype(np.float32)
    b = run(tr, s)
    for prec in (0.7, 0.5):
        th = theta_for("e1", prec)
        for d in (0.0, 0.2, 0.4):
            r = run(tr, s, leads=(1,), scorer=lambda k: A, m=2, theta={1: th}, stage=8, qmax=8, pred_delay=d)
            print(f"{n} E3-like (early k=1 raw score, cap 2, threshold for VAL precision {prec}, delay {d}): crit {r['crit']:.1f} ({100*(r['crit']/b['crit']-1):+.0f}%) copies {r['copies']/b['copies']:.2f}x pf {r['pf']:.1f} wasted {r['wasted']:.1f} x-time {b['ms_tok']/r['ms_tok']:.3f}")

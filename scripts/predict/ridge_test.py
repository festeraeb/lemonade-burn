import numpy as np, time
from load import Set, scores
tr = Set("TRAIN"); va = Set("VAL"); w = tr.w
def hnorm(Wl, x, eps):
    norm = Wl[0]; r = 1/np.sqrt((x*x).mean(1, keepdims=True)+eps); return x*r*norm
for k in (2, 4, 8):
    for j in (10, 20, 30):
        li = tr.layers[j]; Wl = w["W"][li]
        # inputs: x_mid[j-k] normalized with TARGET layer's norm; target: true selection score at j
        Xt = hnorm(Wl, tr.x[:, j-k], w["eps"]); Xv = hnorm(Wl, va.x[:, j-k], w["eps"])
        Yt = scores(Wl, tr.x[:, j], w["eps"])[0]; Yv = scores(Wl, va.x[:, j], w["eps"])[0]
        base_t = 1/(1+np.exp(-(Xt @ Wl[1].T))) + Wl[2]; base_v = 1/(1+np.exp(-(Xv @ Wl[1].T))) + Wl[2]
        def rec(P, Y):
            tp = np.argsort(-P, 1)[:, :4]; ty = np.argsort(-Y, 1)[:, :4]
            return np.mean([len(set(a) & set(b)) / 4 for a, b in zip(tp, ty)])
        res = [f"k={k} j={j} raw {rec(base_v, Yv):.3f}"]
        for lam in (30, 300, 3000):
            A = Xt.T @ Xt + lam * np.eye(2048); D = np.linalg.solve(A, Xt.T @ (Yt - base_t))
            res.append(f"ridge{lam} {rec(base_v + Xv @ D, Yv):.3f}")
        print(" ".join(res), flush=True)

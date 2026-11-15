"""Loader for LB_PRED_DUMP files + LB_LA_TRACE jsonl. Returns arrays for the predictor study."""
import numpy as np, json, os
R = os.environ.get("PRED_RES", "predict-results")

def weights(path):
    b = open(path, "rb").read()
    nl, ne, k, d = np.frombuffer(b[:16], dtype="<u4")
    eps = float(np.frombuffer(b[16:20], dtype="<f4")[0]); o = 20
    W = {}
    per = d + ne * d + ne
    while o < len(b):
        li = int(np.frombuffer(b[o:o+4], dtype="<u4")[0]); o += 4
        a = np.frombuffer(b[o:o+4*per], dtype="<f4"); o += 4 * per
        W[li] = (a[:d].copy(), a[d:d+ne*d].reshape(ne, d).copy(), a[d+ne*d:].copy())
    return dict(nl=int(nl), ne=int(ne), k=int(k), d=int(d), eps=eps, W=W)

def dump(path, k, d):
    dt = np.dtype([("tok", "<u4"), ("pos", "<u4"), ("layer", "<u4"), ("ids", "<u4", (k,)), ("x", "<f4", (d,))])
    return np.fromfile(path, dtype=dt)

def scores(Wl, x, eps):
    """router selection scores of layer weights Wl=(norm, router, bias) on residual rows x [T,d]"""
    norm, R_, b = Wl
    r = 1.0 / np.sqrt((x * x).mean(1, keepdims=True) + eps)
    h = x * r * norm
    return 1.0 / (1.0 + np.exp(-(h @ R_.T))) + b, h @ R_.T

class Set:
    """One collected set: tokens x MoE layers, ordered by (token index, layer)."""
    def __init__(self, name):
        self.name = name
        self.w = weights(f"{R}/{name}.dump.weights")
        w = self.w
        rec = dump(f"{R}/{name}.dump", w["k"], w["d"])
        self.layers = sorted(w["W"]); self.L = len(self.layers)
        n = len(rec) // self.L
        rec = rec[: n * self.L]
        self.T = n
        self.tok = rec["tok"].reshape(n, self.L)[:, 0].copy()
        self.pos = rec["pos"].reshape(n, self.L)[:, 0].copy()
        assert (rec["layer"].reshape(n, self.L) == np.array(self.layers)).all(), "layer order"
        self.ids = rec["ids"].reshape(n, self.L, w["k"]).astype(np.int16)      # as routed
        self.x = rec["x"].reshape(n, self.L, w["d"])                            # residual stream x_mid (f32)
        # prompt segmentation: a new request starts when pos does not follow the previous one
        brk = np.ones(n, bool); brk[1:] = self.pos[1:] != self.pos[:-1] + 1
        self.start = np.flatnonzero(brk); self.prompt = np.cumsum(brk) - 1
        # true expert set as boolean [T, L, 64]
        self.hit = np.zeros((n, self.L, w["ne"]), bool)
        for j in range(w["k"]):
            np.put_along_axis(self.hit, self.ids[:, :, j:j+1].astype(np.int64), True, axis=2)

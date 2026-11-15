"""Predictors for the next-expert study. Every function returns score arrays [T, L, 64] (higher = more
likely) for layer index j (0..L-1; real layer = layers[j]) as it could be computed at ISSUE time."""
import numpy as np
from load import scores

def prev_mask(s, i):
    """[T] bool: token t has a predecessor i tokens back inside the same prompt"""
    ok = np.zeros(s.T, bool)
    ok[i:] = s.prompt[i:] == s.prompt[:-i] if i > 0 else True
    return ok

def recency(s, N, decay=1.0):
    S = np.zeros(s.hit.shape, np.float32)
    for i in range(1, N + 1):
        m = prev_mask(s, i)
        S[i:][m[i:]] += (decay ** (i - 1)) * s.hit[:-i][m[i:]]
    return S

def early(s, k):
    """router of layer j applied to the residual stream x_mid of layer j-k (same token). For j < k the
    source is the previous token's last layer (the end of the previous token); invalid at prompt start."""
    w = s.w; S = np.full(s.hit.shape, -1e9, np.float32)
    for j, li in enumerate(s.layers):
        if j - k >= 0:
            S[:, j] = scores(w["W"][li], s.x[:, j - k], w["eps"])[0]
        else:
            m = prev_mask(s, 1)
            xs = s.x[:-1, s.L - 1]
            sc = scores(w["W"][li], xs, w["eps"])[0]
            S[1:][m[1:], j] = sc[m[1:]]
    return S

def early_prev_end(s):
    """router of layer j on the previous token's last-layer x_mid, for every j (lead = whole token)"""
    w = s.w; S = np.full(s.hit.shape, -1e9, np.float32); m = prev_mask(s, 1)
    for j, li in enumerate(s.layers):
        sc = scores(w["W"][li], s.x[:-1, s.L - 1], w["eps"])[0]
        S[1:][m[1:], j] = sc[m[1:]]
    return S

class Tables:
    """Count tables fitted on a training Set (or several, concatenated by `fit`)."""
    def __init__(self, ne=64, L=38):
        self.ne = ne; self.L = L
        self.same = np.zeros((L, ne, ne)); self.cross = {}      # same[j][a,e]: a in H[t-1,j] -> e in H[t,j]
        self.prior = np.zeros((L, ne)); self.n = 0
        self.tok = {}                                           # token -> [L,ne] counts
        self.ntok = {}
    def fit(self, s, ks=(1, 2, 3, 4, 6, 8), sel=None):
        if sel is None: sel = np.ones(s.T, bool)
        H = s.hit.astype(np.float64) * sel[:, None, None]
        self.prior += H.sum(0); self.n += int(sel.sum())
        m = prev_mask(s, 1) & sel & np.r_[False, sel[:-1]]
        A = H[:-1][m[1:]]; B = H[1:][m[1:]]
        for j in range(self.L):
            self.same[j] += A[:, j].T @ B[:, j]
        for k in ks:
            c = self.cross.setdefault(k, np.zeros((self.L, self.ne, self.ne)))
            for j in range(k, self.L):
                c[j] += H[:, j - k].T @ H[:, j]
        for t in np.flatnonzero(sel):
            tk = int(s.tok[t]); a = self.tok.get(tk)
            if a is None: a = self.tok[tk] = np.zeros((self.L, self.ne))
            a += H[t]; self.ntok[tk] = self.ntok.get(tk, 0) + 1
    def p_same(self, s):
        P = (self.same + 0.5) / (self.same.sum(2, keepdims=True) + 0.5 * self.ne)       # P(e | a)
        S = np.zeros(s.hit.shape, np.float32); m = prev_mask(s, 1)
        for j in range(self.L):
            S[1:, j] = (s.hit[:-1, j].astype(np.float32) @ P[j])
        S[~m] = 0
        return S
    def p_cross(self, s, k):
        c = self.cross[k]; P = (c + 0.5) / (c.sum(2, keepdims=True) + 0.5 * self.ne)
        S = np.zeros(s.hit.shape, np.float32)
        for j in range(k, self.L):
            S[:, j] = s.hit[:, j - k].astype(np.float32) @ P[j]
        return S
    def p_token(self, s, alpha=2.0):
        prior = (self.prior + 1) / (self.n + 16.0)                       # per-layer frequency of each expert
        S = np.zeros(s.hit.shape, np.float32)
        for t in range(s.T):
            tk = int(s.tok[t]); a = self.tok.get(tk); n = self.ntok.get(tk, 0)
            S[t] = ((a if a is not None else 0) + alpha * prior) / (n + alpha)
        return S
    def p_prior(self, s):
        return np.broadcast_to(((self.prior + 1) / (self.n + 16.0)).astype(np.float32), s.hit.shape).copy()

def recall_at(S, H, m, valid=None):
    """S,H [T,L,64] -> per-layer recall@m and precision@m (m predictions per layer pass)"""
    idx = np.argpartition(-S, m - 1, axis=2)[:, :, :m]
    got = np.take_along_axis(H, idx, 2).sum(2)                        # true experts among the m
    tot = H.sum(2)
    if valid is None: valid = np.ones(S.shape[:2], bool)
    rec = np.array([got[valid[:, j], j].sum() / tot[valid[:, j], j].sum() for j in range(S.shape[1])])
    return rec, rec * (tot.mean() / m)

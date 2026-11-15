"""Timed replay of a route trace with a prefetching predictor (docs/ENGINE_PREDICTOR.md).
Model: layer j runs c_pre ms to the routes read, then the experts it needs that are not resident are waited
for (demand copies queue FIFO on one bus behind any prefetch already in flight), then c_post ms. A prefetch
issued when a layer's routes are known copies into a bounded staging area (never into the cache), is repacked
into a cache slot only when its layer really routes it, and costs bus time whether or not it is used."""
import json, numpy as np
from slru import Slru

B = 5.31 / 9.7          # ms of bus per expert (5.31 MB at the 9.7 GB/s measured)
REPACK = 0.05           # ms of main-stream repack per filled miss (copy path measured 0.596 ms/miss)
C_TOKEN = 13.8          # ms per token with every expert resident (flush+fuse, 12500 MiB)

def load_trace(path):
    return [json.loads(l) for l in open(path) if l.strip()]

def run(trace, s, slots=1003, leads=(), scorer=None, m=2, theta=None, stage=8, qmax=4, c_pre_frac=0.5,
        into_cache=False, pred_delay=0.0, first_layer=2, L=38, ne=64, collect=False):
    """scorer(k) -> array [T,L,64] of scores for target layer j+k computed at issue layer j (or None).
    A candidate is fetched if it is in the top-m of its target layer's scores, not cached/staged, and (when
    theta is given) its score >= theta[k]. Returns per-token means."""
    cache = Slru(slots - (0 if into_cache else (stage if leads else 0)))
    c = C_TOKEN / L; cpre = c * c_pre_frac; cpost = c - cpre
    S = {k: scorer(k) for k in leads} if leads else {}
    T = s.T
    t = -1; now = 0.0; bus = 0.0
    staged = {}            # (j, e) -> arrival time
    st = dict(tok=0, crit=0, late=0, cov=0, hit=0, pf=0, used=0, wasted=0, demand=0, stall=0.0, time=0.0, drop=0, evict=0)
    tok_t0 = 0.0; per_tok = []; t_start = None
    npass = 0
    for x in trace:
        j = x["l"] - first_layer
        need = sorted(set(x["need"]))
        if x["k"] != "d":
            cache.plan(x["l"], need); continue
        if x["l"] == first_layer:
            t += 1
            if t >= T: break
            if t > 0 and collect: per_tok.append(now - tok_t0)
            tok_t0 = now; st["tok"] += 1
            if t_start is None: t_start = now
        tr = now + cpre
        arr_max = tr; demand_n = 0; fills = 0
        for e in need:
            key = (j, e)
            if cache.has(x["l"] * ne + e):
                st["hit"] += 1; continue
            fills += 1
            if key in staged:
                a = staged.pop(key); st["used"] += 1
                if a <= tr: st["cov"] += 1
                else: st["late"] += 1; st["crit"] += 1; st["stall"] += a - tr
                arr_max = max(arr_max, a)
            else:
                start = max(tr, bus); bus = start + B; st["crit"] += 1; st["demand"] += 1
                arr_max = max(arr_max, bus)
        if into_cache:
            pass
        cache.plan(x["l"], need)
        now = arr_max + REPACK * fills + cpost
        # expire staged entries of layers that have run
        for key in [k_ for k_ in staged if k_[0] <= j and k_[0] >= 0]:
            if key[0] <= j: del staged[key]; st["wasted"] += 1
        # issue prefetches for the layers ahead
        if leads:
            nq = sum(1 for a in staged.values() if a > tr)
            for k in leads:
                jt = j + k
                if jt >= L: continue
                sc = S[k][t, jt]
                if sc.max() < -1e8: continue
                top = np.argpartition(-sc, m - 1)[:m]
                top = top[np.argsort(-sc[top])]
                for e in top:
                    e = int(e)
                    if sc[e] < -1e8: continue
                    if theta is not None and sc[e] < theta[k]: continue
                    kk = (jt, e)
                    if kk in staged or cache.has((jt + first_layer) * ne + e): continue
                    if nq >= qmax or len(staged) >= stage:
                        st["drop"] += 1; continue
                    start = max(tr + pred_delay, bus); bus = start + B; staged[kk] = bus; st["pf"] += 1; nq += 1
    n = max(st["tok"], 1)
    st["wasted"] += len(staged)
    out = {k_: v / n for k_, v in st.items() if k_ != "tok"}
    out["tokens"] = st["tok"]; out["ms_tok"] = (now - t_start) / n; out["copies"] = out["demand"] + out["pf"]
    if collect: out["per_tok_ms"] = per_tok
    return out

"""Baseline SLRU replay of a trace; returns cached[T,L,64] = was the expert resident when the layer pass began."""
import numpy as np
from slru import Slru
from sim import load_trace
def cached_mask(trace, T, slots=1003, first=2, L=38, ne=64):
    c = Slru(slots); M = np.zeros((T, L, ne), bool); t = -1
    for x in trace:
        need = sorted(set(x["need"]))
        if x["k"] == "d":
            if x["l"] == first:
                t += 1
                if t >= T: break
            j = x["l"] - first
            for e in range(ne): M[t, j, e] = c.has(x["l"] * ne + e)
        c.plan(x["l"], need)
    return M

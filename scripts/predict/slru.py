"""Python port of lb_lfm2::cache::Slru (bookkeeping only) for the predictor simulator."""
from collections import OrderedDict

class Slru:
    def __init__(self, n_slots, n_expert=64):
        self.n = n_slots; self.ne = n_expert
        self.seg = [OrderedDict(), OrderedDict()]   # key -> None, LRU first
        self.where = {}                              # key -> segment (0 probation, 1 protected)
        self.nfree = n_slots                         # empty slots (probation LRU end, taken first)

    def has(self, k): return k in self.where

    def _touch(self, k):
        s = self.where[k]; del self.seg[s][k]; self.seg[1][k] = None; self.where[k] = 1
        if len(self.seg[1]) > self.n // 2:
            h, _ = self.seg[1].popitem(last=False); self.seg[0][h] = None; self.where[h] = 0

    def _victim(self, layer, needed, guard=()):
        # empty slots first (they sit at probation LRU end in the Rust bookkeeping)
        if self.nfree > 0: return None, True
        for sg in (0, 1):
            for k in self.seg[sg]:
                if k in guard: continue
                if k // self.ne == layer and (k % self.ne) in needed: continue
                return k, False
        for sg in (0, 1):                             # guard exhausted: plain victim
            for k in self.seg[sg]:
                if not (k // self.ne == layer and (k % self.ne) in needed): return k, False
        raise RuntimeError("no victim")

    def plan(self, layer, needed, guard=()):
        """needed: sorted distinct experts. Returns (n_hits, fills list of experts)."""
        nd = set(needed); hits = 0; fills = []
        for e in needed:
            k = layer * self.ne + e
            if k in self.where: self._touch(k); hits += 1
        for e in needed:
            k = layer * self.ne + e
            if k in self.where: continue
            v, empty = self._victim(layer, nd, guard)
            if empty: self.nfree -= 1
            else:
                s = self.where.pop(v); del self.seg[s][v]
            self.seg[0][k] = None; self.where[k] = 0
            fills.append(e)
        return hits, fills

    def insert_cold(self, k):
        """admit a prefetched expert on probation (used by the prefetch models); returns evicted key or None"""
        if k in self.where: return None
        ev = None
        if self.nfree > 0: self.nfree -= 1
        else:
            for sg in (0, 1):
                for v in self.seg[sg]: ev = v; break
                if ev is not None: break
            s = self.where.pop(ev); del self.seg[s][ev]
        self.seg[0][k] = None; self.where[k] = 0
        return ev

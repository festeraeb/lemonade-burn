import json, sys
from slru import Slru
def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]
P = load(sys.argv[1]); slots = int(sys.argv[2])
c = Slru(slots); miss = 0; tok = 0; first = min(x["l"] for x in P if x["k"] == "d")
for x in P:
    h, f = c.plan(x["l"], sorted(set(x["need"])))
    if x["k"] == "d":
        miss += len(f)
        if x["l"] == first: tok += 1
print("misses/token", miss / tok, "tokens", tok)

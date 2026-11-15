from load import R
import sys
from load import Set
from sim import *
for n in sys.argv[1:]:
    s = Set(n); tr = load_trace(f"{R}/{n}.jsonl")
    r = run(tr, s); print(n, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})

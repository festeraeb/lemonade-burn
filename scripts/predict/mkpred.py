import numpy as np, pickle, sys
from train import *
d = pickle.load(open(f"{R}/predictor-gbm.pkl", "rb")); models = d["models"]
for n in ("VAL", "TEST_A", "TEST_B"):
    F = ld(n); P = {k: predict(models, F, k) for k in KS}
    np.savez(f"{R}/pred-{n}.npz", **{f"p{k}": v.astype(np.float16) for k, v in P.items()}); print(n, "ok", flush=True)

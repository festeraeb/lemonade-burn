#!/usr/bin/env python3
"""Compare logits of two hf_reference.py output dirs (same token sequences), using the
same metrics as `lb-run compare`: per-position top-1 agreement, agreement where the
second dir's top-1 margin > 0.5, max and mean |dlogit|.

Usage: python -I compare_refs.py A_DIR B_DIR   (B is treated as the reference)
"""
import json
import os
import sys

import numpy as np


def main(a_dir, b_dir):
    pa = json.load(open(os.path.join(a_dir, "prompts.json")))
    pb = json.load(open(os.path.join(b_dir, "prompts.json")))
    tot_agree = tot_n = 0
    tot_max = 0.0
    for i, (ra, rb) in enumerate(zip(pa["records"], pb["records"])):
        assert ra["full_ids"] == rb["full_ids"], f"record {i}: token sequences differ"
        la = np.load(os.path.join(a_dir, f"logits_{i}.npy")).astype(np.float64)
        lb = np.load(os.path.join(b_dir, f"logits_{i}.npy")).astype(np.float64)
        ta, tb = la.argmax(-1), lb.argmax(-1)
        srt = np.sort(lb, axis=-1)
        margin = srt[:, -1] - srt[:, -2]
        agree = int((ta == tb).sum())
        big = margin > 0.5
        d = np.abs(la - lb)
        print(f"[{i}] positions {len(ta)} | top1 agree {agree}/{len(ta)} "
              f"(ref margin>0.5: {int((ta == tb)[big].sum())}/{int(big.sum())}) | "
              f"max|dlogit| {d.max():.4f} | mean|dlogit| {d.mean():.5f}")
        tot_agree += agree
        tot_n += len(ta)
        tot_max = max(tot_max, d.max())
    print(f"TOTAL {pa['dtype']} vs {pb['dtype']}: top1 agree {tot_agree}/{tot_n}, max|dlogit| {tot_max:.4f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

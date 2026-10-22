#!/usr/bin/env python3
"""Independent reference for BitNet b1.58 2B4T using HF transformers on CPU.

Run with `python -I` from a directory that is NOT the model download dir.

Outputs (in --out):
  quant_summary.json  per-linear-tensor ternarization summary computed with torch
                      exactly as transformers' WeightQuant does it (scale, counts,
                      sha256 of the int8 ternary values in row-major order)
  prompts.json        token ids for each fixed prompt (+ greedy continuation)
  logits_<i>.npy      float32 logits [seq, vocab] from one teacher-forced forward
                      over prompt+continuation (dtype given by --dtype)
"""
import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # WeightQuant/ActQuant are @torch.compile'd

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    {"kind": "raw", "text": "The capital of France is"},
    {"kind": "raw", "text": "Search and rescue teams on Lake Michigan use drift models to"},
    {"kind": "chat", "messages": [{"role": "user", "content": "Give me three tips for staying safe while kayaking."}]},
    {"kind": "raw", "text": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"},
]

LINEAR_SUFFIXES = (
    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
    "gate_proj.weight", "up_proj.weight", "down_proj.weight",
)


def quant_summary(model_dir: str) -> dict:
    out = {}
    with safe_open(os.path.join(model_dir, "model.safetensors"), "pt") as f:
        for name in sorted(f.keys()):
            if not name.endswith(LINEAR_SUFFIXES):
                continue
            w = f.get_tensor(name).float()
            scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
            t = (w * scale).round().clamp(-1, 1).to(torch.int8).contiguous()
            vals, counts = torch.unique(t, return_counts=True)
            out[name] = {
                "shape": list(w.shape),
                "scale": float(scale),           # f32 value, exact in a python float
                "dequant": float(1.0 / scale),   # magnitude of +-1 after dequant, f32 math
                "counts": {str(int(v)): int(c) for v, c in zip(vals, counts)},
                "sha256": hashlib.sha256(t.numpy().tobytes()).hexdigest(),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--gen", type=int, default=24)
    ap.add_argument("--skip-quant", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(min(16, os.cpu_count() or 4))

    if not a.skip_quant:
        t0 = time.time()
        qs = quant_summary(a.model)
        json.dump(qs, open(os.path.join(a.out, "quant_summary.json"), "w"), indent=1)
        print(f"quant summary: {len(qs)} tensors in {time.time()-t0:.1f}s", flush=True)

    dtype = getattr(torch, a.dtype)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, attn_implementation="eager")
    model.eval()
    print(type(model.model.layers[0].self_attn.q_proj).__name__, "online_quant=",
          getattr(model.model.layers[0].self_attn.q_proj, "online_quant", None), flush=True)

    records = []
    for i, p in enumerate(PROMPTS):
        if p["kind"] == "chat":
            ids = tok.apply_chat_template(p["messages"], add_generation_prompt=True, tokenize=True)
            if isinstance(ids, dict) or hasattr(ids, "keys"):
                ids = ids["input_ids"]
            ids = list(ids)
            if ids[0] != tok.bos_token_id:
                ids = [tok.bos_token_id] + ids
        else:
            ids = tok(p["text"])["input_ids"]
        inp = torch.tensor([ids])
        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(inp, max_new_tokens=a.gen, do_sample=False, temperature=None, top_p=None)
        full = gen[0].tolist()
        with torch.no_grad():
            logits = model(torch.tensor([full])).logits[0].float().numpy()
        np.save(os.path.join(a.out, f"logits_{i}.npy"), logits)
        cont = tok.decode(full[len(ids):])
        print(f"[{i}] {len(ids)}+{len(full)-len(ids)} tokens {time.time()-t0:.1f}s: {cont!r}", flush=True)
        records.append({"prompt": p, "prompt_ids": ids, "full_ids": full, "continuation": cont})
    json.dump({"dtype": a.dtype, "torch": torch.__version__, "records": records},
              open(os.path.join(a.out, "prompts.json"), "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())

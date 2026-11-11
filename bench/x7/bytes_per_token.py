#!/usr/bin/env python3
"""Weight bytes one decode token reads from a LFM2.5 GGUF: every non-expert tensor (the tied token_embd counts once,
as the output head; the embedding row lookup is negligible), plus k of n experts per MoE layer
(ffn_{gate,up,down}_exps are [.., .., n_expert]). usage: bytes_per_token.py file.gguf [k]  -> JSON line"""
import json, sys
sys.path.insert(0, "/data/llama.cpp-ml4/gguf-py")
from gguf import GGUFReader

path = sys.argv[1]; k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
r = GGUFReader(path)
n_exp = None
for f in r.fields.values():
    if f.name.endswith(".expert_count"):
        n_exp = int(f.parts[f.data[0]][0])
dense = experts = 0; types = {}
for t in r.tensors:
    nb = int(t.n_bytes)
    types[t.tensor_type.name] = types.get(t.tensor_type.name, 0) + nb
    if "_exps" in t.name:
        experts += nb
    else:
        dense += nb
per_tok = dense + experts * k / n_exp
print(json.dumps(dict(file=path.split("/")[-1], n_expert=n_exp, k=k, dense_bytes=dense, expert_bytes=experts,
                      bytes_per_token=int(per_tok), file_bytes_by_type=types)))

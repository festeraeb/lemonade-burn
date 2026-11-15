#!/bin/bash
# Bit-exact dequantisation check against llama.cpp: dequant_check.sh <model.gguf> <llama build bin dir> <llama src dir> <out dir>
set -e; M=$1; L=$2; G=$3; O=$4; mkdir -p $O; S=$(dirname $0)
python3 $S/gguf_table.py $M $O/meta.json >$O/types.txt
python3 - $O <<'P'
import json,sys
o=sys.argv[1]; t=json.load(open(o+'/meta.json.tensors')); ids={'F32':0,'F16':1,'Q8_0':8,'Q4_K':12,'Q5_K':13,'Q6_K':14,'BF16':30}
with open(o+'/t.tsv','w') as f:
    for n,ty,sh,nb,off in t:
        nr=1
        for d in sh[1:]: nr*=d
        f.write(f"{n}\t{ids[ty]}\t{sh[0]}\t{off}\t{nr}\n")
P
gcc -O2 -I$G/ggml/include $S/ggml_dequant.c -o $O/ggml_dequant -L$L -lggml-base -Wl,-rpath,$L
$O/ggml_dequant $M $O/t.tsv $O/ggml.hash
cargo run --release -q -p lb-gguf --example dequant_hash -- $M $O/t.tsv $O/lb.hash
cmp $O/ggml.hash $O/lb.hash && echo "IDENTICAL: every row of every tensor"

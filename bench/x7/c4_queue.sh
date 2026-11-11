#!/usr/bin/env bash
# X7q queue on c4-AF: after the Q8_0 held-out -> KLD text -> Q8_0 base -> per quant: KLD, held-out, llama-bench.
set -u
X=/mnt/t440/fast/x7; C=$X/q/c4.sh; MD=$X/models
while pgrep -f "c4.sh heldout Q8_0" >/dev/null || pgrep -f llama-imatrix >/dev/null; do sleep 15; done
[ "$(grep -c '"llama-Q8_0"' $X/results/heldout-llama.jsonl)" -ge 60 ] || { echo "Q8_0 held-out incomplete"; exit 1; }
[ -s $X/q/kltext.txt ] || python3 $X/q/mk_kltext.py
[ -s $X/kl/base-q8.bin ] || $C klbase
for tag in ${TAGS:-QAD-Q4_0 Q4_0 Q4_0-imat Q4_K_M-imat IQ4_XS-imat IQ4_XS-imat-tc IQ3_M-imat IQ2_M-imat IQ1_S-imat}; do
  until [ -s $MD/LFM2.5-24B-A2B-$tag.gguf ]; do sleep 30; done
  echo "== $(date -u +%T) $tag"
  grep -q "Mean    KLD" $X/logs/kl-$tag.log 2>/dev/null || $C kl $tag AF
  [ "$(grep -c "\"llama-$tag\"" $X/results/heldout-llama.jsonl)" -ge 60 ] || $C heldout $tag AF
  grep -q "\"tag\":\"$tag\"" $X/results/llama-bench.jsonl 2>/dev/null || $C bench $tag AF
done
echo "queue done $(date -u +%T)"

#!/usr/bin/env bash
# X7q quants of LFM2.5-24B-A2B from LiquidAI's BF16 GGUF (run on t440, CPU). Idempotent.
# usage: quantize.sh q8            -> Q8_0 (no imatrix)
#        quantize.sh rest          -> Q4_0 (no imatrix), then with models/imatrix-q8.gguf:
#                                     Q4_0-imat, Q4_K_M, IQ4_XS, IQ3_M, IQ2_M, IQ1_S
set -u
X=/mnt/fast/x7; B=$X/llama-build/bin; M=$X/models; SRC=$M/LFM2.5-24B-A2B-BF16.gguf
T=${THREADS:-24}
q() { # out type [imatrix]
  local out=$M/LFM2.5-24B-A2B-$1.gguf ty=$2 im=${3:-}
  [ -s "$out" ] && { echo "have $out"; return 0; }
  local a=(); [ -n "$im" ] && a=(--imatrix "$im")
  echo "== $(date -u +%H:%M:%S) $1"
  nice $B/llama-quantize "${a[@]}" "$SRC" "$out.tmp" "$ty" $T > $X/logs/quantize-$1.log 2>&1 && mv "$out.tmp" "$out" || { echo "FAILED $1"; tail -5 $X/logs/quantize-$1.log; return 1; }
  ls -la "$out"
}
case $1 in
  q8) q Q8_0 Q8_0;;
  rest)
    IM=$M/imatrix-q8.gguf; [ -s "$IM" ] || { echo "no imatrix $IM"; exit 1; }
    q Q4_0 Q4_0
    for t in Q4_0 Q4_K_M IQ4_XS IQ3_M IQ2_M IQ1_S; do q $t-imat $t "$IM"; done;;
esac

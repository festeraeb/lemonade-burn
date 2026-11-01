#!/usr/bin/env bash
# Probe every GGUF we hold and cross-check each quantization against the highest-bit
# quantization of the same model. CPU only: dequantization is CPU work, so this never
# touches a GPU and cannot collide with photogrammetry on t440.
#
# A wrong dequantizer shows as cosine near 0 with relative RMS near 1.4; a correct one
# tracks bit depth. Rows go to TSV so a later pass can sort on the cosine column.
set -uo pipefail
A=${A:-/data/cargo-target/release/lb-addmodel}
OUT=${OUT:-/data/logs/t440/lb-overnight}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
PROBES="$OUT/probes-$STAMP.tsv"
CMPS="$OUT/compare-$STAMP.tsv"
printf 'file\tarch\tmoe\tn_layer\tn_expert\ttop_k\tmtp\tquants\tmissing\n' > "$PROBES"
printf 'model_group\ttensor\tty_a\tty_b\tcosine\trel_rms\tmax_abs\tn\n' > "$CMPS"

mapfile -t FILES < <(find /mnt/t440/raid0/models /mnt/t440/fast/models -name '*.gguf' -size +100M 2>/dev/null | sort)
echo "$(date -uIs) probing ${#FILES[@]} files" >&2

declare -A GROUP_FILES
for f in "${FILES[@]}"; do
  p=$("$A" probe --gguf "$f" 2>/dev/null) || { echo -e "$f\tPROBE_FAILED" >> "$PROBES"; continue; }
  arch=$(sed -n 's/^arch: *\([^ ]*\).*/\1/p' <<<"$p" | head -1)
  moe=$(sed -n 's/.*moe layout: *\([A-Za-z]*\).*/\1/p' <<<"$p" | head -1)
  mtp=$(grep -c 'MTP: true' <<<"$p")
  g=$(awk '$1=="n_layer"{l=$2} $1=="n_expert"{e=$2} $1=="n_expert_used"{k=$2} END{print l"\t"e"\t"k}' <<<"$p")
  quants=$(grep -oE '\b(F32|F16|Q[2-8]_[01K]|Q[2-6]_K|type#[0-9]+)\b' <<<"$p" | sort -u | paste -sd, -)
  missing=$(grep -c 'MISSING' <<<"$p")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$f" "$arch" "$moe" "$g" "$mtp" "$quants" "$missing" >> "$PROBES"
  # group files by (arch, layers, experts) so same-model quants compare
  key="$arch-$(cut -f1 <<<"$g")-$(cut -f2 <<<"$g")"
  GROUP_FILES[$key]="${GROUP_FILES[$key]:-} $f"
done

# within each group, compare every file against the one with the highest-bit quant present
for key in "${!GROUP_FILES[@]}"; do
  read -ra fs <<< "${GROUP_FILES[$key]}"
  [ "${#fs[@]}" -lt 2 ] && continue
  ref=""
  for want in Q6_K Q8_0 Q5_K F16; do
    for f in "${fs[@]}"; do
      if "$A" probe --gguf "$f" 2>/dev/null | grep -q "$want"; then ref="$f"; break 2; fi
    done
  done
  [ -z "$ref" ] && ref="${fs[0]}"
  for f in "${fs[@]}"; do
    [ "$f" = "$ref" ] && continue
    for t in blk.0.ffn_gate_exps.weight blk.0.ffn_down_exps.weight blk.0.ffn_up_exps.weight token_embd.weight; do
      r=$("$A" compare --a "$f" --b "$ref" --tensor "$t" --rows 32 2>/dev/null) || continue
      ta=$(sed -n '1s/.*: \([^ ]*\) vs \([^ ]*\).*/\1/p' <<<"$r"); tb=$(sed -n '1s/.*: \([^ ]*\) vs \([^ ]*\).*/\2/p' <<<"$r")
      cos=$(awk '/cosine/{print $2}' <<<"$r"); rms=$(awk '/rel RMS/{print $3}' <<<"$r")
      mx=$(awk '/max \|diff\|/{print $2}' <<<"$r"); n=$(sed -n '1s/.*(\([0-9]*\) values).*/\1/p' <<<"$r")
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$key|$(basename "$f")" "$t" "$ta" "$tb" "$cos" "$rms" "$mx" "$n" >> "$CMPS"
      echo "$(date -uIs) $key $(basename "$f") $t $ta/$tb cos=$cos" >&2
    done
  done
done

echo "$(date -uIs) done. probes=$PROBES compares=$CMPS" >&2
# anything with a correct dequantizer must beat 0.90; flag the rest loudly
awk -F'\t' 'NR>1 && $5+0 < 0.90 {print "SUSPECT: "$0}' "$CMPS" >&2

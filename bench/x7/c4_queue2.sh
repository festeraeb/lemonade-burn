#!/usr/bin/env bash
# X7b on c4-AF after c4_queue.sh: lemonade-burn running the IQ4_XS file, transcoded to Q4_0 as it becomes resident.
set -u
X=/mnt/t440/fast/x7; MD=$X/models; AF=${UUID_AF:?card UUID}
until grep -q "queue done" $X/logs/c4-queue.log 2>/dev/null; do sleep 30; done
for tag in ${TAGS:-IQ4_XS-imat}; do
  echo "== $(date -u +%T) lb $tag"
  ROUNDS=3 $X/run_abmath.sh $AF 1 c4AF-$tag $MD/LFM2.5-24B-A2B-$tag.gguf f32m,f16
  [ "$(grep -c "\"lb-$tag-f32-c4AF\"" $X/results/heldout-lb.jsonl)" -ge 60 ] || $X/run_heldout_lb.sh $AF 1 f32 lb-$tag-f32-c4AF $MD/LFM2.5-24B-A2B-$tag.gguf
done
echo "queue2 done $(date -u +%T)"

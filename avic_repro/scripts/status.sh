#!/usr/bin/env bash
# Print per-run progress (questions done / total per chunk) for every run under results/.
cd "$(dirname "$0")/.."
for d in results/*_qc*/; do
  n=0; t=0; parts=""
  nc=$(ls -d "$d"question_chunk_* | wc -l)
  for c in "$d"question_chunk_*/; do
    r="$c/results.json"; idx=${c%/}; idx=${idx##*_}
    cur=$(python3 scripts/progress.py "$r" "$idx" "$nc")
    parts="$parts $cur"; a=${cur%%/*}; b=${cur##*/}; n=$((n+a)); t=$((t+b))
  done
  acc=$( [ -f "$d/results.json" ] && python3 -c "import json;print(round(json.load(open('$d/results.json'))['accuracy']['all']*100,1))" || echo "-" )
  echo "$(basename $d): $n/$t  [$parts ]  merged_acc=$acc"
done

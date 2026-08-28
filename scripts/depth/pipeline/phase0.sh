#!/bin/bash
R=/home/haoming/reason_branch_0824; P=/home/haoming/miniconda3/envs/vlm-ex/bin/python; cd $R; L=$R/results/logs/depth; mkdir -p $L
until grep -q ALL_DONE $R/results/logs/newbench/lanes.done 2>/dev/null; do sleep 120; done
echo "[$(date +%H:%M)] phase0 start" >> $L/phase0.status
i=0; for g in 0 1 2 3 5 6 7; do CUDA_VISIBLE_DEVICES=$g $P scripts/depth/build_posebank.py --manifest data/train3r/manifest.jsonl --shard $i --num-shards 7 > $L/posebank_s$i.log 2>&1 & i=$((i+1)); done; wait
echo "[$(date +%H:%M)] phase0 done" >> $L/phase0.status

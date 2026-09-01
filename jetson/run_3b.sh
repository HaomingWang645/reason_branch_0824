#!/bin/zsh
# 3B campaign: wait for download, then full suite + frames scaling until OOM.
export HF_HOME=/mnt/data/hf_cache VIEWTREE_POSES=human TRANSFORMERS_VERBOSITY=error
PY=/home/ubuntu/miniconda3/envs/mosaic-thinker/bin/python
cd /mnt/data/reason_branch_0824
M=Qwen/Qwen2.5-VL-3B-Instruct

until grep -q DONE3B /mnt/data/hf_cache/qwen3b_dl.log 2>/dev/null; do sleep 15; done
echo "=== 3B download ready"

echo "=== STAGE bench_system 3B"
$PY jetson/bench_system.py --model $M --nq 3 --micro-reps 3 \
  --out jetson/results/bench_raw_3b.json || echo "STAGE_FAILED bench_system"

echo "=== STAGE frames_scale 3B base"
$PY jetson/bench_frames_scale.py --model $M --reps 2 \
  --out jetson/results/frames_scale_3b_raw.json || echo "STAGE_FAILED scale_base"

echo "=== STAGE frames_scale 3B ext"
$PY jetson/bench_frames_scale.py --model $M --reps 1 \
  --counts 192,256,384,512,768,1024 \
  --out jetson/results/frames_scale_3b_ext_raw.json || echo "STAGE_FAILED scale_ext"

echo "ALL_DONE_3B"

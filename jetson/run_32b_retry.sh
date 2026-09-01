#!/bin/zsh
# 32B retry with source-built bitsandbytes (sm_87).
export HF_HOME=/mnt/data/hf_cache VIEWTREE_POSES=human TRANSFORMERS_VERBOSITY=error
PY=/home/ubuntu/miniconda3/envs/mosaic-thinker/bin/python
cd /mnt/data/reason_branch_0824
M=Qwen/Qwen2.5-VL-32B-Instruct

echo "=== STAGE bench_system 32B 4bit (retry)"
$PY jetson/bench_system.py --model $M --quant 4bit --nq 2 --micro-reps 2 \
  --out jetson/results/bench_raw_32b.json || echo "STAGE_FAILED bench_system"

echo "=== STAGE frames_scale 32B 4bit (retry)"
$PY jetson/bench_frames_scale.py --model $M --quant 4bit --reps 1 \
  --counts 16,24,32,48,64,96,128,160,192,256 \
  --out jetson/results/frames_scale_32b_raw.json || echo "STAGE_FAILED scale"

echo "ALL_DONE_32B_RETRY"

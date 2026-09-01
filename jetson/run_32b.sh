#!/bin/zsh
# 32B campaign. bf16 (~66 GB) cannot fit in 64 GB unified memory:
# 1) demonstrate the bf16 load failure (feasibility datum),
# 2) run the suite + frames scaling with bnb NF4 4-bit.
# Waits for the weight download AND for the 3B campaign to finish (GPU exclusivity).
export HF_HOME=/mnt/data/hf_cache VIEWTREE_POSES=human TRANSFORMERS_VERBOSITY=error
PY=/home/ubuntu/miniconda3/envs/mosaic-thinker/bin/python
cd /mnt/data/reason_branch_0824
M=Qwen/Qwen2.5-VL-32B-Instruct

until grep -q DONE32B /mnt/data/hf_cache/qwen32b_dl.log 2>/dev/null; do sleep 60; done
echo "=== 32B download ready"
until grep -q ALL_DONE_3B jetson/results/run_3b.log 2>/dev/null; do sleep 60; done
echo "=== GPU free (3B campaign finished)"

echo "=== STAGE bf16 load attempt (expected to fail: 66 GB > 61 GB)"
$PY - <<'EOF'
import sys, time, torch
sys.path.insert(0, "jetson")
t0 = time.time()
try:
    from bench_system import BenchVLM
    v = BenchVLM("Qwen/Qwen2.5-VL-32B-Instruct", quant="none")
    print(f"BF16_LOAD_OK {time.time()-t0:.0f}s")  # would be a surprise
except Exception as ex:
    print(f"BF16_LOAD_FAILED after {time.time()-t0:.0f}s: "
          f"{type(ex).__name__}: {str(ex)[:200]}")
EOF

echo "=== STAGE bench_system 32B 4bit"
$PY jetson/bench_system.py --model $M --quant 4bit --nq 2 --micro-reps 2 \
  --out jetson/results/bench_raw_32b.json || echo "STAGE_FAILED bench_system"

echo "=== STAGE frames_scale 32B 4bit"
$PY jetson/bench_frames_scale.py --model $M --quant 4bit --reps 1 \
  --counts 16,24,32,48,64,96,128,160,192,256 \
  --out jetson/results/frames_scale_32b_raw.json || echo "STAGE_FAILED scale"

echo "ALL_DONE_32B"

#!/usr/bin/env bash
# Evaluate our own Think3D-RL (GRPO) checkpoint exactly like the released SPAgent-4B:
#   1. stop the px256k ablation server (if still up) to free GPU-6 memory
#   2. serve <ckpt_dir> with vLLM on GPU 6 (port 30061) as "Think3D-RL-4B"
#   3. run the full matrix: base + think3d (--rl-trained) x 3 runs x 3 benchmarks
# Usage: bash scripts/eval_own_ckpt.sh <ckpt_dir>
set -uo pipefail
CKPT="$1"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
for p in $(pgrep -f "served-model-name SPAgent-4B-px256[k]"); do kill $p 2>/dev/null; done
sleep 8
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6)" -lt 70000 ]; do sleep 3; done
echo "[own-ckpt] GPU6 used: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 6)"
nohup bash spagent/logs/serve_vllm.sh "$CKPT" Think3D-RL-4B 30061 0.30 6 > spagent/logs/vllm_think3d_rl_4b.log 2>&1 &
until grep -q "Application startup complete" spagent/logs/vllm_think3d_rl_4b.log; do
  grep -q "initialization failed" spagent/logs/vllm_think3d_rl_4b.log && { echo "[own-ckpt] VLLM FAILED"; exit 1; }; sleep 5; done
echo "[own-ckpt] server ready at $(date +%T)"
bash scripts/run_matrix.sh Think3D-RL-4B 30061 --rl-trained > outputs/logs/matrix_Think3D-RL-4B.log 2>&1
echo "[own-ckpt] matrix finished at $(date +%T)"

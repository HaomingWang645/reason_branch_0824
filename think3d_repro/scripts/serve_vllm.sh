#!/usr/bin/env bash
# Usage: bash serve_vllm.sh <model_path> <served_name> <port> <gpu_frac> [gpu_index=6]
source /home/haoming/reason_branch_0824/think3d_repro/scripts/env.sh
source /home/haoming/miniconda3/etc/profile.d/conda.sh
conda activate vllm
export CUDA_VISIBLE_DEVICES=${5:-6}
exec vllm serve "$1" --served-model-name "$2" --port "$3" --host 0.0.0.0 \
  --gpu-memory-utilization "$4" --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 16}' --max-num-seqs 32 \
  --dtype bfloat16 --trust-remote-code --disable-log-requests

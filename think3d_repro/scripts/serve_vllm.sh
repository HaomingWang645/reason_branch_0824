#!/usr/bin/env bash
# Usage: bash serve_vllm.sh <model_path> <served_name> <port> <gpu_frac> [gpu_index=6]
# --mm-processor-cache-gb 0: vLLM 0.11.2 multimodal processor cache crashed the engine core once
#   (AssertionError "Expected a cached item for mm_hash=...") under concurrent multi-image requests.
source /home/haoming/reason_branch_0824/think3d_repro/scripts/env.sh
source /home/haoming/miniconda3/etc/profile.d/conda.sh
conda activate vllm
export CUDA_VISIBLE_DEVICES=${5:-6}
EXTRA_ARGS="${6:-}"   # e.g. --mm-processor-kwargs '{"max_pixels": 262144}'
exec vllm serve "$1" --served-model-name "$2" --port "$3" --host 0.0.0.0 \
  --gpu-memory-utilization "$4" --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 16}' --max-num-seqs 32 \
  --mm-processor-cache-gb 0 \
  --dtype bfloat16 --trust-remote-code --disable-log-requests $EXTRA_ARGS

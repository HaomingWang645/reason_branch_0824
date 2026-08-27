#!/usr/bin/env bash
# After the GPT matrix (phase2) finishes: InternVL3-14B world-model rows, 1 chunk per GPU (InternVL + SVC + Qwen policy ≈ 60 GB).
cd "$(dirname "$0")/.."
while pgrep -f "scripts/phase2.sh" > /dev/null; do sleep 120; done
echo "[phase3] $(date '+%T') starting InternVL3-14B configs"
CONFIGS="avic_r:OpenGVLab/InternVL3-14B avic:OpenGVLab/InternVL3-14B avic_qwen:OpenGVLab/InternVL3-14B" GPUS="6 7" CHUNKS_PER_GPU=1 bash scripts/run_matrix.sh

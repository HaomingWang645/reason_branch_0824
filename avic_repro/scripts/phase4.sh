#!/usr/bin/env bash
# Queue after the gpt-4o avic_qwen run: repair its crashed chunk on GPU 5 while the GPT-4.1 configs run on GPUs 6/7,
# then the InternVL3-14B world-model configs one chunk per GPU on 5/6/7.
cd "$(dirname "$0")/.."
while pgrep -f "scripts/run_avic.sh" > /dev/null; do sleep 60; done
echo "[phase4] $(date '+%T') gpt-4o avic_qwen main pass finished"
( RUN=gpt-4o_avic_qwen MODE=avic_qwen BACKBONE=gpt-4o GPUS="5" NUM_CHUNKS=4 bash scripts/fix_chunks.sh ) > logs/fix_gpt-4o_avic_qwen.log 2>&1 &
fixpid=$!
CONFIGS="avic_r:gpt-4.1 avic:gpt-4.1 avic_qwen:gpt-4.1" GPUS="6 7 7" bash scripts/run_matrix2.sh
wait $fixpid
echo "[phase4] $(date '+%T') starting InternVL3-14B configs"
CONFIGS="avic_r:OpenGVLab/InternVL3-14B avic:OpenGVLab/InternVL3-14B avic_qwen:OpenGVLab/InternVL3-14B" GPUS="5 7" bash scripts/run_matrix2.sh
echo "[phase4] ALL DONE"

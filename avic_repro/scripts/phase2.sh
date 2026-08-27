#!/usr/bin/env bash
# Wait for the running gpt-4o AVIC-R job, repair its crashed chunks, then run the rest of the GPT matrix at 2 chunks/GPU.
cd "$(dirname "$0")/.."
while pgrep -f "question_chunk_idx" > /dev/null; do sleep 60; done
echo "[phase2] $(date '+%T') previous run finished; fixing gpt-4o_avic_r chunks"
RUN=gpt-4o_avic_r MODE=avic_r BACKBONE=gpt-4o GPUS="6 7" NUM_CHUNKS=6 bash scripts/fix_chunks.sh
CONFIGS="avic:gpt-4o avic_qwen:gpt-4o avic_r:gpt-4.1 avic:gpt-4.1 avic_qwen:gpt-4.1" GPUS="6 7" CHUNKS_PER_GPU=2 bash scripts/run_matrix.sh

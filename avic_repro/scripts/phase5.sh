#!/usr/bin/env bash
# After the InternVL configs (phase4) finish: MindJourney always-on row for GPT-4o on GPUs 5/7/7/6.
cd "$(dirname "$0")/.."
while pgrep -f "scripts/phase4.sh" > /dev/null; do sleep 60; done
echo "[phase5] $(date '+%T') starting MindJourney gpt-4o"
BACKBONE=gpt-4o GPUS="5 7 7 6" CHUNKS_PER_GPU=1 bash scripts/run_mindjourney.sh
echo "[phase5] done"

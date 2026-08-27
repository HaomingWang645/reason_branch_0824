#!/usr/bin/env bash
# Sequentially run a list of "MODE:BACKBONE" configs on the given GPUs (Table 1 reproduction).
#   CONFIGS="avic_r:gpt-4o avic:gpt-4o" GPUS="6 7" CHUNKS_PER_GPU=3 bash scripts/run_matrix.sh
set -u
cd "$(dirname "$0")/.."
GPUS=${GPUS:-"5 6 6 7 7"}; CHUNKS_PER_GPU=1
for cfg in $CONFIGS; do
  mode=${cfg%%:*}; bb=${cfg#*:}
  echo "[matrix] $(date '+%F %T') start $mode $bb"
  MODE=$mode BACKBONE=$bb GPUS="$GPUS" CHUNKS_PER_GPU=$CHUNKS_PER_GPU bash scripts/run_avic2.sh
  # re-run any chunk that crashed (OOM etc.); no-op when everything completed
  read -r -a _g <<< "$GPUS"; nchunks=${#_g[@]}
  bb_tag=$(echo "$bb" | sed 's#.*/##; s#\.##g')
  RUN="${bb_tag}_${mode}" MODE=$mode BACKBONE=$bb GPUS="$GPUS" NUM_CHUNKS=$nchunks bash scripts/fix_chunks.sh
  echo "[matrix] $(date '+%F %T') done  $mode $bb"
done
echo "[matrix] ALL DONE"

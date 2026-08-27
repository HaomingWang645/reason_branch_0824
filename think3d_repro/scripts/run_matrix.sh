#!/usr/bin/env bash
# Launch the full evaluation matrix for one model:
#   settings (base, think3d) x runs (1..3) x datasets (MindCube, BLINK, VSIBench)
# Datasets run in parallel processes; runs are sequential within a dataset.
# Usage: bash scripts/run_matrix.sh <served_model> <port> [--rl-trained] [settings="base think3d"] [runs="1 2 3"]
set -uo pipefail
MODEL="$1"; PORT="$2"; shift 2
EXTRA=()
if [ "${1:-}" = "--rl-trained" ]; then EXTRA=(--rl-trained); shift; fi
SETTINGS="${1:-base think3d}"; RUNS="${2:-1 2 3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/outputs/logs"
for SETTING in $SETTINGS; do
  for DS in MindCube BLINK VSIBench; do
    (
      for RUN in $RUNS; do
        LOG="$ROOT/outputs/logs/${MODEL}_${SETTING}_run${RUN}_${DS}.log"
        echo "[matrix] start $MODEL $SETTING run$RUN $DS -> $LOG"
        if [ "$SETTING" = "think3d" ]; then
          bash "$ROOT/scripts/run_eval.sh" "$MODEL" "$PORT" "$SETTING" "$RUN" "$DS" "${EXTRA[@]}" > "$LOG" 2>&1
        else
          bash "$ROOT/scripts/run_eval.sh" "$MODEL" "$PORT" "$SETTING" "$RUN" "$DS" > "$LOG" 2>&1
        fi
        echo "[matrix] done  $MODEL $SETTING run$RUN $DS (exit $?)"
      done
    ) &
  done
done
wait
echo "[matrix] ALL DONE for $MODEL"

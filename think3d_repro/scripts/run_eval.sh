#!/usr/bin/env bash
# ============================================================
#  Think3D reproduction — evaluation driver
#
#  Usage:
#    bash scripts/run_eval.sh <served_model> <port> <setting> <run_id> <dataset> [extra quick_eval args]
#
#    served_model : vLLM served model name (Qwen3-VL-4B-Instruct | SPAgent-4B | Think3D-RL-4B)
#    port         : vLLM port (30058 / 30059 / 30060)
#    setting      : base    -> no tools, general prompt, 1 iteration   (paper "baseline" rows)
#                   think3d -> pi3x tool, spatial prompt, 3 iterations (paper "Think3D (...)" rows)
#    run_id       : 1|2|3   (paper reports the mean of 3 runs, temperature 1.0)
#    dataset      : MindCube | BLINK | VSIBench
#
#  Paper protocol (Sec 4.1 / App. C): MindCube 40 q / category (120), BLINK all multi-view
#  questions (133, val), VSI-Bench-tiny 4 MC tasks x 50 q with 7 frames / video,
#  temperature 1.0, max 3 interaction rounds, results averaged over 3 runs.
# ============================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source /home/haoming/miniconda3/etc/profile.d/conda.sh
conda activate spagent
cd "$SPAGENT_ROOT"

MODEL="$1"; PORT="$2"; SETTING="$3"; RUN="$4"; DS="$5"; shift 5
export VLLM_BASE_URL="http://localhost:${PORT}/v1"
export SPAGENT_MAX_TOKENS=1024
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"          # never used (no judge, local scoring)

case "$RUN" in [0-9]*) SEED=$((41 + RUN)) ;; *) SEED=42 ;; esac
WORK_DIR="$REPRO_ROOT/outputs/run${RUN}/vlmeval_runs"
TRACE_DIR="$REPRO_ROOT/outputs/run${RUN}/spagent_traces"
mkdir -p "$WORK_DIR" "$TRACE_DIR"

COMMON=(--model "$MODEL" --model-backend qwen-vllm
        --temperature 1.0 --seed "$SEED"
        --work-dir "$WORK_DIR" --trace-dir "$TRACE_DIR"
        --mindcube-path dataset/MindCube_eval120.jsonl
        --vsibench-path dataset/VSI_Bench_tiny.jsonl
        --num-video-frames 7
        --judge-model exact_matching --no-score)

case "$SETTING" in
  base)    ARGS=(--prompt general --max-iterations 1) ;;
  think3d) ARGS=(--prompt spatial --tools pi3x --pi3x-url http://localhost:20031 --max-iterations 3) ;;
  *) echo "unknown setting $SETTING"; exit 1 ;;
esac

case "$DS" in
  BLINK) DSARGS=(--datasets BLINK --task-filter Multi-view_Reasoning --limit 1000) ;;
  MindCube) DSARGS=(--datasets MindCube) ;;
  VSIBench) DSARGS=(--datasets VSIBench) ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac

echo "[run_eval] model=$MODEL setting=$SETTING run=$RUN dataset=$DS extra=$*"
python scripts/quick_eval.py "${COMMON[@]}" "${ARGS[@]}" "${DSARGS[@]}" "$@"

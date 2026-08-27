#!/usr/bin/env bash
# Run one Table-1 configuration of AVIC on SAT-Real (150 test questions), split over GPUs.
#
#   MODE=baseline|avic|avic_qwen|avic_r  BACKBONE=gpt-4o|gpt-4.1|o1|OpenGVLab/InternVL3-14B \
#   GPUS="6 7" CHUNKS_PER_GPU=2 bash scripts/run_avic.sh
#
# baseline  : no world model, direct QA (pipeline_baseline.py)
# avic      : training-free AVIC, backbone is policy + verifier + QA (pipeline_avic.py, --policy_model_type gpt)
# avic_qwen : AVIC with zero-shot Qwen2.5-VL-7B policy, backbone as verifier + QA
# avic_r    : AVIC-R, released GRPO LoRA adapter_step140 on Qwen2.5-VL-7B as policy
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$(dirname "$0")/env.sh"
MODE=${MODE:-avic}
BACKBONE=${BACKBONE:-gpt-4o}
GPUS=${GPUS:-"5 6 6 7 7"}
CHUNKS_PER_GPU=1   # one chunk per GPU slot; repeat ids in GPUS for more chunks per GPU
NUM_QUESTIONS=${NUM_QUESTIONS:-150}
TAG=${TAG:-}
read -r -a gpu_arr <<< "$GPUS"
num_chunks=$(( ${#gpu_arr[@]} * CHUNKS_PER_GPU ))
bb_tag=$(echo "$BACKBONE" | sed 's#.*/##; s#\.##g')
run_name="${bb_tag}_${MODE}${TAG}"
out_dir="$AVIC_ROOT/results/${run_name}"
log_dir="$AVIC_ROOT/logs/${run_name}"; mkdir -p "$log_dir"
echo "[run_avic] $run_name -> $out_dir  (chunks=$num_chunks gpus=$GPUS)"

policy_args=""
case $MODE in
  avic)      policy_args="--policy_model_type gpt" ;;
  avic_qwen) policy_args="--policy_model_type qwen2.5vl --policy_model_name Qwen/Qwen2.5-VL-7B-Instruct" ;;
  avic_r)    policy_args="--policy_model_type qwen2.5vl --policy_model_name Qwen/Qwen2.5-VL-7B-Instruct --policy_lora_ckpt $AVIC_ROOT/checkpoints/AVIC-Qwen2.5-VL-7B-policy" ;;
  baseline)  ;;
  *) echo "bad MODE $MODE"; exit 1 ;;
esac
common="--vlm_model_name=$BACKBONE --vlm_qa_model_name=None --num_questions $NUM_QUESTIONS \
  --output_dir $out_dir --input_dir data --question_type None --max_images 2 --max_tries_gpt 4 --split test \
  --num_question_chunks $num_chunks --task img2trajvid_s-prob --replace_or_include_input True --cfg 4.0 \
  --guider 1 --L_short 576 --num_targets 8 --use_traj_prior True --chunk_strategy interp"
if [ "$MODE" = baseline ]; then
  script=pipelines/pipeline_baseline.py
  extra="--max_steps_per_question 1"
else
  script=pipelines/pipeline_avic.py
  # settings from README 'Eval setting' / scripts/inference_avic_rl_parallel.sh
  extra="--scaling_strategy spatial_beam_search --helpful_score_threshold 8 --exploration_score_threshold 8 \
    --sampling_interval_angle 9 --sampling_interval_meter 0.25 --fixed_rotation_magnitudes 27 --fixed_forward_magnitudes 0.75 \
    --max_steps_per_question 3 --num_top_candidates 6 --num_beams 3 --num_frames 9 --frame_interval 3 --max_inference_batch_size 1 \
    --num_policy_samples 5 --max_wm_candidates 5 --max_action_ids_cap 6 \
    --policy_temperature 0.7 --policy_top_p 1.0 --policy_max_new_tokens 512 $policy_args"
fi
pids=()
for ((c=0; c<num_chunks; c++)); do
  gpu=${gpu_arr[$(( c % ${#gpu_arr[@]} ))]}
  log="$log_dir/chunk_${c}_gpu${gpu}.log"
  echo "  chunk $c -> GPU $gpu  ($log)"
  CUDA_VISIBLE_DEVICES=$gpu python $script $common $extra --question_chunk_idx $c > "$log" 2>&1 &
  pids+=($!)
  sleep 20   # stagger model loading
done
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
qc_dir="${out_dir}$( [ "$MODE" = baseline ] && echo "" || echo "_spatial_beam_search")_qc${num_chunks}"
if [ "$num_chunks" -gt 1 ]; then
  python tools/aggregate_chunks.py "$qc_dir" --csv --label "$run_name" | tee -a "$AVIC_ROOT/results/summary.csv"
else
  python -c "import json,sys; d=json.load(open(sys.argv[1]+'/results.json')); print(d['current'], d['accuracy'])" "${out_dir}$( [ "$MODE" = baseline ] && echo "" || echo "_spatial_beam_search")"
fi
echo "[run_avic] finished $run_name fail=$fail"

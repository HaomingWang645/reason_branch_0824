#!/usr/bin/env bash
# Re-run chunks of a finished run that did not process all their questions (crashed chunks).
# The pipeline resumes from results.json, so this is idempotent.
#   RUN=gpt-4o_avic_r MODE=avic_r BACKBONE=gpt-4o GPUS="6 7" NUM_CHUNKS=6 bash scripts/fix_chunks.sh
set -u
source "$(dirname "$0")/env.sh"
RUN=${RUN:?}; MODE=${MODE:?}; BACKBONE=${BACKBONE:?}; GPUS=${GPUS:-"6 7"}; NUM_CHUNKS=${NUM_CHUNKS:-6}
read -r -a gpu_arr <<< "$GPUS"
out_dir="$AVIC_ROOT/results/${RUN}"
suffix=$( [ "$MODE" = baseline ] && echo "" || echo "_spatial_beam_search")
qc_dir="${out_dir}${suffix}_qc${NUM_CHUNKS}"
policy_args=""
case $MODE in
  avic)      policy_args="--policy_model_type gpt" ;;
  avic_qwen) policy_args="--policy_model_type qwen2.5vl --policy_model_name Qwen/Qwen2.5-VL-7B-Instruct" ;;
  avic_r)    policy_args="--policy_model_type qwen2.5vl --policy_model_name Qwen/Qwen2.5-VL-7B-Instruct --policy_lora_ckpt $AVIC_ROOT/checkpoints/AVIC-Qwen2.5-VL-7B-policy" ;;
esac
common="--vlm_model_name=$BACKBONE --vlm_qa_model_name=None --num_questions 150 \
  --output_dir $out_dir --input_dir data --question_type None --max_images 2 --max_tries_gpt 4 --split test \
  --num_question_chunks $NUM_CHUNKS --task img2trajvid_s-prob --replace_or_include_input True --cfg 4.0 \
  --guider 1 --L_short 576 --num_targets 8 --use_traj_prior True --chunk_strategy interp"
if [ "$MODE" = baseline ]; then script=pipelines/pipeline_baseline.py; extra="--max_steps_per_question 1"; else
  script=pipelines/pipeline_avic.py
  extra="--scaling_strategy spatial_beam_search --helpful_score_threshold 8 --exploration_score_threshold 8 \
    --sampling_interval_angle 9 --sampling_interval_meter 0.25 --fixed_rotation_magnitudes 27 --fixed_forward_magnitudes 0.75 \
    --max_steps_per_question 3 --num_top_candidates 6 --num_beams 3 --num_frames 9 --frame_interval 3 --max_inference_batch_size 1 \
    --num_policy_samples 5 --max_wm_candidates 5 --max_action_ids_cap 6 \
    --policy_temperature 0.7 --policy_top_p 1.0 --policy_max_new_tokens 512 $policy_args"
fi
pids=(); i=0
for ((c=0; c<NUM_CHUNKS; c++)); do
  r="$qc_dir/question_chunk_$c/results.json"
  cur=$(python $AVIC_ROOT/scripts/progress.py "$r" "$c" "$NUM_CHUNKS")
  if [ "${cur%%/*}" = "${cur##*/}" ]; then continue; fi
  gpu=${gpu_arr[$(( i % ${#gpu_arr[@]} ))]}; i=$((i+1))
  log="$AVIC_ROOT/logs/${RUN}/chunk_${c}_gpu${gpu}_fix.log"
  echo "[fix_chunks] rerun chunk $c on GPU $gpu ($log)"
  CUDA_VISIBLE_DEVICES=$gpu python $script $common $extra --question_chunk_idx $c > "$log" 2>&1 &
  pids+=($!); sleep 30
done
for p in "${pids[@]:-}"; do [ -n "$p" ] && wait $p; done
python tools/aggregate_chunks.py "$qc_dir" --csv --label "$RUN" | tee -a "$AVIC_ROOT/results/summary.csv"

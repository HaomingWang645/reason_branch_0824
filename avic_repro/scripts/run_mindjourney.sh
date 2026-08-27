#!/usr/bin/env bash
# "+ MindJourney" rows of Table 1: always-on SVC spatial beam search from the MindJourney repo,
# with the SAT-test settings of mindjourney/scripts/inference_pipeline_svc_scaling_parallel_sat_test.sh.
#   BACKBONE=gpt-4o GPUS="6 7" CHUNKS_PER_GPU=3 bash scripts/run_mindjourney.sh
set -u
source "$(dirname "$0")/env.sh"
cd $AVIC_ROOT/mindjourney
export PYTHONPATH=$AVIC_ROOT/mindjourney:${PYTHONPATH:-}
export QUESTION_DATASET_TYPE="SAT_test"
export NUM_OF_FRAMES=20
BACKBONE=${BACKBONE:-gpt-4o}; GPUS=${GPUS:-"6 7"}; CHUNKS_PER_GPU=${CHUNKS_PER_GPU:-3}; NUM_QUESTIONS=${NUM_QUESTIONS:-150}; TAG=${TAG:-}
read -r -a gpu_arr <<< "$GPUS"; num_chunks=$(( ${#gpu_arr[@]} * CHUNKS_PER_GPU ))
bb_tag=$(echo "$BACKBONE" | sed 's#.*/##; s#\.##g'); run_name="${bb_tag}_mindjourney${TAG}"
out_dir="$AVIC_ROOT/results/${run_name}"; log_dir="$AVIC_ROOT/logs/${run_name}"; mkdir -p "$log_dir"
echo "[run_mindjourney] $run_name -> $out_dir (chunks=$num_chunks gpus=$GPUS)"
pids=()
for ((c=0; c<num_chunks; c++)); do
  gpu=${gpu_arr[$(( c % ${#gpu_arr[@]} ))]}; log="$log_dir/chunk_${c}_gpu${gpu}.log"
  echo "  chunk $c -> GPU $gpu ($log)"
  CUDA_VISIBLE_DEVICES=$gpu python pipelines/pipeline_svc_scaling_spatial_beam_search.py \
    --vlm_model_name $BACKBONE --vlm_qa_model_name None --num_questions $NUM_QUESTIONS \
    --output_dir $out_dir --input_dir data --scaling_strategy beam_search_double_rank --question_type None \
    --helpful_score_threshold 8 --exploration_score_threshold 8 --max_images 2 \
    --sampling_interval_angle 3 --sampling_interval_meter 0.25 --fixed_rotation_magnitudes 27 --fixed_forward_magnitudes 0.75 \
    --max_steps_per_question 3 --num_top_candidates 18 --num_beams 2 --max_tries_gpt 5 \
    --num_frames $((NUM_OF_FRAMES+1)) --frame_interval 3 --max_inference_batch_size 1 --split test \
    --num_question_chunks $num_chunks --question_chunk_idx $c \
    --task img2trajvid_s-prob --replace_or_include_input True --cfg 4.0 --guider 1 --L_short 576 \
    --num_targets 8 --use_traj_prior True --chunk_strategy interp > "$log" 2>&1 &
  pids+=($!); sleep 20
done
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
qc_dir="${out_dir}_beam_search_double_rank_qc${num_chunks}"
python $AVIC_SRC/tools/aggregate_chunks.py "$qc_dir" --csv --label "$run_name" | tee -a "$AVIC_ROOT/results/summary.csv"
echo "[run_mindjourney] finished $run_name fail=$fail"

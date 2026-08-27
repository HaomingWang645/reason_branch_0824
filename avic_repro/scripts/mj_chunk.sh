#!/usr/bin/env bash
# Run (or resume) a single MindJourney chunk: BACKBONE=gpt-4.1 NUM_CHUNKS=4 CHUNK=0 GPU=7 bash scripts/mj_chunk.sh
set -u
source "$(dirname "$0")/env.sh"; cd $AVIC_ROOT/mindjourney
export PYTHONPATH=$AVIC_ROOT/mindjourney:${PYTHONPATH:-}; export QUESTION_DATASET_TYPE="SAT_test"; export NUM_OF_FRAMES=20
bb_tag=$(echo "$BACKBONE" | sed 's#.*/##; s#\.##g'); run_name="${bb_tag}_mindjourney"; out_dir="$AVIC_ROOT/results/${run_name}"
log="$AVIC_ROOT/logs/${run_name}/chunk_${CHUNK}_gpu${GPU}_fix.log"
CUDA_VISIBLE_DEVICES=$GPU python pipelines/pipeline_svc_scaling_spatial_beam_search.py \
    --vlm_model_name $BACKBONE --vlm_qa_model_name None --num_questions 150 \
    --output_dir $out_dir --input_dir data --scaling_strategy beam_search_double_rank --question_type None \
    --helpful_score_threshold 8 --exploration_score_threshold 8 --max_images 2 \
    --sampling_interval_angle 3 --sampling_interval_meter 0.25 --fixed_rotation_magnitudes 27 --fixed_forward_magnitudes 0.75 \
    --max_steps_per_question 3 --num_top_candidates 18 --num_beams 2 --max_tries_gpt 5 \
    --num_frames $((NUM_OF_FRAMES+1)) --frame_interval 3 --max_inference_batch_size 1 --split test \
    --num_question_chunks $NUM_CHUNKS --question_chunk_idx $CHUNK \
    --task img2trajvid_s-prob --replace_or_include_input True --cfg 4.0 --guider 1 --L_short 576 \
    --num_targets 8 --use_traj_prior True --chunk_strategy interp > "$log" 2>&1
echo "[mj_chunk] chunk $CHUNK done"

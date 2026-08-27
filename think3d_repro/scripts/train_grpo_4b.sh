#!/usr/bin/env bash
# ============================================================
#  Think3D-RL reproduction: GRPO on Qwen3-VL-4B-Instruct (single GPU 5)
#
#  Follows spagent/train/train_grpo_4b.sh + paper App. C:
#    1 epoch over the 977-sample crossviewQA RL set (976 here: 24 samples with
#    images missing from the public MindCube release are dropped),
#    8 rollouts / prompt, lr 1e-6 cosine w/ 5% warmup, max_completion_length 1024,
#    rewards = accuracy + multi-turn format + zero-angle penalty (equal weights),
#    max 3 tool-calling turns, offline Pi3X point-cloud cache for rendering.
#  Paper used 8xH200 with per-GPU batch 8 x grad-accum 4 (= 256 completions /
#  optimizer step = 32 prompts). Here: 1 GPU x batch 4 x grad-accum 32
#  = 128 completions / step = 16 prompts / step (61 optimizer steps);
#  rollouts are generated 32 at a time (generation_batch_size) to fit HF-generate prefill in memory.
#  DeepSpeed ZeRO-2 with optimizer CPU offload keeps the full-parameter
#  fine-tune inside one 95 GB GPU without changing the optimization.
# ============================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source /home/haoming/miniconda3/etc/profile.d/conda.sh
conda activate spagent
cd "$SPAGENT_ROOT"

export PI3X_CACHE_DIR="$SPAGENT_ROOT/dataset/pi3x_cache"
export PI3X_OFFLINE_MAX_POINTS=150000   # render budget per view (see pi3x_offline_tool.py)
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=${GPU:-5}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=$(ls -d /home/haoming/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/*)
OUT="$SPAGENT_ROOT/output/grpo_think3d_4b"

MAX_PIXELS=262144 \
MASTER_PORT=29611 \
NPROC_PER_NODE=1 \
swift rlhf \
    --rlhf_type grpo \
    --model "$MODEL" \
    --external_plugins plugin/plugin.py \
    --multi_turn_scheduler spagent_tool_call_scheduler \
    --max_turns 3 \
    --reward_funcs external_r1v_acc external_multiturn_format external_angle_penalty \
    --reward_weights 1.0 1.0 1.0 \
    --tuner_type full \
    --freeze_vit true \
    --torch_dtype bfloat16 \
    --dataset dataset/crossviewQA_train_rl_fixed.jsonl \
    --load_from_cache_file true \
    --max_completion_length 1024 \
    --max_length 32768 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine \
    --gradient_accumulation_steps 64 \
    --save_strategy steps \
    --save_steps 20 \
    --save_total_limit 4 \
    --logging_steps 1 \
    --output_dir "$OUT" \
    --warmup_ratio 0.05 \
    --num_generations 8 \
    --generation_batch_size 32 \
    --vit_gradient_checkpointing false \
    --temperature 1.0 \
    --system train/system_prompt/system_prompt_grpo.txt \
    --log_completions true \
    --report_to tensorboard \
    --num_iterations 1 \
    --dataloader_num_workers 4 \
    --beta 0.001 \
    --deepspeed zero2_offload \
    --max_grad_norm 0.5 \
    --truncation_strategy left \
    --gradient_checkpointing true \
    "$@"

#!/bin/bash
R=/home/haoming/reason_branch_0824; P=/home/haoming/miniconda3/envs/vlm-ex/bin/python; TR=/home/haoming/miniconda3/envs/vlm-ex/bin/torchrun; cd $R
export CUDA_DEVICE_ORDER=PCI_BUS_ID; L=$R/results/logs/scalevlm; O=$R/results/scalevlm
M3=Qwen/Qwen2.5-VL-3B-Instruct; M32=Qwen/Qwen2.5-VL-32B-Instruct
echo "[$(date +%H:%M)] baselines start" >> $L/baselines.status
# 3B SFT-plain (2 ranks) and 32B SFT-plain attempt (1 rank) in parallel
DSE_GPUS=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $TR --nproc_per_node=2 --master_port=29711 scripts/train_sft.py --model $M3 --data data/sft_plain.jsonl --out checkpoints/scalevlm/sft_plain_3b --epochs 1 --lr 1e-4 --accum 16 > $L/sft_plain_3b.log 2>&1 &
DSE_GPUS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $TR --nproc_per_node=1 --master_port=29712 scripts/train_sft.py --model $M32 --data data/sft_plain.jsonl --out checkpoints/scalevlm/sft_plain_32b --epochs 1 --lr 1e-4 --accum 32 > $L/sft_plain_32b.log 2>&1 &
wait
echo "[$(date +%H:%M)] sft done: 3b $(grep -ac '^SAVED' $L/sft_plain_3b.log) 32b $(grep -ac '^SAVED' $L/sft_plain_32b.log)" >> $L/baselines.status
# evals + 3B GRPO-plain
CUDA_VISIBLE_DEVICES=0 $P scripts/run_eval.py --condition frames16 --model $M3 --adapter checkpoints/scalevlm/sft_plain_3b --parity odd --out $O/frames16_3b_sftplain.jsonl > $L/f16_3b_sftplain.log 2>&1 &
if grep -aq '^SAVED' $L/sft_plain_32b.log; then
  CUDA_VISIBLE_DEVICES=4,5 VIEWTREE_DEVICE_MAP=auto $P scripts/run_eval.py --condition frames16 --model $M32 --adapter checkpoints/scalevlm/sft_plain_32b --parity odd --out $O/frames16_32b_sftplain.jsonl > $L/f16_32b_sftplain.log 2>&1 &
fi
DSE_GPUS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $TR --nproc_per_node=1 --master_port=29713 scripts/train_grpo_plain.py --model $M3 --init checkpoints/scalevlm/sft_plain_3b --out checkpoints/scalevlm/grpo_plain_3b > $L/grpo_plain_3b.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=1 $P scripts/run_eval.py --condition frames16 --model $M3 --adapter checkpoints/scalevlm/grpo_plain_3b --parity odd --out $O/frames16_3b_grpoplain.jsonl > $L/f16_3b_grpoplain.log 2>&1
echo "[$(date +%H:%M)] BASELINES_DONE" >> $L/baselines.status

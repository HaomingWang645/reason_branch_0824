#!/bin/bash
# GRPO walks resumed on GPUs 4,5,7 (0-3 released; 6 in use by another user). Budget 30k items, first 9.1k already done.
R=/home/haoming/reason_branch_0824; P=/home/haoming/miniconda3/envs/vlm-ex/bin/python; TR=/home/haoming/miniconda3/envs/vlm-ex/bin/torchrun; cd $R
export CUDA_DEVICE_ORDER=PCI_BUS_ID; L=$R/results/logs/depth; st(){ echo "[$(date +%H:%M)] $1" >> $L/chain.status; }
st "phase3 GRPO walks (resumed on GPUs 4,5,7; 30k budget, skip 0.304)"
DSE_GPUS=4,5,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $TR --nproc_per_node=3 --master_port=29673 scripts/depth/train_grpo_walk.py --init checkpoints/depth/grpo_walk --out checkpoints/depth/grpo_walk --items 30000 --skip-frac 0.304 --curriculum 0 --per-scene 50 --group 6 --accum-items 8 > $L/train_grpo_walk2.log 2>&1
grep -aq "^SAVED" $L/train_grpo_walk2.log || { st "GRPO FAILED"; exit 1; }
st "CHAIN_DONE"

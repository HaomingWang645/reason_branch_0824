#!/bin/bash
# ViewTree-D chain from the value-head stage on GPUs 0-5 (6 and 7 hold other users' jobs).
R=/home/haoming/reason_branch_0824; P=/home/haoming/miniconda3/envs/vlm-ex/bin/python; TR=/home/haoming/miniconda3/envs/vlm-ex/bin/torchrun; cd $R
export CUDA_DEVICE_ORDER=PCI_BUS_ID; G=0,1,2,3,4,5; NP=6; L=$R/results/logs/depth; st(){ echo "[$(date +%H:%M)] $1" >> $L/chain.status; }
st "phase2 value head (GPU 0)"
CUDA_VISIBLE_DEVICES=0 $P scripts/depth/train_value_head.py > $L/train_value_head.log 2>&1
grep -aq "^SAVED" $L/train_value_head.log || st "value head FAILED (continuing; inference can fall back to conf_head_v2_human)"
st "phase2b train SFT-C (6 GPUs)"
DSE_GPUS=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $TR --nproc_per_node=$NP --master_port=29662 scripts/train_sft.py --data data/train3r/phase2.jsonl --init checkpoints/depth/sft_a --out checkpoints/depth/sft_c --epochs 1 --lr 5e-5 --accum 16 > $L/train_sft_c.log 2>&1
grep -aq "^SAVED" $L/train_sft_c.log || { st "SFT-C FAILED"; exit 1; }
st "phase3 GRPO walks (6 GPUs)"
DSE_GPUS=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $TR --nproc_per_node=$NP --master_port=29663 scripts/depth/train_grpo_walk.py --init checkpoints/depth/sft_c --out checkpoints/depth/grpo_walk --items 60000 --per-scene 50 --group 6 --accum-items 8 > $L/train_grpo_walk.log 2>&1
grep -aq "^SAVED" $L/train_grpo_walk.log || { st "GRPO FAILED"; exit 1; }
st "CHAIN_DONE"

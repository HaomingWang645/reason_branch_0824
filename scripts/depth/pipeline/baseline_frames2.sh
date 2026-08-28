#!/bin/bash
R=/home/haoming/reason_branch_0824; P=/home/haoming/miniconda3/envs/vlm-ex/bin/python; TR=/home/haoming/miniconda3/envs/vlm-ex/bin/torchrun; cd $R; L=$R/results/logs/depth
export CUDA_DEVICE_ORDER=PCI_BUS_ID
echo "[$(date +%H:%M)] frames baseline resumed on GPUs 6,7" >> $L/baseline.status
DSE_GPUS=6,7 $TR --nproc_per_node=2 --master_port=29634 scripts/train_sft.py --data data/train3r/frames_only.jsonl --init checkpoints/depth/sft_frames --skip-frac 0.24 --out checkpoints/depth/sft_frames --epochs 1 --lr 8e-5 --accum 16 > $L/train_sft_frames_resume.log 2>&1
grep -aq "^SAVED" $L/train_sft_frames_resume.log && echo "[$(date +%H:%M)] frames baseline DONE" >> $L/baseline.status || echo "[$(date +%H:%M)] frames baseline FAILED" >> $L/baseline.status

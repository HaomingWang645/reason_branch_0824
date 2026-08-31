#!/bin/bash
R=/home/haoming/reason_branch_0824; P=/home/haoming/miniconda3/envs/vlm-ex/bin/python; cd $R; export CUDA_DEVICE_ORDER=PCI_BUS_ID VIEWTREE_POSES=human
O=$R/results/depth; L=$R/results/logs/depth_eval; A=checkpoints/depth/grpo_walk_interim5000
echo "[$(date +%H:%M)] interim grpo eval start" >> $L/grpo_interim.status
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i $P scripts/depth/run_tree_d.py --adapter $A --value-head checkpoints/depth/value_head.pt --parity odd --shard $i --num-shards 4 --out $O/treeD_grpoI_s$i.jsonl > $L/treeD_grpoI_s$i.log 2>&1 & done
wait
CUDA_VISIBLE_DEVICES=0 $P scripts/run_eval.py --condition frames16 --adapter $A --parity odd --out $O/frames16_grpoI.jsonl > $L/frames_grpoI.log 2>&1
echo "[$(date +%H:%M)] GRPO_INTERIM_DONE" >> $L/grpo_interim.status

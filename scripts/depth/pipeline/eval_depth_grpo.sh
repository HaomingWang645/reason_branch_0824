#!/bin/bash
R=/home/haoming/reason_branch_0824; P=/home/haoming/miniconda3/envs/vlm-ex/bin/python; cd $R; export CUDA_DEVICE_ORDER=PCI_BUS_ID VIEWTREE_POSES=human
O=$R/results/depth; L=$R/results/logs/depth_eval
until tail -n1 $R/results/logs/depth/chain.status | grep -q "CHAIN_DONE\|GRPO FAILED"; do sleep 300; done
until grep -q ALL_DONE $L/lanes.done 2>/dev/null; do sleep 120; done
echo "[$(date +%H:%M)] grpo eval start" >> $L/grpo_eval.status
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i $P scripts/depth/run_tree_d.py --adapter checkpoints/depth/grpo_walk --value-head checkpoints/depth/value_head.pt --parity odd --shard $i --num-shards 4 --out $O/treeD_grpo_s$i.jsonl > $L/treeD_grpo_s$i.log 2>&1 & done
CUDA_VISIBLE_DEVICES=4 $P scripts/run_eval.py --condition frames16 --adapter checkpoints/depth/grpo_walk --parity odd --out $O/frames16_grpo.jsonl > $L/frames_grpo.log 2>&1 &
wait; echo "[$(date +%H:%M)] GRPO_EVAL_DONE" >> $L/grpo_eval.status

#!/bin/bash
R=/home/haoming/reason_branch_0824; P=/home/haoming/miniconda3/envs/vlm-ex/bin/python; cd $R
export CUDA_DEVICE_ORDER=PCI_BUS_ID VIEWTREE_POSES=human
O=$R/results/depth; L=$R/results/logs/depth_eval
until [ "$(grep -c '^DONE' $L/sti_treeD_s0.log $L/sti_treeD_s1.log $L/sti_sft_a.log 2>/dev/null | awk -F: '{s+=$2} END{print s}')" = "3" ]; do sleep 180; done
pkill -f "eval_ost_treeD.sh"; pkill -f "run_tree_d_bench.py --bench vsti"; sleep 20
seed=$(mktemp); cat $O/vsti_treeD_s0.jsonl $O/vsti_treeD_s1.jsonl $O/vsti_treeD_s2.jsonl $O/vsti_treeD_s3.jsonl > $seed 2>/dev/null
GPUS=(0 1 2 3 4 5 7)
for i in 0 1 2 3 4 5 6; do
  [ -f $O/vsti_treeD_s${i}b.jsonl ] || cp $seed $O/vsti_treeD_s${i}b.jsonl
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} $P scripts/depth/run_tree_d_bench.py --bench vsti --adapter checkpoints/depth/sft_c --value-head checkpoints/depth/value_head.pt --shard $i --num-shards 7 --out $O/vsti_treeD_s${i}b.jsonl > $L/vsti_treeD_s${i}b.log 2>&1 &
done
wait; echo VSTI_TREED_DONE >> $L/newmodel_bench.status
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i $P scripts/depth/run_tree_d_ost.py --adapter checkpoints/depth/sft_c --value-head checkpoints/depth/value_head.pt --shard $i --num-shards 4 --out $O/ost_treeD_s$i.jsonl > $L/ost_treeD_s$i.log 2>&1 & done
wait; echo OST_TREED_DONE >> $L/newmodel_bench.status

#!/bin/bash
R=/home/haoming/reason_branch_0824; P=/home/haoming/miniconda3/envs/vlm-ex/bin/python; cd $R
export CUDA_DEVICE_ORDER=PCI_BUS_ID VIEWTREE_POSES=human
O=$R/results/depth; L=$R/results/logs/depth_eval
until [ "$(for i in 0 1 2 3; do grep -c '^DONE' $L/vsti_treeD_s$i.log 2>/dev/null; done | paste -sd+ | bc)" = "4" ]; do sleep 300; done
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i $P scripts/depth/run_tree_d_ost.py --adapter checkpoints/depth/sft_c --value-head checkpoints/depth/value_head.pt --shard $i --num-shards 4 --out $O/ost_treeD_s$i.jsonl > $L/ost_treeD_s$i.log 2>&1 & done
wait; echo OST_TREED_DONE >> $L/newmodel_bench.status

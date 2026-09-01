#!/bin/bash
R=/home/haoming/reason_branch_0824; P=/home/haoming/miniconda3/envs/vlm-ex/bin/python; cd $R
export CUDA_DEVICE_ORDER=PCI_BUS_ID VIEWTREE_POSES=human
O=$R/results/depth; L=$R/results/logs/depth_eval
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i $P scripts/depth/run_tree_d_bench.py --bench vsti --adapter checkpoints/depth/sft_c --value-head checkpoints/depth/value_head.pt --shard $i --num-shards 4 --out $O/vsti_treeD_s$i.jsonl > $L/vsti_treeD_s$i.log 2>&1 & done
for i in 0 1; do CUDA_VISIBLE_DEVICES=$((4+i)) $P scripts/depth/run_tree_d_bench.py --bench sti --adapter checkpoints/depth/sft_c --value-head checkpoints/depth/value_head.pt --shard $i --num-shards 2 --out $O/sti_treeD_s$i.jsonl > $L/sti_treeD_s$i.log 2>&1 & done
( CUDA_VISIBLE_DEVICES=7 $P scripts/eval_video_bench.py --bench sti --system direct --adapter checkpoints/depth/sft_frames --out $O/sti_sft_frames_s0.jsonl > $L/sti_sft_frames.log 2>&1
  CUDA_VISIBLE_DEVICES=7 $P scripts/eval_video_bench.py --bench sti --system direct --adapter checkpoints/depth/sft_a --out $O/sti_sft_a_s0.jsonl > $L/sti_sft_a.log 2>&1 ) &
wait; echo NEWBENCH_DONE >> $L/newmodel_bench.status

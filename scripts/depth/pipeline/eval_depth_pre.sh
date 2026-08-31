#!/bin/bash
R=/home/haoming/reason_branch_0824; P=/home/haoming/miniconda3/envs/vlm-ex/bin/python; cd $R; export CUDA_DEVICE_ORDER=PCI_BUS_ID VIEWTREE_POSES=human
O=$R/results/depth; L=$R/results/logs/depth_eval
lane(){ g=$1; shift; ( for c in "$@"; do bash -c "$c"; done; echo "LANE_DONE gpu$g" >> $L/lanes.done ) & }
for i in 0 1 2 3; do
  lane $i "CUDA_VISIBLE_DEVICES=$i $P scripts/depth/run_tree_d.py --adapter checkpoints/depth/sft_c --value-head checkpoints/depth/value_head.pt --parity odd --shard $i --num-shards 4 --out $O/treeD_sftc_s$i.jsonl > $L/treeD_sftc_s$i.log 2>&1" \
         "CUDA_VISIBLE_DEVICES=$i $P scripts/run_tree.py --parity odd --adapter checkpoints/depth/sft_a --conf-head checkpoints/depth/value_head.pt --shard $i --num-shards 4 --out $O/tree1_sfta_s$i.jsonl > $L/tree1_sfta_s$i.log 2>&1"
done
lane 6 "CUDA_VISIBLE_DEVICES=6 $P scripts/run_eval.py --condition frames16 --adapter checkpoints/depth/sft_frames --parity odd --shard 0 --num-shards 2 --out $O/frames16_sftframes_s0.jsonl > $L/frames_s0.log 2>&1" \
       "CUDA_VISIBLE_DEVICES=6 $P scripts/run_eval.py --condition frames16 --adapter checkpoints/depth/sft_frames --parity odd --shard 1 --num-shards 2 --out $O/frames16_sftframes_s1.jsonl > $L/frames_s1.log 2>&1" \
       "CUDA_VISIBLE_DEVICES=6 $P scripts/run_eval.py --condition frames16 --adapter checkpoints/depth/sft_a --parity odd --shard 0 --num-shards 1 --out $O/frames16_sfta.jsonl > $L/frames_sfta.log 2>&1"
wait; echo ALL_DONE >> $L/lanes.done

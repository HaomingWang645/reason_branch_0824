#!/bin/bash
R=/home/haoming/reason_branch_0824; P=/home/haoming/miniconda3/envs/vlm-ex/bin/python; TR=/home/haoming/miniconda3/envs/vlm-ex/bin/torchrun; cd $R
export CUDA_DEVICE_ORDER=PCI_BUS_ID VIEWTREE_POSES=human
O=$R/results/scalevlm; L=$R/results/logs/scalevlm; ST=$R/results/logs/depth_eval
M3=Qwen/Qwen2.5-VL-3B-Instruct; M32=Qwen/Qwen2.5-VL-32B-Instruct
until grep -q OST_TREED_DONE $ST/newmodel_bench.status 2>/dev/null && [ -f $ST/models_dl.done ]; do sleep 300; done
echo "[$(date +%H:%M)] stage1" >> $L/scale.status
CUDA_VISIBLE_DEVICES=0 $P scripts/run_eval.py --condition frames16 --model $M3 --parity odd --out $O/frames16_3b.jsonl > $L/frames16_3b.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $P scripts/run_tree.py --model $M3 --parity odd --shard 0 --num-shards 2 --out $O/tree_3b_s0.jsonl > $L/tree_3b_s0.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $P scripts/run_tree.py --model $M3 --parity odd --shard 1 --num-shards 2 --out $O/tree_3b_s1.jsonl > $L/tree_3b_s1.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $P scripts/run_tree.py --parity odd --shard 0 --num-shards 2 --out $O/tree_7b_s0.jsonl > $L/tree_7b_s0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 $P scripts/run_tree.py --parity odd --shard 1 --num-shards 2 --out $O/tree_7b_s1.jsonl > $L/tree_7b_s1.log 2>&1 &
CUDA_VISIBLE_DEVICES=5,7 VIEWTREE_DEVICE_MAP=auto $P scripts/run_eval.py --condition frames16 --model $M32 --parity odd --out $O/frames16_32b.jsonl > $L/frames16_32b.log 2>&1 &
wait
echo "[$(date +%H:%M)] stage2" >> $L/scale.status
CUDA_VISIBLE_DEVICES=5,7 VIEWTREE_DEVICE_MAP=auto $P scripts/run_tree.py --model $M32 --parity odd --shard 0 --num-shards 2 --out $O/tree_32b_s0.jsonl > $L/tree_32b_s0.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1 VIEWTREE_DEVICE_MAP=auto $P scripts/run_tree.py --model $M32 --parity odd --shard 1 --num-shards 2 --out $O/tree_32b_s1.jsonl > $L/tree_32b_s1.log 2>&1 &
DSE_GPUS=2,3,4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $TR --nproc_per_node=3 --master_port=29701 scripts/train_sft.py --model $M3 --data data/train3r/frames_only_30k.jsonl --out checkpoints/scalevlm/sft3b_30k --epochs 1 --lr 1e-4 --accum 16 > $L/train_sft3b.log 2>&1 &
wait
echo "[$(date +%H:%M)] stage3" >> $L/scale.status
CUDA_VISIBLE_DEVICES=0 $P scripts/run_eval.py --condition frames16 --model $M3 --adapter checkpoints/scalevlm/sft3b_30k --parity odd --out $O/frames16_3b_sft30k.jsonl > $L/frames16_3b_sft.log 2>&1
echo "[$(date +%H:%M)] SCALE_DONE" >> $L/scale.status

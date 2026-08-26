#!/bin/bash
# Large-scale run of one RL design: 8-GPU DDP training on the full train set,
# then the full evaluation cascade. Usage:
#   scripts/dse_scale.sh NAME "TRAIN_ARGS" "EVAL_ARGS"
# e.g. scripts/dse_scale.sh E_select_10k "--mode select --cost 0.05" "--mode select"
set -u
R=/home/haoming/reason_branch_0824; P=/home/haoming/miniconda3/envs/vlm-ex/bin/python
NAME=$1; TRAIN=$2; EVAL=$3
CK=$R/checkpoints/scale/$NAME; OUT=$R/results/scale; LOG=$R/results/logs/scale
mkdir -p $CK $OUT $LOG
echo "[$(date +%H:%M)] train $NAME" >> $LOG/$NAME.status
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/haoming/miniconda3/envs/vlm-ex/bin/torchrun \
  --nproc_per_node=8 --master_port=29590 $R/scripts/train_grpo_v2.py --items 9995 \
  --accum-items 8 --out $CK $TRAIN > $LOG/train_$NAME.log 2>&1
grep -aq "^SAVED" $LOG/train_$NAME.log || { echo "TRAIN FAILED" >> $LOG/$NAME.status; exit 1; }
echo "[$(date +%H:%M)] eval $NAME" >> $LOG/$NAME.status
# MindCube per-task rollouts (2 GPUs), VSI odd-half memory (2 shards), VSI odd-half tree + head v2 (4 shards)
CUDA_VISIBLE_DEVICES=0 $P $R/scripts/eval_policy_v2.py --adapter $CK $EVAL --split MindCube_tinybench --out $OUT/${NAME}_tiny.jsonl > $LOG/eval_${NAME}_tiny.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $P $R/scripts/eval_policy_v2.py --adapter $CK $EVAL --split MindCube_rest_clean --out $OUT/${NAME}_rest.jsonl > $LOG/eval_${NAME}_rest.log 2>&1 &
for i in 0 1; do CUDA_VISIBLE_DEVICES=$((i+2)) $P $R/scripts/run_eval.py --condition memory --adapter $CK --parity odd --shard $i --num-shards 2 --out $OUT/${NAME}_vsimem_s$i.jsonl > $LOG/eval_${NAME}_vsimem_s$i.log 2>&1 & done
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$((i+4)) $P $R/scripts/run_tree.py --parity odd --adapter $CK --conf-head $R/checkpoints/conf_head_v2.pt --shard $i --num-shards 4 --out $OUT/${NAME}_tree_s$i.jsonl > $LOG/eval_${NAME}_tree_s$i.log 2>&1 & done
wait
echo "[$(date +%H:%M)] SCALE_DONE $NAME" >> $LOG/$NAME.status

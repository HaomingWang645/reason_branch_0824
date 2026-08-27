#!/bin/bash
# Large-scale run of one RL design on an arbitrary GPU subset: DDP training on
# the full train set, then the evaluation cascade run in NP sequential lanes
# (one lane per GPU) so Exclusive_Process GPUs never see two processes.
#   GPUS=0,1,2 PORT=29593 scripts/dse_scale3.sh NAME "TRAIN_ARGS" "EVAL_ARGS"
set -u
R=/home/haoming/reason_branch_0824; P=/home/haoming/miniconda3/envs/vlm-ex/bin/python
NAME=$1; TRAIN=$2; EVAL=$3
GPUS=${GPUS:-0,1,2,3,4,5,6,7}; IFS=',' read -ra G <<< "$GPUS"; NP=${#G[@]}
PORT=${PORT:-29590}
CK=$R/checkpoints/scale/$NAME; OUT=$R/results/scale; LOG=$R/results/logs/scale
mkdir -p $CK $OUT $LOG
echo "[$(date +%H:%M)] train $NAME" >> $LOG/$NAME.status
DSE_GPUS=$GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/haoming/miniconda3/envs/vlm-ex/bin/torchrun \
  --nproc_per_node=$NP --master_port=$PORT $R/scripts/train_grpo_v2.py --items 9995 \
  --accum-items 8 --out $CK $TRAIN > $LOG/train_$NAME.log 2>&1
grep -aq "^SAVED" $LOG/train_$NAME.log || { echo "TRAIN FAILED" >> $LOG/$NAME.status; exit 1; }
echo "[$(date +%H:%M)] eval $NAME" >> $LOG/$NAME.status
TASKS=(
 "$P $R/scripts/eval_policy_v2.py --adapter $CK $EVAL --split MindCube_tinybench --out $OUT/${NAME}_tiny.jsonl > $LOG/eval_${NAME}_tiny.log 2>&1"
 "$P $R/scripts/eval_policy_v2.py --adapter $CK $EVAL --split MindCube_rest_clean --out $OUT/${NAME}_rest.jsonl > $LOG/eval_${NAME}_rest.log 2>&1"
 "$P $R/scripts/run_eval.py --condition memory --adapter $CK --parity odd --shard 0 --num-shards 2 --out $OUT/${NAME}_vsimem_s0.jsonl > $LOG/eval_${NAME}_vsimem_s0.log 2>&1"
 "$P $R/scripts/run_eval.py --condition memory --adapter $CK --parity odd --shard 1 --num-shards 2 --out $OUT/${NAME}_vsimem_s1.jsonl > $LOG/eval_${NAME}_vsimem_s1.log 2>&1"
)
for i in 0 1 2 3; do TASKS+=("$P $R/scripts/run_tree.py --parity odd --adapter $CK --conf-head $R/checkpoints/conf_head_v2.pt --shard $i --num-shards 4 --out $OUT/${NAME}_tree_s$i.jsonl > $LOG/eval_${NAME}_tree_s$i.log 2>&1"); done
for ((l=0; l<NP; l++)); do
  ( for ((t=l; t<${#TASKS[@]}; t+=NP)); do CUDA_VISIBLE_DEVICES=${G[$l]} bash -c "${TASKS[$t]}"; done ) &
done
wait
echo "[$(date +%H:%M)] SCALE_DONE $NAME" >> $LOG/$NAME.status

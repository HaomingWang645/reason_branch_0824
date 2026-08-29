#!/bin/bash
# ViewTree-D chain on all 8 GPUs (PCI order). Waits for the frames baseline and the next SFT-A checkpoint, stops the 3-GPU run, resumes on 8.
R=/home/haoming/reason_branch_0824; P=/home/haoming/miniconda3/envs/vlm-ex/bin/python; TR=/home/haoming/miniconda3/envs/vlm-ex/bin/torchrun; cd $R
export CUDA_DEVICE_ORDER=PCI_BUS_ID; G=0,1,2,3,4,5,6,7; NP=8; L=$R/results/logs/depth; st(){ echo "[$(date +%H:%M)] $1" >> $L/chain.status; }
st "phase2 oracle walks (8 shards, 5 QA/scene)"
for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i $P scripts/depth/oracle_walks.py --adapter checkpoints/depth/sft_a --per-scene 5 --beam 2 --depth 3 --shard $i --num-shards 8 --out data/train3r/oracle_s$i.jsonl --feats data/train3r/feats_oracle > $L/oracle_s$i.log 2>&1 & done; wait
st "phase2 value head"
$P scripts/depth/train_value_head.py > $L/train_value_head.log 2>&1
st "phase2b build SFT-C"
$P scripts/depth/train_sft_c.py build --out data/train3r/phase2.jsonl > $L/phase2_build.log 2>&1
st "phase2b train SFT-C"
DSE_GPUS=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $TR --nproc_per_node=$NP --master_port=29652 scripts/train_sft.py --data data/train3r/phase2.jsonl --init checkpoints/depth/sft_a --out checkpoints/depth/sft_c --epochs 1 --lr 5e-5 --accum 16 > $L/train_sft_c.log 2>&1
grep -aq "^SAVED" $L/train_sft_c.log || { st "SFT-C FAILED"; exit 1; }
st "phase3 GRPO walks"
DSE_GPUS=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $TR --nproc_per_node=$NP --master_port=29653 scripts/depth/train_grpo_walk.py --init checkpoints/depth/sft_c --out checkpoints/depth/grpo_walk --items 60000 --per-scene 50 --group 6 --accum-items 8 > $L/train_grpo_walk.log 2>&1
grep -aq "^SAVED" $L/train_grpo_walk.log || { st "GRPO FAILED"; exit 1; }
st "CHAIN_DONE"

#!/bin/bash
# ViewTree-D chain: Phase 0 (pose banks, separate script) -> Phase 1 SFT-A -> Phase 2 oracle walks (+SFT-C, value head) 
R=/home/haoming/reason_branch_0824; P=/home/haoming/miniconda3/envs/vlm-ex/bin/python; TR=/home/haoming/miniconda3/envs/vlm-ex/bin/torchrun; cd $R
L=$R/results/logs/depth; st(){ echo "[$(date +%H:%M)] $1" >> $L/chain.status; }
until grep -q "phase0 done" $L/phase0.status 2>/dev/null; do sleep 300; done
st "phase1 build"
$P scripts/depth/build_phase1.py --per-scene 120 --out data/train3r/phase1.jsonl > $L/phase1_build.log 2>&1
st "phase1 train SFT-A"
DSE_GPUS=0,1,2,3,5,6,7 $TR --nproc_per_node=7 --master_port=29611 scripts/train_sft.py --data data/train3r/phase1.jsonl --out checkpoints/depth/sft_a --epochs 1 --lr 1e-4 --accum 16 > $L/train_sft_a.log 2>&1
grep -aq "^SAVED" $L/train_sft_a.log || { st "SFT-A FAILED"; exit 1; }
st "phase2 oracle walks"
i=0; for g in 0 1 2 3 5 6 7; do CUDA_VISIBLE_DEVICES=$g $P scripts/depth/oracle_walks.py --adapter checkpoints/depth/sft_a --per-scene 30 --beam 2 --depth 3 --shard $i --num-shards 7 --out data/train3r/oracle_s$i.jsonl --feats data/train3r/feats_oracle > $L/oracle_s$i.log 2>&1 & i=$((i+1)); done; wait
st "phase2 oracle done"
st "phase2 value head"
$P scripts/depth/train_value_head.py > $L/train_value_head.log 2>&1
st "phase2b build SFT-C"
$P scripts/depth/train_sft_c.py build --out data/train3r/phase2.jsonl > $L/phase2_build.log 2>&1
st "phase2b train SFT-C"
DSE_GPUS=0,1,2,3,5,6,7 $TR --nproc_per_node=7 --master_port=29612 scripts/train_sft.py --data data/train3r/phase2.jsonl --out checkpoints/depth/sft_c --epochs 1 --lr 1e-4 --accum 16 > $L/train_sft_c.log 2>&1
grep -aq "^SAVED" $L/train_sft_c.log || { st "SFT-C FAILED"; exit 1; }
st "phase3 GRPO walks"
DSE_GPUS=0,1,2,3,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $TR --nproc_per_node=7 --master_port=29613 scripts/depth/train_grpo_walk.py --init checkpoints/depth/sft_c --out checkpoints/depth/grpo_walk --items 120000 --per-scene 80 --group 6 --accum-items 8 > $L/train_grpo_walk.log 2>&1
grep -aq "^SAVED" $L/train_grpo_walk.log || { st "GRPO FAILED"; exit 1; }
st "CHAIN_DONE"

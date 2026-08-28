#!/bin/bash
# Data-matched no-memory baseline: SFT on the same corpus, frames only (8 context frames).
R=/home/haoming/reason_branch_0824; P=/home/haoming/miniconda3/envs/vlm-ex/bin/python; TR=/home/haoming/miniconda3/envs/vlm-ex/bin/torchrun; cd $R; L=$R/results/logs/depth
echo "[$(date +%H:%M)] frames baseline build" >> $L/baseline.status
$P - <<'PY'
import json,random,os,collections
R="/home/haoming/reason_branch_0824"; rng=random.Random(0); by=collections.defaultdict(list)
for l in open(f"{R}/data/train3r/qa_all.jsonl"):
    r=json.loads(l); by[r["scene"]].append(r)
n=0
with open(f"{R}/data/train3r/frames_only.jsonl","w") as f:
    for src in os.listdir(f"{R}/data/posebank"):
        for scene in os.listdir(f"{R}/data/posebank/{src}"):
            if scene not in by or not os.path.exists(f"{R}/data/posebank/{src}/{scene}/bank.json"): continue
            qs=by[scene]; rng.shuffle(qs)
            frames=[f"{R}/data/posebank/{src}/{scene}/frame_{i:02d}.jpg" for i in range(8)]
            for r in qs[:60]:
                f.write(json.dumps(dict(id=f"{scene}::f{n}",kind="answer",images=frames,render=None,prompt="These are frames of a video.\n"+r["q"],target=r["a"]))+"\n"); n+=1
print("frames-only examples", n)
PY
echo "[$(date +%H:%M)] frames baseline train" >> $L/baseline.status
DSE_GPUS=5,6,7 $TR --nproc_per_node=3 --master_port=29621 scripts/train_sft.py --data data/train3r/frames_only.jsonl --out checkpoints/depth/sft_frames --epochs 1 --lr 1e-4 --accum 16 > $L/train_sft_frames.log 2>&1
grep -aq "^SAVED" $L/train_sft_frames.log && echo "[$(date +%H:%M)] frames baseline DONE" >> $L/baseline.status || echo "[$(date +%H:%M)] frames baseline FAILED" >> $L/baseline.status

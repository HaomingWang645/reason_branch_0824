#!/usr/bin/env python3
"""Build MindCube evaluation subsets following the Think3D protocol.

Paper: "We sample 40 questions from each category (rotation, around, among),
resulting in 120 questions in total". The exact sample is not released, so we
draw a seeded random sample from MindCube_tinybench, EXCLUDING any question whose
scene (image tuple) appears in the RL training set (crossviewQA_train_rl), so the
RL model is never evaluated on scenes it was trained on.

Outputs (repo-format JSONL usable by scripts/quick_eval.py --mindcube-path):
  dataset/MindCube_eval120.jsonl        40 / category  (paper-size protocol)
  dataset/MindCube_eval_noverlap.jsonl  all non-overlapping tinybench questions
"""
import json, random, collections, sys, os
root = os.path.join(os.path.dirname(__file__), '..', 'spagent')
src = os.path.join(root, 'dataset/MindCube_data.jsonl')
train = os.path.join(root, 'dataset/crossviewQA_train_rl_fixed.jsonl')
train_scenes = set()
for l in open(train):
    d = json.loads(l)
    train_scenes.add(tuple(p.split('mindcube/data/')[-1] for p in d['images']))
items = [json.loads(l) for l in open(src)]
def scene(it): return tuple(p.split('mindcube/data/')[-1] for p in it['image'])
keep = [it for it in items if scene(it) not in train_scenes]
print(f'tinybench {len(items)} -> non-overlapping {len(keep)}')
by_cat = collections.defaultdict(list)
for it in keep: by_cat[it['task']].append(it)
print({k: len(v) for k, v in by_cat.items()})
rng = random.Random(42)
sub = []
for cat in ['rotation', 'around', 'among']:
    pool = by_cat[cat][:]
    rng.shuffle(pool)
    sub.extend(pool[:40])
with open(os.path.join(root, 'dataset/MindCube_eval120.jsonl'), 'w') as f:
    for it in sub: f.write(json.dumps(it, ensure_ascii=False) + '\n')
with open(os.path.join(root, 'dataset/MindCube_eval_noverlap.jsonl'), 'w') as f:
    for it in keep: f.write(json.dumps(it, ensure_ascii=False) + '\n')
print('eval120:', collections.Counter(it['task'] for it in sub))

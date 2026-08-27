#!/usr/bin/env python3
"""Build a VSI-Bench-tiny style split for the Think3D protocol.

Think3D (Table 1) evaluates the four multiple-choice VSI-Bench tasks
(route planning, relative direction, relative distance, appearance order) on
"VSI-Bench-tiny" (the 400-question random subset of VSI-Bench used for the
human study in "Thinking in Space"; 50 questions per task). The exact ids are
not public, so we draw a seeded random sample of 50 questions per MC task from
the official test split (the three relative-direction difficulty levels are
merged into one task, as in the paper's table).

Output: spagent/dataset/VSI_Bench_tiny.jsonl  (200 questions, repo JSONL format)
"""
import json, random, collections, os
root = os.path.join(os.path.dirname(__file__), '..', 'spagent')
src = os.path.join(root, 'dataset/VSI_Bench.jsonl')
items = [json.loads(l) for l in open(src)]
def group(task):
    return 'object_rel_direction' if task.startswith('object_rel_direction') else task
by = collections.defaultdict(list)
for it in items:
    by[group(it['task'])].append(it)
print({k: len(v) for k, v in by.items()})
rng = random.Random(42)
out = []
for task in ['route_planning', 'object_rel_direction', 'object_rel_distance', 'obj_appearance_order']:
    pool = by[task][:]
    rng.shuffle(pool)
    sel = pool[:50]
    for it in sel:
        it['task'] = task          # merged task name for per-task reporting
    out.extend(sel)
dst = os.path.join(root, 'dataset/VSI_Bench_tiny.jsonl')
with open(dst, 'w') as f:
    for it in out:
        f.write(json.dumps(it, ensure_ascii=False) + '\n')
print('wrote', dst, len(out), collections.Counter(it['task'] for it in out))
print(out[0]['conversations'][0]['value'][:300])

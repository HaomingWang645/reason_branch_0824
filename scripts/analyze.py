"""Aggregate condition results: per-type metrics + paired scene-bootstrap CIs."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viewtree.data import load_questions

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPES_ORDER = [
    "object_counting", "object_abs_distance", "object_size_estimation",
    "room_size_estimation", "object_rel_distance", "object_rel_direction_easy",
    "object_rel_direction_medium", "object_rel_direction_hard",
    "obj_appearance_order", "route_planning",
]


def load_condition(name):
    preds = {}
    for f in glob.glob(os.path.join(REPO, "results", f"{name}_s*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            preds[r["id"]] = r
    return preds


def main():
    rows = {r["id"]: r for r in load_questions()}
    conds = sys.argv[1:] or ["current", "frames12", "frames16", "renders_only", "memory"]
    data = {c: load_condition(c) for c in conds}
    for c in conds:
        print(f"{c}: {len(data[c])} answers")
    common = set(rows)
    for c in conds:
        common &= set(data[c])
    print(f"paired on {len(common)} questions\n")

    # per-type table
    header = f"{'question_type':32s}" + "".join(f"{c:>14s}" for c in conds)
    print(header)
    for t in TYPES_ORDER:
        ids = [i for i in common if rows[i]["question_type"] == t]
        line = f"{t:32s}"
        for c in conds:
            line += f"{np.mean([data[c][i]['score'] for i in ids]):14.3f}"
        print(line + f"   (n={len(ids)})")
    line = f"{'MEAN OF TYPES':32s}"
    means = {}
    for c in conds:
        m = np.mean([
            np.mean([data[c][i]["score"] for i in common if rows[i]["question_type"] == t])
            for t in TYPES_ORDER
        ])
        means[c] = m
        line += f"{m:14.3f}"
    print(line)

    # scene-level bootstrap CI on mean-of-types
    scenes = sorted({(rows[i]["dataset"], rows[i]["scene_name"]) for i in common})
    by_scene = {s: [] for s in scenes}
    for i in common:
        by_scene[(rows[i]["dataset"], rows[i]["scene_name"])].append(i)
    rng = np.random.default_rng(0)
    B = 2000
    boot = {c: [] for c in conds}
    for _ in range(B):
        pick = rng.choice(len(scenes), len(scenes), replace=True)
        ids = [i for k in pick for i in by_scene[scenes[k]]]
        for c in conds:
            per_t = [
                np.mean([data[c][i]["score"] for i in ids if rows[i]["question_type"] == t] or [0])
                for t in TYPES_ORDER
            ]
            boot[c].append(np.mean(per_t))
    print("\nscene-bootstrap 95% CI (mean of types):")
    for c in conds:
        lo, hi = np.percentile(boot[c], [2.5, 97.5])
        print(f"  {c:14s} {means[c]:.3f}  [{lo:.3f}, {hi:.3f}]")
    if "frames16" in conds:
        for c in conds:
            if c == "frames16":
                continue
            d = np.array(boot[c]) - np.array(boot["frames16"])
            lo, hi = np.percentile(d, [2.5, 97.5])
            print(f"  {c} - frames16: {means[c]-means['frames16']:+.3f}  [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()

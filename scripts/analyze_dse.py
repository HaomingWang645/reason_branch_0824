"""Per-variant, per-task analysis of the RL design-space sweep.

Joins rollout results with MindCube metadata (category prefix among/around/
rotation and the item 'type' field) so designs can be compared per task.
Baseline rows: SFT-v2 greedy policy (results/policy_sft2_s*.jsonl,
results/policy_rest_sft2.jsonl). Prints markdown tables.
"""
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MC = os.path.join(REPO, "data", "mindcube", "data", "raw")
VARIANTS = ["A_baseline", "B_depth2", "C_costfree", "D_highcost", "E_select",
            "F_cot", "G_cot_short", "H_group12"]


def meta(split):
    m = {}
    for l in open(os.path.join(MC, split + ".jsonl")):
        r = json.loads(l)
        m[r["id"]] = {"cat": r["id"].split("_")[0], "type": str(r.get("type", "?")),
                      "n": len(r["images"])}
    return m


def load(paths):
    d = {}
    for p in paths:
        if os.path.exists(p):
            for l in open(p):
                r = json.loads(l)
                d[r["id"]] = r
    return d


def table(split, base_paths, tag):
    md = meta(split)
    runs = {"SFT-v2 (ref)": load(base_paths)}
    for v in VARIANTS:
        runs[v] = load([os.path.join(REPO, "results", "dse", f"{v}_{tag}.jsonl")])
    runs = {k: v for k, v in runs.items() if v}
    ids = set(md)
    for v in runs.values():
        ids &= set(v)
    ids = sorted(ids)
    cats = sorted({md[i]["cat"] for i in ids})
    types = sorted({md[i]["type"] for i in ids})
    print(f"\n### {split}: paired n={len(ids)}\n")
    hdr = "| variant | acc | views | toks | " + " | ".join(
        f"{c} (n={sum(md[i]['cat']==c for i in ids)})" for c in cats) + " |"
    print(hdr)
    print("|---|---|---|---|" + "---|" * len(cats))
    for name, d in runs.items():
        acc = np.mean([d[i]["correct"] for i in ids])
        views = np.mean([d[i]["views"] for i in ids])
        toks = np.mean([d[i].get("ntok", 0) for i in ids])
        cells = []
        for c in cats:
            sub = [i for i in ids if md[i]["cat"] == c]
            cells.append(f"{np.mean([d[i]['correct'] for i in sub]):.3f}")
        print(f"| {name} | {acc:.3f} | {views:.2f} | {toks:.0f} | " + " | ".join(cells) + " |")
    print(f"\nBy question type ({split}):\n")
    print("| variant | " + " | ".join(f"{t} (n={sum(md[i]['type']==t for i in ids)})" for t in types) + " |")
    print("|---|" + "---|" * len(types))
    for name, d in runs.items():
        cells = []
        for t in types:
            sub = [i for i in ids if md[i]["type"] == t]
            cells.append(f"{np.mean([d[i]['correct'] for i in sub]):.3f}" if sub else "—")
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    table("MindCube_tinybench",
          glob.glob(os.path.join(REPO, "results", "policy_sft2_s*.jsonl")), "tiny")
    table("MindCube_rest_clean",
          [os.path.join(REPO, "results", "policy_rest_sft2.jsonl")], "rest")

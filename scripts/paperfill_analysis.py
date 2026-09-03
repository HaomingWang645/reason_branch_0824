"""Paired scene-bootstrap deltas for the paper-fill ablations, plus the
late-consensus subset check (questions the deployed beam answered via
consensus at depth >= 2: does beam width pay there?).

Deltas (B=2000 scene bootstrap, mean-of-10-types statistic):
  greedy  - beam      random  - beam
  memstatic - SFT-A frames-only      memzs_7b - zero-shot(7B)
"""
import json, os, sys
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R)
from viewtree.data import load_questions

TYPES = ["object_counting", "object_abs_distance", "object_size_estimation",
         "room_size_estimation", "object_rel_distance", "object_rel_direction_easy",
         "object_rel_direction_medium", "object_rel_direction_hard",
         "obj_appearance_order", "route_planning"]

rows = load_questions()
scenes = sorted({(r["dataset"], r["scene_name"]) for r in rows})
odd = set(scenes[1::2])
byid = {r["id"]: r for r in rows if (r["dataset"], r["scene_name"]) in odd}

def load(paths, base="results/paperfill"):
    d = {}
    for p in paths:
        fp = os.path.join(R, base, p) if not p.startswith("results") else os.path.join(R, p)
        if not os.path.exists(fp):
            print("missing", fp); return None
        for l in open(fp):
            try: r = json.loads(l)
            except Exception: continue
            if r.get("id") in byid and r["id"] not in d:
                d[r["id"]] = r
    return d

S = {
    "beam": load([f"results/depth/treeD_sftc_s{i}.jsonl" for i in range(4)]),
    "greedy": load(["greedy_7b_s0.jsonl", "greedy_7b_s1.jsonl", "greedy_7b.jsonl"]),
    "random": load(["random_7b_s0.jsonl", "random_7b_s1.jsonl"]),
    "memstatic": load(["memstatic_7b.jsonl"]),
    "sfta": load(["results/depth/frames16_sfta.jsonl"]),
    "zs7b": load(["results/frames16_s0.jsonl", "results/frames16_s1.jsonl"]),
    "memzs": load(["memzs_7b.jsonl"]),
}

def mean10(sc, ids):
    per = []
    for t in TYPES:
        v = [sc[i]["score"] for i in ids if byid[i]["question_type"] == t]
        if v: per.append(np.mean(v))
    return float(np.mean(per))

def paired(a, b, B=2000, seed=0):
    ids = sorted(set(S[a]) & set(S[b]))
    sc_by = {}
    for i in ids:
        sc_by.setdefault((byid[i]["dataset"], byid[i]["scene_name"]), []).append(i)
    keys = sorted(sc_by)
    rng = np.random.default_rng(seed)
    d0 = mean10(S[a], ids) - mean10(S[b], ids)
    ds = []
    for _ in range(B):
        pick = rng.choice(len(keys), len(keys), replace=True)
        bids = [i for k in pick for i in sc_by[keys[k]]]
        ds.append(mean10(S[a], bids) - mean10(S[b], bids))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    print(f"{a:10s} - {b:10s}: {d0*100:+.1f}  [{lo*100:+.1f}, {hi*100:+.1f}]  (n={len(ids)})")
    return d0, lo, hi

for a, b in [("greedy", "beam"), ("random", "beam"), ("memstatic", "sfta"),
             ("memzs", "zs7b"), ("memstatic", "beam")]:
    if S[a] and S[b]: paired(a, b)

# late-consensus subset: beam solved via consensus at depth >= 2
late = [i for i, r in S["beam"].items() if str(r.get("mode", "")).startswith("consensus_d") and int(str(r["mode"])[-1]) >= 2]
late = [i for i in late if i in S["greedy"] and i in S["random"]]
print(f"\nlate-consensus subset (beam consensus at d>=2): n={len(late)}")
for name in ["beam", "greedy", "random", "sfta"]:
    v = [S[name][i]["score"] for i in late if i in S[name]]
    print(f"  {name:8s} mean score {np.mean(v):.3f} (n={len(v)})")
# and the walk-vs-direct subset: questions the beam gate sent exploring
walked = [i for i, r in S["beam"].items() if r.get("gate", "").upper().find("YES") < 0]
walked = [i for i in walked if i in S["greedy"] and i in S["random"]]
print(f"\ngate-explored subset: n={len(walked)}")
for name in ["beam", "greedy", "random", "sfta"]:
    v = [S[name][i]["score"] for i in walked if i in S[name]]
    print(f"  {name:8s} mean score {np.mean(v):.3f} (n={len(v)})")

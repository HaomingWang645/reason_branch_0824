"""Score the paper-fill runs (results/paperfill/*.jsonl) and print per-type numbers in
the paper's column layout: A.D., O.S., R.Dt., R.Dr. (mean of easy/med/hard), O.C.,
Ap.Or., Avg (mean of all ten type means). Also mean VLM calls where recorded."""
import json, os, sys
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(R, "results/paperfill")

TYPES = ["object_counting", "object_abs_distance", "object_size_estimation",
         "room_size_estimation", "object_rel_distance", "object_rel_direction_easy",
         "object_rel_direction_medium", "object_rel_direction_hard",
         "obj_appearance_order", "route_planning"]

def load(paths):
    rows = {}
    for p in paths:
        fp = os.path.join(D, p)
        if not os.path.exists(fp):
            return None
        for l in open(fp):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if "id" in r and r.get("pred", "") != "" or r.get("score") is not None:
                rows.setdefault(r["id"], r)
    return rows

def stats(rows, expect=2557):
    per = {}
    for t in TYPES:
        v = [r["score"] for r in rows.values() if r.get("question_type") == t]
        per[t] = float(np.mean(v)) if v else float("nan")
    mean10 = float(np.mean([per[t] for t in TYPES]))
    rdr = float(np.mean([per["object_rel_direction_easy"], per["object_rel_direction_medium"], per["object_rel_direction_hard"]]))
    calls = [r["calls"] for r in rows.values() if "calls" in r]
    cols = dict(AD=per["object_abs_distance"], OS=per["object_size_estimation"],
                RDt=per["object_rel_distance"], RDr=rdr, OC=per["object_counting"],
                AO=per["obj_appearance_order"], AVG=mean10,
                room=per["room_size_estimation"], route=per["route_planning"],
                n=len(rows), calls=float(np.mean(calls)) if calls else None)
    return cols

RUNS = {
    "cot_7b": ["cot_7b_s0.jsonl", "cot_7b_s1.jsonl"],
    "cot_3b": ["cot_3b_s0.jsonl", "cot_3b_s1.jsonl", "cot_3b.jsonl"],
    "cot_32b": ["cot_32b_s0b.jsonl", "cot_32b_s1b.jsonl", "cot_32b.jsonl"],
    "memzs_7b": ["memzs_7b.jsonl"],
    "memzs_3b": ["memzs_3b.jsonl"],
    "memzs_32b": ["memzs_32b.jsonl"],
    "memstatic_7b": ["memstatic_7b.jsonl"],
    "greedy_7b": ["greedy_7b_s0.jsonl", "greedy_7b_s1.jsonl", "greedy_7b.jsonl"],
    "random_7b": ["random_7b_s0.jsonl", "random_7b_s1.jsonl"],
}

if __name__ == "__main__":
    out = {}
    for name, paths in RUNS.items():
        have = [p for p in paths if os.path.exists(os.path.join(D, p))]
        rows = load(have) if have else None
        if not rows:
            print(f"{name:14s}  (no files yet)")
            continue
        c = stats(rows)
        out[name] = c
        cs = f"  calls {c['calls']:.1f}" if c["calls"] else ""
        print(f"{name:14s} n={c['n']:4d}  AD {c['AD']:.3f}  OS {c['OS']:.3f}  RDt {c['RDt']:.3f}  "
              f"RDr {c['RDr']:.3f}  OC {c['OC']:.3f}  AO {c['AO']:.3f}  room {c['room']:.3f}  "
              f"route {c['route']:.3f}  AVG {c['AVG']:.3f}{cs}")
    json.dump(out, open(os.path.join(D, "paperfill_scores.json"), "w"), indent=1)

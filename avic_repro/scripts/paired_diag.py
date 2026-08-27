"""Paired per-question comparison of a WM run against the same backbone's no-WM baseline.
Usage: python scripts/paired_diag.py results/gpt-4o_baseline_qc2 results/gpt-4o_avic_r_spatial_beam_search_qc6"""
import glob, json, os, sys
from collections import Counter, defaultdict

def load(run):
    out = {}
    for r in glob.glob(os.path.join(run, "**", "results.json"), recursive=True):
        if r == os.path.join(run, "results.json") and glob.glob(os.path.join(run, "question_chunk_*")):
            continue
        d = json.load(open(r))
        for qt, p in d["progress"].items():
            for q in p["correct"]: out[q] = (qt, True)
            for q in p["wrong"]: out[q] = (qt, False)
    return out

def decisions(run):
    dec = {}
    for g in glob.glob(os.path.join(run, "**", "step_0", "gpt.json"), recursive=True):
        qid = int(g.split(os.sep)[-3]); d = json.load(open(g))
        dec[qid] = (d.get("planning") or {}).get("decision", "skip")
    return dec

base, wm = load(sys.argv[1]), load(sys.argv[2]); dec = decisions(sys.argv[2])
common = sorted(set(base) & set(wm))
print(f"paired questions: {len(common)}  base acc={100*sum(base[q][1] for q in common)/len(common):.1f}  wm-run acc={100*sum(wm[q][1] for q in common)/len(common):.1f}")
tab = defaultdict(Counter)
for q in common:
    tab[dec.get(q, "?")][(base[q][1], wm[q][1])] += 1
for d, c in tab.items():
    n = sum(c.values())
    print(f"decision={d:8s} n={n:3d}  base✓wm✓={c[(True,True)]:3d}  base✓wm✗={c[(True,False)]:3d}  base✗wm✓={c[(False,True)]:3d}  base✗wm✗={c[(False,False)]:3d}"
          f"  -> base acc {100*(c[(True,True)]+c[(True,False)])/n:.1f}  wm acc {100*(c[(True,True)]+c[(False,True)])/n:.1f}")
per = defaultdict(lambda: [0,0,0])
for q in common:
    qt = base[q][0]; per[qt][0]+=1; per[qt][1]+=base[q][1]; per[qt][2]+=wm[q][1]
for qt,(n,b,w) in sorted(per.items()): print(f"  {qt:15s} n={n:2d} base={100*b/n:5.1f} wm={100*w/n:5.1f}")

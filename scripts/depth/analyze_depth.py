"""Depth ablation table on the VSI held-out odd half: mean of 10 types, paired,
scene-bootstrap CIs, path/cost statistics.  Systems are result globs."""
import glob, json, os, sys, collections
import numpy as np
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, R)
from viewtree.data import load_questions
rows = {r["id"]: r for r in load_questions()}
scenes = sorted({(r["dataset"], r["scene_name"]) for r in rows.values()}); odd = set(scenes[1::2])
def load(pat):
    d = {}
    for p in glob.glob(os.path.join(R, pat)):
        for l in open(p):
            r = json.loads(l)
            if "score" in r: d[r["id"]] = r
    return d
SYS = collections.OrderedDict([
    ("zero-shot frames16", "results/frames16_s*.jsonl"),
    ("SFT-plain frames16 (MindCube)", "results/plain/sft_plain_frames16_s*.jsonl"),
    ("depth-1 tree: D_10k + human views + matched head (prev best)", "results/human/tree_d10k_headH_s*.jsonl"),
    ("corpus frames-only SFT (data-matched baseline)", "results/depth/frames16_sftframes_s*.jsonl"),
    ("SFT-A frames-only", "results/depth/frames16_sfta.jsonl"),
    ("depth-1 tree with SFT-A + value head", "results/depth/tree1_sfta_s*.jsonl"),
    ("ViewTree-D no-RL: SFT-C + value head + beam (d<=3)", "results/depth/treeD_sftc_s*.jsonl"),
    ("ViewTree-D: GRPO walks + value head + beam (d<=3)", "results/depth/treeD_grpo_s*.jsonl"),
])
D = collections.OrderedDict((k, load(v)) for k, v in SYS.items()); D = collections.OrderedDict((k, v) for k, v in D.items() if v)
ids = [i for i in rows if (rows[i]["dataset"], rows[i]["scene_name"]) in odd and all(i in v for v in D.values())]
types = sorted({rows[i]["question_type"] for i in ids}); print(f"paired odd-half n = {len(ids)}; systems = {list(D)}")
g = collections.defaultdict(list)
for i in ids: g[(rows[i]["dataset"], rows[i]["scene_name"])].append(i)
sc = list(g); rng = np.random.default_rng(0); boots = [[i for j in rng.choice(len(sc), len(sc)) for i in g[sc[j]]] for _ in range(1000)]
def mot(d, sel): return np.mean([np.mean([d[i]["score"] for i in sel if rows[i]["question_type"] == t]) for t in types])
ref_key = "corpus frames-only SFT (data-matched baseline)" if "corpus frames-only SFT (data-matched baseline)" in D else list(D)[0]
print("| system | mean of types | Δ vs " + ref_key + " [95% CI] | " + " | ".join(t.replace("object_", "").replace("_estimation", "") for t in types) + " | calls | depth |")
print("|---|---|---|" + "---|" * len(types) + "---|---|")
for k, v in D.items():
    per = [np.mean([v[i]["score"] for i in ids if rows[i]["question_type"] == t]) for t in types]
    dd = [mot(v, b) - mot(D[ref_key], b) for b in boots]
    calls = np.mean([v[i].get("calls", np.nan) for i in ids]); dep = np.mean([v[i].get("depth", np.nan) for i in ids])
    print(f"| {k} | **{np.mean(per):.3f}** | {mot(v, ids)-mot(D[ref_key], ids):+.3f} [{np.percentile(dd,2.5):+.3f}, {np.percentile(dd,97.5):+.3f}] | " + " | ".join(f"{x:.3f}" for x in per) + f" | {calls:.1f} | {dep:.2f} |")
for k, v in D.items():
    if "mode" in next(iter(v.values())): print(k, "modes:", dict(collections.Counter(v[i]["mode"] for i in ids)))

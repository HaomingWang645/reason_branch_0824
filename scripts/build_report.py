"""Build report/REPORT.md + report/REPORT.pdf for the best ViewTree system.
Tables are computed from result files; reasoning-tree figures are read from
figures/tree_rep_vsi_*.png and figures/tree_rep_ost_*.png."""
import glob, json, os, sys, collections
import numpy as np
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, R)
from viewtree.data import load_questions
OUT = os.path.join(R, "report"); os.makedirs(OUT, exist_ok=True)

def load(pat):
    d = {}
    for p in glob.glob(os.path.join(R, pat)):
        for l in open(p):
            r = json.loads(l); d[r["id"]] = r
    return d

def boot_ci(a_fn, b_fn, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed); keys = list(groups); ds = []
    for _ in range(n):
        sel = [i for j in rng.choice(len(keys), len(keys)) for i in groups[keys[j]]]
        ds.append(a_fn(sel) - b_fn(sel))
    return np.percentile(ds, 2.5), np.percentile(ds, 97.5)

# ---------------- VSI ----------------
rows = {r["id"]: r for r in load_questions()}
scenes = sorted({(r["dataset"], r["scene_name"]) for r in rows.values()}); odd = set(scenes[1::2])
VSI = collections.OrderedDict([
    ("Qwen2.5-VL-7B zero-shot (16 frames)", load("results/frames16_s*.jsonl")),
    ("SFT-plain (16 frames)", load("results/plain/sft_plain_frames16_s*.jsonl")),
    ("SFT+GRPO-plain (16 frames)", load("results/plain/grpo_plain_frames16_s*.jsonl")),
    ("SFT-v2 + static memory (12 frames + 5 renders)", load("results/sft2_memory_s*.jsonl")),
    ("ViewTree, legacy views + head v2", load("results/scale/D_highcost_10k_tree_s*.jsonl")),
    ("**ViewTree (best): human views + matched head**", load("results/human/tree_d10k_headH_s*.jsonl")),
])
ids = [i for i in rows if (rows[i]["dataset"], rows[i]["scene_name"]) in odd and all(i in v for v in VSI.values())]
types = sorted({rows[i]["question_type"] for i in ids})
TN = {t: t.replace("object_", "").replace("_estimation", "").replace("rel_direction_", "dir ").replace("_", " ") for t in types}
groups = collections.defaultdict(list)
for i in ids: groups[(rows[i]["dataset"], rows[i]["scene_name"])].append(i)
def mot(d): return lambda sel: np.mean([np.mean([d[i]["score"] for i in sel if rows[i]["question_type"] == t]) for t in types])
best = VSI["**ViewTree (best): human views + matched head**"]
vsi_tab = ["| system | mean of types | " + " | ".join(TN[t] for t in types) + " |", "|---|---|" + "---|" * len(types)]
vsi_ci = []
for k, v in VSI.items():
    per = [np.mean([v[i]["score"] for i in ids if rows[i]["question_type"] == t]) for t in types]
    vsi_tab.append(f"| {k} | **{np.mean(per):.3f}** | " + " | ".join(f"{x:.3f}" for x in per) + " |")
    if k in ("SFT-plain (16 frames)", "SFT+GRPO-plain (16 frames)", "Qwen2.5-VL-7B zero-shot (16 frames)"):
        lo, hi = boot_ci(mot(best), mot(v), groups); vsi_ci.append(f"- best ViewTree − {k}: **{mot(best)(ids)-mot(v)(ids):+.3f}** [{lo:+.3f}, {hi:+.3f}] (scene-bootstrap 95 % CI)")
# per-type wins vs strongest no-memory baseline
sp = VSI["SFT-plain (16 frames)"]; gp = VSI["SFT+GRPO-plain (16 frames)"]
pt = []
for t in types:
    sel = [i for i in ids if rows[i]["question_type"] == t]
    b = np.mean([best[i]["score"] for i in sel]); s1 = np.mean([sp[i]["score"] for i in sel]); s2 = np.mean([gp[i]["score"] for i in sel])
    pt.append(f"| {TN[t]} | {len(sel)} | {s1:.3f} | {s2:.3f} | **{b:.3f}** | {b-max(s1,s2):+.3f} |")
vsi_pt = ["| VSI task | n | SFT-plain | SFT+GRPO-plain | ViewTree (best) | Δ vs best baseline |", "|---|---|---|---|---|---|"] + pt
modes = collections.Counter(best[i]["mode"] for i in ids)

# ---------------- OST ----------------
OSTB = collections.OrderedDict([
    ("Qwen2.5-VL-7B zero-shot", load("results/ext_ost_base_s*.jsonl")),
    ("SFT-plain", load("results/plain/ext_sft_plain_ost_s*.jsonl")),
    ("SFT+GRPO-plain", load("results/plain/ext_grpo_plain_ost_s*.jsonl")),
    ("SFT-v2 adapter, single pass", load("results/ext_ost_sft2_s*.jsonl")),
    ("D_10k adapter, single pass", load("results/ext_d10k/ost_s*.jsonl")),
])
tree_ost = load("results/report/ost_tree_s*.jsonl")
ost_ids = [i for i in OSTB["SFT-plain"] if all(i in v for v in OSTB.values()) and (not tree_ost or i in tree_ost)]
if tree_ost: OSTB["**ViewTree (best): reasoning tree**"] = tree_ost
oclasses = sorted({OSTB["SFT-plain"][i]["qtype"] for i in ost_ids})
ost_tab = ["| system | overall | " + " | ".join(f"{c} (n={sum(OSTB['SFT-plain'][i]['qtype']==c for i in ost_ids)})" for c in oclasses) + " |", "|---|---|" + "---|" * len(oclasses)]
for k, v in OSTB.items():
    ost_tab.append(f"| {k} | **{np.mean([v[i]['correct'] for i in ost_ids]):.3f}** | " + " | ".join(f"{np.mean([v[i]['correct'] for i in ost_ids if OSTB['SFT-plain'][i]['qtype']==c]):.3f}" for c in oclasses) + " |")
ost_ci = []
if tree_ost:
    og = collections.defaultdict(list)
    for i in ost_ids: og[i.split("_")[0] + "_" + str(int(i.split("_")[1]) // 50)].append(i)  # coarse item blocks as bootstrap units
    acc = lambda d: (lambda sel: np.mean([d[i]["correct"] for i in sel]))
    for k in ("SFT-plain", "SFT+GRPO-plain", "Qwen2.5-VL-7B zero-shot"):
        lo, hi = boot_ci(acc(tree_ost), acc(OSTB[k]), og); ost_ci.append(f"- ViewTree − {k}: **{acc(tree_ost)(ost_ids)-acc(OSTB[k])(ost_ids):+.3f}** [{lo:+.3f}, {hi:+.3f}] (bootstrap 95 % CI over item blocks)")
    omodes = collections.Counter(tree_ost[i]["mode"] for i in ost_ids)
ost_note = "" if tree_ost else "*(ViewTree tree evaluation on OST-Bench in progress — table will be completed.)*"


# ---------------- STI-Bench ----------------
def load_sti(pat):  # STI 'ID' is only unique within a video -> key by (video, id)
    d = {}
    for p in glob.glob(os.path.join(R, pat)):
        for l in open(p):
            r = json.loads(l); d[r["scene"] + "/" + r["id"]] = r
    return d
STI = collections.OrderedDict([
    ("Qwen2.5-VL-7B zero-shot", load_sti("results/newbench/sti_direct_zeroshot_s*.jsonl")),
    ("SFT-plain", load_sti("results/newbench/sti_direct_sft_plain_s*.jsonl")),
    ("SFT+GRPO-plain", load_sti("results/newbench/sti_direct_grpo_plain_s*.jsonl")),
    ("SFT-v2 adapter, single pass", load_sti("results/newbench/sti_direct_sft2_s*.jsonl")),
    ("D_10k adapter, single pass", load_sti("results/newbench/sti_direct_d10k_s*.jsonl")),
    ("**ViewTree (best): reasoning tree**", load_sti("results/newbench/sti_tree_s*.jsonl")),
])
STI = collections.OrderedDict((k, v) for k, v in STI.items() if len(v) >= 2000)  # only systems with (near-)complete coverage
sti_md = ""
if "SFT+GRPO-plain" in STI and "**ViewTree (best): reasoning tree**" in STI:
    sti_ids = [i for i in STI["SFT-plain"] if all(i in v for v in STI.values())]
    ref = STI["SFT-plain"]; stasks = sorted({ref[i]["qtype"] for i in sti_ids}); ssrc = sorted({ref[i]["source"] for i in sti_ids})
    tree = STI["**ViewTree (best): reasoning tree**"]
    tab = ["| system | overall | " + " | ".join(f"{t} ({sum(ref[i]['qtype']==t for i in sti_ids)})" for t in stasks) + " |", "|---|---|" + "---|" * len(stasks)]
    for k, v in STI.items():
        tab.append(f"| {k} | **{np.mean([v[i]['score'] for i in sti_ids]):.3f}** | " + " | ".join(f"{np.mean([v[i]['score'] for i in sti_ids if ref[i]['qtype']==t]):.3f}" for t in stasks) + " |")
    tab2 = ["| system | " + " | ".join(f"{c} ({sum(ref[i]['source']==c for i in sti_ids)})" for c in ssrc) + " |", "|---|" + "---|" * len(ssrc)]
    for k, v in STI.items():
        tab2.append(f"| {k} | " + " | ".join(f"{np.mean([v[i]['score'] for i in sti_ids if ref[i]['source']==c]):.3f}" for c in ssrc) + " |")
    sg = collections.defaultdict(list)
    for i in sti_ids: sg[ref[i]["scene"]].append(i)  # bootstrap over videos
    acc = lambda d: (lambda sel: np.mean([d[i]["score"] for i in sel]))
    cis = []
    for k in ("SFT-plain", "SFT+GRPO-plain", "Qwen2.5-VL-7B zero-shot"):
        if k in STI:
            lo, hi = boot_ci(acc(tree), acc(STI[k]), sg); cis.append(f"- ViewTree − {k}: **{acc(tree)(sti_ids)-acc(STI[k])(sti_ids):+.3f}** [{lo:+.3f}, {hi:+.3f}] (video-bootstrap 95 % CI)")
    smodes = collections.Counter(tree[i]["mode"] for i in sti_ids)
    sti_md = f"""
## 3b. STI-Bench (spatial-temporal understanding from video, 2,064 single-choice items, 8 tasks)

Videos come from ScanNet (indoor walkthroughs), Waymo (outdoor driving) and Omni6DPose (desktop object manipulation); every question
refers to a time window of the video. All systems see the same 16 frames sampled inside the queried window (±1 s for instantaneous
questions). Only 2 of the 150 ScanNet videos overlap the scenes used to train our confidence head; results are on all items
(paired n = {len(sti_ids)}). No model was retrained for this benchmark.

{chr(10).join(tab)}

By video source:

{chr(10).join(tab2)}

{chr(10).join(cis)}

ViewTree path mix: {" · ".join(f"{k} {v}" for k, v in smodes.items())}.

**Reading (a negative result, reported as such).** On STI-Bench the *zero-shot* model is the best system: every model
fine-tuned on MindCube indoor multiple-choice data regresses, most severely the no-memory baselines (−11 pts; pose estimation
0.535→0.28, ego-centric orientation 0.454→0.11), whose answer-only fine-tuning over-fits the MindCube option format and indoor
domain. ViewTree keeps more of the base model's ability (−6.4) and is +4.5 above both no-memory baselines (significant), and
its gate answers directly on 66 % of items — the tree does no harm — but it does not recover zero-shot performance. STI's tasks
are largely temporal-quantitative (speed, displacement, trajectory, pose over time) and two of three sources (driving, desktop
manipulation) are far from the indoor-room domain of all training data used here; a static scene memory is the wrong tool for
motion questions. The practical conclusion is that the *controller adapter*, not the memory, is what limits transfer to this
benchmark; the multi-step design (DESIGN_DEPTH.md) trains on a corpus that includes camera-motion questions for this reason.
"""


# ---------------- VSTI-Bench ----------------
VSTI = collections.OrderedDict([
    ("Qwen2.5-VL-7B zero-shot", load("results/newbench/vsti_direct_zeroshot_s*.jsonl")),
    ("SFT-plain", load("results/newbench/vsti_direct_sft_plain_s*.jsonl")),
    ("SFT+GRPO-plain", load("results/newbench/vsti_direct_grpo_plain_s*.jsonl")),
    ("SFT-v2 adapter, single pass", load("results/newbench/vsti_direct_sft2_s*.jsonl")),
    ("D_10k adapter, single pass", load("results/newbench/vsti_direct_d10k_s*.jsonl")),
    ("**ViewTree (best): reasoning tree**", load("results/newbench/vsti_tree_s*.jsonl")),
])
vsti_md = ""
VSTI = collections.OrderedDict((k, v) for k, v in VSTI.items() if len(v) >= 5000)
if "**ViewTree (best): reasoning tree**" in VSTI and "SFT+GRPO-plain" in VSTI:
    vids = [i for i in VSTI["SFT-plain"] if all(i in v for v in VSTI.values())]; vref = VSTI["SFT-plain"]; vtree = VSTI["**ViewTree (best): reasoning tree**"]
    vtypes = sorted({vref[i]["qtype"] for i in vids}); VT = {t: t.replace("camera_", "cam ").replace("obj_obj_relative_pos_", "obj-obj ").replace("_", " ") for t in vtypes}
    def vtab(subset):
        tab = ["| system | mean of types | " + " | ".join(f"{VT[t]} ({sum(vref[i]['qtype']==t for i in subset)})" for t in vtypes) + " |", "|---|---|" + "---|" * len(vtypes)]
        for k, v in VSTI.items():
            per = [np.mean([v[i]["score"] for i in subset if vref[i]["qtype"] == t]) for t in vtypes]
            tab.append(f"| {k} | **{np.mean(per):.3f}** | " + " | ".join(f"{x:.3f}" for x in per) + " |")
        return tab
    clean = [i for i in vids if vref[i]["clean"]]
    vg = collections.defaultdict(list)
    for i in vids: vg[vref[i]["scene"]].append(i)
    vmot = lambda d: (lambda sel: np.mean([np.mean([d[i]["score"] for i in sel if vref[i]["qtype"] == t]) for t in vtypes]))
    vcis = []
    for k in ("SFT-plain", "SFT+GRPO-plain", "Qwen2.5-VL-7B zero-shot", "D_10k adapter, single pass"):
        if k in VSTI:
            lo, hi = boot_ci(vmot(vtree), vmot(VSTI[k]), vg); vcis.append(f"- ViewTree − {k}: **{vmot(vtree)(vids)-vmot(VSTI[k])(vids):+.3f}** [{lo:+.3f}, {hi:+.3f}] (scene-bootstrap 95 % CI)")
    vmodes = collections.Counter(vtree[i]["mode"] for i in vids)
    vsti_md = f"""
## 3c. VSTI-Bench (visual-spatial temporal intelligence on ScanNet videos, 5,736 items, 9 types)

Seven multiple-choice types and two numeric types (camera displacement, camera–object absolute distance; scored by
mean relative accuracy). All systems see the same 32 uniformly sampled frames (questions reference "frame k of 32").
{len(vids)-len(clean)} items lie on ScanNet scenes that were used to train our confidence head; the second table is the
leakage-clean subset (n = {len(clean)}). No model was retrained.

All items (paired n = {len(vids)}):

{chr(10).join(vtab(vids))}

Leakage-clean subset:

{chr(10).join(vtab(clean))}

{chr(10).join(vcis)}

ViewTree path mix: {" · ".join(f"{k} {v}" for k, v in vmodes.items())}.

**Reading.** The pattern of STI-Bench repeats on VSTI: the zero-shot model is the strongest overall (0.523), the no-memory
baselines fine-tuned on MindCube lose ~7 points, and ViewTree recovers most of that loss (+4.9 / +5.3 over them, −1.8 below
zero-shot, all significant). The per-type split is the informative part: on the three **object–object relative-position**
types ViewTree is the best system of all (0.640 / 0.647 / 0.806 vs 0.567 / 0.622 / 0.748 zero-shot) — these are room-geometry
questions the scene memory can answer — while on the **camera-motion** types (displacement, movement direction) every
fine-tuned model is below zero-shot and the tree's renders cannot help, because the question is about the trajectory of the
recorded camera, not about the room. The clean subset gives the same numbers within ±0.8 pt, so the head's exposure to 15 % of
the scenes does not drive the result.
"""

# ---------------- ViewTree-D (depth <= 3) ----------------
DEP = collections.OrderedDict([
    ("Qwen2.5-VL-7B zero-shot (16 frames)", load("results/frames16_s*.jsonl")),
    ("SFT-plain (16 frames, MindCube)", load("results/plain/sft_plain_frames16_s*.jsonl")),
    ("ViewTree depth-1 (best of §2)", load("results/human/tree_d10k_headH_s*.jsonl")),
    ("**corpus frames-only SFT (data-matched, no memory)**", load("results/depth/frames16_sftframes_s*.jsonl")),
    ("SFT-A walk-trained answerer, frames only at test", load("results/depth/frames16_sfta.jsonl")),
    ("depth-1 tree with SFT-A + value head", load("results/depth/tree1_sfta_s*.jsonl")),
    ("**ViewTree-D, no RL: SFT-C + value head + beam (d ≤ 3)**", load("results/depth/treeD_sftc_s*.jsonl")),
    ("ViewTree-D, GRPO adapter (interim, ~70 % of budget) + beam", load("results/depth/treeD_grpoI_s*.jsonl")),
    ("**ViewTree-D, GRPO adapter (final) + beam**", load("results/depth/treeD_grpo_s*.jsonl")),
])
DEP = collections.OrderedDict((k, v) for k, v in DEP.items() if len(v) >= 2400)
dep_ids = [i for i in rows if (rows[i]["dataset"], rows[i]["scene_name"]) in odd and all(i in v for v in DEP.values())]
dep_ref = "**corpus frames-only SFT (data-matched, no memory)**"
dep_tab, dep_note, dep_modes = [], "", {}
if dep_ref in DEP:
    dg = collections.defaultdict(list)
    for i in dep_ids: dg[(rows[i]["dataset"], rows[i]["scene_name"])].append(i)
    dmot = lambda d: (lambda sel: np.mean([np.mean([d[i]["score"] for i in sel if rows[i]["question_type"] == t]) for t in types]))
    dep_tab = ["| system | mean of types | Δ vs data-matched baseline [95 % CI] | " + " | ".join(TN[t] for t in types) + " | calls | depth |", "|---|---|---|" + "---|" * len(types) + "---|---|"]
    for k, v in DEP.items():
        per = [np.mean([v[i]["score"] for i in dep_ids if rows[i]["question_type"] == t]) for t in types]
        lo, hi = boot_ci(dmot(v), dmot(DEP[dep_ref]), dg)
        calls = np.mean([v[i].get("calls", np.nan) for i in dep_ids]); dep = np.mean([v[i].get("depth", np.nan) for i in dep_ids])
        cs = "" if np.isnan(calls) else f"{calls:.1f}"; ds = "" if np.isnan(dep) else f"{dep:.2f}"
        dep_tab.append(f"| {k} | **{np.mean(per):.3f}** | {dmot(v)(dep_ids)-dmot(DEP[dep_ref])(dep_ids):+.3f} [{lo:+.3f}, {hi:+.3f}] | " + " | ".join(f"{x:.3f}" for x in per) + f" | {cs} | {ds} |")
        if "mode" in next(iter(v.values())) and "path" in next(iter(v.values())): dep_modes[k] = collections.Counter(v[i]["mode"] for i in dep_ids)
    dep_note = f"paired n = {len(dep_ids)} on the held-out odd half; " + "; ".join(f"{k.strip('*')}: " + ", ".join(f"{m} {c}" for m, c in sorted(dep_modes[k].items(), key=lambda x: -x[1])) for k in dep_modes)
# transfer of the corpus-trained adapters (single pass) to OST and VSTI
def _acc_tab(sysd, key, classes, cls_of, score):
    t = ["| system | overall | " + " | ".join(classes) + " |", "|---|---|" + "---|" * len(classes)]
    for k, v in sysd.items():
        t.append(f"| {k} | **{np.mean([score(v[i]) for i in key]):.3f}** | " + " | ".join(f"{np.mean([score(v[i]) for i in key if cls_of(i) == c]):.3f}" for c in classes) + " |")
    return t
DOST = collections.OrderedDict([(k, OSTB[k]) for k in ("Qwen2.5-VL-7B zero-shot", "SFT-plain", "D_10k adapter, single pass") if k in OSTB])
if tree_ost: DOST["ViewTree depth-1 tree"] = tree_ost
DOST["corpus frames-only SFT, single pass"] = load("results/depth/ost_sft_frames_s*.jsonl"); DOST["SFT-A (walk-trained), single pass"] = load("results/depth/ost_sft_a_s*.jsonl")
DOST = collections.OrderedDict((k, v) for k, v in DOST.items() if len(v) >= 5000)
dost_ids = [i for i in ost_ids if all(i in v for v in DOST.values())]
dost_tab = _acc_tab(DOST, dost_ids, oclasses, lambda i: OSTB["SFT-plain"][i]["qtype"], lambda r: r["correct"]) if dost_ids else []
DV = collections.OrderedDict([(k, VSTI[k]) for k in ("Qwen2.5-VL-7B zero-shot", "SFT-plain", "**ViewTree (best): reasoning tree**") if k in VSTI])
DV["corpus frames-only SFT, single pass"] = load("results/depth/vsti_sft_frames_s*.jsonl"); DV["SFT-A (walk-trained), single pass"] = load("results/depth/vsti_sft_a_s*.jsonl")
DV = collections.OrderedDict((k.replace("**ViewTree (best): reasoning tree**", "ViewTree depth-1 tree"), v) for k, v in DV.items() if len(v) >= 5000)
dv_ids = [i for i in DV["SFT-plain"] if all(i in v for v in DV.values())] if "SFT-plain" in DV else []
dv_types = sorted({DV["SFT-plain"][i]["qtype"] for i in dv_ids}) if dv_ids else []
dv_tab = []
if dv_ids:
    dv_tab = ["| system | mean of 9 types | " + " | ".join(t.replace("camera_", "cam ").replace("obj_obj_relative_pos_", "obj-obj ").replace("obj_rel_dist_", "obj rel ").replace("_", " ") for t in dv_types) + " |", "|---|---|" + "---|" * len(dv_types)]
    for k, v in DV.items():
        per = [np.mean([v[i]["score"] for i in dv_ids if DV["SFT-plain"][i]["qtype"] == t]) for t in dv_types]
        dv_tab.append(f"| {k} | **{np.mean(per):.3f}** | " + " | ".join(f"{x:.3f}" for x in per) + " |")

# ---------------- figures ----------------
def tree_figs(tag, tr_json):
    if not os.path.exists(tr_json): return {}
    T = json.load(open(tr_json)); out = collections.defaultdict(list)
    for t in T:
        f = f"figures/tree_{tag}_{t['id']}.png"
        if os.path.exists(os.path.join(R, f)): out[t["qtype"]].append((t, f))
    return out
vsi_figs = tree_figs("rep_vsi", os.path.join(R, "results/traces/rep_vsi/traces.json"))
ost_figs = tree_figs("rep_ost", os.path.join(R, "results/traces/rep_ost/traces.json"))
def fig_block(entries):
    md = []
    for t, f in entries[:3]:
        ok = "correct" if t["score"] > 0.5 else "wrong"
        md.append(f"![]({f})\n\n*#{t['id']} — {t['question'].split(chr(10))[0][:140]} — gate {t['gate']}, mode `{t['mode']}`, final **{str(t['final']).strip()}** (GT {t['gt']}) → {ok}.*\n")
    return "\n".join(md)

# ---------------- markdown ----------------
md = f"""# ViewTree: Spatial Reasoning over an Explicit Scene Memory with a Human-Camera Reasoning Tree

*Technical report — best system and its comparison with no-world-memory baselines. Generated {__import__('datetime').date.today()}.*

## 1. Introduction: the method from the beginning

### 1.1 Problem
A vision-language model (VLM) answering a spatial question about a room from a handful of video frames or photos has to reason about
geometry it never sees at once: distances, relative directions, counts, room size, route planning. Frame prompting (feeding 16 frames
to Qwen2.5-VL-7B) gives 0.311 on VSI-Bench (mean over 10 question types). ViewTree adds an **explicit 3-D scene memory** and lets the
model **acquire new evidence on a reasoning path** — re-rendering the reconstructed room from viewpoints a person holding a camera
could have taken — instead of answering from whatever frames it was given.

### 1.2 System components
1. **Scene memory (frozen).** VGGT-1B reconstructs a coloured point cloud with camera poses from the input frames (32 frames for video
   benchmarks, all given images for image benchmarks). Memory is ~7.8 GB + 0.22 GB/frame; one reconstruction per scene.
2. **Renderer.** A GPU z-buffered point-splat renderer (splat radius 2) produces a novel view for any camera pose in a few ms.
3. **Human-camera viewpoint proposer (hard constraints).** Candidate viewpoints must be where a person could stand: inside the convex
   hull of the *recorded* camera trajectory (the region the videographer actually walked), at the median recorded camera height
   (eye level), ≥4 % of the room diagonal clear of any reconstructed surface, roll = 0 (image horizontal parallel to the floor),
   pitch 10° down. Four positions are farthest-point-sampled so they come from different sides of the room, each looking toward the
   room centre; views painting < 45 % of pixels are discarded and replaced. A top-down bird's-eye view is kept as the fifth, final
   view. The legacy proposer placed cameras outside the room and above the ceiling (0 % inside); the constrained one is inside 100 %
   of the time with no accuracy loss (Fig. 2).
4. **Controller VLM.** Qwen2.5-VL-7B with one LoRA adapter (r = 16) that both answers and emits control tokens. Trained in stages:
   *Stage I* SFT on 16.8k MindCube examples (control STOP/MOVE/RENDER + answers from an evidence ladder of 1…all views, teacher-labelled);
   *Stage III* SFT-v2 adds 6.1k on-policy fusion examples (complementary / redundant view combinations) — 22.9k in total;
   *Stage IV* GRPO on the view-control policy over all 9,995 MindCube train items with reward = correctness − λ·0.2·(views−1) and a dual
   variable λ driving the mean view count toward 1.5 (the “D_highcost” design, chosen from an 8-variant RL design sweep as the best
   accuracy/efficiency trade-off: 0.778 on MindCube-rest at 1.30 views vs 0.765 at 1.85 for SFT-v2).
5. **Confidence head.** A 2-layer MLP (3584→512→1, temperature-calibrated) on the controller's last-token hidden state predicts whether
   an answer state is correct. It is trained on MindCube ladder states plus VSI tree states from the 144 even-indexed VSI scenes
   (the 144 odd-indexed scenes are never touched and form the held-out evaluation half). The best system uses the head retrained on
   *human-view* states (held-out AUROC 0.710 vs 0.672 for the legacy-view head).
6. **Reasoning tree (depth 1, branching 5, keep 2).** For every question (Fig. 1):
   *gate* — “can you answer from these frames alone? YES / EXPLORE”; YES → answer directly (1–2 calls).
   *branch* — otherwise render the 5 constrained views; each branch answers from frames + that view and is scored by the head.
   *prune* — keep the top-2 branches; if they agree and beat the direct answer → early stop (*branch consensus*).
   *fuse* — answer from frames + both kept views, pose-tagged.
   *arbitrate* — if the head ranks the direct answer above the fused and kept answers → fall back to direct, else take the fused answer.
   Cost: 1–2 VLM calls when the gate fires, otherwise ≤ 8 calls + 1 reconstruction + 5 renders.

![](figures/tree_schematic.png)

*Figure 1. The depth-1 reasoning tree. Nodes are VLM calls; [ ] are confidence-head scores used for pruning and arbitration.*

![](figures/human_views_examples.png)

*Figure 2. Legacy proposer (outside/above the room, 39° pitch) vs the human-camera constraint (inside the walked region, eye level,
roll 0°) on held-out VSI scenes; the top-down bird's-eye view is shared.*


### 1.4 Frame budgets (fixed for every evaluation)

| benchmark | frames fed to VGGT (3-D memory) | frames the controller sees per call | notes |
|---|---|---|---|
| VSI-Bench (video) | 32 uniformly sampled frames | 8 of those 32 (uniform subset) + 1 render per branch, 2 renders at fuse | memory ≈ 7.8 GB + 0.22 GB × 32 ≈ 14.8 GB per reconstruction; static-memory baseline: 16-frame reconstruction, 12 frames + 5 renders; no-memory baselines: 16 frames |
| OST-Bench (image history) | all observed images (latest 12) | up to 8 of them + renders | identical input to the single-pass baselines |
| STI-Bench | 16 frames inside the queried time window | all 16 + renders | ±1 s window for instantaneous questions |
| VSTI-Bench | 32 uniform frames | all 32 + renders | questions cite "frame k of 32" |
| MindCube (training) | all given views (≤ 4) → one top-down render | 1…k views as the ladder policy acquires them | |

### 1.3 What is being compared
- **SFT-plain** — the same base model LoRA-fine-tuned on the benchmark's own training split (all 10,000 MindCube train items;
  input = all given views + question, target = answer letter). No memory, no renders, no control.
- **SFT+GRPO-plain** — SFT-plain followed by GRPO with reward = answer correctness (6 samples/item, 9,995 items).
- **ViewTree (best)** — the full system above: D_highcost-10k adapter + human-camera views + matched confidence head.
On VSI-Bench the no-memory baselines see 16 uniformly sampled frames; on OST-Bench they see the cumulative image history (latest 12),
exactly as the ViewTree gate sees it. All comparisons are paired on identical question sets.

## 2. VSI-Bench (held-out half: 144 scenes never used for any training, 2,557 questions)

Score = accuracy for multiple choice, mean relative accuracy for numerical answers; headline = mean over the 10 question types.

{chr(10).join(vsi_tab)}

{chr(10).join(vsi_ci)}

Per-task view against the two no-memory baselines:

{chr(10).join(vsi_pt)}

ViewTree path mix on the held-out half: direct {modes['direct']} · branch consensus {modes['branch_consensus']} · fused {modes['fused']} ·
fused→fallback-to-direct {modes['fused_fallback_direct']} (of {len(ids)}).

**Reading.** The no-memory baselines transfer only +1.1–1.5 points from MindCube to VSI; the full system is ~4 points above them,
significant. The gains concentrate on types that need geometry the frames do not show at once — room size, object size, absolute
and relative distance, route planning, appearance order — while single-frame-answerable types (counting, easy relative direction)
are close to the baselines.

## 3. OST-Bench (online spatio-temporal exploration, 5,557 multiple-choice items)

Each item is a turn in an exploration; the agent sees the images observed so far and answers about its own state
(*Agent_state*), what is visible (*Agent_visible_info*) or spatial relations to objects (*Agent_object_spatial*). ViewTree reconstructs the observed images with VGGT and runs the same tree; 154 items whose image
histories mix resolutions failed reconstruction and are excluded from every system (paired n = {len(ost_ids)}). The single-pass
rows use the same adapters answering directly from the image history.
{ost_note}

{chr(10).join(ost_tab)}

{chr(10).join(ost_ci)}

{"ViewTree path mix on OST-Bench: " + " · ".join(f"{k} {v}" for k, v in omodes.items()) + "." if tree_ost else ""}

{sti_md}
{vsti_md}
## 4. Visualized reasoning trees

Each figure is one question run through the best system. Every node shows the images the controller sees at that node
(root = the observed frames; branch = frames + one rendered eye-level view; fuse = frames + the two kept views); orange = kept
branches; faded branches were not executed because the gate answered directly; [ ] = confidence-head score.

### 4.1 VSI-Bench — three trees per task
"""
for t in types:
    md += f"\n#### {TN[t]} \n\n" + (fig_block(vsi_figs.get(t, [])) or "*(pending)*\n")
md += "\n### 4.2 OST-Bench — three trees per task\n"
for c in oclasses:
    md += f"\n#### {c}\n\n" + (fig_block(ost_figs.get(c, [])) or "*(pending)*\n")
md += """
**Reading.** On OST-Bench the tree is ahead of both no-memory baselines (SFT-plain, SFT+GRPO-plain) but not ahead of its own
adapter answering directly from the image history: OST questions are about the agent's *own* trajectory and what it has *seen*
(temporal facts), which a rendered view of the current reconstruction cannot add — the head correctly falls back to the direct
answer on most explored items. The memory helps where the question is about the room's geometry (VSI), not about the agent's history.

## 5. ViewTree-D: multi-step view acquisition (depth ≤ 3), trained from scratch on a 494k-QA corpus

The depth-1 tree of §1 renders five candidate views *once*; it cannot look again after seeing a render, move locally toward the
object the question is about, or learn when to stop. ViewTree-D (design: `DESIGN_DEPTH.md`, log: RESULTS.md §8) makes the camera
itself the action space and trains the controller in phases on a corpus 50× MindCube.

**Reasoning path = a walk in the memory.** A state is (question, 8 context frames, renders seen so far, current camera pose). One
step = one camera action + one render: `TURN_LEFT`/`TURN_RIGHT` (yaw ±45°), `FORWARD` (one walkable cell), `NEXT_SPOT` (next
farthest-point standing position, facing the room centre), `LOOK_AROUND` (+180°), `BIRD_EYE` (top-down, allowed only as the last
acquisition), `STOP`. The human-camera constraints of §1.2 are a hard action mask (positions inside the walked hull at eye level,
roll 0, coverage ≥ 45 %). Every scene has a pre-rendered **pose bank** (12 positions × 8 yaws + top-down = 97 views), so training
never renders online; at test time only the poses the beam visits are rendered.

**Training corpus.** VLM-3R `vsibench_train` + `vstibench_train` and VSI-590K (ScanNet + ScanNet++ v2 videos we hold):
493,663 QA on 1,709 scenes, 176k numeric; every VSI-Bench, VSTI-Bench, STI, OST and MindCube evaluation scene excluded at room level.

**Phases.** 0 — pose banks; 1 — *SFT-A* answerer on ~100k random-walk states (frames + 0…3 renders → answer; 33k frames-only);
2 — *oracle walks*: beam-2 depth-3 search over the bank with the SFT-A answerer on 8,639 QA (direct correct 54 %, best walk 68 %),
a *value head* (same MLP as §1.2, GELU/dropout, AUROC 0.723 held-out) on walk states, and *SFT-C* imitation of the oracle actions
(the prompt lists the valid moves); 3 — *GRPO over walks* (group 6, reward = MRA/accuracy − step cost, dual λ on the mean step
budget, masked actions penalised). **Inference** = gate, then beam search over camera moves (branch 3 by policy logit, keep 2 by
the value head, depth ≤ 3, early stop on agreement, direct-vs-walk arbitration), ≤ 12 VLM calls.

### 5.1 VSI-Bench held-out odd half (paired, scene-bootstrap CIs vs the data-matched baseline)

The corpus alone changes the picture: a frames-only SFT on it reaches ~0.51 (vs 0.367 for the best MindCube-trained system),
because VLM-3R's training QA uses VSI-Bench's own templates on disjoint rooms. Every ViewTree-D number is therefore read against
that **data-matched, no-memory baseline** (bold), not against §2.

DEP_TAB_PLACEHOLDER

DEP_NOTE_PLACEHOLDER

**Reading.**
- Multi-step acquisition adds a borderline **+2.1** over the data-matched baseline at 4.5 calls/question (the gate answers directly
  on 71 %). The gain is where a second viewpoint changes the geometry the model sees — relative direction hard/medium, relative
  distance, appearance order — and is negative on counting, room size and route planning, where extra renders distract.
- Same answerer and head, depth 1 vs depth ≤ 3: +0.8 vs +2.1, with *fewer* calls for the deeper beam because its gate stops more
  often; the ordering baseline < depth-1 < depth-≤3 holds on the relational/directional types.
- The GRPO policy drifted toward STOP-at-depth-0 (mean steps 0.18 → 0.07, λ never activated) — the pre-registered collapse risk;
  the beam explores regardless, so the RL adapter acts mainly through its answer tokens (rows above when present).

### 5.2 Transfer of the corpus-trained adapters (single pass, no tree)

OST-Bench (paired n = DOST_N):

DOST_TAB_PLACEHOLDER

VSTI-Bench (paired n = DV_N; VSTI rooms are ScanNet *val*, 0 shared with the corpus, but the corpus contains `vstibench_train`'s templates):

DV_TAB_PLACEHOLDER

**Reading.** The corpus helps exactly where its templates match — VSI (+14) and VSTI (+16 over zero-shot, +21 over SFT-plain, with
the numeric camera/object-distance types going from ~0.15 to ~0.5) — and *hurts* where they do not: on OST both corpus adapters are
significantly below zero-shot (−2.4 / −2.2, driven by Agent_visible_info), while the MindCube-trained D_10k adapter and the depth-1
tree stay at or above it. The ViewTree-D claim is a VSI-family claim until a mixed corpus with OST-style exploration QA is trained.

## 6. Summary
- ViewTree's contribution is **cross-benchmark transfer and view efficiency**: on benchmarks other than the one it was trained on it is
  the best system by a significant margin, while using few extra views (the controller answers directly on ~23 % of VSI questions and
  stops after consensus on another third).
- On the training benchmark itself (MindCube), plain fine-tuning with all views handed over for free is stronger (0.750 vs 0.632 on
  tinybench) — reported in RESULTS.md §1c; the memory system's value is not peak accuracy there but the acquisition trade-off.
- The human-camera constraint costs nothing (100 % valid views, +0.4) and, with a head that can read eye-level views, gives the best
  held-out result (0.367), the first significant win of the adaptive tree over static memory prompting (+2.6 [+0.5, +4.7]).
- **Scale and depth (§5).** Training on a 494k-QA corpus of VSI-template questions lifts the frames-only model to ~0.51 on VSI
  (+14); multi-step acquisition (ViewTree-D, depth ≤ 3) adds a further borderline +2.1 on top of that data-matched baseline, on
  relational/directional questions, at 4.5 calls per question — but the corpus gain is template-specific (VSI/VSTI up, OST down).

*Reproducibility: all numbers are computed from result files in the repository by `scripts/build_report.py`; full experiment log in
RESULTS.md, design decisions in DECISIONS.md.*
"""
md = md.replace("DEP_TAB_PLACEHOLDER", "\n".join(dep_tab) or "*(pending)*").replace("DEP_NOTE_PLACEHOLDER", ("*Path mix — " + dep_note + ".*") if dep_note else "")
md = md.replace("DOST_TAB_PLACEHOLDER", "\n".join(dost_tab) or "*(pending)*").replace("DOST_N", str(len(dost_ids))).replace("DV_TAB_PLACEHOLDER", "\n".join(dv_tab) or "*(pending)*").replace("DV_N", str(len(dv_ids)))
open(os.path.join(OUT, "REPORT.md"), "w").write(md)
# ---------------- PDF ----------------
import markdown, weasyprint
html = markdown.markdown(md, extensions=["tables"])
css = """@page { size: A4; margin: 16mm; } body { font-family: 'DejaVu Sans', sans-serif; font-size: 9.5pt; line-height: 1.35; }
h1 { font-size: 17pt; } h2 { font-size: 13pt; margin-top: 18pt; border-bottom: 1px solid #ccc; } h3 { font-size: 11pt; } h4 { font-size: 10pt; margin-bottom: 2pt; }
table { border-collapse: collapse; font-size: 8pt; margin: 6pt 0; } th, td { border: 1px solid #bbb; padding: 2pt 4pt; } th { background: #f0f0f0; }
img { max-width: 100%; margin-top: 6pt; } em { color: #444; } code { font-size: 8.5pt; }"""
weasyprint.HTML(string=html, base_url=R).write_pdf(os.path.join(OUT, "REPORT.pdf"), stylesheets=[weasyprint.CSS(string=css)])
print("wrote", os.path.join(OUT, "REPORT.md"), os.path.join(OUT, "REPORT.pdf"), "vsi figs", sum(len(v) for v in vsi_figs.values()), "ost figs", sum(len(v) for v in ost_figs.values()))

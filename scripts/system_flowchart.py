"""Flowchart of the entire ViewTree system: offline scene pipeline, per-question inference
(depth-1 tree + ViewTree-D beam), training lanes, planned mobile runtime.
  python scripts/system_flowchart.py  -> figures/system_flowchart.png"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, MUT, GRID = "#1f2328", "#57606a", "#c9d1d9"
TEAL, TEALBG = "#0f6e73", "#e3f0ef"      # trained
BLUE, BLUEBG = "#0969da", "#ddf4ff"      # frozen
AMB, AMBBG = "#9a6700", "#fff8c5"        # decisions
GRN, GRNBG = "#1a7f37", "#dafbe1"        # outputs
PUR, PURBG = "#8250df", "#fbefff"        # data/training
fig, ax = plt.subplots(figsize=(18.5, 11.5)); ax.set_xlim(0, 18.5); ax.set_ylim(0, 11.5); ax.axis("off")

def box(x, y, w, h, title, sub="", fc="white", ec=GRID, lw=1.4, fs=8.6, sfs=7.0, ls="-", tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.09", fc=fc, ec=ec, lw=lw, ls=ls))
    if sub:
        ax.text(x+w/2, y+h-0.06, title, ha="center", va="top", fontsize=fs, color=tc, weight="bold")
        ax.text(x+w/2, y+0.07, sub, ha="center", va="bottom", fontsize=sfs, color=MUT, linespacing=1.35)
    else:
        ax.text(x+w/2, y+h/2, title, ha="center", va="center", fontsize=fs, color=tc, weight="bold")
    return (x, y, w, h)

def A(p, q, color=MUT, ls="-", lw=1.5, label=None, lx=0, ly=0.12, rad=0.0, fs=7.2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=3, shrinkB=3))
    if label: ax.text((p[0]+q[0])/2+lx, (p[1]+q[1])/2+ly, label, fontsize=fs, color=color, ha="center")

def band(x, y, w, h, label):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc="#f6f8fa", ec="#e3e8ee", lw=1.1, zorder=0))
    ax.text(x+0.14, y+h-0.10, label, fontsize=9.5, color=MUT, weight="bold", va="top",
            family="DejaVu Sans", zorder=1)

ax.text(0.15, 11.32, "ViewTree — full system", fontsize=15, weight="bold", color=INK, va="top")
ax.text(0.15, 10.92, "blue = frozen geometry   ·   teal = trained (LoRA / heads)   ·   amber = decision   ·   purple = training data & phases   ·   dashed = planned (Jetson)", fontsize=8, color=MUT, va="top")

# ---- Band 1: offline scene pipeline -------------------------------------------------
band(0.1, 8.05, 18.3, 2.55, "OFFLINE · once per scene")
b_vid  = box(0.35, 8.45, 2.0, 1.5, "Input video /\nimage set", "16–32 frames uniform\n(≤4 views MindCube)", fc="white")
b_vggt = box(2.95, 8.45, 2.1, 1.5, "VGGT-1B", "frozen reconstruction\npoints + colors + poses\n≈7.8 GB + 0.22 GB/frame", fc=BLUEBG, ec=BLUE)
b_mem  = box(5.65, 8.45, 2.1, 1.5, "3-D scene memory", "point cloud, reused by\nevery question & branch", fc=BLUEBG, ec=BLUE)
b_pose = box(8.35, 8.30, 3.6, 1.8, "Human-camera pose proposer", "hard constraints: inside walked hull ·\neye level · clearance ≥4% diag · roll 0 ·\npitch 10° · render coverage ≥45%", fc=BLUEBG, ec=BLUE)
b_cand = box(12.55, 9.35, 2.6, 0.85, "5 candidate views", "4 FPS spots → centre + top-down", fc="white")
b_bank = box(12.55, 8.30, 2.6, 0.85, "Pose bank · 97 views", "12 spots × 8 yaws + top-down", fc="white")
b_rend = box(15.75, 8.45, 2.4, 1.5, "Point-splat renderer", "GPU z-buffer, splat 2, few ms\nlazy render + pose-keyed cache", fc=BLUEBG, ec=BLUE)
A((2.35,9.2),(2.95,9.2)); A((5.05,9.2),(5.65,9.2)); A((7.75,9.2),(8.35,9.2))
A((11.95,9.5),(12.55,9.77), label="depth-1", ly=0.16); A((11.95,8.9),(12.55,8.72), label="ViewTree-D", ly=-0.22)
A((15.15,9.77),(15.75,9.4)); A((15.15,8.72),(15.75,9.0))

# ---- Band 2: per-question inference --------------------------------------------------
band(0.1, 2.55, 18.3, 5.3, "INFERENCE · per question")
b_q    = box(0.35, 5.6, 2.0, 1.5, "Question +\ncontext frames", "8 of 16–32 frames\n(pose-tagged prompt)", fc="white")
b_vlm  = box(2.95, 5.6, 2.5, 1.5, "Qwen2.5-VL-7B\n+ LoRA r16", "controller & answerer\n(one adapter per system)", fc=TEALBG, ec=TEAL)
b_gate = box(6.0, 5.75, 1.85, 1.2, "Gate", "answer from frames\nalone? YES / EXPLORE", fc=AMBBG, ec=AMB)
b_dir  = box(6.9, 3.0, 1.85, 1.0, "Direct answer", "1–2 VLM calls\n(~23–71% of questions)", fc="white")
# depth-1 tree
b_t1 = box(8.6, 6.35, 2.5, 1.15, "Branch ×5", "answer from frames +\neach candidate view", fc="white")
b_t2 = box(11.5, 6.35, 2.3, 1.15, "Keep-2 by head", "consensus? → early stop", fc=AMBBG, ec=AMB)
b_t3 = box(14.2, 6.35, 2.0, 1.15, "Fuse", "frames + both kept\nviews, pose-tagged", fc="white")
ax.text(8.6, 7.72, "Depth-1 tree  (best MindCube-trained system)", fontsize=8.6, color=TEAL, weight="bold")
# viewtree-d beam
b_d1 = box(8.6, 4.35, 2.5, 1.3, "Propose actions", "top-3 by policy logit:\nTURN / FORWARD / NEXT_SPOT /\nLOOK_AROUND / BIRD_EYE / STOP", fc="white")
b_d2 = box(11.5, 4.35, 2.3, 1.3, "Score states", "answer + value head\nper new render", fc="white")
b_d3 = box(14.2, 4.35, 2.0, 1.3, "Keep-2 · depth<3?", "consensus → stop\nelse walk deeper ↺", fc=AMBBG, ec=AMB)
ax.text(8.6, 5.85, "ViewTree-D beam  (corpus-trained, depth ≤ 3 walks)", fontsize=8.6, color=TEAL, weight="bold")
b_head = box(3.55, 3.0, 2.6, 1.15, "Confidence / value head", "MLP 3584→512→1, temperature-\ncalibrated · prunes & arbitrates", fc=TEALBG, ec=TEAL)
b_arb  = box(11.5, 3.0, 2.3, 1.0, "Arbitrate", "head ranks direct vs\nfused / best walk state", fc=AMBBG, ec=AMB)
b_ans  = box(14.6, 2.95, 3.4, 1.1, "Answer + confidence + trace", "mean 4.5 calls (beam) · ≤8 (depth-1)", fc=GRNBG, ec=GRN)
A((2.35,6.35),(2.95,6.35)); A((5.45,6.35),(6.0,6.35))
A((6.92,5.75),(7.6,4.0), label="YES", lx=0.38, ly=0.1, color=GRN)
A((7.85,6.6),(8.6,6.9), label="EXPLORE", ly=0.14, color=AMB)
A((7.85,6.1),(8.6,5.2), label="EXPLORE (D)", lx=-0.15, ly=0.16, color=AMB)
A((11.1,6.9),(11.5,6.9)); A((13.8,6.9),(14.2,6.9)); A((15.2,6.35),(12.9,4.0), rad=-0.25)
A((11.1,5.0),(11.5,5.0)); A((13.8,5.0),(14.2,5.0))
A((15.2,5.65),(9.85,5.68), rad=0.22, color=MUT, ls=":", label="walk deeper (≤3)", ly=0.42, lx=0.4)
A((16.2,4.35),(16.2,4.05), color=GRN)
A((13.8,3.5),(14.6,3.5)); A((8.75,3.5),(11.5,3.5))
A((5.6,4.15),(11.6,6.35), rad=0.12, color=TEAL, ls="--", lw=1.2)
A((5.9,4.15),(11.6,4.6), rad=0.06, color=TEAL, ls="--", lw=1.2, label="scores", lx=-1.9, ly=0.12)

# ---- Band 3: training ----------------------------------------------------------------
band(0.1, 0.15, 12.3, 2.2, "")
ax.text(0.24, 2.2, "TRAINING (offline)", fontsize=9.5, color=MUT, weight="bold", va="top")
b_mc = box(0.35, 0.35, 2.15, 1.45, "MindCube 10k", "Stage I ladder SFT 16.8k →\nStage III fusion SFT-v2 →\nStage IV GRPO ladder (D_10k)", fc=PURBG, ec=PUR)
b_co = box(2.95, 0.35, 3.3, 1.45, "Corpus 494k QA · 1,709 scenes", "Phase 0 pose banks → Phase 1 SFT-A →\nPhase 2 oracle walks + value head + SFT-C →\nPhase 3 GRPO over walks", fc=PURBG, ec=PUR)
b_hd = box(6.85, 0.35, 3.0, 1.45, "Head training", "state → P(correct), outcome labels,\ntemperature on held-out scenes\n(AUROC 0.710 / 0.723)", fc=PURBG, ec=PUR)
ax.text(10.35, 1.15, "scene-level splits everywhere;\nall evaluation scenes excluded", fontsize=7.2, color=MUT, va="center")
A((1.4,1.8),(3.6,5.6), rad=-0.15, color=PUR, ls="--", lw=1.2, label="LoRA adapters", lx=-0.9, ly=0.3)
A((4.6,1.8),(4.2,5.6), rad=0.1, color=PUR, ls="--", lw=1.2)
A((8.35,1.8),(5.2,3.0), rad=0.12, color=PUR, ls="--", lw=1.2, label="heads", lx=0.7, ly=-0.05)

# ---- Planned mobile runtime ----------------------------------------------------------
box(12.7, 0.55, 5.7, 1.6, "Mobile runtime — Jetson (planned, M6)", "versioned scene store · pose/token caches · shared KV prefix per\nbranch point · per-level batching · scheduler (B_t, D_t, C_t) from\ndeadline / battery / memory / thermal — caps beam width & depth", fc="white", ec=MUT, ls="--", tc=MUT)
A((15.5,2.15),(15.5,2.55), color=MUT, ls="--", lw=1.2)
fig.savefig(os.path.join(R, "figures/system_flowchart.png"), dpi=150, bbox_inches="tight")
print("wrote figures/system_flowchart.png")

"""Design-section figures for the ViewTree paper, styled after the SpatialMind
example (dashed stage boxes, numbered badges, real-image film strips, prompt
boxes).  Outputs PDF+PNG into the paper's figures/ dir."""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(R, "paper/ViewTree__Resource_Aware_On_Device_Spatial_Reasoning_via_Adaptive_Viewpoint_Branching_and_Fusion__1_/figures")
KA = os.path.join(R, "figures/motivation_kitchen_assets")
INK, MUT = "#1f2328", "#57606a"
RED, MAG, PUR, TEALD, OK, BAD, AMB = "#d62828", "#c2185b", "#7b1fa2", "#155e63", "#1a7f37", "#cf222e", "#b26a00"

def box(ax, x, y, w, h, text, fc="white", ec="#9aa4b1", lw=1.1, fs=7.6, color=INK, ls="-", weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05", fc=fc, ec=ec, lw=lw, ls=ls, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, linespacing=1.25, zorder=4, weight=weight)
def group(ax, x, y, w, h, color=RED):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", fc="none", ec=color, lw=1.4, ls=(0, (5, 3)), zorder=1))
def badge(ax, x, y, n, label, color=RED):
    ax.add_patch(Circle((x, y), 0.09, fc=color, ec="none", zorder=6))
    ax.text(x, y, str(n), fontsize=8, color="white", ha="center", va="center", zorder=7, weight="bold")
    ax.text(x + 0.14, y, label, fontsize=8.2, color=color, ha="left", va="center", zorder=7, weight="bold")
def arr(ax, p, q, color=MUT, ls="-", lw=1.5, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, color=color, lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2, zorder=5))
def img(ax, path, x, y, w, ec="#6b7280", lw=1.1, ls="-"):
    im = mpimg.imread(path); h = w * im.shape[0] / im.shape[1]
    ax.imshow(im, extent=(x, x + w, y, y + h), aspect="auto", zorder=2)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", fc="none", ec=ec, lw=lw, ls=ls, zorder=4)); return h
def strip(ax, paths, x, y, w=0.62, gap=0.05):
    for j, p in enumerate(paths): img(ax, p, x + j * (w + gap), y, w)
    return x + len(paths) * (w + gap) - gap

# ================ fig_action_space ================
fig, ax = plt.subplots(figsize=(7.2, 3.4)); ax.set_xlim(0, 7.2); ax.set_ylim(0, 3.4); ax.axis("off")
group(ax, 0.08, 0.42, 2.9, 2.85, RED); badge(ax, 0.62, 0.26, 1, "Discrete camera actions", RED)
ax.text(1.53, 3.12, "standing spots at eye level,\ninside the recorded region", fontsize=6.9, color=TEALD, ha="center", va="center")
img(ax, f"{KA}/bev_no_ceiling.jpg", 0.22, 0.85, 2.35)
spots = [(0.95, 2.4), (1.22, 2.0), (1.55, 1.62), (1.85, 1.3)]
for sx, sy in spots: ax.add_patch(Circle((sx, sy), 0.055, fc=TEALD, ec="white", lw=1, zorder=6))
arr(ax, spots[0], spots[1], color=TEALD, lw=1.7); arr(ax, spots[1], spots[2], color=TEALD, lw=1.7)
for ang in (40, -40):
    a0 = np.deg2rad(ang); p0 = spots[2]
    arr(ax, p0, (p0[0] + 0.45*np.cos(a0), p0[1] + 0.45*np.sin(a0)), color=AMB, lw=1.3, ls="--")
ax.text(0.56, 1.5, "Forward /\nNext-Spot", fontsize=6.8, color=TEALD, ha="center", zorder=7)
ax.text(2.4, 1.28, "turns", fontsize=6.8, color=AMB, ha="center", zorder=7)
ax.text(1.53, 0.66, "Turn-Left / Turn-Right / Look-Around rotate in place; Bird-Eye once, last", fontsize=6.2, color=MUT, ha="center")
group(ax, 3.2, 0.42, 2.0, 2.85, MAG); badge(ax, 3.75, 0.26, 2, "Rendering & validity", MAG)
img(ax, f"{KA}/reconstruction_views/render_spot03_dir2.jpg", 3.42, 2.15, 1.25, ec=OK, lw=1.6)
ax.text(4.05, 2.06, "valid view: offered\nto the controller", fontsize=6.8, color=OK, ha="center", va="top")
img(ax, f"{R}/figures/motivation_kitchen_assets/render_invalid_example.jpg", 3.42, 0.95, 1.25, ec=BAD, lw=1.6, ls="--")
ax.text(4.05, 0.86, "invalid (34% coverage):\naction removed", fontsize=6.8, color=BAD, ha="center", va="top")
box(ax, 5.5, 0.62, 1.6, 2.45, "pose checks\n\ninside recorded\nregion\n\neye height,\nlevel horizon\n\nclearance from\nsurfaces\n\nrendered\ncoverage", fs=6.9, fc="#f6f8fa")
arr(ax, (5.22, 1.85), (5.5, 1.85), color=MAG)
fig.savefig(f"{OUT}/fig_action_space.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_action_space.png", dpi=170, bbox_inches="tight"); plt.close(fig); print("action_space")

# ================ fig_control_prompt (SpatialMind Fig-6 style) ================
fig, ax = plt.subplots(figsize=(7.2, 2.6)); ax.set_xlim(0, 7.2); ax.set_ylim(0, 2.6); ax.axis("off")
xe = strip(ax, [f"{KA}/frames_64/frame_{k:02d}.jpg" for k in (6, 22, 38, 54)], 0.15, 1.75, w=0.66)
ax.text(0.15 + 1.4, 1.62, "context frames", fontsize=7, color=RED, ha="center", va="top", weight="bold")
arr(ax, (xe + 0.05, 2.0), (xe + 0.4, 2.0))
img(ax, f"{KA}/reconstruction_views/render_spot03_dir2.jpg", xe + 0.45, 1.7, 0.9, ec=TEALD, lw=1.5)
ax.text(xe + 0.9, 1.6, "view acquired at step 1\n(pose tag: spot 3, direction 2)", fontsize=6.8, color=TEALD, ha="center", va="top")
box(ax, 4.6, 1.55, 2.5, 0.95, "Instruction: you may answer now or move\nthe camera. Valid moves from here:\nStop, Turn-Left, Turn-Right, Forward,\nNext-Spot. Reply with exactly one token.", fc="#faf5ff", ec=PUR, ls=(0, (4, 2)), fs=6.8, color=PUR)
box(ax, 0.15, 0.25, 3.3, 0.9, "Question: which of these objects (heater,\ntrash can, door, cup) is the closest\nto the microwave?", fc="#fff8f8", ec=RED, ls=(0, (4, 2)), fs=7.0)
box(ax, 3.9, 0.25, 3.2, 0.9, "Controller output (one token, from prefill logits):\n\nTurn-Right   [Stop 0.21  Forward 0.14  ...]", fc="#f6f8fa", ec=INK, fs=7.0)
arr(ax, (3.45, 0.7), (3.9, 0.7))
fig.savefig(f"{OUT}/fig_control_prompt.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_control_prompt.png", dpi=170, bbox_inches="tight"); plt.close(fig); print("control_prompt")

# ================ fig_training_pipeline ================
fig, ax = plt.subplots(figsize=(7.2, 3.1)); ax.set_xlim(0, 7.2); ax.set_ylim(0, 3.1); ax.axis("off")
group(ax, 0.08, 0.5, 2.15, 2.3, RED); badge(ax, 0.75, 0.34, 1, "Answering", RED)
strip(ax, [f"{KA}/frames_64/frame_{k:02d}.jpg" for k in (10, 28, 46)], 0.2, 2.15, w=0.56)
img(ax, f"{KA}/reconstruction_views/render_spot08_dir2.jpg", 1.02, 1.32, 0.75, ec=TEALD, lw=1.3)
ax.text(0.62, 1.62, "+ 0..3\nrendered\nviews", fontsize=6.8, color=TEALD, ha="center")
ax.text(1.15, 1.2, "same correct answer supervises\nevery trajectory prefix", fontsize=6.8, color=MUT, ha="center", va="top")
group(ax, 2.5, 0.5, 2.15, 2.3, MAG); badge(ax, 3.2, 0.34, 2, "Oracle search", MAG)
r0 = (3.55, 2.35); ax.add_patch(Circle(r0, 0.055, fc=INK, ec="white", zorder=6))
for (kx, ky), c, t in [((2.95, 1.75), BAD, "✗"), ((3.55, 1.75), OK, "✓ shortest"), ((4.15, 1.75), OK, "✓")]:
    arr(ax, r0, (kx, ky), color=c, lw=1.4)
    ax.add_patch(Circle((kx, ky), 0.05, fc=c, ec="white", zorder=6))
    ax.text(kx, ky - 0.16, t, fontsize=6.8, color=c, ha="center")
ax.text(3.57, 1.25, "try valid movements; keep the\nshortest trajectory that fixes\nthe answer (ground truth known);\ncorrect at start → Stop, depth 0", fontsize=6.8, color=MUT, ha="center", va="top")
group(ax, 4.95, 0.5, 2.15, 2.3, PUR); badge(ax, 5.6, 0.34, 3, "Policy & confidence", PUR)
box(ax, 5.05, 2.15, 1.95, 0.55, "camera controller:\nimitate oracle actions + Stop", fc="#e3f0ef", ec=TEALD, fs=6.9)
box(ax, 5.05, 1.45, 1.95, 0.55, "confidence head: predict\nanswer correctness per state", fc="#e3f0ef", ec=TEALD, fs=6.9)
box(ax, 5.05, 0.72, 1.95, 0.55, "optional RL over trajectories\n(kept as ablation)", fc="white", ec=MUT, ls=(0, (4, 2)), fs=6.8, color=MUT)
arr(ax, (2.23, 1.65), (2.5, 1.65)); arr(ax, (4.65, 2.4), (5.05, 2.4)); arr(ax, (4.65, 1.72), (5.05, 1.72))
ax.text(0.08, 3.04, "all labels come from answer outcomes on the training corpus; no human annotation of camera trajectories", fontsize=7, color=MUT, va="top")
fig.savefig(f"{OUT}/fig_training_pipeline.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_training_pipeline.png", dpi=170, bbox_inches="tight"); plt.close(fig); print("training_pipeline")

# ================ fig_runtime ================
fig, ax = plt.subplots(figsize=(7.2, 2.75)); ax.set_xlim(0, 7.2); ax.set_ylim(0, 2.75); ax.axis("off")
group(ax, 0.08, 0.35, 2.15, 2.1, RED); badge(ax, 0.85, 0.2, 1, "Scene reuse", RED)
box(ax, 0.18, 1.5, 1.95, 0.8, "reconstruct once per scene\n11.3 s warm → 0.7 s and 32 J\nper question amortized", fc="#ddf4ff", ec="#0969da", fs=6.9)
box(ax, 0.18, 0.55, 1.95, 0.8, "pose-keyed caches: rendered\nviews (0.12 s each) and their\ntokens; repeats are free", fc="#ddf4ff", ec="#0969da", fs=6.9)
group(ax, 2.5, 0.35, 2.15, 2.1, MAG); badge(ax, 3.3, 0.2, 2, "Cheap calls", MAG)
box(ax, 2.6, 1.5, 1.95, 0.8, "per-level batching: sibling\nviews rendered and encoded\ntogether, no model switching", fc="#fdf2f8", ec=MAG, fs=6.9)
box(ax, 2.6, 0.55, 1.95, 0.8, "decode-free control: action\nscores from prefill logits;\nfused confidence readout\nsaves 4.0 s per scored answer", fc="#fdf2f8", ec=MAG, fs=6.6)
group(ax, 4.95, 0.35, 2.15, 2.1, PUR); badge(ax, 5.75, 0.2, 3, "Bounded search", PUR)
box(ax, 5.05, 1.5, 1.95, 0.8, "width 3, keep 2, depth ≤ 3;\ngate exits after 2 calls on\n71% of questions", fc="#faf5ff", ec=PUR, fs=6.9)
box(ax, 5.05, 0.55, 1.95, 0.8, "expected cost 4.7 calls:\n24.5 s / 1104 J per question\n(Jetson AGX Orin, 7B)", fc="#faf5ff", ec=PUR, fs=6.9)
arr(ax, (2.25, 1.4), (2.5, 1.4)); arr(ax, (4.67, 1.4), (4.95, 1.4))
ax.text(0.08, 2.68, "one reconstruction serves every question and every search path in the scene; prefill dominates VLM cost, so the runtime batches and shortens prefills", fontsize=7, color=MUT, va="top")
fig.savefig(f"{OUT}/fig_runtime.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_runtime.png", dpi=170, bbox_inches="tight"); plt.close(fig); print("runtime")

# ================ fig_search_example ================
fig, ax = plt.subplots(figsize=(7.4, 4.7)); ax.axis("off")
im = mpimg.imread(f"{R}/figures/treeD_object_rel_distance_2027.png")
ax.imshow(im); fig.savefig(f"{OUT}/fig_search_example.pdf", bbox_inches="tight", dpi=200); plt.close(fig); print("search_example")

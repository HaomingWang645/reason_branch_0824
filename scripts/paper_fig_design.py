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

# ================ fig_overview (replaces the tikz placeholder) ================
TR = os.path.join(R, "figures/treeD_trace")
fig, ax = plt.subplots(figsize=(14.2, 3.4)); ax.set_xlim(0, 14.2); ax.set_ylim(0, 3.4); ax.axis("off")
box(ax, 0.15, 2.3, 1.75, 0.75, "User query:\n\"which object is the\nclosest to the microwave?\"", fc="#fff8f8", ec=RED, ls=(0,(4,2)), fs=7.2)
strip(ax, [f"{KA}/frames_64/frame_{k:02d}.jpg" for k in (6, 30, 54)], 0.2, 1.35, w=0.56)
ax.text(1.05, 1.22, "recent video frames", fontsize=7, color=MUT, ha="center", va="top")
group(ax, 2.35, 0.5, 3.1, 2.7, RED); badge(ax, 3.15, 0.34, 1, "Scene memory (once per scene)", RED)
box(ax, 2.5, 2.35, 1.3, 0.6, "frozen 3D\nreconstruction", fc="#155e63", color="white", ec="#155e63", fs=7.3)
img(ax, f"{KA}/bev_no_ceiling.jpg", 2.5, 0.75, 1.35)
ax.text(3.17, 0.62, "explicit scene memory", fontsize=6.8, color=MUT, ha="center", va="top")
img(ax, f"{KA}/reconstruction_views/render_spot03_dir2.jpg", 4.0, 1.9, 1.3, ec=TEALD, lw=1.3)
ax.text(4.65, 1.8, "renderable from 97\ndiscrete camera poses\n(validity-checked)", fontsize=6.8, color=TEALD, ha="center", va="top")
arr(ax, (1.95, 2.0), (2.35, 2.0))
group(ax, 5.75, 0.5, 2.0, 2.7, MAG); badge(ax, 6.35, 0.34, 2, "Exploration gate", MAG)
box(ax, 5.87, 2.2, 1.75, 0.75, "direct answer from\nframes + learned gate:\nenough information?", fc="#fdf2f8", ec=MAG, fs=7.0)
box(ax, 5.87, 1.25, 1.75, 0.55, "Yes (71%):\nreturn direct answer", fc="#dafbe1", ec=OK, fs=7.0)
box(ax, 5.87, 0.65, 1.75, 0.42, "Explore: start search", fc="white", ec=MAG, fs=7.0)
arr(ax, (5.45, 2.0), (5.75, 2.0)); arr(ax, (6.75, 2.2), (6.75, 1.8), color=OK); arr(ax, (6.75, 1.25), (6.75, 1.07), color=MAG)
group(ax, 8.1, 0.5, 4.4, 2.7, PUR); badge(ax, 9.0, 0.34, 3, "Confidence-guided camera walks", PUR)
for j, (v, yb) in enumerate([("view16", 2.15), ("view23", 1.25), ("view0", 0.72)]):
    w = 0.95
    hh = img(ax, f"{TR}/2027/{v}.jpg", 8.25, yb, w, ec=("#2f6feb" if j < 2 else "#9aa4b1"), lw=(1.6 if j < 2 else 1.0))
arr(ax, (7.62, 0.86), (8.25, 1.6), color=MAG)
ax.text(9.25, 2.6, "expand 3 actions per path,\nkeep 2 by calibrated confidence", fontsize=6.9, color=MUT, ha="left")
img(ax, f"{TR}/2027/view57.jpg", 10.55, 1.5, 1.0, ec=OK, lw=1.6)
arr(ax, (9.25, 2.0), (10.55, 2.0), color=PUR)
ax.text(9.9, 2.08, "walk deeper\n(depth ≤ 3)", fontsize=6.8, color=PUR, ha="center")
ax.text(11.05, 1.4, "retained paths agree →\nstop early", fontsize=6.8, color=OK, ha="center", va="top")
box(ax, 11.75, 2.15, 0.62, 0.55, "cup ✓", fc="#dafbe1", ec=OK, fs=7.5)
arr(ax, (11.6, 2.2), (11.75, 2.35), color=OK)
box(ax, 12.62, 0.8, 1.45, 2.0, "final arbitration:\nhighest calibrated\nconfidence among\ndirect + explored\nanswers\n\n(falls back to the\ndirect answer when\nrenders mislead)", fc="#faf5ff", ec=PUR, fs=6.8)
arr(ax, (12.38, 2.4), (12.62, 2.4), color=PUR)
fig.savefig(f"{OUT}/fig_overview.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_overview.png", dpi=150, bbox_inches="tight"); plt.close(fig); print("overview")

# ================ fig_gate_example ================
fig, ax = plt.subplots(figsize=(7.2, 2.5)); ax.set_xlim(0, 7.2); ax.set_ylim(0, 2.5); ax.axis("off")
strip(ax, [f"{TR}/1014/frame{k}.jpg" for k in (1, 3, 6)], 0.15, 1.5, w=0.62)
box(ax, 2.35, 1.5, 2.6, 0.72, "\"standing by the fireplace facing the TV,\nis the washer to my front-left,\nfront-right, back-left, or back-right?\"", fc="#fff8f8", ec=RED, ls=(0,(4,2)), fs=6.6)
box(ax, 5.2, 1.62, 0.95, 0.5, "gate: Yes", fc="#dafbe1", ec=OK, fs=7.5)
box(ax, 6.3, 1.62, 0.82, 0.5, "B ✓\n2 calls", fc="#dafbe1", ec=OK, fs=7.2)
arr(ax, (4.95, 1.86), (5.2, 1.86)); arr(ax, (6.15, 1.86), (6.3, 1.86), color=OK)
ax.text(0.15, 1.38, "the recorded frames already cover the living-room and appliance areas: the gate answers directly", fontsize=6.8, color=MUT, va="top")
strip(ax, [f"{TR}/2205/frame{k}.jpg" for k in (0, 4, 7)], 0.15, 0.35, w=0.62)
box(ax, 2.35, 0.35, 2.6, 0.72, "\"what is the longest dimension of\nthe laptop, in centimeters?\"", fc="#fff8f8", ec=RED, ls=(0,(4,2)), fs=6.8)
box(ax, 5.2, 0.47, 0.95, 0.5, "gate:\nExplore", fc="#fdf2f8", ec=MAG, fs=7.2)
box(ax, 6.3, 0.47, 0.82, 0.5, "start\nsearch", fc="white", ec=MAG, fs=7.2)
arr(ax, (4.95, 0.71), (5.2, 0.71)); arr(ax, (6.15, 0.71), (6.3, 0.71), color=MAG)
ax.text(0.15, 0.23, "the laptop is a few pixels wide at a distance: the frames cannot support a metric answer", fontsize=6.8, color=MUT, va="top")
fig.savefig(f"{OUT}/fig_gate_example.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_gate_example.png", dpi=170, bbox_inches="tight"); plt.close(fig); print("gate_example")

# ================ fig_stopping (flowchart) ================
fig, ax = plt.subplots(figsize=(7.2, 2.3)); ax.set_xlim(0, 7.2); ax.set_ylim(0, 2.3); ax.axis("off")
box(ax, 0.15, 1.45, 1.6, 0.65, "expand paths, keep\ntop-2 by calibrated\nconfidence", fc="#f6f8fa", fs=7.2)
box(ax, 2.35, 1.45, 2.0, 0.65, "retained answers agree,\nboth beat the direct\nanswer's confidence?", fc="#fff8c5", ec=AMB, fs=7.0)
box(ax, 5.15, 1.5, 1.85, 0.55, "return the common\nanswer (early stop)", fc="#dafbe1", ec=OK, fs=7.2)
box(ax, 2.35, 0.3, 2.0, 0.6, "depth budget left\nand paths still moving?", fc="#fff8c5", ec=AMB, fs=7.0)
box(ax, 0.15, 0.3, 1.6, 0.6, "continue both paths\n(disagreement =\nkeep exploring)", fc="white", ec=TEALD, fs=6.9)
box(ax, 5.15, 0.3, 1.85, 0.65, "final arbitration:\nhighest confidence among\ndirect + explored answers", fc="#faf5ff", ec=PUR, fs=7.0)
arr(ax, (1.75, 1.78), (2.35, 1.78)); arr(ax, (4.35, 1.78), (5.15, 1.78), color=OK); ax.text(4.72, 1.88, "yes", fontsize=6.9, color=OK)
arr(ax, (3.35, 1.45), (3.35, 0.9), color=AMB); ax.text(3.43, 1.12, "no", fontsize=6.9, color=AMB)
arr(ax, (2.35, 0.6), (1.75, 0.6), color=TEALD); ax.text(2.03, 0.68, "yes", fontsize=6.9, color=TEALD)
arr(ax, (0.95, 0.9), (0.95, 1.45), color=TEALD); ax.text(1.03, 1.14, "next depth", fontsize=6.6, color=TEALD)
arr(ax, (4.35, 0.6), (5.15, 0.6), color=PUR); ax.text(4.74, 0.68, "no / all paths\nchose Stop", fontsize=6.4, color=PUR, ha="center")
fig.savefig(f"{OUT}/fig_stopping.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_stopping.png", dpi=170, bbox_inches="tight"); plt.close(fig); print("stopping")

# ================ fig_oracle_example ================
fig, ax = plt.subplots(figsize=(7.2, 2.5)); ax.set_xlim(0, 7.2); ax.set_ylim(0, 2.5); ax.axis("off")
strip(ax, [f"{TR}/2205/frame{k}.jpg" for k in (4, 7)], 0.15, 1.55, w=0.72)
ax.text(0.9, 1.42, "original frames", fontsize=6.9, color=MUT, ha="center", va="top")
box(ax, 1.85, 1.62, 1.25, 0.55, "direct answer:\n35 cm  ✗ (GT 40)", fc="#fff5f5", ec=BAD, fs=7.0)
h1 = img(ax, f"{TR}/2205/view16.jpg", 3.45, 1.45, 1.0, ec="#9aa4b1")
ax.text(3.95, 1.35, "start at spot 3:\nstill wrong (35)", fontsize=6.7, color=MUT, ha="center", va="top")
h2 = img(ax, f"{TR}/2205/view80.jpg", 4.95, 1.45, 1.0, ec=OK, lw=1.7)
ax.text(5.45, 1.35, "after Forward the answer\nflips to 40 cm ✓", fontsize=6.7, color=OK, ha="center", va="top")
arr(ax, (3.1, 1.9), (3.45, 1.9)); arr(ax, (4.45, 1.9), (4.95, 1.9), color=OK)
ax.text(4.7, 2.0, "Forward", fontsize=6.8, color=OK, ha="center")
box(ax, 6.15, 1.55, 0.95, 0.75, "oracle labels:\nForward,\nthen Stop", fc="#faf5ff", ec=PUR, fs=6.9)
arr(ax, (5.95, 1.9), (6.15, 1.9), color=PUR)
ax.text(0.15, 0.85, "the shortest trajectory whose answer becomes correct supervises the controller; every visited state -- including the", fontsize=7.0, color=MUT, va="top")
ax.text(0.15, 0.62, "wrong ones (35 cm) -- is labelled by answer correctness and trains the confidence head; questions that are already", fontsize=7.0, color=MUT, va="top")
ax.text(0.15, 0.39, "correct from the frames contribute depth-zero Stop labels, teaching the controller not to over-explore.", fontsize=7.0, color=MUT, va="top")
fig.savefig(f"{OUT}/fig_oracle_example.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_oracle_example.png", dpi=170, bbox_inches="tight"); plt.close(fig); print("oracle_example")

# ================ fig_frames_scaling (measured, sec 2.1) ================
fr = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384]
lat = [12.3, 16.6, 21.0, 31.2, 40.8, 61.6, 83.4, 129.9, 179.2, 299.7]
mem = [22.6, 23.1, 23.5, 24.5, 25.4, 27.2, 29.1, 32.8, 36.5, 43.8]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.1))
for a in (a1, a2):
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    a.grid(True, color="#e3e2de", lw=0.7); a.set_axisbelow(True); a.set_xlabel("input frames", fontsize=8); a.tick_params(labelsize=7.5)
a1.plot(fr, lat, "-o", color="#d62828", ms=3.5, lw=1.6)
a1.scatter([512], [340], marker="x", s=60, color="#d62828"); a1.annotate("OOM", (512, 340), textcoords="offset points", xytext=(-6, 8), fontsize=8, color="#d62828", weight="bold")
a1.axvline(128, color="#7b1fa2", ls=":", lw=1.2); a1.annotate("position-embedding\nlimit (~128)", (128, 250), textcoords="offset points", xytext=(4, 0), fontsize=6.8, color="#7b1fa2")
a1.set_ylabel("latency per question (s)", fontsize=8); a1.set_xlim(0, 560)
a2.plot(fr, mem, "-o", color="#155e63", ms=3.5, lw=1.6)
a2.scatter([512], [45.3], marker="x", s=60, color="#d62828"); a2.annotate("alloc fails\n(64 GB unified)", (512, 45.3), textcoords="offset points", xytext=(-58, -22), fontsize=6.8, color="#d62828")
a2.set_ylabel("peak GPU memory (GB)", fontsize=8); a2.set_xlim(0, 560)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_frames_scaling.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_frames_scaling.png", dpi=170, bbox_inches="tight"); plt.close(fig); print("frames_scaling")

# ================ fig_view_quality (sec 2.2) ================
fig, ax = plt.subplots(figsize=(7.2, 2.35)); ax.set_xlim(0, 7.2); ax.set_ylim(0, 2.35); ax.axis("off")
h1 = img(ax, f"{R}/data/renders/scannetpp_3864514494/view1.png", 0.15, 0.55, 2.15, ec=BAD, lw=1.7, ls="--")
ax.text(1.22, 0.44, "unconstrained proposal: outside the room,\nlooking down through the ceiling", fontsize=6.9, color=BAD, ha="center", va="top")
h2 = img(ax, f"{R}/data/renders_human/scannetpp_3864514494/view0.png", 2.75, 0.55, 2.15, ec=OK, lw=1.7)
ax.text(3.82, 0.44, "constrained to a human-reachable pose:\neye level inside the recorded region", fontsize=6.9, color=OK, ha="center", va="top")
box(ax, 5.25, 0.62, 1.85, 1.55, "on held-out scenes, a naive\nproposer placed 0% of\ncameras inside the room;\nthe constraints reach 100%\nvalid poses, and ~25% of\ncandidate poses still fail\nthe support check and\nmust be masked", fs=6.8, fc="#f6f8fa")
ax.text(0.15, 2.27, "the same reconstruction, two viewpoint proposals for the same scene", fontsize=7.2, color=MUT, va="top")
fig.savefig(f"{OUT}/fig_view_quality.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_view_quality.png", dpi=170, bbox_inches="tight"); plt.close(fig); print("view_quality")

# ================ fig_paths_example (sec 2.3) ================
D28 = os.path.join(R, "figures/treeD_trace/2288")
fig, ax = plt.subplots(figsize=(7.2, 3.2)); ax.set_xlim(0, 7.2); ax.set_ylim(0, 3.2); ax.axis("off")
ax.text(0.15, 3.12, '"What is the longest dimension of the exhaust fan, in centimeters?"  (ground truth: 12)', fontsize=7.6, color=INK, va="top", weight="bold")
h1 = img(ax, f"{D28}/view0.jpg", 0.3, 1.5, 1.45, ec=BAD, lw=1.8)
ax.text(1.02, 1.4, "path A, step 1 (conf 0.700):\nanswers 20 ✗ (67% off)\n$\\bf{a\\ single{-}path\\ search}$\n$\\bf{commits\\ here}$", fontsize=6.7, color=BAD, ha="center", va="top")
h2 = img(ax, f"{D28}/view8.jpg", 2.15, 1.5, 1.45, ec="#2f6feb", lw=1.6)
ax.text(2.87, 1.4, "path B, step 1 (conf 0.697):\nanswers 14 --- nearly tied,\nretained as the second path", fontsize=6.7, color="#2f6feb", ha="center", va="top")
h3 = img(ax, f"{D28}/view88.jpg", 4.35, 1.6, 1.3, ec=OK, lw=1.5)
h4 = img(ax, f"{D28}/view1.jpg", 4.35, 0.35, 1.3, ec=OK, lw=1.5)
ax.text(5.15, 2.72, "step 2: every continuation of $\\it{both}$ paths answers 14", fontsize=6.9, color=OK, ha="center")
arr(ax, (1.78, 2.2), (4.35, 2.35), color=MUT, rad=-0.1); arr(ax, (3.62, 2.1), (4.35, 1.0), color=MUT, rad=0.12)
box(ax, 5.85, 1.15, 1.28, 1.25, "paths agree:\nfinal 14\n(17% off) ✓\n\nsingle path:\n20 ✗", fc="#dafbe1", ec=OK, fs=6.9)
arr(ax, (5.65, 2.3), (5.85, 2.0), color=OK); arr(ax, (5.65, 0.85), (5.85, 1.35), color=OK)
ax.text(0.15, 0.14, "a real VSI-Bench search: the most confident first movement is wrong; keeping two paths lets later observations correct it", fontsize=7.0, color=MUT, va="top")
fig.savefig(f"{OUT}/fig_paths_example.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_paths_example.png", dpi=170, bbox_inches="tight"); plt.close(fig); print("paths_example")

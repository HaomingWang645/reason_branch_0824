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

# ================ fig_training_pipeline (block flow-chart style) ================
BLUF, PURF, ARC = "#BDD7EE", "#C08BBB", "#2F5773"
fig, ax = plt.subplots(figsize=(3.6, 4.5)); ax.set_xlim(0, 3.6); ax.set_ylim(0, 4.5); ax.axis("off")

def bb2(x, y, w, h, text, fill, fs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                                fc=fill, ec="none", zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color="#111111", weight="bold", zorder=4)

def ta2(p_, q, lw=2.8, col=None):
    ax.add_patch(FancyArrowPatch(p_, q, arrowstyle="-|>", mutation_scale=18,
                                 color=col or ARC, lw=lw, shrinkA=1, shrinkB=1, zorder=5))

def flame2(x, y, k=0.8):
    import matplotlib.patches as mp
    ax.add_patch(mp.Polygon([(x, y), (x - 0.09 * k, y + 0.10 * k), (x - 0.045 * k, y + 0.085 * k),
                             (x - 0.02 * k, y + 0.24 * k), (x + 0.05 * k, y + 0.10 * k),
                             (x + 0.09 * k, y + 0.14 * k), (x + 0.10 * k, y + 0.02 * k)],
                            closed=True, fc="#f4801f", ec="#c9560a", lw=0.7, zorder=7))
    ax.add_patch(mp.Polygon([(x, y + 0.01), (x - 0.035 * k, y + 0.09 * k), (x + 0.01 * k, y + 0.13 * k),
                             (x + 0.045 * k, y + 0.06 * k)], closed=True, fc="#ffd166", ec="none", zorder=8))

def num(x, y, n):
    ax.add_patch(Circle((x, y), 0.13, fc="#111111", ec="none", zorder=6))
    ax.text(x, y, str(n), fontsize=10, color="white", ha="center", va="center", weight="bold", zorder=7)

# stage 1: answer from rendered views
num(0.32, 4.2, 1)
bb2(0.55, 3.85, 1.85, 0.62, "Frames +\nRendered Views", BLUF, 10)
ta2((2.42, 4.16), (2.62, 4.16))
bb2(2.62, 3.85, 0.82, 0.62, "VLM", PURF, 13)
flame2(3.36, 4.36, 0.7)
ta2((3.03, 3.83), (3.03, 3.42))
# stage 2: oracle search over camera walks
num(0.32, 3.1, 2)
bb2(0.55, 2.75, 1.7, 0.62, "Oracle\nSearch", PURF, 11)
r0 = (2.75, 3.28)
ax.add_patch(Circle(r0, 0.055, fc="#111111", ec="white", zorder=6))
for (kx, ky), col, t in [((2.42, 2.86), "#cf222e", "\u2717"), ((2.78, 2.82), "#1a7f37", "\u2713"), ((3.16, 2.9), "#1a7f37", "\u2713")]:
    ta2(r0, (kx, ky), lw=1.6, col=col)
    ax.add_patch(Circle((kx, ky), 0.05, fc=col, ec="white", zorder=6))
    ax.text(kx, ky - 0.14, t, fontsize=9, color=col, ha="center", va="top", weight="bold")
ta2((1.4, 2.73), (1.4, 2.32))
bb2(0.55, 1.75, 2.5, 0.55, "Walks + State Labels", BLUF, 10.5)
# stage 3: supervise controller and confidence head
num(0.32, 1.42, 3)
ta2((1.25, 1.73), (0.95, 1.22))
ta2((2.35, 1.73), (2.7, 1.22))
bb2(0.2, 0.6, 1.55, 0.6, "Controller", PURF, 11)
flame2(1.66, 1.1, 0.65)
bb2(1.95, 0.6, 1.55, 0.6, "Confidence\nHead", PURF, 10)
flame2(3.41, 1.1, 0.65)
ta2((1.0, 0.58), (1.0, 0.28)); ta2((2.72, 0.58), (2.72, 0.28))
bb2(1.1, 0.02, 1.55, 0.24, "guide the search", "#eef2f6", 8)
fig.savefig(f"{OUT}/fig_training_pipeline.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_training_pipeline.png", dpi=170, bbox_inches="tight"); plt.close(fig); print("training_pipeline")

# ================ fig_runtime (block flow-chart style) ================
fig, ax = plt.subplots(figsize=(3.6, 4.1)); ax.set_xlim(0, 3.6); ax.set_ylim(0, 4.1); ax.axis("off")

def mech(x, y, w, h, title, line, fill, fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                                fc=fill, ec="none", zorder=3))
    ax.text(x + w / 2, y + h - 0.30, title, ha="center", va="center", fontsize=fs,
            color="#111111", weight="bold", zorder=4)
    ax.text(x + w / 2, y + 0.34, line, ha="center", va="center", fontsize=8,
            color="#222222", zorder=4)

bb2(0.95, 3.6, 1.7, 0.45, "Question + Video", BLUF, 10.5)
ta2((1.35, 3.58), (1.15, 3.30)); ta2((2.25, 3.58), (2.45, 3.30))
mech(0.2, 2.25, 1.55, 1.0, "Reconstruct\nonce  \u2744", "11.3 s / scene \u2192\n0.7 s / question", PURF)
mech(1.9, 2.25, 1.5, 1.0, "Pose-keyed\ncaches", "render 0.12 s;\nrepeats are free", BLUF)
ta2((0.97, 2.23), (0.97, 1.95)); ta2((2.65, 2.23), (2.65, 1.95))
mech(0.2, 0.9, 1.55, 1.0, "Gated, bounded\nsearch", "71% exit in 9.6 s;\nb=3, keep 2, d\u22643", PURF, 9.5)
flame2(1.62, 1.75, 0.6)
mech(1.9, 0.9, 1.5, 1.0, "Decode-free\ncontrol", "prefill logits only;\nfused head \u22124.0 s", BLUF, 10)
ta2((2.65, 0.88), (2.65, 0.62), lw=2.2); ta2((0.97, 0.88), (0.97, 0.62), lw=2.2)
bb2(0.75, 0.1, 2.1, 0.5, "Answer: 24.5 s / 1.1 kJ avg.", BLUF, 9.5)
fig.savefig(f"{OUT}/fig_runtime.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_runtime.png", dpi=170, bbox_inches="tight"); plt.close(fig); print("runtime")

# ================ fig_search_example (single-column, vertical flow) ================
TD = os.path.join(R, "figures/treeD_trace/2027")
fig, ax = plt.subplots(figsize=(3.5, 4.9)); ax.set_xlim(0, 3.5); ax.set_ylim(0, 4.9); ax.axis("off")
KEPT, PRUN = "#0969da", "#9aa4b1"

def cell(x, ytop, w, path, action, score, state):
    im = mpimg.imread(path); h = w * im.shape[0] / im.shape[1]
    y = ytop - h
    ec, lw, ls = {"kept": (KEPT, 1.6, "-"), "pruned": (PRUN, 0.9, "-"), "invalid": (BAD, 1.1, (0, (3, 2)))}[state]
    ax.imshow(im, extent=(x, x + w, y, ytop), aspect="auto", zorder=2, alpha=1.0 if state == "kept" else 0.88)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", fc="none", ec=ec, lw=lw, ls=ls, zorder=4))
    ax.text(x + w / 2, ytop + 0.015, action, fontsize=4.8, color=MUT, ha="center", va="bottom", zorder=7,
            bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.78))
    if state == "invalid":
        ax.text(x + w / 2, y - 0.02, "invalid", fontsize=4.9, color=BAD, ha="center", va="top")
    else:
        ax.text(x + w / 2, y - 0.02, score, fontsize=5.1, color=KEPT if state == "kept" else MUT,
                ha="center", va="top", weight="bold" if state == "kept" else "normal")
    return (x + w / 2, ytop, y)  # center-x, top, bottom

# context frames + gate
for j2, k in enumerate((0, 3, 6)):
    im = mpimg.imread(f"{TD}/frame{k}.jpg"); h = 0.5 * im.shape[0] / im.shape[1]
    ax.imshow(im, extent=(0.10 + j2 * 0.54, 0.60 + j2 * 0.54, 4.82 - h, 4.82), aspect="auto", zorder=2)
    ax.add_patch(FancyBboxPatch((0.10 + j2 * 0.54, 4.82 - h), 0.5, h, boxstyle="square,pad=0", fc="none", ec="#6b7280", lw=0.8, zorder=4))
ax.text(0.86, 4.82 - 0.5 * 0.75 - 0.03, "8 context frames (3 shown)", fontsize=5.0, color=MUT, ha="center", va="top")
box(ax, 1.86, 4.56, 0.72, 0.24, "gate: EXPLORE", fc="#eaf3fb", ec=KEPT, fs=5.6)
box(ax, 2.68, 4.56, 0.74, 0.24, "direct: D [0.60]", fc="#f6f8fa", ec="#9aa4b1", fs=5.6)
arr(ax, (2.24, 4.56), (1.75, 4.20), color=MUT, lw=1.0, rad=0.15)

def rlabel(y, t):
    ax.text(0.045, y, t, fontsize=5.6, color=INK, ha="center", va="center", rotation=90, weight="bold")

# depth 1: three starting spots (two kept)
rlabel(3.55, "depth 1")
t1 = 3.94; w1 = 0.86
d1 = [cell(0.14, t1, w1, f"{TD}/view0.jpg", "spot 1", "D  [0.59]", "pruned"),
      cell(1.14, t1, w1, f"{TD}/view8.jpg", "spot 2", "D  [0.59]", "kept"),
      cell(2.14, t1, w1, f"{TD}/view16.jpg", "spot 3", "D  [0.60]", "kept")]

# depth 2: three children per kept spot; both kept children come from spot 3
rlabel(2.62, "depth 2")
t2 = 2.90
d2spec = [("view15.jpg", "turn left", "D  [0.57]", "pruned", 1), ("view9.jpg", "turn right", "D  [0.57]", "pruned", 1),
          ("view16.jpg", "next spot", "D  [0.58]", "pruned", 1), ("view23.jpg", "turn left", "D  [0.58]", "pruned", 2),
          ("view17.jpg", "turn right", "D  [0.59]", "kept", 2), ("view24.jpg", "next spot", "D  [0.59]", "kept", 2)]
x = 0.14; d2 = []
for f, act, sc, st, par in d2spec:
    w = 0.62 if st == "kept" else 0.46
    d2.append(cell(x, t2, w, f"{TD}/{f}", act, sc, st) + (st, par)); x += w + 0.055
for cx, top, bot, st, par in d2:
    px, ptop, pbot = d1[par]
    arr(ax, (px, pbot - 0.10), (cx, top + 0.10), color=KEPT if st == "kept" else PRUN, lw=0.9 if st == "kept" else 0.6, rad=0.0)

# depth 3: children of the two kept depth-2 nodes
rlabel(1.66, "depth 3")
t3 = 1.94
d3spec = [("view31.jpg", "turn left", "D  [0.60]", "pruned", 5), ("view25.jpg", "turn right", "", "invalid", 5),
          ("view28.jpg", "look around", "D  [0.61]", "kept", 5), ("view18.jpg", "turn right", "", "invalid", 4),
          ("view57.jpg", "forward", "D  [0.61]", "kept", 4)]
x = 0.14; d3 = []
for f, act, sc, st, par in d3spec:
    w = 0.62 if st == "kept" else 0.46
    d3.append(cell(x, t3, w, f"{TD}/{f}", act, sc, st) + (st, par)); x += w + 0.055
for cx, top, bot, st, par in d3:
    px, ptop, pbot = d2[par][:3]
    arr(ax, (px, pbot - 0.10), (cx, top + 0.10), color=KEPT if st == "kept" else PRUN, lw=0.9 if st == "kept" else 0.6, rad=0.0)

# consensus
box(ax, 0.95, 0.14, 1.66, 0.44, "consensus at depth 3\nfinal: D = GT $\checkmark$ (18 VLM calls)", fc="#eef7f1", ec=OK, fs=6.0, color=OK)
for cx, top, bot, st, par in d3:
    if st == "kept":
        arr(ax, (cx, bot - 0.10), (1.78, 0.60), color=OK, lw=1.1, rad=0.1 if cx > 1.78 else -0.1)
fig.savefig(f"{OUT}/fig_search_example.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_search_example.png", dpi=200, bbox_inches="tight"); plt.close(fig); print("search_example")

# ================ fig_overview (block flow-chart style) ================
TR = os.path.join(R, "figures/treeD_trace")
BLUF, PURF, ARC = "#BDD7EE", "#C08BBB", "#2F5773"
fig, ax = plt.subplots(figsize=(13.6, 4.4)); ax.set_xlim(0, 13.6); ax.set_ylim(0, 4.4); ax.axis("off")

def bb(x, y, w, h, text, fill, fs=13):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.14",
                                fc=fill, ec="none", zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color="#111111", weight="bold", zorder=4)

def tarr(p_, q, lw=3.4, ls="-"):
    ax.add_patch(FancyArrowPatch(p_, q, arrowstyle="-|>", mutation_scale=24, color=ARC,
                                 lw=lw, ls=ls, shrinkA=1, shrinkB=1, zorder=5))

def flame(x, y, k=1.0):
    import matplotlib.patches as mp
    ax.add_patch(mp.Polygon([(x, y), (x - 0.09 * k, y + 0.10 * k), (x - 0.045 * k, y + 0.085 * k),
                             (x - 0.02 * k, y + 0.24 * k), (x + 0.05 * k, y + 0.10 * k),
                             (x + 0.09 * k, y + 0.14 * k), (x + 0.10 * k, y + 0.02 * k)],
                            closed=True, fc="#f4801f", ec="#c9560a", lw=0.8, zorder=7))
    ax.add_patch(mp.Polygon([(x, y + 0.01), (x - 0.035 * k, y + 0.09 * k), (x + 0.01 * k, y + 0.13 * k),
                             (x + 0.045 * k, y + 0.06 * k)], closed=True, fc="#ffd166", ec="none", zorder=8))

# ---- inputs and VLM ----
bb(0.25, 2.75, 1.5, 0.55, "Query", BLUF, 14)
bb(1.95, 2.75, 1.75, 0.55, "Context\nFrames", BLUF, 12)
bb(0.70, 1.15, 2.0, 1.15, "VLM", PURF, 27)
flame(2.52, 2.02, 1.1)
tarr((1.0, 2.73), (1.35, 2.34)); tarr((2.82, 2.73), (2.15, 2.34))
# action out, confidence below
bb(3.65, 1.95, 1.5, 0.55, "Action", BLUF, 14)
tarr((2.72, 1.85), (3.63, 2.15))
bb(3.65, 0.75, 1.9, 0.55, "Confidence", PURF, 13)
flame(5.42, 1.18, 0.85)
tarr((2.72, 1.35), (3.63, 1.05))
bb(6.30, 0.75, 1.35, 0.55, "Answer", BLUF, 14)
tarr((5.58, 1.02), (6.28, 1.02))
tarr((4.60, 1.32), (4.40, 1.93))
ax.text(4.30, 1.70, "explore", fontsize=9, color=ARC, ha="right", va="center", weight="bold")
ax.text(5.92, 1.36, "answer", fontsize=9, color=ARC, ha="center", va="bottom", weight="bold")

# ---- explicit scene memory (dashed group) ----
ax.add_patch(FancyBboxPatch((8.6, 0.35), 4.75, 3.85, boxstyle="round,pad=0.02,rounding_size=0.06",
                            fc="none", ec="#111111", lw=2.0, ls=(0, (6, 4)), zorder=2))
ax.text(10.45, 4.06, "Explicit Scene Memory", fontsize=12, color="#111111", ha="center", va="center", weight="bold")
bb(9.0, 3.42, 2.0, 0.5, "Video Frames", BLUF, 12.5)
tarr((10.0, 3.40), (10.0, 3.26))
bb(9.0, 2.72, 2.0, 0.5, "VGGT  \u2744", PURF, 12.5)
im = mpimg.imread(f"{R}/figures/wm_failure_assets/bev.jpg")[60:700, 112:1035]
iw = 2.3; ih = iw * im.shape[0] / im.shape[1]
ax.imshow(im, extent=(10.0 - iw / 2, 10.0 + iw / 2, 2.62 - ih, 2.62), aspect="auto", zorder=2)
ax.add_patch(FancyBboxPatch((10.0 - iw / 2, 2.62 - ih), iw, ih, boxstyle="square,pad=0", fc="none", ec="#6b7280", lw=1.0, zorder=4))
tarr((10.0, 2.70), (10.0, 2.64))
tarr((10.0, 2.62 - ih - 0.02), (10.0, 1.04))
bb(9.0, 0.48, 2.0, 0.52, "Scene Memory", PURF, 12.5)
himg = mpimg.imread(f"{R}/figures/wm_failure_assets/render_25.jpg")
rw = 1.3; rh = rw * himg.shape[0] / himg.shape[1]
ax.imshow(himg, extent=(11.9, 11.9 + rw, 1.58, 1.58 + rh), aspect="auto", zorder=2)
ax.add_patch(FancyBboxPatch((11.9, 1.58), rw, rh, boxstyle="square,pad=0", fc="none", ec="#6b7280", lw=1.0, zorder=4))
tarr((11.02, 0.80), (12.25, 1.56))
# action into the memory; rendered view + pose back to the VLM along the bottom rail
tarr((5.17, 2.05), (9.0, 0.90))
ax.plot([12.9, 12.9, 9.32], [1.56, 0.30, 0.30], color=ARC, lw=3.0, zorder=5)
bb(7.75, 0.06, 1.55, 0.5, "View + Pose", BLUF, 11.5)
ax.plot([7.73, 1.55], [0.30, 0.30], color=ARC, lw=3.0, zorder=5)
tarr((1.55, 0.30), (1.55, 1.13))
# example imagery in / under the blocks
BFR = f"{R}/figures/motivation_assets/vsi_arkit_47334117/frames_64"
for j, k in enumerate((9, 25, 57)):
    fim = mpimg.imread(f"{BFR}/frame_{k:02d}.jpg")
    fx = 1.10 + j * 0.80
    ax.imshow(fim, extent=(fx, fx + 0.75, 3.52, 3.52 + 0.56), aspect="auto", zorder=2)
    ax.add_patch(FancyBboxPatch((fx, 3.52), 0.75, 0.56, boxstyle="square,pad=0", fc="none", ec="#6b7280", lw=0.9, zorder=4))
tarr((2.82, 3.50), (2.82, 3.34), lw=2.4)
for j, k in enumerate((14, 29)):
    fim = mpimg.imread(f"{BFR}/frame_{k:02d}.jpg")
    fx = 11.30 + j * 0.86
    ax.imshow(fim, extent=(fx, fx + 0.80, 3.26, 3.26 + 0.60), aspect="auto", zorder=2)
    ax.add_patch(FancyBboxPatch((fx, 3.26), 0.80, 0.60, boxstyle="square,pad=0", fc="none", ec="#6b7280", lw=0.9, zorder=4))
vim = mpimg.imread(f"{R}/figures/wm_failure_assets/render_25.jpg")
vh = 0.78 * vim.shape[0] / vim.shape[1]
ax.imshow(vim, extent=(6.52, 7.30, 0.02, 0.02 + vh), aspect="auto", zorder=6)
ax.add_patch(FancyBboxPatch((6.52, 0.02), 0.78, vh, boxstyle="square,pad=0", fc="none", ec="#6b7280", lw=0.9, zorder=7))
ax.text(4.40, 2.62, "e.g., Turn-Left, Forward", fontsize=8.5, color="#555555", ha="center", va="bottom", style="italic")
# legend
flame(0.35, 0.52, 0.8); ax.text(0.52, 0.60, "trained", fontsize=9, color="#444444", ha="left", va="center")
ax.text(1.25, 0.60, "\u2744 frozen", fontsize=9, color="#444444", ha="left", va="center")
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

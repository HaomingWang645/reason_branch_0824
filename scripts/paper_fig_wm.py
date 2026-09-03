"""Intro figures: (1) fig_wm_failure -- expanded world-model failure comparison built
from real assets (ARKit bedroom 47334117: captured frames, SVC/Stable-Virtual-Camera
predictions, VGGT point renders); (2) fig_operations -- the four reasoning operations.
Same visual language as paper_fig_design.py. Outputs PDF+PNG into the paper figures dir."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(R, "paper/ViewTree__Resource_Aware_On_Device_Spatial_Reasoning_via_Adaptive_Viewpoint_Branching_and_Fusion__1_/figures")
WA = os.path.join(R, "figures/wm_failure_assets")
FR = os.path.join(R, "figures/motivation_assets/vsi_arkit_47334117/frames_64")
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

# ================ fig_wm_failure (single column, vertical) ================
meta = json.load(open(f"{WA}/pose_meta.json"))
bev = mpimg.imread(f"{WA}/bev.jpg")
cy0, cy1, cx0, cx1 = 0, 760, 112, 1035
bevc = bev[cy0:cy1, cx0:cx1]

fig, ax = plt.subplots(figsize=(3.5, 5.15)); ax.set_xlim(0, 3.5); ax.set_ylim(0.74, 5.86); ax.axis("off")

# -- captured scene group ---------------------------------------------------
group(ax, 0.06, 3.80, 3.38, 1.84, TEALD); badge(ax, 0.62, 5.76, 1, "Captured scene", TEALD)
for j, k in enumerate((14, 21, 29, 57)):
    img(ax, f"{FR}/frame_{k:02d}.jpg", 0.14 + j * 0.82, 5.02, 0.78)
ax.text(1.72, 4.985, "4 of 64 recorded frames: the bed area is covered", fontsize=5.6, color=MUT, ha="center", va="top")
img(ax, f"{FR}/frame_09.jpg", 0.14, 4.06, 0.95, ec=INK, lw=1.4)
ax.text(0.615, 4.01, "current view", fontsize=5.9, color=INK, ha="center", va="top", weight="bold")
bw = 1.04; bh = bw * bevc.shape[0] / bevc.shape[1]
bx, by = 1.24, 4.02
ax.imshow(bevc, extent=(bx, bx + bw, by, by + bh), aspect="auto", zorder=2)
ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="square,pad=0", fc="none", ec="#6b7280", lw=1.0, zorder=4))
def bev_pt(k):
    px, py = meta["cam_bev"][k]
    return (bx + (px - cx0) / (cx1 - cx0) * bw, by + bh - (py - cy0) / (cy1 - cy0) * bh)
def bev_dir(k, L=0.24):
    pp = bev_pt(k); f = meta["fwd_bev"][k]; c = meta["cam_bev"][k]
    d = np.array([f[0] - c[0], -(f[1] - c[1])]); d = d / (np.linalg.norm(d) + 1e-9)
    return pp, (pp[0] + L * d[0], pp[1] + L * d[1])
p9, q9 = bev_dir(9, 0.20); p25, q25 = bev_dir(25, 0.24); p23, q23 = bev_dir(23, 0.20)
ax.add_patch(Circle(p9, 0.028, fc=INK, ec="white", lw=0.8, zorder=6))
arr(ax, p9, q9, color="#6d7681", lw=1.2)
arr(ax, q9, q25, color=RED, lw=1.5, rad=0.5)
arr(ax, p25, p23, color=PUR, lw=1.5, ls="--")
ax.text(bx + bw / 2, by - 0.05, "reconstruction (top view)", fontsize=5.5, color=MUT, ha="center", va="top")
box(ax, 2.52, 4.22, 0.86, 0.52, "request 1\nturn left\n$\\approx$85$^\\circ$", fc="#fff5f5", ec=RED, ls=(0, (4, 2)), fs=6.0, color=RED, weight="bold")
arr(ax, (2.44, 4.48), (2.52, 4.48), color=MUT, lw=1.1)

# -- comparison rows --------------------------------------------------------
cw = 1.05; ch = cw * 3 / 4
colx = [0.10, 1.25, 2.40]
heads = [("World-model\nprediction", BAD), ("Actual view\n(captured)", INK), ("ViewTree\nrender", OK)]
for cx, (t, c) in zip(colx, heads):
    ax.text(cx + cw / 2, 3.44, t, fontsize=6.0, color=c, ha="center", va="bottom", weight="bold")
r1y, r2y = 2.60, 1.06
cells = [(0, r1y, f"{WA}/svc/hop1/final.png", BAD), (1, r1y, f"{FR}/frame_25.jpg", "#6b7280"),
         (2, r1y, f"{WA}/render_25.jpg", OK),
         (0, r2y, f"{WA}/svc/hop2/final.png", BAD), (1, r2y, f"{FR}/frame_23.jpg", "#6b7280"),
         (2, r2y, f"{WA}/render_23.jpg", OK)]
for ci, yy, path, ec in cells:
    img(ax, path, colx[ci], yy, cw, ec=ec, lw=1.5)
ax.text(colx[0] + cw / 2, r1y - 0.045, "$\\times$ invents a bed & walls that\ncontradict the recording", fontsize=5.4, color=BAD, ha="center", va="top")
ax.text(colx[2] + cw / 2, r1y - 0.045, "$\\checkmark$ same geometry,\nsparser texture", fontsize=5.4, color=OK, ha="center", va="top")
ax.text(colx[0] + cw / 2, r2y - 0.045, "$\\times$ drifts further from its\nown invented scene", fontsize=5.4, color=BAD, ha="center", va="top")
ax.text(colx[2] + cw / 2, r2y - 0.045, "$\\checkmark$ no error\naccumulation", fontsize=5.4, color=OK, ha="center", va="top")

# middle band, aligned per column
arr(ax, (colx[0] + 0.16, r1y - 0.34), (colx[0] + 0.16, r2y + ch + 0.03), color=BAD, lw=1.5, rad=-0.12)
ax.text(colx[0] + 0.34, 2.10, "fed back as\nnext input:\nerrors compound", fontsize=5.0, color=BAD, ha="left", va="center", weight="bold")
box(ax, colx[1] + 0.02, 1.92, 1.01, 0.40, "request 2\nforward $\\approx$0.7 m", fc="#faf5ff", ec=PUR, ls=(0, (4, 2)), fs=5.5, color=PUR, weight="bold")
gx = colx[2] + cw / 2
box(ax, colx[2] + 0.02, 1.94, 1.01, 0.36, "same stored\ngeometry", fc="#eef7f1", ec=OK, fs=5.4, color=OK)
arr(ax, (gx, 2.30), (gx, r1y - 0.30), color=OK, lw=1.3)
arr(ax, (gx, 1.94), (gx, r2y + ch + 0.03), color=OK, lw=1.3)
fig.savefig(f"{OUT}/fig_wm_failure.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_wm_failure.png", dpi=170, bbox_inches="tight"); plt.close(fig); print("wm_failure")

# ================ fig_operations ================
fig, ax = plt.subplots(figsize=(3.5, 2.62)); ax.set_xlim(0, 3.5); ax.set_ylim(0, 2.62); ax.axis("off")
def node(x, y, r=0.055, fc=TEALD):
    ax.add_patch(Circle((x, y), r, fc=fc, ec="white", lw=0.9, zorder=6))
panels = [(0.06, 1.40, "Move", TEALD), (1.84, 1.40, "Branch", MAG), (0.06, 0.08, "Fuse", PUR), (1.84, 0.08, "Stop", OK)]
for px, py, name, col in panels:
    group(ax, px, py, 1.58, 1.12, col)
    ax.text(px + 0.10, py + 1.00, name, fontsize=8.2, color=col, ha="left", va="center", weight="bold")
# Move: pose -> new pose, a rendered view comes back
node(0.30, 2.10); node(0.86, 2.10); arr(ax, (0.36, 2.10), (0.79, 2.10), color=TEALD, lw=1.6)
img(ax, f"{WA}/render_25.jpg", 1.02, 1.92, 0.50, ec=TEALD, lw=1.2)
ax.text(0.58, 2.17, "action", fontsize=5.8, color=TEALD, ha="center", va="bottom")
ax.text(0.85, 1.51, "a camera action returns one\nrendered view + its pose", fontsize=5.6, color=MUT, ha="center", va="bottom")
# Branch: fork into independent paths
node(2.22, 2.10, fc=MAG)
for dy, a in ((0.24, 0.25), (0.0, 0.0), (-0.24, -0.25)):
    node(2.94, 2.10 + dy, fc=MAG)
    arr(ax, (2.28, 2.10), (2.87, 2.10 + dy), color=MAG, lw=1.3, rad=a)
ax.text(2.63, 1.51, "the same history forks into\nindependent paths", fontsize=5.6, color=MUT, ha="center", va="bottom")
# Fuse: two branches -> one answer
node(0.30, 0.98, fc=PUR); node(0.30, 0.64, fc=PUR)
box(ax, 0.80, 0.61, 0.70, 0.40, "combined\nanswer", fc="#faf5ff", ec=PUR, fs=6.0, color=PUR)
arr(ax, (0.36, 0.96), (0.80, 0.87), color=PUR, lw=1.3, rad=-0.15)
arr(ax, (0.36, 0.66), (0.80, 0.75), color=PUR, lw=1.3, rad=0.15)
ax.text(0.85, 0.19, "complementary views\nanswer together", fontsize=5.6, color=MUT, ha="center", va="bottom")
# Stop: commit the answer with its confidence
node(2.22, 0.88, fc=OK)
box(ax, 2.46, 0.70, 0.86, 0.36, "answer: 12 cm\nconfidence 0.81", fc="#eef7f1", ec=OK, fs=5.8, color=OK)
arr(ax, (2.28, 0.88), (2.46, 0.88), color=OK, lw=1.3)
ax.text(2.63, 0.19, "the path ends; high confidence\nends the whole search", fontsize=5.6, color=MUT, ha="center", va="bottom")
fig.savefig(f"{OUT}/fig_operations.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_operations.png", dpi=170, bbox_inches="tight"); plt.close(fig); print("operations")

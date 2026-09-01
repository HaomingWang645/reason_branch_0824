"""Motivation figure: (a) frame prompting scatters spatial evidence across views;
(b) ViewTree reconstructs once and chooses what to observe. Real frames/renders
from VSI scene of trace #3099.   -> figures/motivation.png/.pdf"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(R, "figures/treeD_trace/3099")
INK, MUT, GRID = "#1f2328", "#57606a", "#c9d1d9"
TEAL, BLUE, AMB, BAD, OK = "#0f6e73", "#0969da", "#9a6700", "#cf222e", "#1a7f37"
fig, ax = plt.subplots(figsize=(16.6, 8.6)); ax.set_xlim(0, 16.6); ax.set_ylim(0, 8.6); ax.axis("off")

def img(path, x, y, w, ec=GRID, lw=1.4):
    im = mpimg.imread(path); h = w * im.shape[0] / im.shape[1]
    ax.imshow(im, extent=(x, x + w, y, y + h), aspect="auto", zorder=2)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", fc="none", ec=ec, lw=lw, zorder=4))
    return h

def ell(x, y, w, h, fx, fy, fw, fh, color, label=None, lx=0, ly=-0.14, fs=7.6):
    # fx,fy = fraction from top-left of the image; fw,fh = fractional size
    cx = x + (fx + fw / 2) * w; cy = y + h * (1 - fy - fh / 2)
    ax.add_patch(Ellipse((cx, cy), fw * w, fh * h, fill=False, ec=color, lw=1.8, zorder=5))
    if label: ax.text(cx + lx, y + h * (1 - fy - fh) + ly if ly < 0 else cy + ly, label, fontsize=fs, color=color, ha="center", va="top", zorder=6,
                      bbox=dict(fc="white", ec="none", alpha=0.75, pad=1))

def arr(p, q, color=MUT, ls="-", lw=1.6, label=None, ly=0.12, rad=0.0, fs=8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2, zorder=3))
    if label: ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + ly, label, fontsize=fs, color=color, ha="center", zorder=6)

def box(x, y, w, h, text, fc="white", ec=GRID, lw=1.3, fs=8.2, color=INK, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, linespacing=1.35, zorder=3)

Q = 'Q: "Which of these objects (backpack, pillow, bed, lamp) is the closest to the table?"'
ax.text(0.15, 8.45, Q, fontsize=10.5, color=INK, weight="bold", va="top")

# ---------------- (a) ----------------
ax.text(0.15, 7.98, "(a)  Frame prompting: the evidence is scattered across viewpoints", fontsize=10, color=BAD, weight="bold", va="top")
W = 3.1; y0 = 5.05
h = img(f"{D}/frame0.jpg", 0.35, y0, W)
ell(0.35, y0, W, h, 0.10, 0.30, 0.62, 0.42, BLUE, "table", ly=-0.10)
ell(0.35, y0, W, h, 0.67, 0.00, 0.19, 0.32, TEAL, "lamp")
x2 = 0.35 + W + 0.75
h2 = img(f"{D}/frame2.jpg", x2, y0, W)
ell(x2, y0, W, h2, 0.63, 0.42, 0.25, 0.30, AMB, "backpack: dark, occluded", ly=-0.10)
x3 = x2 + W + 0.75
h3 = img(f"{D}/frame4.jpg", x3, y0, W)
ell(x3, y0, W, h3, 0.75, 0.00, 0.25, 0.30, TEAL, "same lamp:\nnew position & scale", ly=-0.10)
ell(x3, y0, W, h3, 0.00, 0.16, 0.18, 0.16, AMB, "same backpack:\ntiny at distance", ly=-0.66)
ell(x3, y0, W, h3, 0.26, 0.12, 0.44, 0.20, MUT, "pillow / bed", ly=-0.10)
for xa in (0.35 + W, x2 + W):
    arr((xa + 0.08, y0 + 1.15), (xa + 0.67, y0 + 1.15), color=BAD, ls=":", label="camera\nmotion?", ly=0.22, fs=7.6)
box(x3 + W + 0.55, y0 + 0.15, 4.6, 2.0,
    "To answer, the VLM must implicitly:\n"
    "estimate camera motion between frames ·\n"
    "re-identify objects across views ·\n"
    "place everything in one 3-D layout ·\n"
    "ignore redundant / distracting views\n"
    "$\\bf{— unreliable\\ for\\ (small)\\ VLMs}$  ✗", fc="#fff5f5", ec=BAD, fs=8.4)
ax.text(0.35, y0 - 0.28, "frames arrive as unrelated 2-D projections: each is encoded in its own camera coordinates; the table and the backpack never appear together", fontsize=8, color=MUT, va="top")

# ---------------- (b) ----------------
yb = 3.55
ax.text(0.15, yb + 0.9, "(b)  ViewTree: reconstruct once into an explicit 3-D memory, then $\\it{choose}$ what to observe", fontsize=10, color=TEAL, weight="bold", va="top")
y1 = 0.75; TH = 0.62
for j, fr in enumerate((0, 2, 4)):
    img(f"{D}/frame{fr}.jpg", 0.35 + j * 0.32, y1 + 1.3 - j * 0.18, 0.9, ec=GRID, lw=1)
ax.text(0.95, y1 + 0.9, "input frames\n(one pass)", fontsize=7.8, color=MUT, ha="center", va="top")
arr((1.75, y1 + 1.6), (2.45, y1 + 1.6), color=TEAL, label="VGGT\n(frozen)", ly=0.16, fs=7.8)
hv = img(f"{D}/view8.jpg", 2.5, y1 + 0.55, 3.0, ec=BLUE, lw=1.8)
ax.text(4.0, y1 + 0.4, "one unified 3-D space — camera poses known by construction;\nbed, chair and backpack in a single coordinate frame", fontsize=7.8, color=BLUE, ha="center", va="top")
box(5.95, y1 + 1.0, 3.0, 1.65, "virtual camera\n(human-constrained:\ninside the walked area,\neye level, roll 0;\ninvalid views masked)", fc="#eef7f7", ec=TEAL, fs=8)
arr((5.5, y1 + 1.8), (5.95, y1 + 1.8), color=TEAL)
arr((8.95, y1 + 1.8), (9.6, y1 + 1.8), color=TEAL)
ax.text(9.27, y1 + 2.95, 'controller: "TURN LEFT at standing spot 3"', fontsize=7.8, color=TEAL, ha='center', zorder=6)
hw = img(f"{D}/view23.jpg", 9.65, y1 + 0.55, 3.0, ec=TEAL, lw=1.8)
ax.text(11.15, y1 + 0.4, "requested view: the table area beside the bed, all queried\nobjects in one view — the missing evidence, rendered on demand", fontsize=7.8, color=TEAL, ha="center", va="top")
arr((12.7, y1 + 1.8), (13.35, y1 + 1.8), color=OK)
box(13.4, y1 + 1.15, 2.9, 1.3, "answer: D (lamp)  ✓\nkept by confidence head;\nstops when views agree", fc="#dafbe1", ec=OK, fs=8.4)
ax.text(0.35, y1 - 0.25, "reconstruction runs once per scene; the model explores by moving a virtual camera under physical constraints, acquires only the views it needs (mean < 5 VLM calls), and falls back to the direct answer when the memory adds nothing", fontsize=8, color=MUT, va="top")
fig.savefig(os.path.join(R, "figures/motivation.png"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(R, "figures/motivation.pdf"), bbox_inches="tight")
print("wrote figures/motivation.png/.pdf")

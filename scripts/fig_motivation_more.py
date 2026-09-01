"""Two more motivation figures from real traces:
  motivation_room    (#3822, library aisle, room size: direct 12.4 X -> depth-3 walk 14.5 ~ GT 15.6)
  motivation_kitchen (#2027, kitchenette, relative distance: candidates scattered one-per-frame)"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, MUT, GRID = "#1f2328", "#57606a", "#c9d1d9"
TEAL, BLUE, AMB, BAD, OK = "#0f6e73", "#0969da", "#9a6700", "#cf222e", "#1a7f37"

def new(figsize=(16.6, 8.2)):
    fig, ax = plt.subplots(figsize=figsize); ax.set_xlim(0, figsize[0]); ax.set_ylim(0, figsize[1]); ax.axis("off"); return fig, ax
def img(ax, path, x, y, w, ec=GRID, lw=1.4):
    im = mpimg.imread(path); h = w * im.shape[0] / im.shape[1]
    ax.imshow(im, extent=(x, x + w, y, y + h), aspect="auto", zorder=2)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", fc="none", ec=ec, lw=lw, zorder=4)); return h
def ell(ax, x, y, w, h, fx, fy, fw, fh, color, label=None, ly=-0.14, fs=7.6):
    cx = x + (fx + fw / 2) * w; cy = y + h * (1 - fy - fh / 2)
    ax.add_patch(Ellipse((cx, cy), fw * w, fh * h, fill=False, ec=color, lw=1.9, zorder=5))
    if label: ax.text(cx, y + h * (1 - fy - fh) + ly, label, fontsize=fs, color=color, ha="center", va="top", zorder=6,
                      bbox=dict(fc="white", ec="none", alpha=0.78, pad=1))
def arr(ax, p, q, color=MUT, ls="-", lw=1.6, label=None, ly=0.12, fs=8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, color=color, lw=lw, ls=ls, shrinkA=2, shrinkB=2, zorder=3))
    if label: ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + ly, label, fontsize=fs, color=color, ha="center", zorder=6)
def box(ax, x, y, w, h, text, fc="white", ec=GRID, lw=1.3, fs=8.2, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, linespacing=1.35, zorder=3)

# ================= motivation_room (#3822) =================
D = os.path.join(R, "figures/treeD_trace/3822")
fig, ax = new()
ax.text(0.15, 8.05, 'Q: "What is the size of this room (in square meters)?"', fontsize=10.5, weight="bold", color=INK, va="top")
ax.text(0.15, 7.6, "(a)  Frame prompting: no frame shows the room's extent — and more frames only add near-duplicates", fontsize=10, color=BAD, weight="bold", va="top")
W = 3.0; y0 = 4.85
for j, fr in enumerate((0, 3, 6)):
    x = 0.35 + j * (W + 0.75); h = img(ax, f"{D}/frame{fr}.jpg", x, y0, W)
    if j < 2: arr(ax, (x + W + 0.08, y0 + 1.1), (x + W + 0.67, y0 + 1.1), color=BAD, ls=":", label="camera\nmotion?", ly=0.2, fs=7.4)
ax.text(0.35 + W / 2, y0 - 0.18, "shelf close-up", fontsize=7.8, color=MUT, ha="center", va="top")
ax.text(0.35 + W + 0.75 + W / 2, y0 - 0.18, "another shelf — or the same one?", fontsize=7.8, color=MUT, ha="center", va="top")
ax.text(0.35 + 2 * (W + 0.75) + W / 2, y0 - 0.18, "position along the aisle unknown", fontsize=7.8, color=MUT, ha="center", va="top")
box(ax, 0.35 + 3 * (W + 0.75) - 0.2, y0 + 0.15, 4.5, 2.0,
    "all 8 input frames are close-ups taken\nwhile walking along the shelves;\nthe aisle's length and the far wall\nare never visible, and the near-duplicate\nframes add no constraint\n$\\bf{direct\\ answer:\\ 12.4\\ m^2}$  ✗  (GT 15.6)", fc="#fff5f5", ec=BAD, fs=8.3)
# (b)
ax.text(0.15, 4.15, "(b)  ViewTree-D: a depth-3 walk steps back and along the aisle until its extent is visible", fontsize=10, color=TEAL, weight="bold", va="top")
y1 = 0.6
for j, fr in enumerate((0, 3, 6)): img(ax, f"{D}/frame{fr}.jpg", 0.35 + j * 0.3, y1 + 1.35 - j * 0.17, 0.85, ec=GRID, lw=1)
ax.text(0.9, y1 + 0.95, "input frames", fontsize=7.8, color=MUT, ha="center", va="top")
arr(ax, (1.7, y1 + 1.7), (2.35, y1 + 1.7), color=TEAL, label="VGGT\n(frozen)", ly=0.16, fs=7.8)
img(ax, f"{D}/view0.jpg", 2.4, y1 + 0.75, 2.6, ec=BLUE, lw=1.8)
ax.text(3.7, y1 + 0.6, "explicit 3-D memory:\none unified reconstruction of the aisle", fontsize=7.8, color=BLUE, ha="center", va="top")
box(ax, 5.5, y1 + 1.05, 2.6, 1.5, "walk (human-constrained):\nstart at spot 2 →\nFORWARD to spot 8 →\nTURN RIGHT", fc="#eef7f7", ec=TEAL, fs=8)
arr(ax, (5.05, y1 + 1.8), (5.5, y1 + 1.8), color=TEAL)
arr(ax, (8.1, y1 + 1.8), (8.7, y1 + 1.8), color=TEAL)
img(ax, f"{D}/view56.jpg", 8.75, y1 + 0.75, 2.7, ec=TEAL, lw=1.8)
ax.text(10.1, y1 + 0.6, "acquired view: both shelf rows and the end\nwall in one frame — the extent is now visible", fontsize=7.8, color=TEAL, ha="center", va="top")
arr(ax, (11.5, y1 + 1.8), (12.1, y1 + 1.8), color=OK)
box(ax, 12.15, y1 + 1.25, 3.1, 1.15, "answer: 14.5 m²  ✓\n(GT 15.6; both retained walks agree,\nconfidence above the direct guess)", fc="#dafbe1", ec=OK, fs=8.2)
fig.savefig(os.path.join(R, "figures/motivation_room.png"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(R, "figures/motivation_room.pdf"), bbox_inches="tight"); plt.close(fig); print("room")

# ================= motivation_kitchen (#2027) =================
D = os.path.join(R, "figures/treeD_trace/2027")
fig, ax = new()
ax.text(0.15, 8.05, 'Q: "Which of these objects (heater, trash can, door, cup) is the closest to the microwave?"', fontsize=10.5, weight="bold", color=INK, va="top")
ax.text(0.15, 7.6, "(a)  Frame prompting: every candidate object sits in its own frame — no pairwise distance is ever visible", fontsize=10, color=BAD, weight="bold", va="top")
W = 3.0; y0 = 4.85
h0 = img(ax, f"{D}/frame0.jpg", 0.35, y0, W)
ell(ax, 0.35, y0, W, h0, 0.30, 0.05, 0.18, 0.24, TEAL, "microwave", ly=-0.08)
x2 = 0.35 + W + 0.75; h2 = img(ax, f"{D}/frame2.jpg", x2, y0, W)
ell(ax, x2, y0, W, h2, 0.82, 0.02, 0.17, 0.58, AMB, "door: alone in its frame,\ndepth ambiguous", ly=-0.10)
x3 = x2 + W + 0.75; h3 = img(ax, f"{D}/frame5.jpg", x3, y0, W)
ell(ax, x3, y0, W, h3, 0.26, 0.72, 0.46, 0.26, BLUE, "heater: extreme close-up,\nno surrounding context", ly=-0.08)
for xa in (0.35 + W, x2 + W):
    arr(ax, (xa + 0.08, y0 + 1.1), (xa + 0.67, y0 + 1.1), color=BAD, ls=":", label="camera\nmotion?", ly=0.2, fs=7.4)
box(ax, x3 + W + 0.55, y0 + 0.15, 4.4, 2.0,
    "the microwave, door, heater (and cup,\ntrash can) each appear at a different\ndepth in a different frame; comparing\ntheir distances requires fusing all\nviews into one 3-D layout — implicitly\n$\\bf{— unreliable\\ for\\ (small)\\ VLMs}$", fc="#fff5f5", ec=BAD, fs=8.3)
# (b)
ax.text(0.15, 4.15, "(b)  ViewTree-D: walk to the reference object; the nearest candidate becomes directly visible", fontsize=10, color=TEAL, weight="bold", va="top")
y1 = 0.6
for j, fr in enumerate((0, 2, 5)): img(ax, f"{D}/frame{fr}.jpg", 0.35 + j * 0.3, y1 + 1.35 - j * 0.17, 0.85, ec=GRID, lw=1)
ax.text(0.9, y1 + 0.95, "input frames", fontsize=7.8, color=MUT, ha="center", va="top")
arr(ax, (1.7, y1 + 1.7), (2.35, y1 + 1.7), color=TEAL, label="VGGT\n(frozen)", ly=0.16, fs=7.8)
img(ax, f"{D}/view28.jpg", 2.4, y1 + 0.75, 2.6, ec=BLUE, lw=1.8)
ax.text(3.7, y1 + 0.6, "explicit 3-D memory — the red door,\nnow at a known 3-D position", fontsize=7.8, color=BLUE, ha="center", va="top")
box(ax, 5.5, y1 + 1.05, 2.6, 1.5, "walk (beam over actions):\nstart at spot 3 →\nTURN RIGHT →\nFORWARD to spot 8", fc="#eef7f7", ec=TEAL, fs=8)
arr(ax, (5.05, y1 + 1.8), (5.5, y1 + 1.8), color=TEAL)
arr(ax, (8.1, y1 + 1.8), (8.7, y1 + 1.8), color=TEAL)
hv = img(ax, f"{D}/view57.jpg", 8.75, y1 + 0.75, 2.7, ec=TEAL, lw=1.8)
ell(ax, 8.75, y1 + 0.75, 2.7, hv, 0.12, 0.62, 0.26, 0.24, OK, None)
ax.text(10.1, y1 + 0.6, "acquired view: the counter under the microwave —\ncups by the sink sit directly beneath it", fontsize=7.8, color=TEAL, ha="center", va="top")
arr(ax, (11.5, y1 + 1.8), (12.1, y1 + 1.8), color=OK)
box(ax, 12.15, y1 + 1.25, 3.1, 1.15, "answer: D (cup)  ✓\nconsensus of the retained depth-3\nwalks, confidence above direct", fc="#dafbe1", ec=OK, fs=8.2)
fig.savefig(os.path.join(R, "figures/motivation_kitchen.png"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(R, "figures/motivation_kitchen.pdf"), bbox_inches="tight"); plt.close(fig); print("kitchen")

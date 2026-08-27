"""Draw the depth-1 ViewTree as a graph: a generic schematic and, for each
captured trace (results/traces/<tag>/traces.json), an instantiated tree with
the real frames/renders, head confidences, kept branches and the path taken.
  python scripts/tree_diagram.py --tag d10k --label "tree v4 + D_highcost 10k"
"""
import argparse, json, os, textwrap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); FIG = f"{R}/figures"
ORANGE, INK, MUT, GRID, OK, BAD, BLUE, AQUA = "#eb6834", "#26292e", "#6b6f76", "#d9d9d4", "#008300", "#c0392b", "#2a78d6", "#1baf7a"
VIEW_NAMES = ["side 1", "side 2", "side 3", "side 4", "top-down"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK})

def box(ax, x, y, w, h, text, fc="white", ec=GRID, lw=1.2, fs=8.5, weight="normal", color=INK):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight=weight, color=color, zorder=4)

def arrow(ax, p, q, color=MUT, lw=1.3, ls="-", label=None, lpos=0.5, dashed=False):
    a = FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, color=color, lw=lw, ls=(":" if dashed else ls), zorder=2, shrinkA=2, shrinkB=2)
    ax.add_patch(a)
    if label:
        ax.text(p[0] + (q[0]-p[0])*lpos, p[1] + (q[1]-p[1])*lpos + 0.12, label, fontsize=7.5, color=color, ha="center", va="bottom", zorder=5,
                bbox=dict(fc="white", ec="none", pad=0.5))

def schematic():
    fig, ax = plt.subplots(figsize=(13, 7.4), dpi=170); ax.set_xlim(0, 13); ax.set_ylim(0, 7.4); ax.axis("off")
    ax.text(0.2, 7.15, "ViewTree depth-1 tree (tree v4): branching factor 5, keep 2, depth 1", fontsize=12, fontweight="bold", va="top")
    ax.text(0.2, 6.78, "Node = one VLM call by the controller (Qwen2.5-VL-7B + LoRA); numbers in [ ] = confidence-head v2 score used for ranking / arbitration", fontsize=8.5, color=MUT, va="top")
    # root
    box(ax, 2.0, 5.6, 3.0, 1.0, "ROOT STATE\n8 video frames + question\n(VGGT reconstruction built once)", fc="#eef3fb", ec=BLUE, lw=1.6, fs=8.5)
    box(ax, 6.0, 5.6, 2.4, 0.8, "GATE (call 1)\n\"answer from frames alone?\"", fc="#fff6e8", ec=ORANGE, lw=1.6)
    arrow(ax, (3.5, 5.6), (4.8, 5.6))
    box(ax, 10.6, 5.6, 2.6, 0.8, "DIRECT answer (call 2)\nfrom frames only  [c_direct]", fc="white", ec=BLUE, lw=1.4)
    arrow(ax, (7.2, 5.6), (9.3, 5.6), color=OK, label="YES  →  mode: direct (stop, 1–2 calls)")
    # branches
    xs = [1.3, 3.65, 6.0, 8.35, 10.7]
    for i, x in enumerate(xs):
        arrow(ax, (6.0, 5.2), (x, 3.75), color=ORANGE if i in (1, 2) else MUT, lw=1.6 if i in (1, 2) else 1.0)
        box(ax, x, 3.3, 2.15, 0.9, f"BRANCH {i+1} (call {3+i})\nframes + render: {VIEW_NAMES[i]}\nanswer  [c_{i+1}]", fc="#fff6e8" if i in (1, 2) else "white", ec=ORANGE if i in (1, 2) else GRID, lw=2 if i in (1, 2) else 1.0, fs=7.8)
    ax.text(6.0, 4.55, "EXPLORE → render 5 heuristic viewpoints (4 elevated sides + top-down), one (action, view) pair each", fontsize=8, color=ORANGE, ha="center", va="center", bbox=dict(fc="white", ec="none", pad=0.8))
    ax.text(5.0, 2.62, "PRUNE: keep top-2 by head confidence (orange).", fontsize=8.5, color=ORANGE, ha="center", va="center", fontweight="bold")
    # consensus
    box(ax, 2.4, 1.55, 3.4, 0.85, "CONSENSUS early stop\nkept-2 agree AND c_kept > c_direct\n→ mode: branch_consensus (7 calls)", fc="#e9f7f1", ec=AQUA, lw=1.5, fs=7.8)
    box(ax, 7.6, 1.55, 3.2, 0.85, "FUSE (call 8)\nframes + 2 kept renders, pose-tagged\nanswer  [c_fused]", fc="#fff6e8", ec=ORANGE, lw=1.5, fs=7.8)
    arrow(ax, (3.65, 2.85), (2.6, 2.0), color=AQUA); arrow(ax, (6.0, 2.85), (2.9, 2.0), color=AQUA)
    arrow(ax, (3.65, 2.85), (7.2, 2.0), color=ORANGE); arrow(ax, (6.0, 2.85), (7.5, 2.0), color=ORANGE)
    ax.text(5.0, 2.2, "else", fontsize=7.5, color=ORANGE)
    box(ax, 11.5, 1.55, 2.4, 0.85, "ARBITRATE\nc_direct > c_fused and > c_kept ?\n→ direct : fused", fc="white", ec=INK, lw=1.2, fs=7.6)
    arrow(ax, (9.2, 1.55), (10.3, 1.55))
    arrow(ax, (10.6, 5.2), (11.7, 2.0), color=BLUE, dashed=True, label="c_direct kept for arbitration", lpos=0.45)
    box(ax, 7.0, 0.4, 5.0, 0.55, "FINAL: direct | branch_consensus | fused | fused_fallback_direct", fc="#f4f4f1", ec=GRID, fs=8.5, weight="bold")
    arrow(ax, (11.5, 1.12), (8.5, 0.68)); arrow(ax, (2.4, 1.12), (5.0, 0.68), color=AQUA)
    ax.text(0.2, 0.15, "Cost per question: 1–2 VLM calls if the gate says YES, otherwise up to 8 (gate, direct, 5 branches, fuse) + 1 VGGT reconstruction + 5 renders. "
            "Held-out VSI mode mix (D_10k): direct 597 · consensus 820 · fused 546 · fallback 594.", fontsize=7.5, color=MUT, va="bottom")
    fig.savefig(f"{FIG}/tree_schematic.png", bbox_inches="tight"); plt.close(fig)

def img(ax, path, cx, cy, w, alpha=1.0, ec=GRID, lw=1, max_h=None):
    im = Image.open(path); h = w * im.height / im.width
    if max_h and h > max_h: h = max_h; w = h * im.width / im.height
    ax.imshow(im, extent=(cx - w/2, cx + w/2, cy - h/2, cy + h/2), alpha=alpha, zorder=3, aspect="auto")
    ax.add_patch(plt.Rectangle((cx - w/2, cy - h/2), w, h, fc="none", ec=ec, lw=lw, zorder=4))
    return h

def instance(t, tag, label):
    TR = f"{R}/results/traces/{tag}"; qid = t["id"]; mode = t["mode"]; executed = mode != "direct"
    fig, ax = plt.subplots(figsize=(13, 9.2), dpi=170); ax.set_xlim(0, 13); ax.set_ylim(0, 9.2); ax.axis("off")
    ok = t["score"] > .5
    ax.text(0.2, 9.05, f"{label}  ·  #{qid}  ·  {t['qtype'].replace('_',' ')}  ·  {t['scene']}", fontsize=9, color=MUT, va="top")
    ax.text(0.2, 8.75, textwrap.fill(t["question"].split("\n")[0], 150), fontsize=10, fontweight="bold", va="top")
    ax.text(0.2, 8.4, f"path taken: {mode}   →   final {t['final'].strip()}   (GT {t['gt']})   {'CORRECT' if ok else 'WRONG'}", fontsize=10, color=OK if ok else BAD, fontweight="bold", va="top")
    # root frames (data coords)
    for i in range(4): img(ax, f"{TR}/{qid}_frame{i}.jpg", 0.75 + i*0.95, 7.0, 0.9, max_h=0.85)
    ax.text(2.2, 7.65, "ROOT: 8 video frames (4 shown) + question", fontsize=8, color=BLUE, ha="center", va="center")
    d = t["direct"]; gate_yes = "YES" in t["gate"].upper()
    box(ax, 6.0, 7.05, 2.3, 0.7, f"GATE → {t['gate']}", fc="#fff6e8", ec=ORANGE, lw=1.6, weight="bold")
    arrow(ax, (3.7, 7.05), (4.85, 7.05))
    dwin = mode in ("direct", "fused_fallback_direct")
    box(ax, 10.6, 7.05, 2.8, 0.7, f"DIRECT: {d['pred'].strip()}   [{d['conf']:.3f}]", fc="#e9f7f1" if dwin else "white", ec=OK if dwin else BLUE, lw=2 if dwin else 1.2)
    arrow(ax, (7.15, 7.05), (9.2, 7.05), color=OK if gate_yes else MUT, lw=2 if gate_yes else 1, label="YES: answer directly" if gate_yes else "(direct answer scored for arbitration)")
    xs = [1.3, 3.65, 6.0, 8.35, 10.7]; W = 2.15
    for b in t["branches"]:
        i = b["view"]; x = xs[i]; kept = executed and i in t["kept"]
        h = img(ax, f"{TR}/{qid}_view{i}.jpg", x, 4.85, W, alpha=1 if executed else 0.3, ec=ORANGE if kept else GRID, lw=3 if kept else 1)
        arrow(ax, (6.0, 6.7), (x, 4.85 + h/2 + 0.05), color=(ORANGE if kept else MUT), lw=(1.8 if kept else 0.9), dashed=not executed)
        ax.text(x, 4.85 - h/2 - 0.12, f"{VIEW_NAMES[i]}: {b['pred'].strip()[:10]}  [{b['conf']:.3f}]" + ("  KEPT" if kept else ""), fontsize=8, ha="center", va="top", color=ORANGE if kept else MUT, fontweight="bold" if kept else "normal")
    ybot = 4.85 - (W * 0.75) / 2 - 0.35
    if not executed:
        ax.text(6.0, 2.6, "gate said YES → branches were NOT executed (shown faded for reference)", fontsize=9, color=MUT, ha="center", style="italic")
    else:
        f = t["fuse"]; kc = [t["branches"][i]["conf"] for i in t["kept"]]; kp = {t["branches"][i]["pred"].strip() for i in t["kept"]}
        if mode == "branch_consensus":
            box(ax, 6.0, 2.4, 7.0, 0.7, f"CONSENSUS: kept branches agree on '{list(kp)[0]}', conf {max(kc):.3f} > direct {d['conf']:.3f} → stop (fusion not executed)", fc="#e9f7f1", ec=OK, lw=2, fs=8.5)
            for i in t["kept"]: arrow(ax, (xs[i], ybot), (6.0, 2.8), color=ORANGE, lw=1.8)
        else:
            box(ax, 3.6, 2.4, 4.6, 0.7, f"FUSE (frames + 2 kept renders): {f['pred'].strip()}   [{f['conf']:.3f}]", fc="#fff6e8", ec=ORANGE, lw=1.8, fs=8.5)
            for i in t["kept"]: arrow(ax, (xs[i], ybot), (3.6, 2.8), color=ORANGE, lw=1.8)
            win_direct = mode == "fused_fallback_direct"
            box(ax, 9.7, 2.4, 5.9, 0.8, f"ARBITRATE: direct {d['conf']:.3f} vs fused {f['conf']:.3f} vs kept {max(kc):.3f}\n→ " + ("DIRECT wins (fallback)" if win_direct else "FUSED wins"), fc="white", ec=INK, lw=1.2, fs=8.2)
            arrow(ax, (5.95, 2.4), (6.75, 2.4)); arrow(ax, (10.6, 6.7), (10.6, 2.8), color=BLUE, dashed=True)
    box(ax, 6.5, 1.0, 5.6, 0.7, f"FINAL ({mode}): {t['final'].strip()}    GT: {t['gt']}", fc="#e9f7f1" if ok else "#fdecea", ec=OK if ok else BAD, lw=2, fs=10, weight="bold")
    fig.savefig(f"{FIG}/tree_{tag}_{qid}.png", bbox_inches="tight"); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="d10k"); ap.add_argument("--label", default="tree v4 + D_highcost 10k"); ap.add_argument("--ids", type=int, nargs="*")
    a = ap.parse_args(); schematic()
    traces = json.load(open(f"{R}/results/traces/{a.tag}/traces.json"))
    for t in traces:
        if not a.ids or t["id"] in a.ids: instance(t, a.tag, a.label)
    print("written", sorted(f for f in os.listdir(FIG) if f.startswith("tree_")))

if __name__ == "__main__":
    main()

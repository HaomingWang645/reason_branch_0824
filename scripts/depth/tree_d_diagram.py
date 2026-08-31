"""Draw ViewTree-D walk diagrams from results/depth/treeD_trace.jsonl (+ figures/treeD_trace/<id>/ images).
Each figure: context frames -> gate -> direct answer [value]; then one column per beam level with the
rendered view of every explored state (kept = green, pruned = grey, invalid = red dashed), the action that
produced it and its answer [value]; final decision box.   python scripts/depth/tree_d_diagram.py"""
import json, os, sys, textwrap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TR = os.path.join(R, "results/depth/treeD_trace.jsonl"); FD = os.path.join(R, "figures/treeD_trace"); OUT = os.path.join(R, "figures")
INK, MUT, GRID, OK, BAD, KEEP = "#1f2328", "#57606a", "#d0d7de", "#1a7f37", "#cf222e", "#0969da"
TN = {"obj_appearance_order": "appearance order", "object_abs_distance": "abs distance", "object_counting": "counting", "object_rel_direction_easy": "dir easy",
      "object_rel_direction_hard": "dir hard", "object_rel_direction_medium": "dir medium", "object_rel_distance": "rel distance", "object_size_estimation": "size",
      "room_size_estimation": "room size", "route_planning": "route planning"}

def box(ax, x, y, w, h, text, fc="white", ec=GRID, lw=1.2, fs=8, color=INK, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, linespacing=1.25)

def img(ax, path, x, y, w, ec=GRID, lw=1.2, ls="-"):
    im = mpimg.imread(path); h = w * im.shape[0] / im.shape[1]
    ax.imshow(im, extent=(x, x + w, y, y + h), aspect="auto", zorder=2)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", fc="none", ec=ec, lw=lw, ls=ls, zorder=3)); return h

def arrow(ax, p, q, color=MUT, ls="-"):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, ls=ls, shrinkA=2, shrinkB=2), zorder=1)

def draw(t):
    fd = os.path.join(FD, str(t["id"])); levels = t.get("levels", [])
    ncol = 1 + len(levels); nmax = max([1] + [len(l["beam"]) for l in levels])
    W = 1.9; G = 0.55; IW = 1.55; IH = IW * 0.75; RH = IH + 0.62
    fig_w = 2.6 + ncol * (W + G) + 2.2; fig_h = 1.5 + max(2.6, nmax * RH)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h)); ax.set_xlim(0, fig_w); ax.set_ylim(0, fig_h); ax.axis("off")
    ok = t["score"] >= 0.5
    head = f"#{t['id']} · {TN.get(t['question_type'], t['question_type'])} · {t['scene']}\n" + textwrap.fill(t["question"].split("\n")[0], 150)
    ax.text(0.15, fig_h - 0.18, head, ha="left", va="top", fontsize=8.5, color=INK)
    top = fig_h - 0.95
    # context frames strip
    x0 = 0.15; fw = 0.62; y_strip = top - fw * 0.75
    for j in [0, 2, 5, 7]:
        p = os.path.join(fd, f"frame{j}.jpg")
        if os.path.exists(p): img(ax, p, x0, y_strip, fw); x0 += fw + 0.06
    ax.text(0.15, y_strip - 0.16, "8 context frames (4 shown)", fontsize=7.5, color=MUT, va="top")
    # gate + direct
    gx, gy = 0.15, y_strip - 1.05
    gate = t["gate"].upper(); box(ax, gx, gy, 1.1, 0.42, f"gate: {gate}", fc="#fff8c5" if "YES" in gate else "#ddf4ff", fs=8)
    dx = gx + 1.3; box(ax, dx, gy, 1.15, 0.42, f"direct: {t['direct']}\n[{t['dconf']:.2f}]", fs=7.8)
    arrow(ax, (gx + 1.1, gy + 0.21), (dx, gy + 0.21))
    pos = {(): (dx + 1.15, gy + 0.21)}
    # beam levels
    xl = 2.75
    for li, lev in enumerate(levels):
        n = len(lev["beam"]); col_h = n * RH; ytop = top + 0.1
        ax.text(xl + W / 2, top + 0.25, f"depth {lev['depth']}", ha="center", fontsize=8.5, color=INK, weight="bold")
        for k, b in enumerate(lev["beam"]):
            y = ytop - (k + 1) * RH; path = [tuple(p) for p in b["path"]]; act, idx = path[-1]
            ec = KEEP if b["kept"] else (BAD if not b["valid"] else GRID); ls = "--" if not b["valid"] else "-"; lw = 2.0 if b["kept"] else 1.2
            p = os.path.join(fd, f"view{idx}.jpg")
            if os.path.exists(p): h = img(ax, p, xl, y + 0.5, IW, ec=ec, lw=lw, ls=ls)
            else: box(ax, xl, y + 0.5, IW, IH, "(no render)", ec=ec, ls=ls); h = IH
            lab = act.replace("_", " ").lower(); desc = t.get("views", {}).get(str(idx), "")
            desc = desc.replace("eye-level view from standing spot", "spot").replace("facing direction", "dir").replace(" of 8", "")
            ans = "invalid view" if not b["valid"] else f"ans {b['answer']}  [{b['value']:.2f}]"
            ax.text(xl + IW / 2, y + 0.5 + h + 0.05, f"{lab} → {desc}", ha="center", va="bottom", fontsize=6.6, color=MUT)
            ax.text(xl + IW / 2, y + 0.42, ans, ha="center", va="top", fontsize=7.6, color=KEEP if b["kept"] else (BAD if not b["valid"] else INK), weight="bold" if b["kept"] else "normal")
            c = (xl, y + 0.5 + h / 2); pos[tuple(path)] = (xl + IW, y + 0.5 + h / 2)
            parent = pos.get(tuple(path[:-1]), pos[()]); arrow(ax, parent, c, color=KEEP if b["kept"] else MUT)
        xl += W + G
    # final
    fx = xl + 0.1; fy = gy
    mode = t["mode"]; txt = f"{mode.replace('_', ' ')}\nfinal: {t['pred']}\nGT {t['gt']}  {'✓' if ok else '✗'}"
    box(ax, fx, fy - 0.15, 1.9, 0.78, txt, fc="#dafbe1" if ok else "#ffebe9", ec=OK if ok else BAD, lw=1.6, fs=8)
    src = pos.get(tuple(tuple(p) for p in t["path"]), pos[()]); arrow(ax, src, (fx, fy + 0.24), color=OK if ok else BAD)
    ax.text(fx, fy - 0.3, f"{t['calls']} VLM calls · depth {t['depth']}", fontsize=7.2, color=MUT, va="top")
    ax.text(0.15, 0.12, "blue border = kept by value head · grey = pruned · red dashed = invalid (coverage < 45 %) · [ ] = value-head score", fontsize=7, color=MUT)
    out = os.path.join(OUT, f"treeD_{t['question_type']}_{t['id']}.png"); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig); return out

if __name__ == "__main__":
    ids = set(int(x) for x in sys.argv[1:]) if len(sys.argv) > 1 else None
    for l in open(TR):
        t = json.loads(l)
        if ids and t["id"] not in ids: continue
        print(draw(t))

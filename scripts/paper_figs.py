"""Paper result figures (PDF+PNG each):
  fig_cross_bench  - grouped bars, 5 systems x 4 benchmarks
  fig_depth_delta  - where multi-step acquisition helps (per VSI type + per benchmark)
  fig_cost_acc     - accuracy vs mean VLM calls on VSI odd half
  fig_scale        - backbone scale study
Values computed from result files where per-item data exists; cross-benchmark
means match RESULTS.md §6/§8."""
import json, glob, os, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys; sys.path.insert(0, R)
from viewtree.data import load_questions
INK, SEC, GRID = "#0b0b0b", "#52514e", "#e3e2de"
C = {"zs": "#2a78d6", "plain": "#eb6834", "tree1": "#1baf7a", "single": "#eda100", "beam": "#e87ba4"}
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": GRID, "axes.linewidth": 0.8, "text.color": INK,
                     "axes.labelcolor": SEC, "xtick.color": SEC, "ytick.color": SEC, "figure.facecolor": "white"})
def save(fig, name):
    fig.savefig(os.path.join(R, f"figures/{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(R, f"figures/{name}.png"), dpi=200, bbox_inches="tight"); plt.close(fig); print(name)

rows = {r["id"]: r for r in load_questions()}
def load(pat):
    d = {}
    for f in glob.glob(os.path.join(R, pat)):
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            if "score" in r: d[r["id"]] = r
    return d

# ---------- fig A: cross-benchmark ----------
sysnames = ["Qwen2.5-VL-7B\nzero-shot", "SFT-plain\n(benchmark data)", "ViewTree depth-1\n(MindCube-trained)", "corpus SFT,\nsingle pass", "ViewTree-D beam\n(corpus-trained)"]
cols = [C["zs"], C["plain"], C["tree1"], C["single"], C["beam"]]
bench = ["VSI (held-out)", "OST", "STI", "VSTI"]
vals = np.array([[0.313, 0.540, 0.371, 0.523], [0.327, 0.524, 0.261, 0.456], [0.367, 0.541, 0.306, 0.505],
                 [0.524, 0.518, 0.341, 0.685], [0.530, 0.496, 0.345, 0.680]])
fig, ax = plt.subplots(figsize=(7.2, 2.9))
x = np.arange(4); w = 0.16
for i, (n, c) in enumerate(zip(sysnames, cols)):
    b = ax.bar(x + (i - 2) * w, vals[i], w * 0.9, color=c, label=n.replace("\n", " "), zorder=3)
    for xi, v in zip(x + (i - 2) * w, vals[i]): ax.text(xi, v + 0.008, f"{v:.2f}".lstrip("0"), ha="center", fontsize=6.4, color=SEC)
ax.set_ylim(0, 0.78); ax.set_xticks(x); ax.set_xticklabels(bench); ax.set_ylabel("accuracy (mean over question types)")
ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(ncol=3, frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 1.22))
save(fig, "fig_cross_bench")

# ---------- fig B: where depth helps ----------
base = load("results/depth/frames16_sftframes_s*.jsonl"); beam = load("results/depth/treeD_sftc_s*.jsonl")
ids = [i for i in beam if i in base]
types = sorted({rows[i]["question_type"] for i in ids})
TN = {t: t.replace("object_", "").replace("_estimation", "").replace("rel_direction_", "direction ").replace("_", " ") for t in types}
dts = sorted([(100 * (np.mean([beam[i]["score"] for i in ids if rows[i]["question_type"] == t]) -
                      np.mean([base[i]["score"] for i in ids if rows[i]["question_type"] == t])), t) for t in types])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.55, 1], "wspace": 0.32})
y = np.arange(len(dts))
a1.barh(y, [d for d, _ in dts], 0.62, color=[C["zs"] if d > 0 else C["plain"] for d, _ in dts], zorder=3)
for yi, (d, t) in zip(y, dts): a1.text(d + (0.25 if d > 0 else -0.25), yi, f"{d:+.1f}", va="center", ha="left" if d > 0 else "right", fontsize=6.8, color=SEC)
a1.set_yticks(y); a1.set_yticklabels([TN[t] for _, t in dts], fontsize=7.5)
a1.axvline(0, color=SEC, lw=0.9); a1.set_xlim(-4.5, 9.5); a1.xaxis.grid(True, color=GRID, lw=0.7); a1.set_axisbelow(True)
a1.set_xlabel("Δ vs data-matched baseline (pts), VSI held-out"); a1.set_title("(a) per question type", fontsize=8.5, loc="left")
for s in ("top", "right"): a1.spines[s].set_visible(False)
bn = ["VSI", "STI", "VSTI", "OST"]; d = [2.1, 0.4, -0.5, -2.1]; lo = [-0.0, -1.0, -1.2, -3.1]; hi = [4.3, 1.8, 0.2, -1.2]
y2 = np.arange(4)
a2.barh(y2, d, 0.55, color=[C["zs"] if v > 0 else C["plain"] for v in d], zorder=3)
a2.errorbar(d, y2, xerr=[np.array(d) - np.array(lo), np.array(hi) - np.array(d)], fmt="none", ecolor=INK, elinewidth=1, capsize=2.5, zorder=4)
for yi, v, l_, h_ in zip(y2, d, lo, hi): a2.text((h_ + 0.5) if v > 0 else (l_ - 0.5), yi, f"{v:+.1f}", fontsize=7, color=SEC, ha="left" if v > 0 else "right", va="center")
a2.set_yticks(y2); a2.set_yticklabels(bn, fontsize=8); a2.axvline(0, color=SEC, lw=0.9); a2.set_xlim(-5.6, 6.6)
a2.xaxis.grid(True, color=GRID, lw=0.7); a2.set_axisbelow(True)
a2.set_xlabel("Δ beam − single pass (pts, 95% CI)"); a2.set_title("(b) per benchmark", fontsize=8.5, loc="left")
for s in ("top", "right"): a2.spines[s].set_visible(False)
save(fig, "fig_depth_delta")

# ---------- fig C: accuracy vs cost ----------
def calls_from_modes(d, direct=2, cons=7, full=8):
    m = collections.Counter(v["mode"] for v in d.values() if "mode" in v)
    tot = sum(m.values())
    return (m.get("direct", 0) * direct + m.get("branch_consensus", 0) * cons + (m.get("fused", 0) + m.get("fused_fallback_direct", 0)) * full) / max(tot, 1)
tree1 = load("results/human/tree_d10k_headH_s*.jsonl"); tree1A = load("results/depth/tree1_sfta_s*.jsonl")
grpoF = load("results/depth/frames16_grpoI.jsonl")
mot = lambda d: np.mean([np.mean([d[i]["score"] for i in d if rows[i]["question_type"] == t]) for t in types])
pts = [("zero-shot", 1, 0.313, C["zs"], "o"), ("SFT-plain", 1, 0.327, C["plain"], "o"), ("static memory", 1, 0.342, C["plain"], "s"),
       ("ViewTree depth-1 (best)", calls_from_modes(tree1), 0.367, C["tree1"], "^"),
       ("corpus SFT", 1, 0.509, C["single"], "o"), ("SFT-A", 1, 0.524, C["single"], "D"),
       ("depth-1 + SFT-A", calls_from_modes(tree1A), 0.517, C["beam"], "^"), ("ViewTree-D beam", 4.5, 0.530, C["beam"], "*")]
fig, ax = plt.subplots(figsize=(4.6, 3.0))
for n, cx, cy, c, mk in pts:
    ax.scatter(cx, cy, s=90 if mk == "*" else 45, color=c, marker=mk, zorder=4, edgecolors="white", linewidths=1.2)
    dy = {"zero-shot": -0.023, "SFT-A": 0.009, "corpus SFT": -0.031, "SFT-plain": -0.004}.get(n, 0.011)
    ax.text(cx + 0.1, cy + dy, n, fontsize=7.2, color=INK)
ax.set_xlabel("mean VLM calls per question"); ax.set_ylabel("VSI held-out accuracy")
ax.set_xlim(0.4, 8.6); ax.set_ylim(0.28, 0.57); ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.annotate("", xy=(4.35, 0.529), xytext=(1.3, 0.514), arrowprops=dict(arrowstyle="-|>", color=SEC, ls=":", lw=1))
ax.text(3.6, 0.487, "+0.6 for 3.5 extra calls\n(gate stops 71%)", fontsize=6.8, color=SEC, ha="center")
save(fig, "fig_cost_acc")

# ---------- fig D: backbone scale ----------
fig, ax = plt.subplots(figsize=(4.6, 2.9))
g = np.arange(3); w = 0.3
f16 = [0.321, 0.313, 0.380]; lite = [0.314, 0.341, 0.348]
ax.bar(g - w / 2, f16, w * 0.92, color=C["zs"], label="frames only (zero-shot)", zorder=3)
ax.bar(g + w / 2, lite, w * 0.92, color=C["tree1"], label="+ training-free tree", zorder=3)
ax.bar([3.05], [0.497], w * 0.92, color=C["single"], label="3B + corpus SFT (30k)", zorder=3)
for xi, v in list(zip(g - w / 2, f16)) + list(zip(g + w / 2, lite)) + [(3.05, 0.497)]:
    ax.text(xi, v + 0.007, f"{v:.2f}".lstrip("0"), ha="center", fontsize=7, color=SEC)
dl = ["−0.7 n.s.", "+2.8", "−3.2"]
for xi, t, v in zip(g, dl, lite): ax.text(xi + w / 2, v + 0.036, t, ha="center", fontsize=7.2, color=INK, style="italic")
ax.axhline(0.509, color=C["single"], lw=1, ls=":"); ax.text(1.9, 0.515, "7B corpus SFT (100k): 0.51", fontsize=6.6, color=SEC, va="bottom", ha="center")
ax.set_xticks(list(g) + [3.05]); ax.set_xticklabels(["3B", "7B", "32B", "3B\n+SFT"]); ax.set_ylim(0, 0.6)
ax.set_ylabel("VSI held-out accuracy"); ax.yaxis.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=7, loc="upper left")
save(fig, "fig_scale")

"""Render per-question trace montages for a captured tag (results/traces/<tag>),
optionally with a comparison tag's decision on the same question.
  python scripts/trace_figures_v2.py --tag d10k --label "tree v4 + D_highcost 10k" --compare tree4 --compare-label "tree v4 (SFT-v2 adapter)"
Writes figures/trace_<tag>_<id>.png and figures/trace_<tag>_summary.png."""
import argparse, json, os, textwrap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); FIG = f"{R}/figures"
ORANGE, INK, MUT, GRID, OK, BAD, BLUE = "#eb6834", "#26292e", "#6b6f76", "#e8e8e4", "#008300", "#c0392b", "#2a78d6"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "font.size": 10})
VIEW_NAMES = ["side 1", "side 2", "side 3", "side 4", "top-down"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--label", required=True)
    ap.add_argument("--compare"); ap.add_argument("--compare-label", default="")
    a = ap.parse_args()
    TR = f"{R}/results/traces/{a.tag}"
    traces = json.load(open(f"{TR}/traces.json"))
    cmp = {t["id"]: t for t in json.load(open(f"{R}/results/traces/{a.compare}/traces.json"))} if a.compare else {}
    for t in traces:
        qid = t["id"]; ok = t["score"] > .5
        fig = plt.figure(figsize=(11, 6.6), dpi=170)
        gs = fig.add_gridspec(2, 20, hspace=.36, top=.68, bottom=.12, left=.02, right=.98)
        fig.text(.02, .975, f"{a.label}   |   #{qid} - {t['qtype'].replace('_', ' ')} - {t['scene']}", fontsize=8.5, color=MUT, va="top")
        fig.text(.02, .945, textwrap.fill(t["question"].split("\n")[0], 140), fontsize=9.5, color=INK, fontweight="bold", va="top")
        verdict = ("CORRECT" if ok else "WRONG") + f" - mode {t['mode']} - {t['executed']}"
        fig.text(.02, .77, textwrap.fill(verdict, 150), fontsize=9.5, color=(OK if ok else BAD), fontweight="bold", va="top")
        for i in range(4):
            ax = fig.add_subplot(gs[0, i*4:(i+1)*4]); ax.imshow(Image.open(f"{TR}/{qid}_frame{i}.jpg")); ax.axis("off")
            if i == 0: ax.set_title("video evidence (4 of 8 frames)", fontsize=8.5, loc="left", color=MUT)
        gax = fig.add_subplot(gs[0, 16:]); gax.axis("off"); d = t["direct"]
        lines = [f"gate:   {t['gate']}", f"direct: {d['pred'].strip()}  ({d['conf']:.3f})",
                 f"fused:  {t['fuse']['pred'].strip()}  ({t['fuse']['conf']:.3f})", f"mode:   {t['mode']}",
                 f"final:  {t['final'].strip()}   GT: {t['gt']}"]
        gax.text(0, .95, "\n".join(lines), fontsize=9, family="monospace", va="top", color=INK)
        executed = t["mode"] != "direct"
        for b in t["branches"]:
            vi = b["view"]; ax = fig.add_subplot(gs[1, vi*4:(vi+1)*4])
            im = Image.open(f"{TR}/{qid}_view{vi}.jpg")
            ax.imshow(im, alpha=1.0 if executed else 0.35)
            kept = executed and vi in t["kept"]
            for sp in ax.spines.values(): sp.set_edgecolor(ORANGE if kept else GRID); sp.set_linewidth(3 if kept else 1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(f"{VIEW_NAMES[vi]} - {b['pred'].strip()[:10]} ({b['conf']:.3f})" + ("  KEPT" if kept else ""), fontsize=8, color=(ORANGE if kept else MUT))
            if vi == 0: ax.set_title("rendered branch viewpoints (kept = orange" + ("" if executed else "; faded = not executed, gate stopped early") + ")", fontsize=8.5, loc="left", color=MUT)
        if qid in cmp:
            c = cmp[qid]; cok = c["score"] > .5
            fig.text(.02, .035, f"{a.compare_label} on the same question: gate {c['gate']}, mode {c['mode']}, final {c['final'].strip()} -> " + ("CORRECT" if cok else "WRONG")
                     + f"   (direct {c['direct']['conf']:.3f}, fused {c['fuse']['conf']:.3f})", fontsize=9, color=(OK if cok else BAD), va="bottom")
        fig.savefig(f"{FIG}/trace_{a.tag}_{qid}.png"); plt.close(fig)
    # summary table figure
    fig, ax = plt.subplots(figsize=(13, 0.9 + 0.42 * len(traces)), dpi=170); ax.axis("off")
    cols = ["id", "type", "gate", "mode", "final / GT", "score"] + (["cmp: mode", "cmp: final", "cmp: score"] if cmp else [])
    rows = []
    for t in traces:
        r = [t["id"], t["qtype"].replace("object_", "").replace("_estimation", ""), t["gate"], t["mode"], f"{t['final'].strip()} / {t['gt']}", f"{t['score']:.1f}"]
        if t["id"] in cmp: c = cmp[t["id"]]; r += [c["mode"], c["final"].strip(), f"{c['score']:.1f}"]
        rows.append(r)
    widths = [.05, .17, .08, .17, .1, .06] + ([.17, .1, .08] if cmp else [])
    tb = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="left", colWidths=widths); tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.3)
    ax.set_title(f"{a.label}: decisions on the six example questions" + (f"   (cmp = {a.compare_label})" if cmp else ""), fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(f"{FIG}/trace_{a.tag}_summary.png"); plt.close(fig)
    print("written", sorted(f for f in os.listdir(FIG) if a.tag in f))

if __name__ == "__main__":
    main()

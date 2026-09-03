"""Evaluation-section figures with real data.

Efficiency (measured on Jetson AGX Orin, 7B backbone; jetson/JETSON_MEASUREMENTS.md):
  vt_eff_methods.pdf   per-question latency + energy by method
  vt_eff_routes.pdf    cost of each adaptive route with its frequency
  vt_eff_breakdown.pdf where the time goes (phase breakdown per route)

Complexity (VSI-Bench held-out odd half, per-question scores):
  vt_object_number.pdf     counting questions binned by ground-truth object count
  vt_spatial_scale.pdf     all questions binned by the scene's room area
  vt_temporal_duration.pdf all questions binned by video duration
Systems: Direct input (zero-shot frames16), Standard SFT (corpus frames-only),
ViewTree (SFT-C beam). Bin stats cached in results/paperfill/complexity_bins.json.
"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R)
OUT = os.path.join(R, "paper/ViewTree__Resource_Aware_On_Device_Spatial_Reasoning_via_Adaptive_Viewpoint_Branching_and_Fusion__1_/figures")
CACHE = os.path.join(R, "results/paperfill/complexity_bins.json")
INK, MUT, RED, TEALD, BLUE = "#1f2328", "#57606a", "#d62828", "#155e63", "#0969da"
GREY = "#9aa4b1"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#888", "axes.linewidth": 0.8})

# ---------------- efficiency (measured) ----------------
def eff_methods():
    meth = ["Direct\ninput", "Static\nmemory", "Tree d=1\n(non-adpt.)", "ViewTree\n(ours)"]
    lat = [11.7, 11.4, 46.3, 24.5]
    en = [529, 518, 2110, 1104]
    fig, ax = plt.subplots(figsize=(2.8, 1.9))
    x = np.arange(4)
    b1 = ax.bar(x - 0.19, lat, 0.38, color=TEALD, label="latency (s)")
    ax.set_ylabel("latency (s)", color=TEALD)
    ax.tick_params(axis="y", labelcolor=TEALD)
    ax2 = ax.twinx()
    b2 = ax2.bar(x + 0.19, [e / 1000 for e in en], 0.38, color=RED, label="energy (kJ)")
    ax2.set_ylabel("energy (kJ)", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax.set_xticks(x); ax.set_xticklabels(meth, fontsize=5.7)
    for xi, v in zip(x, lat): ax.text(xi - 0.24, v + 1.2, f"{v:.0f}s", ha="center", fontsize=5.7, color=TEALD)
    for xi, v in zip(x, en): ax2.text(xi + 0.24, v / 1000 + 0.06, f"{v/1000:.1f}", ha="center", fontsize=5.7, color=RED)
    ax.set_ylim(0, 55); ax2.set_ylim(0, 2.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/vt_eff_methods.pdf"); fig.savefig(f"{OUT}/vt_eff_methods.png", dpi=170); plt.close(fig)

def eff_routes():
    routes = ["depth 0\n(gated)", "depth 1", "depth 2", "depth 3"]
    share = [71.0, 14.8, 6.1, 8.1]
    lat = [9.6, 26.0, 70.9, 117.2]
    fig, ax = plt.subplots(figsize=(2.45, 1.9))
    x = np.arange(4)
    bars = ax.bar(x, lat, 0.55, color=TEALD)
    for xi, v, s in zip(x, lat, share):
        ax.text(xi, v + 3.2, f"{v:.0f}s", ha="center", fontsize=6.4, color=INK)
        ax.text(xi, 3.5, f"{s:.0f}%", ha="center", fontsize=6.8, color="white", weight="bold")
    l1 = ax.axhline(24.5, color=RED, lw=1.2, ls="--", label="ViewTree expected mix: 24.5 s")
    l2 = ax.axhline(11.7, color=GREY, lw=1.0, ls=":", label="one 16-frame call: 11.7 s")
    ax.legend(handles=[l1, l2], fontsize=5.5, frameon=False, loc="upper left", handlelength=1.6)
    ax.set_xticks(x); ax.set_xticklabels(routes, fontsize=6.8)
    ax.set_ylabel("latency (s)"); ax.set_ylim(0, 132)
    fig.tight_layout()
    fig.savefig(f"{OUT}/vt_eff_routes.pdf"); fig.savefig(f"{OUT}/vt_eff_routes.png", dpi=170); plt.close(fig)

def eff_breakdown():
    # phases per route, measured call shapes (s): recon amortized 0.7; gate 4.33;
    # answer(8f) 5.19; answer+renders ~5.4-6.1; control prefill 4.30; render 0.12 each
    rows = [
        ("Direct\n(16 frames)", [("VLM call", 11.7, TEALD)]),
        ("ViewTree\ndepth 0", [("gate", 4.33, BLUE), ("answer", 5.19, TEALD)]),
        ("ViewTree\ndepth 1", [("gate", 4.33, BLUE), ("render+encode", 0.6, "#b26a00"),
                               ("answers (3 views)", 16.8, TEALD), ("control", 4.30, RED)]),
    ]
    fig, ax = plt.subplots(figsize=(2.45, 1.9))
    labels = []
    for i, (name, parts) in enumerate(rows):
        xoff = 0
        for pname, v, c in parts:
            ax.barh(i, v, left=xoff, height=0.52, color=c, ec="white", lw=0.4)
            xoff += v
        ax.text(xoff + 0.6, i, f"{xoff:.1f}s", va="center", fontsize=6.8, color=INK)
        labels.append(name)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels, fontsize=6.8)
    ax.set_xlabel("time per question (s)"); ax.set_xlim(0, 32); ax.invert_yaxis()
    import matplotlib.patches as mpatches
    leg = [mpatches.Patch(color=c, label=l) for l, c in
           [("gate", BLUE), ("render+encode", "#b26a00"), ("answer call(s)", TEALD), ("control call", RED)]]
    ax.legend(handles=leg, fontsize=5.3, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.52), handlelength=1.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/vt_eff_breakdown.pdf"); fig.savefig(f"{OUT}/vt_eff_breakdown.png", dpi=170); plt.close(fig)


# ---------------- per-backbone cost facets (example-paper style) ----------------
EBLUE, ERED = "#4C72B0", "#C44E52"
# Video-CoT is DERIVED, not a full system run: direct-input call + max(0, mean CoT
# decode length - the harness's 32-token budget) x the backbone's measured decode
# rate (0.097 / 0.11 / 0.27 s/token) at its measured average power (37 / 45 / 52 W).
# Mean CoT decode lengths from the evaluation runs: 38.6 / 12.8 / 61.1 tokens.
BB_DATA = {
    "3b":  ("Qwen2.5-VL-3B (bf16)",  [8.4, 9.0, 8.0, 35.3, 17.6],  [0.338, 0.362, 0.319, 1.321, 0.681], None),
    "7b":  ("Qwen2.5-VL-7B (bf16)",  [11.7, 11.7, 11.4, 46.3, 24.5], [0.529, 0.529, 0.518, 2.110, 1.104], None),
    "32b": ("Qwen2.5-VL-32B (NF4)",  [32.2, 40.0, 31.4, 126.5, 69.6], [1.690, 2.100, 1.673, 6.717, 3.690], "bf16: OOM"),
}
METHS = ["Direct", "Video\nCoT*", "Static\nMem.", "Tree\nd=1", "Ours"]
COT_IDX = 1

def eff_backbone(tag):
    name, lat, en, note = BB_DATA[tag]
    fig, ax = plt.subplots(figsize=(2.6, 1.95))
    x = np.arange(5)
    hat = ["" if i != COT_IDX else "///" for i in range(5)]
    for xi, v, h in zip(x, lat, hat):
        ax.bar(xi - 0.19, v, 0.32, color=EBLUE, hatch=h, edgecolor="white" if h else EBLUE, lw=0.4)
    ax.bar(0, 0, 0, color=EBLUE, label="Latency (s)")
    ax.set_ylabel("Latency (s)", color=EBLUE, fontsize=8)
    ax.tick_params(axis="y", labelcolor=EBLUE, labelsize=7)
    ax2 = ax.twinx()
    for xi, v, h in zip(x, en, hat):
        ax2.bar(xi + 0.19, v, 0.32, color=ERED, hatch=h, edgecolor="white" if h else ERED, lw=0.4)
    ax2.bar(0, 0, 0, color=ERED, label="Energy (kJ)")
    ax2.set_ylabel("Energy (kJ)", color=ERED, fontsize=8)
    ax2.tick_params(axis="y", labelcolor=ERED, labelsize=7)
    ymax = max(lat) * 1.45; ax.set_ylim(0, ymax)
    y2max = max(en) * 1.45; ax2.set_ylim(0, y2max)
    for xi, v in zip(x, lat):
        ax.text(xi - 0.23, v + ymax * 0.015, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", fontsize=5.3, color=EBLUE, weight="bold")
    for xi, v in zip(x, en):
        ax2.text(xi + 0.23, v + y2max * 0.015, f"{v:.1f}" if v < 10 else f"{v:.0f}", ha="center", fontsize=5.3, color=ERED, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(METHS, fontsize=6.2)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=5.6, loc="upper left", frameon=True, framealpha=0.9, edgecolor="#cccccc", handlelength=1.0, borderpad=0.25, labelspacing=0.25, handletextpad=0.4)
    if note:
        ax.text(0.985, 0.985, note, transform=ax.transAxes, fontsize=5.8, color=ERED, ha="right", va="top", weight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/vt_eff_{tag}.pdf"); fig.savefig(f"{OUT}/vt_eff_{tag}.png", dpi=170); plt.close(fig)

# ---------------- complexity (per-question analysis) ----------------
def compute_bins():
    import cv2
    from viewtree.data import load_questions
    rows = load_questions()
    scenes = sorted({(r["dataset"], r["scene_name"]) for r in rows})
    odd = set(scenes[1::2])
    byid = {r["id"]: r for r in rows if (r["dataset"], r["scene_name"]) in odd}

    def read(paths):
        d = {}
        for p in paths:
            for l in open(os.path.join(R, p)):
                r = json.loads(l)
                if r["id"] in byid:
                    d[r["id"]] = r["score"]
        return d
    S = {
        "direct": read(["results/frames16_s0.jsonl", "results/frames16_s1.jsonl"]),
        "sft": read(["results/depth/frames16_sftframes_s0.jsonl", "results/depth/frames16_sftframes_s1.jsonl"]),
        "vt": read([f"results/depth/treeD_sftc_s{i}.jsonl" for i in range(4)]),
    }
    ids = sorted(set(S["direct"]) & set(S["sft"]) & set(S["vt"]))
    print("paired ids:", len(ids))

    # scene room area from room-size questions' GT; video duration via cv2
    area, dur = {}, {}
    for r in rows:
        k = (r["dataset"], r["scene_name"])
        if k in odd and r["question_type"] == "room_size_estimation":
            area[k] = float(r["ground_truth"])
    for k in odd:
        v = next((r["video"] for r in rows if (r["dataset"], r["scene_name"]) == k), None)
        cap = cv2.VideoCapture(v)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        dur[f"{k[0]}|{k[1]}"] = float(n / max(fps, 1))
        cap.release()

    out = {"n_paired": len(ids), "bins": {}}
    def binstat(name, groups):
        res = []
        for label, qids in groups:
            if not qids: continue
            res.append({"label": label, "n": len(qids),
                        **{s: float(np.mean([S[s][i] for i in qids])) for s in S}})
        out["bins"][name] = res

    cnt = [i for i in ids if byid[i]["question_type"] == "object_counting"]
    gtc = {i: float(byid[i]["ground_truth"]) for i in cnt}
    binstat("object_number", [
        ("1–2", [i for i in cnt if gtc[i] <= 2]),
        ("3–4", [i for i in cnt if 3 <= gtc[i] <= 4]),
        ("≥5", [i for i in cnt if gtc[i] >= 5])])

    wa = [i for i in ids if (byid[i]["dataset"], byid[i]["scene_name"]) in area]
    ar = {i: area[(byid[i]["dataset"], byid[i]["scene_name"])] for i in wa}
    qs = np.quantile(sorted(area.values()), [1 / 3, 2 / 3])
    binstat("spatial_scale", [
        (f"<{qs[0]:.0f} m²", [i for i in wa if ar[i] < qs[0]]),
        (f"{qs[0]:.0f}–{qs[1]:.0f} m²", [i for i in wa if qs[0] <= ar[i] < qs[1]]),
        (f"≥{qs[1]:.0f} m²", [i for i in wa if ar[i] >= qs[1]])])
    out["area_cuts"] = [float(q) for q in qs]

    dr = {i: dur[f"{byid[i]['dataset']}|{byid[i]['scene_name']}"] for i in ids}
    qd = np.quantile(sorted(dur.values()), [1 / 3, 2 / 3])
    binstat("temporal_duration", [
        (f"<{qd[0]:.0f} s", [i for i in ids if dr[i] < qd[0]]),
        (f"{qd[0]:.0f}–{qd[1]:.0f} s", [i for i in ids if qd[0] <= dr[i] < qd[1]]),
        (f"≥{qd[1]:.0f} s", [i for i in ids if dr[i] >= qd[1]])])
    out["dur_cuts"] = [float(q) for q in qd]
    json.dump(out, open(CACHE, "w"), indent=1)
    return out

def complexity_figs(data):
    names = {"object_number": ("vt_object_number", "ground-truth object count"),
             "spatial_scale": ("vt_spatial_scale", "room area"),
             "temporal_duration": ("vt_temporal_duration", "video duration")}
    for key, (fname, xlab) in names.items():
        bins = data["bins"][key]
        fig, ax = plt.subplots(figsize=(2.35, 1.85))
        x = np.arange(len(bins)); w = 0.26
        for off, skey, lab, c in [(-w, "direct", "Direct input", GREY), (0, "sft", "Standard SFT", TEALD), (w, "vt", "ViewTree", RED)]:
            vals = [b[skey] for b in bins]
            ax.bar(x + off, vals, w, color=c, label=lab)
        ax.set_xticks(x); ax.set_xticklabels([b["label"] for b in bins], fontsize=7)
        ax.set_xlabel(xlab, fontsize=7.4); ax.set_ylabel("score", fontsize=7.4)
        ax.set_ylim(0, max(b[s] for b in bins for s in ("direct", "sft", "vt")) * 1.48)
        ax.legend(fontsize=5.3, frameon=False, ncol=1, loc="upper right")
        for xi, b in zip(x, bins):
            ax.text(xi, ax.get_ylim()[1] * 0.025, f"n={b['n']}", ha="center", fontsize=5.4, color="white", weight="bold")
        fig.tight_layout()
        fig.savefig(f"{OUT}/{fname}.pdf"); fig.savefig(f"{OUT}/{fname}.png", dpi=170); plt.close(fig)
        print(fname, [(b["label"], round(b["direct"], 3), round(b["sft"], 3), round(b["vt"], 3), b["n"]) for b in bins])

if __name__ == "__main__":
    eff_methods(); eff_routes(); eff_breakdown()
    for t in ("3b", "7b", "32b"): eff_backbone(t)
    print("efficiency figs done")
    data = json.load(open(CACHE)) if os.path.exists(CACHE) and "--recompute" not in sys.argv else compute_bins()
    complexity_figs(data)

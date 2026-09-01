"""Summarize jetson/results/bench_raw.json into a markdown report.

Expected-mix weights come from the server run's measured path mix
(RESULTS.md:821-822, n=2557 VSI odd half):
  ViewTree-D: direct 1815, consensus d1 379, d2 157, d3+beststate+fallback 206
  depth-1 tree (human views, RESULTS.md:644): direct 599, consensus 846,
  fused+fallback 1112  (consensus route = full minus the fuse call)
"""
import json, sys
import numpy as np

RAW = sys.argv[1] if len(sys.argv) > 1 else "jetson/results/bench_raw.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "jetson/results/REPORT.md"
recs = json.load(open(RAW))

def med(xs):
    return float(np.median(xs)) if xs else float("nan")

def rows(kind, **match):
    out = []
    for r in recs:
        if r.get("kind") != kind:
            continue
        if all(r.get(k) == v for k, v in match.items()):
            out.append(r)
    return out

RAILS = ("VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0")
def tot_e(r):  # 3-rail sum (lower bound on board energy)
    return sum(r.get(f"{x}_energy_j", 0) for x in RAILS)
def tot_p(r):
    return sum(r.get(f"{x}_avg_w", 0) for x in RAILS)
GE = "VDD_GPU_SOC_energy_j"

lines = []
meta = rows("meta")[0]
lines.append(f"# Jetson AGX Orin system measurements — {meta.get('model','Qwen2.5-VL-7B')} ({meta.get('quant','none')})\n")
lines.append(f"Device: {meta['device']}, power mode {meta['powermode']}, "
             f"idle draw {meta['idle_total_w']:.1f} W (sum of VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0; energy figures below are the same 3-rail sum, a lower bound on board power). "
             f"Run: {meta['ts']}.\n")

# phases
lines.append("## Scene-setup and one-time phases\n")
lines.append("| phase | latency s | energy J (3-rail) | avg W | peak CUDA GB | notes |")
lines.append("|---|---|---|---|---|---|")
for r in rows("phase"):
    note = ""
    if "n_points" in r: note = f"{r['n_points']/1e6:.1f}M points"
    if "n_entries" in r: note = f"{r['n_entries']} poses"
    if "vlm_load_s" in r: note = f"VLM {r['vlm_load_s']:.0f}s + VGGT {r['vggt_load_s']:.0f}s"
    if "mean_s" in r:
        lines.append(f"| {r['label']} | {r['mean_s']:.3f} (mean/view, n={r['n']}) | | | | p95 {r['p95_s']:.3f}s |")
        continue
    lines.append(f"| {r['label']} | {r.get('latency_s', 0):.1f} | {tot_e(r):.0f} | "
                 f"{tot_p(r):.1f} | {r.get('cuda_peak_gb', 0):.1f} | {note} |")

# micro
lines.append("\n## VLM call shapes (microbenchmark, median of reps)\n")
lines.append("| shape | images | prompt tok | decode tok | latency s | preproc s | energy J | avg W |")
lines.append("|---|---|---|---|---|---|---|---|")
micro = rows("micro")
labels = []
for r in micro:
    if r["label"] not in labels:
        labels.append(r["label"])
shape_cost = {}
for lb in labels:
    rs = [r for r in micro if r["label"] == lb]
    lat = med([r["total_s"] for r in rs])
    shape_cost[lb] = (lat, med([tot_e(r) for r in rs]))
    lines.append(f"| {lb} | {rs[0]['n_images']} | {rs[0]['n_prompt_tokens']} | "
                 f"{rs[0]['n_decode_tokens']} | {lat:.2f} | "
                 f"{med([r['preproc_s'] for r in rs]):.2f} | "
                 f"{med([tot_e(r) for r in rs]):.0f} | "
                 f"{med([tot_p(r) for r in rs]):.1f} |")

# questions
lines.append("\n## Per-question end-to-end (median across questions)\n")
lines.append("| method / route | VLM calls | latency s | energy J (3-rail) | GPU rail J | peak mem MB |")
lines.append("|---|---|---|---|---|---|")
qmethods = []
for r in rows("question"):
    if r["method"] not in qmethods:
        qmethods.append(r["method"])
mcost = {}
for m in qmethods:
    rs = rows("question", method=m)
    lat = med([r["latency_s"] for r in rs])
    en = med([tot_e(r) for r in rs])
    ge = med([r.get(GE, 0) for r in rs])
    calls = med([r["calls"] for r in rs])
    mem = max(r.get("mem_peak_mb", 0) for r in rs)
    mcost[m] = (lat, en, calls)
    lines.append(f"| {m} | {calls:.0f} | {lat:.1f} | {en:.0f} | {ge:.0f} | {mem:.0f} |")

# expected mixes
lines.append("\n## Expected per-question cost at the server-measured path mix\n")
if all(f"vtd_d{d}" in mcost for d in range(4)):
    w = np.array([1815, 379, 157, 206], float); w /= w.sum()
    lat = sum(wi * mcost[f"vtd_d{d}"][0] for d, wi in zip(range(4), w))
    en = sum(wi * mcost[f"vtd_d{d}"][1] for d, wi in zip(range(4), w))
    calls = sum(wi * mcost[f"vtd_d{d}"][2] for d, wi in zip(range(4), w))
    lines.append(f"- **ViewTree-D (ours)**: {lat:.1f} s, {en:.0f} J, {calls:.1f} calls "
                 f"(mix 71.0/14.8/6.1/8.1 % for depth 0/1/2/3, RESULTS.md:821)")
if "tree1_direct" in mcost and "tree1_full" in mcost:
    fuse = shape_cost.get("ans_10i_d12", (0, 0))
    w = np.array([599, 846, 1112], float); w /= w.sum()
    lat3 = [mcost["tree1_direct"][0],
            mcost["tree1_full"][0] - fuse[0],
            mcost["tree1_full"][0]]
    en3 = [mcost["tree1_direct"][1],
           mcost["tree1_full"][1] - fuse[1],
           mcost["tree1_full"][1]]
    lines.append(f"- **depth-1 tree**: {float(np.dot(w, lat3)):.1f} s, "
                 f"{float(np.dot(w, en3)):.0f} J "
                 f"(mix 23.4 % direct / 33.1 % consensus / 43.5 % fused, RESULTS.md:644)")
for m in ("frames16", "memory32"):
    if m in mcost:
        lines.append(f"- **{m}**: {mcost[m][0]:.1f} s, {mcost[m][1]:.0f} J (deterministic 1 call)")

# amortization
lines.append("\n## Scene-setup amortization\n")
recon = rows("phase", label="recon_32f")
pb = rows("phase", label="posebank_build")
r5 = rows("phase", label="renders_5")
if recon:
    rl, re_ = recon[0]["latency_s"], tot_e(recon[0])
    pl = pb[0]["latency_s"] if pb else 0
    pe = tot_e(pb[0]) if pb else 0
    vl = r5[0]["latency_s"] if r5 else 0
    ve = tot_e(r5[0]) if r5 else 0
    lines.append(f"VGGT(32f) = {rl:.1f} s / {re_:.0f} J; pose bank {pl:.1f} s; "
                 f"5 renders {vl:.1f} s. VSI mean ≈ 17.8 questions/scene:")
    lines.append(f"- ViewTree-D adds (recon+bank)/17.8 = **{(rl+pl)/17.8:.1f} s, "
                 f"{(re_+pe)/17.8:.0f} J per question** amortized "
                 f"(upfront {rl+pl:.0f} s cold).")
    lines.append(f"- memory32/depth-1 add (recon+5 renders)/17.8 = "
                 f"**{(rl+vl)/17.8:.1f} s per question** (upfront {rl+vl:.0f} s).")
    lines.append(f"- frames16 adds nothing (no reconstruction).")

open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

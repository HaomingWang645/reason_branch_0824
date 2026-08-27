"""Summarize every finished run under results/ into a Table-1-style markdown table.

Columns follow the paper: EgoM ObjM EgoAct Goal Pers Avg, plus #Token (K) (avg VLM tokens per
question, from the '[ChatGPT] ... usage_tokens=' log lines for API backbones) and Avg. WM
(avg number of world-model renders per question), plus WM-call rate and imagined views/question.

Usage:  python scripts/summarize.py [--md results/RESULTS_TABLE.md]
"""
import argparse, glob, json, os, re, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QTYPES = [("ego_movement", "EgoM"), ("obj_movement", "ObjM"), ("action_conseq", "EgoAct"),
          ("goal_aim", "Goal"), ("perspective", "Pers")]
USAGE_RE = re.compile(r"usage_tokens[=: ]+(\d+)")


def chunk_dirs(run_dir):
    cs = sorted(glob.glob(os.path.join(run_dir, "question_chunk_*")))
    return cs if cs else [run_dir]


def per_question_stats(qdir):
    """Return (decision, n_wm_renders, n_views) for one question dir, or None."""
    g = os.path.join(qdir, "step_0", "gpt.json")
    if not os.path.exists(g):
        return None
    try:
        d = json.load(open(g))
    except Exception:
        return None
    planning = d.get("planning") or {}
    decision = planning.get("decision", "skip")
    step0 = os.path.join(qdir, "step_0")
    plans = [p for p in os.listdir(step0) if p.startswith("plan_") and os.path.isdir(os.path.join(step0, p))]
    n_renders = len(plans)
    # imagined views actually shown to the QA model = frames sampled from the *selected* plan
    views = 0
    if decision != "skip" and planning.get("best_folder"):
        bf = os.path.join(step0, planning["best_folder"])
        if os.path.isdir(bf):
            views = len([f for f in os.listdir(bf) if f.endswith(".png")])
    return decision, n_renders, views


def summarize_run(run_dir):
    prog = defaultdict(lambda: {"correct": 0, "wrong": 0})
    n_q = 0; n_call = 0; n_renders = 0; n_views = 0; n_stats = 0
    for c in chunk_dirs(run_dir):
        r = os.path.join(c, "results.json")
        if not os.path.exists(r):
            continue
        d = json.load(open(r))
        for qt, p in d["progress"].items():
            prog[qt]["correct"] += len(p["correct"]); prog[qt]["wrong"] += len(p["wrong"])
        for q in os.listdir(c):
            qd = os.path.join(c, q)
            if not (q.isdigit() and os.path.isdir(qd)):
                continue
            st = per_question_stats(qd)
            if st is None:
                continue
            n_stats += 1
            dec, nr, nv = st
            n_call += dec != "skip"; n_renders += nr; n_views += nv
    n_q = sum(v["correct"] + v["wrong"] for v in prog.values())
    if n_q == 0:
        return None
    # tokens from logs
    run_name = os.path.basename(run_dir.rstrip("/")).split("_spatial_beam_search")[0].split("_qc")[0]
    tok = 0
    for lf in glob.glob(os.path.join(ROOT, "logs", run_name, "chunk_*.log")):
        for line in open(lf, errors="ignore"):
            m = USAGE_RE.search(line)
            if m:
                tok += int(m.group(1))
    row = {"run": run_name, "n": n_q,
           "acc": 100 * sum(v["correct"] for v in prog.values()) / n_q,
           "tokens_k": tok / 1000 / n_q if tok else None,
           "wm_rate": 100 * n_call / n_stats if n_stats else None,
           "avg_wm": n_renders / n_stats if n_stats else None,
           "views": n_views / n_stats if n_stats else None}
    for qt, short in QTYPES:
        v = prog.get(qt); row[short] = 100 * v["correct"] / (v["correct"] + v["wrong"]) if v and (v["correct"] + v["wrong"]) else None
    return row


def fmt(x, nd=1):
    return "–" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--md", default=None); a = ap.parse_args()
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(ROOT, "results", "*"))):
        if not os.path.isdir(run_dir) or run_dir.endswith("smoke"):
            continue
        r = summarize_run(run_dir)
        if r:
            rows.append(r)
    hdr = "| run | n | EgoM | ObjM | EgoAct | Goal | Pers | **Avg** | #Token (K) | WM-call % | Avg. WM | views/q |"
    lines = [hdr, "|" + "---|" * 12]
    for r in rows:
        lines.append(f"| {r['run']} | {r['n']} | " + " | ".join(fmt(r[s]) for _, s in QTYPES) +
                     f" | **{fmt(r['acc'])}** | {fmt(r['tokens_k'])} | {fmt(r['wm_rate'])} | {fmt(r['avg_wm'], 2)} | {fmt(r['views'], 2)} |")
    out = "\n".join(lines)
    print(out)
    if a.md:
        open(a.md, "w").write(out + "\n")


if __name__ == "__main__":
    main()

"""Render the full reasoning path of a few SAT questions (one or more per task type) as a self-contained HTML page.

For every selected question the page shows, in pipeline order:
  1. the observation(s) and the question / choices
  2. the policy samples (decision, reason, plan) and the majority vote
  3. every world-model candidate that was rendered, its verifier score, and the imagined frames
  4. the imagined trajectory handed to the QA model, the QA answer, and the verdict

Usage:
  python scripts/viz_traces.py --runs gpt-4o_avic_r_spatial_beam_search_qc6 gpt-41_avic_r_spatial_beam_search_qc3 \
      --per-type 2 --out results/traces.html
"""
import argparse, base64, glob, html, io, json, os, re
from collections import defaultdict
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPE_NAMES = {"ego_movement": "Egocentric movement", "obj_movement": "Object movement",
              "action_conseq": "Action consequence", "goal_aim": "Goal aiming", "perspective": "Perspective taking"}
TYPE_ORDER = ["ego_movement", "obj_movement", "action_conseq", "goal_aim", "perspective"]
CAND_RE = re.compile(r"\[CAND (\d+)\] score=(\S+) status=(\S+) plan=(\[.*\])")
SCENE_RE = re.compile(r"scenes \['.*?/(\d+)/step_0/img_0\.png'\]")


def img_uri(path, max_w=420):
    im = Image.open(path).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def cand_scores(run_name):
    """qid -> {plan_tuple: (score, status)} parsed from the chunk logs."""
    out = defaultdict(dict)
    for lf in glob.glob(os.path.join(ROOT, "logs", run_name, "chunk_*.log")):
        qid = None
        for line in open(lf, errors="ignore"):
            m = SCENE_RE.search(line)
            if m:
                qid = int(m.group(1)); continue
            m = CAND_RE.search(line)
            if m and qid is not None:
                plan = tuple(json.loads(m.group(4).replace("'", '"')))
                out[qid][plan] = (m.group(2), m.group(3))
    return out


def plan_key(folder):
    parts = folder[len("plan_"):].split("_")
    return tuple(f"{parts[i]} {parts[i+1]}" for i in range(0, len(parts) - 1, 2))


def collect(run_dir, run_name):
    scores = cand_scores(run_name)
    items = []
    for g in glob.glob(os.path.join(run_dir, "**", "step_0", "gpt.json"), recursive=True):
        qdir = os.path.dirname(os.path.dirname(g)); qid = int(os.path.basename(qdir))
        d = json.load(open(g)); step0 = os.path.join(qdir, "step_0")
        plans = sorted(p for p in os.listdir(step0) if p.startswith("plan_") and os.path.isdir(os.path.join(step0, p)))
        cands = []
        for p in plans:
            frames = sorted((f for f in os.listdir(os.path.join(step0, p)) if f.endswith(".png")),
                            key=lambda f: [float(x) for x in re.findall(r"\d+(?:\.\d+)?", f[:-4])])
            sc = scores.get(qid, {}).get(plan_key(p), (None, None))
            cands.append({"folder": p, "plan": list(plan_key(p)), "score": sc[0], "status": sc[1],
                          "frames": [os.path.join(step0, p, f) for f in frames]})
        items.append({"qid": qid, "q": d["question"], "result": d["result"], "answer": d["llm_response"],
                      "planning": d.get("planning") or {"decision": "skip"}, "policies": d.get("policies") or [],
                      "imgs": [os.path.join(step0, f) for f in ("img_0.png", "helper_img.png") if os.path.exists(os.path.join(step0, f))],
                      "cands": cands})
    return items


def pick(items, per_type):
    by = defaultdict(list)
    for it in items:
        by[it["q"]["question_type"]].append(it)
    chosen = []
    for t in TYPE_ORDER:
        pool = by.get(t, [])
        # prefer world-model traces; alternate correct / wrong so both outcomes are visible
        wm_ok = [i for i in pool if i["cands"] and i["result"] == "correct"]
        wm_bad = [i for i in pool if i["cands"] and i["result"] != "correct"]
        skip = [i for i in pool if not i["cands"]]
        order = []
        for k in range(per_type):
            src = [wm_ok, wm_bad, skip][k % 3] if k % 3 != 2 or skip else (wm_ok if k % 2 == 0 else wm_bad)
            for cand in (src, wm_ok, wm_bad, skip):
                if cand:
                    order.append(cand.pop(0)); break
        chosen += sorted(order, key=lambda i: i["qid"])
    return chosen


def esc(s):
    return html.escape(str(s))


def render_item(it, run_label):
    q = it["q"]; pl = it["planning"]; votes = pl.get("vote_stats", {})
    ok = it["result"] == "correct"
    h = [f'<article class="sample" id="{esc(run_label)}-{it["qid"]}">']
    h.append(f'<header class="sample-head"><span class="type">{esc(TYPE_NAMES.get(q["question_type"], q["question_type"]))}</span>'
             f'<span class="qid">SAT-test #{it["qid"]}</span><span class="run">{esc(run_label)}</span>'
             f'<span class="verdict {"ok" if ok else "bad"}">{"correct" if ok else "wrong"}</span></header>')
    # 1 observation
    h.append('<section class="stage"><div class="rail"><b>1</b><span>Observation</span></div><div class="body">')
    h.append('<div class="strip">' + "".join(
        f'<figure><img src="{img_uri(p)}" alt="input view {i+1}"><figcaption>{"Image 1 · current view" if i == 0 else "Image 2"}</figcaption></figure>'
        for i, p in enumerate(it["imgs"])) + '</div>')
    h.append(f'<p class="question">{esc(q["question"])}</p><ul class="choices">' + "".join(
        f'<li class="{"gt" if c == q["correct_answer"] else ""}">{esc(c)}</li>' for c in q["answer_choices"]) + '</ul></div></section>')
    # 2 policy
    h.append('<section class="stage"><div class="rail"><b>2</b><span>Policy samples</span></div><div class="body">')
    if it["policies"]:
        h.append('<ol class="policies">')
        for pol, raw in it["policies"]:
            dec = pol.get("decision", "?"); acts = pol.get("actions") or []
            plan = ", ".join(f'{a["type"]} {a["value"]:g}' for a in acts) if acts else "—"
            h.append(f'<li><span class="dec {esc(dec)}">{esc(dec)}</span><code>{esc(plan)}</code><em>{esc(pol.get("reason", ""))}</em></li>')
        h.append('</ol>')
    else:
        h.append('<p class="muted">no parsable policy sample</p>')
    h.append(f'<p class="vote">Majority vote: <b>{esc(pl.get("decision"))}</b> '
             f'(call_wm {votes.get("call_wm", 0)} · skip {votes.get("skip", 0)}); '
             f'{pl.get("num_call_candidates", 0)} unique plan(s) sent to the world model</p></div></section>')
    # 3 world model candidates
    h.append('<section class="stage"><div class="rail"><b>3</b><span>Imagination &amp; verifier</span></div><div class="body">')
    if it["cands"]:
        best = pl.get("best_folder")
        for c in it["cands"]:
            sel = c["folder"] == best
            score = "—" if c["score"] is None else c["score"]
            chosen = '<span class="chosen">selected</span>' if sel else ""
            h.append(f'<div class="cand {"sel" if sel else ""}"><div class="cand-head"><code>{esc(" → ".join(c["plan"]))}</code>'
                     f'<span class="score">verifier {esc(score)}/10</span>{chosen}</div>'
                     '<div class="strip small">' + "".join(
                         f'<figure><img src="{img_uri(f, 300)}" alt="imagined frame"><figcaption>{esc(os.path.basename(f)[:-4].replace("_", " "))}</figcaption></figure>'
                         for f in c["frames"]) + '</div></div>')
        if len(it["cands"]) == 1:
            h.append('<p class="muted">single candidate — used without verifier scoring</p>')
    else:
        h.append('<p class="muted">world model not called — the QA model answers from the observation alone</p>')
    h.append('</div></section>')
    # 4 answer
    h.append('<section class="stage"><div class="rail"><b>4</b><span>Answer</span></div><div class="body">')
    h.append(f'<p class="answer"><span class="lbl">QA model</span> {esc(it["answer"].strip())}</p>'
             f'<p class="answer"><span class="lbl">ground truth</span> {esc(q["correct_answer"])}</p></div></section>')
    h.append('</article>')
    return "".join(h)


CSS = """
:root{--paper:#F5F4EF;--ink:#1C1F24;--ink2:#4A4F57;--muted:#7C8189;--line:#DAD8D0;--card:#FFFFFF;--accent:#1F7A74;--accent-soft:#E2F0EE;--ok:#2E7D4F;--ok-soft:#E3F1E8;--bad:#B3402F;--bad-soft:#F6E4E0;--sel:#F4EFD9}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--paper:#14171B;--ink:#E8E6E0;--ink2:#B9BCC2;--muted:#8A8F98;--line:#2C3138;--card:#1C2026;--accent:#5FBDB5;--accent-soft:#1D3331;--ok:#6FCB93;--ok-soft:#1B3325;--bad:#EF8A76;--bad-soft:#3A2220;--sel:#2E2A1C}}
:root[data-theme="dark"]{--paper:#14171B;--ink:#E8E6E0;--ink2:#B9BCC2;--muted:#8A8F98;--line:#2C3138;--card:#1C2026;--accent:#5FBDB5;--accent-soft:#1D3331;--ok:#6FCB93;--ok-soft:#1B3325;--bad:#EF8A76;--bad-soft:#3A2220;--sel:#2E2A1C}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:15px;line-height:1.5}
.wrap{max-width:1080px;margin:0 auto;padding:40px 24px 80px}
h1{font-family:Sora,"IBM Plex Sans",sans-serif;font-weight:600;font-size:30px;letter-spacing:-.01em;margin:0 0 8px;text-wrap:balance}
.lede{color:var(--ink2);max-width:68ch;margin:0 0 24px}
.legend{display:flex;flex-wrap:wrap;gap:8px 20px;font-size:13px;color:var(--muted);margin-bottom:32px}
.legend b{color:var(--ink)}
nav.toc{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:6px 16px;font-size:13px;margin-bottom:40px;padding:16px;border:1px solid var(--line);border-radius:6px;background:var(--card)}
nav.toc a{color:var(--ink2);text-decoration:none}nav.toc a:hover,nav.toc a:focus-visible{color:var(--accent);outline:none;text-decoration:underline}
nav.toc .h{grid-column:1/-1;font-family:Sora,sans-serif;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-top:6px}
.sample{background:var(--card);border:1px solid var(--line);border-radius:8px;margin-bottom:36px;overflow:hidden}
.sample-head{display:flex;flex-wrap:wrap;align-items:center;gap:12px;padding:14px 20px;border-bottom:1px solid var(--line);font-size:13px}
.sample-head .type{font-family:Sora,sans-serif;font-weight:600;font-size:15px;color:var(--ink)}
.sample-head .qid,.sample-head .run{color:var(--muted);font-family:"IBM Plex Mono",monospace;font-size:12px}
.verdict{margin-left:auto;padding:3px 10px;border-radius:999px;font-family:Sora,sans-serif;font-weight:600;font-size:12px;letter-spacing:.04em;text-transform:uppercase}
.verdict.ok{background:var(--ok-soft);color:var(--ok)}.verdict.bad{background:var(--bad-soft);color:var(--bad)}
.stage{display:grid;grid-template-columns:120px 1fr;border-bottom:1px solid var(--line)}.stage:last-child{border-bottom:0}
.rail{padding:18px 16px;border-right:1px solid var(--line);display:flex;flex-direction:column;gap:4px}
.rail b{font-family:Sora,sans-serif;font-size:22px;font-weight:600;color:var(--accent);line-height:1}
.rail span{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.body{padding:18px 20px;min-width:0}
.strip{display:flex;gap:12px;overflow-x:auto;padding-bottom:6px}.strip figure{margin:0;flex:0 0 auto}
.strip img{display:block;max-width:100%;width:300px;border-radius:4px;border:1px solid var(--line)}
.strip.small img{width:220px}
figcaption{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--muted);margin-top:4px}
.question{font-size:16px;margin:14px 0 8px;max-width:70ch}
.choices{list-style:none;padding:0;margin:0;display:flex;flex-wrap:wrap;gap:8px}
.choices li{padding:4px 10px;border:1px solid var(--line);border-radius:4px;font-size:14px}
.choices li.gt{border-color:var(--ok);color:var(--ok)}
.policies{list-style:none;padding:0;margin:0 0 12px;display:flex;flex-direction:column;gap:6px}
.policies li{display:grid;grid-template-columns:80px minmax(120px,220px) 1fr;gap:12px;align-items:baseline;font-size:13px}
.policies em{color:var(--ink2);font-style:normal}
.policies code,.cand code{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--ink)}
.dec{font-family:Sora,sans-serif;font-size:11px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;padding:2px 8px;border-radius:999px;text-align:center}
.dec.call_wm{background:var(--accent-soft);color:var(--accent)}.dec.skip{background:var(--line);color:var(--ink2)}
.vote{margin:0;font-size:14px;color:var(--ink2)}
.cand{border:1px solid var(--line);border-radius:6px;padding:10px 12px;margin-bottom:10px}
.cand.sel{background:var(--sel);border-color:var(--accent)}
.cand-head{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:8px;font-size:13px}
.score{font-family:"IBM Plex Mono",monospace;color:var(--ink2);font-variant-numeric:tabular-nums}
.chosen{font-family:Sora,sans-serif;font-size:11px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;color:var(--accent)}
.answer{margin:0 0 6px;font-size:15px}.answer .lbl{display:inline-block;width:110px;font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.muted{color:var(--muted);font-size:13px;margin:0}
@media (max-width:640px){.stage{grid-template-columns:1fr}.rail{flex-direction:row;align-items:baseline;border-right:0;border-bottom:1px solid var(--line);padding:10px 16px}.policies li{grid-template-columns:1fr}}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="result dir names under results/")
    ap.add_argument("--per-type", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "traces.html"))
    a = ap.parse_args()
    parts = []; toc = []
    for run in a.runs:
        run_name = run.split("_spatial_beam_search")[0]
        items = pick(collect(os.path.join(ROOT, "results", run), run_name), a.per_type)
        toc.append(f'<span class="h">{esc(run_name)}</span>' + "".join(
            f'<a href="#{esc(run_name)}-{i["qid"]}">{esc(TYPE_NAMES[i["q"]["question_type"]])} · #{i["qid"]} · {"✓" if i["result"] == "correct" else "✗"}</a>' for i in items))
        parts.append(f'<h2 class="runh">{esc(run_name)}</h2>' + "".join(render_item(i, run_name) for i in items))
    page = f"""<title>AVIC Reasoning Traces</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Sora:wght@600&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono&display=swap">
<style>{CSS}.runh{{font-family:Sora,sans-serif;font-size:18px;font-weight:600;margin:32px 0 14px;color:var(--ink2)}}</style>
<div class="wrap">
<h1>AVIC reasoning traces on SAT-Real</h1>
<p class="lede">Every stage of the adaptive-imagination pipeline for a few test questions of each task type: the observation, the five sampled policy decisions and their majority vote, every trajectory the world model rendered with its verifier score, and the final answer of the QA model against the ground truth. Selected trajectories are highlighted; imagined frames are shown in the order they were generated.</p>
<div class="legend"><span><b>call_wm</b> policy wants imagined views</span><span><b>skip</b> answer from the observation</span><span><b>verifier n/10</b> trajectory usefulness score from the backbone</span><span>green choice = ground truth</span></div>
<nav class="toc">{"".join(toc)}</nav>
{"".join(parts)}
</div>"""
    open(a.out, "w").write(page)
    print(a.out, f"{os.path.getsize(a.out)/1e6:.1f} MB")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate Think3D-reproduction results into paper-style tables.

Scans  outputs/run{N}/vlmeval_runs/<model_tag>/<dataset>/  and computes
  MindCube : accuracy per category (rotation / among / around)   [Table 2]
  BLINK    : Multi-view_Reasoning accuracy                        [Table 2]
  VSIBench : accuracy per MC task (route_planning, object_rel_direction,
             object_rel_distance, obj_appearance_order)           [Table 1]
Scoring rule (identical for all benchmarks): take the option letter inside
<answer>...</answer> (fallback: first stand-alone A-E letter) and compare with GT.
Results are averaged over runs (paper: mean of 3 runs).
"""
import json, re, glob, os, sys, collections
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')

def letter(text):
    """Mirror spagent/scripts/quick_eval.py::_normalize_answer_local (the repo's scorer):
    option letter inside <answer>...</answer>; otherwise first "(X)" / "X." match;
    otherwise first A-E character.  Truncated / unparseable outputs count as wrong."""
    if not isinstance(text, str):
        return ''
    s = text.strip()
    a, b = s.find('<answer>'), s.find('</answer>')
    if a != -1 and b > a:
        s = s[a + 8:b].strip()
    m = re.search(r'\(([A-E])\)|([A-E])\.', s)
    if m:
        return m.group(1) or m.group(2)
    for ch in s:
        if ch in 'ABCDE':
            return ch
    return ''

def score_blink(xlsx):
    df = pd.read_excel(xlsx)
    gt = df['answer'].map(letter); pr = df['prediction'].map(letter)
    fails = df['prediction'].astype(str).str.startswith('INFER_FAIL').sum() + (df['prediction'].astype(str).str.contains('推理失败')).sum()
    return {'Multi-view': ((gt == pr).mean(), len(df), int(fails))}

def score_local(results_json):
    d = json.load(open(results_json))
    per = collections.defaultdict(lambda: [0, 0, 0])
    for r in d['detailed_results']:
        task = r.get('task', '')
        pred = str(r.get('prediction', ''))
        ok = letter(pred) == letter(str(r.get('answer', '')))
        per[task][0] += ok; per[task][1] += 1
        per[task][2] += pred.startswith('INFER_FAIL') or ('推理失败' in pred) or (not r.get('success', True))
    return {t: (v[0] / v[1], v[1], v[2]) for t, v in per.items()}

def main():
    rows = collections.defaultdict(dict)   # (model_tag, dataset) -> run -> {task: (acc,n,fail)}
    for run_dir in sorted(glob.glob(os.path.join(ROOT, 'run[0-9]*'))):
        run = os.path.basename(run_dir)
        for tag_dir in sorted(glob.glob(os.path.join(run_dir, 'vlmeval_runs', '*'))):
            tag = os.path.basename(tag_dir)
            for ds_dir in glob.glob(os.path.join(tag_dir, '*')):
                ds = os.path.basename(ds_dir)
                if ds == 'BLINK':
                    xs = glob.glob(os.path.join(ds_dir, '*_BLINK.xlsx'))
                    if xs: rows[(tag, ds)][run] = score_blink(xs[0])
                else:
                    js = glob.glob(os.path.join(ds_dir, '*_results.json'))
                    if js: rows[(tag, ds)][run] = score_local(js[0])
    out = []
    for (tag, ds), runs in sorted(rows.items()):
        tasks = sorted({t for r in runs.values() for t in r})
        line = {'model_tag': tag, 'dataset': ds, 'runs': len(runs)}
        accs = []
        for t in tasks:
            vals = [runs[r][t][0] for r in runs if t in runs[r]]
            n = runs[next(iter(runs))][t][1] if t in runs[next(iter(runs))] else 0
            fails = sum(runs[r][t][2] for r in runs if t in runs[r])
            line[t] = f"{100*sum(vals)/len(vals):.2f} (n={n}{', fail='+str(fails) if fails else ''})"
            accs.append(sum(vals)/len(vals))
        line['avg_over_tasks'] = f"{100*sum(accs)/len(accs):.2f}" if accs else ''
        out.append(line)
    df = pd.DataFrame(out)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(ROOT, 'summary.csv'), index=False)
    print('\nsaved', os.path.join(ROOT, 'summary.csv'))

if __name__ == '__main__' and '--tables' not in sys.argv:
    main()


# ── Paper-layout markdown tables ─────────────────────────────────────────────
PAPER = {
    # model_tag -> (row label, paper Table 2 numbers, paper Table 1 numbers)
    'Qwen3_VL_4B_Instruct_no_tools_general': ('Qwen3-VL-4B (no tool)',
        [47.87, 34.17, 20.00, 41.67, 35.92], [34.69, 40.67, 35.33, 42.44, 38.28]),
    'Qwen3_VL_4B_Instruct_pi3x_spatial': ('Think3D (Qwen3-VL-4B)',
        [48.62, 35.83, 28.33, 33.33, 36.53], [30.61, 44.00, 29.33, 52.38, 39.08]),
    'SPAgent_4B_no_tools_general': ('Qwen3-VL-4B-T3RL, released SPAgent-4B (no tool)',
        [46.11, 30.83, 25.83, 35.83, 34.65], [27.89, 30.67, 32.00, 42.86, 33.36]),
    'SPAgent_4B_pi3x_spatial': ('Think3D (released SPAgent-4B)',
        [53.39, 42.50, 37.47, 42.50, 43.97], [36.73, 39.00, 44.67, 61.22, 45.41]),
    'SPAgent_4B_px256k_pi3x_spatial': ('Think3D (released SPAgent-4B), eval images capped at 262144 px (= RL training MAX_PIXELS)',
        [53.39, 42.50, 37.47, 42.50, 43.97], [36.73, 39.00, 44.67, 61.22, 45.41]),
    'Qwen3_VL_4B_Instruct_px256k_pi3x_spatial': ('Think3D (Qwen3-VL-4B), eval images capped at 262144 px',
        [48.62, 35.83, 28.33, 33.33, 36.53], [30.61, 44.00, 29.33, 52.38, 39.08]),
    'Think3D_RL_4B_no_tools_general': ('Qwen3-VL-4B-T3RL, our GRPO run (no tool)',
        [46.11, 30.83, 25.83, 35.83, 34.65], [27.89, 30.67, 32.00, 42.86, 33.36]),
    'Think3D_RL_4B_pi3x_spatial': ('Think3D (our GRPO run)',
        [53.39, 42.50, 37.47, 42.50, 43.97], [36.73, 39.00, 44.67, 61.22, 45.41]),
}
T2_COLS = [('BLINK', 'Multi-view'), ('MindCube', 'rotation'), ('MindCube', 'among'), ('MindCube', 'around')]
T1_COLS = [('VSIBench', 'route_planning'), ('VSIBench', 'object_rel_direction'),
           ('VSIBench', 'object_rel_distance'), ('VSIBench', 'obj_appearance_order')]

def collect_runs():
    """(tag, dataset) -> run -> {task: (acc, n, fails)}"""
    rows = collections.defaultdict(dict)
    for run_dir in sorted(glob.glob(os.path.join(ROOT, 'run[0-9]*'))):
        run = os.path.basename(run_dir)
        for tag_dir in sorted(glob.glob(os.path.join(run_dir, 'vlmeval_runs', '*'))):
            tag = os.path.basename(tag_dir)
            for ds_dir in glob.glob(os.path.join(tag_dir, '*')):
                ds = os.path.basename(ds_dir)
                if ds == 'BLINK':
                    xs = glob.glob(os.path.join(ds_dir, '*_BLINK.xlsx'))
                    if xs: rows[(tag, ds)][run] = score_blink(xs[0])
                else:
                    js = glob.glob(os.path.join(ds_dir, '*_results.json'))
                    if js: rows[(tag, ds)][run] = score_local(js[0])
    return rows

def md_tables():
    import statistics
    rows = collect_runs()
    tags = [t for t in PAPER if any(k[0] == t for k in rows)]
    out = []
    for title, cols in [('Table 2 — BLINK Multi-view + MindCube (accuracy %, mean ± std over runs; paper value in brackets)', T2_COLS),
                        ('Table 1 — VSI-Bench-tiny, 4 MC tasks (accuracy %, mean ± std over runs; paper value in brackets)', T1_COLS)]:
        hdr = ['Model'] + [c[1].replace('_', ' ') for c in cols] + ['Avg', 'runs']
        out.append(f'\n### {title}\n'); out.append('| ' + ' | '.join(hdr) + ' |'); out.append('|' + '---|' * len(hdr))
        for tag in tags:
            label, p2, p1 = PAPER[tag]; paper = p2 if cols is T2_COLS else p1
            cells = []; means = []; nruns = 0
            for i, (ds, task) in enumerate(cols):
                runs = rows.get((tag, ds), {})
                vals = [100 * runs[r][task][0] for r in runs if task in runs[r]]
                if vals:
                    nruns = max(nruns, len(vals)); m = statistics.mean(vals); means.append(m)
                    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                    cells.append(f'{m:.2f} ± {sd:.1f} [{paper[i]:.2f}]')
                else:
                    cells.append(f'— [{paper[i]:.2f}]')
            avg = f'{statistics.mean(means):.2f} [{paper[4]:.2f}]' if len(means) == len(cols) else f'— [{paper[4]:.2f}]'
            out.append('| ' + ' | '.join([label] + cells + [avg, str(nruns)]) + ' |')
    text = '\n'.join(out)
    with open(os.path.join(ROOT, 'results_tables.md'), 'w') as f:
        f.write(text + '\n')
    print(text)

if __name__ == '__main__' and '--tables' in sys.argv:
    md_tables()

#!/usr/bin/env python3
"""Build an HTML page that visualizes Think3D reasoning traces (inputs -> reasoning -> tool call ->
rendered 3D view -> answer) for a hand-picked set of samples from outputs/runviz."""
import json, base64, io, os, re, html, sys
from PIL import Image
import pandas as pd
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
T = os.path.join(R, 'outputs/runviz/spagent_traces/SPAgent_4B_pi3x_spatial')
SP = os.path.join(R, 'spagent')

def data_uri(path, maxw):
    p = path if os.path.isabs(path) else os.path.join(SP, path)
    im = Image.open(p).convert('RGB')
    w, h = im.size
    if w > maxw: im = im.resize((maxw, int(h * maxw / w)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, 'JPEG', quality=72, optimize=True)
    return 'data:image/jpeg;base64,' + base64.b64encode(b.getvalue()).decode(), (w, h)

def letter(s):
    s = str(s).strip(); a, b = s.find('<answer>'), s.find('</answer>')
    if a != -1 and b > a: s = s[a+8:b].strip()
    m = re.search(r'\(([A-E])\)|([A-E])\.', s)
    if m: return m.group(1) or m.group(2)
    for ch in s:
        if ch in 'ABCDE': return ch
    return s[:1]

vsi_task = {json.loads(l)['id']: json.loads(l)['task'] for l in open(os.path.join(SP, 'dataset/VSI_Bench_tiny.jsonl'))}
blink = pd.read_excel(os.path.join(R, 'outputs/runviz/vlmeval_runs/SPAgent_4B_pi3x_spatial/BLINK/SPAgent_4B_pi3x_spatial_BLINK.xlsx'))
blink_gt = dict(zip(blink['index'].astype(str), blink['answer']))

SAMPLES = [  # (dataset, idx, label)
    ('MindCube', 0, 'MindCube · rotation'), ('MindCube', 2, 'MindCube · rotation'),
    ('MindCube', 8, 'MindCube · among'), ('MindCube', 6, 'MindCube · among'),
    ('MindCube', 3, 'MindCube · around'), ('MindCube', 5, 'MindCube · around'),
    ('BLINK', 0, 'BLINK · multi-view reasoning'), ('BLINK', 1, 'BLINK · multi-view reasoning'),
    ('BLINK', 2, 'BLINK · multi-view reasoning'), ('BLINK', 3, 'BLINK · multi-view reasoning'),
    ('VSIBench', 0, None), ('VSIBench', 7, None), ('VSIBench', 4, None), ('VSIBench', 10, None), ('VSIBench', 3, None),
]
TASKN = {'route_planning': 'route planning', 'object_rel_direction': 'relative direction',
         'object_rel_distance': 'relative distance', 'obj_appearance_order': 'appearance order'}

def fmt_text(t):
    t = html.escape(t.strip())
    t = re.sub(r'&lt;think&gt;|&lt;/think&gt;', '', t)
    t = re.sub(r'&lt;answer&gt;(.*?)&lt;/answer&gt;', r'<span class="ans">answer: \1</span>', t, flags=re.S)
    t = re.sub(r'&lt;tool_call&gt;.*?&lt;/tool_call&gt;', '', t, flags=re.S)
    return t.strip().replace('\n', '<br>')

cards = []
for ds, idx, label in SAMPLES:
    conv = json.load(open(os.path.join(T, ds, f'{idx:05d}_conv.json')))['conversation']
    tr = json.load(open(os.path.join(T, ds, f'{idx:05d}.json')))
    if ds == 'BLINK':
        gt = str(blink_gt.get(str(tr['index']), '')); label = label
    elif ds == 'VSIBench':
        gt = str(tr['ground_truth']); label = 'VSI-Bench · ' + TASKN.get(vsi_task.get(tr['id'], ''), '?')
    else:
        gt = str(tr['ground_truth'])
    pred = letter(tr['answer']); ok = pred == letter(gt)
    q = html.escape(tr['question'])
    inputs = tr['image_paths']
    if ds == 'VSIBench':   # quick_eval extracts frames to a temp dir that is deleted afterwards: re-extract identically
        import cv2
        fr_dir = os.path.join(R, 'outputs/runviz/frames'); os.makedirs(fr_dir, exist_ok=True)
        stem = os.path.basename(inputs[0]).rsplit('_frame_', 1)[0]
        cap = cv2.VideoCapture(os.path.join(SP, 'dataset/VSI_videos', stem + '.mp4'))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); interval = max(1, total // 7); new_inputs = []
        for i in range(7):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval); ret, frame = cap.read()
            if ret:
                fp = os.path.join(fr_dir, f'{stem}_frame_{i}.jpg'); cv2.imwrite(fp, frame); new_inputs.append(fp)
        cap.release(); inputs = new_inputs
    steps = []
    steps.append(('input', f'{len(inputs)} input image{"s" if len(inputs)>1 else ""}', inputs, q))
    turn = 0
    for m in conv:
        if m['role'] == 'assistant':
            turn += 1; steps.append(('model', f'model turn {turn}', None, fmt_text(m.get('text', ''))))
        elif m['role'] == 'tool' and m['type'] == 'tool_call':
            a = m['metadata']['arguments']
            view = 'ego view from cam %s' % a.get('rotation_reference_camera', 1) if a.get('camera_view') else 'global view around cam %s' % a.get('rotation_reference_camera', 1)
            steps.append(('call', 'Pi3X call', None, f'azimuth {a.get("azimuth_angle", 0)}°, elevation {a.get("elevation_angle", 0)}° · {view} · {len(a.get("image_path", []))} image(s) reconstructed'))
        elif m['role'] == 'tool' and m['type'] == 'tool_result':
            imgs = m.get('images') or []
            pts = re.search(r'points_count: (\d+)', m.get('text', ''))
            steps.append(('render', 'rendered 3D view', imgs, f'{int(pts.group(1)):,} points' if pts else ''))
    steps.append(('verdict', 'verdict', None, (pred, gt, ok)))
    cards.append((label, ds, idx, steps, ok, tr.get('elapsed_s', 0)))

def img_tag(p, maxw, cls=''):
    try:
        u, (w, h) = data_uri(p, maxw)
        return f'<img class="{cls}" src="{u}" alt="" loading="lazy">'
    except Exception as e:
        return f'<div class="missing">image unavailable</div>'

out = []
for label, ds, idx, steps, ok, el in cards:
    body = []
    for kind, name, imgs, content in steps:
        if kind == 'input':
            strip = ''.join(f'<figure><figcaption>cam {i+1}</figcaption>{img_tag(p, 360, "thumb")}</figure>' for i, p in enumerate(imgs))
            body.append(f'<section class="step step-input"><div class="rail"><span class="dot"></span><span class="lbl">question</span></div><div class="body"><p class="q">{content}</p><div class="strip">{strip}</div></div></section>')
        elif kind == 'model':
            body.append(f'<section class="step"><div class="rail"><span class="dot"></span><span class="lbl">{name}</span></div><div class="body"><div class="think">{content}</div></div></section>')
        elif kind == 'call':
            body.append(f'<section class="step step-call"><div class="rail"><span class="dot dot-call"></span><span class="lbl">tool call</span></div><div class="body"><code class="call">pi3x_tool → {html.escape(content)}</code></div></section>')
        elif kind == 'render':
            r = ''.join(img_tag(p, 720, "render") for p in imgs) if imgs else '<div class="missing">no image returned</div>'
            body.append(f'<section class="step step-render"><div class="rail"><span class="dot dot-render"></span><span class="lbl">{name}</span></div><div class="body">{r}<p class="cap">{html.escape(content)} · camera frustums mark the input viewpoints</p></div></section>')
        elif kind == 'verdict':
            pred, gt, ok = content
            body.append(f'<section class="step step-verdict"><div class="rail"><span class="dot {"dot-ok" if ok else "dot-bad"}"></span><span class="lbl">verdict</span></div><div class="body"><span class="pill {"ok" if ok else "bad"}">{"correct" if ok else "wrong"}</span> <span class="pv">model {html.escape(pred or "—")} · ground truth {html.escape(gt)}</span></div></section>')
    out.append(f'<article class="card {"is-ok" if ok else "is-bad"}"><header><h2>{html.escape(label)}</h2><span class="meta">{ds} #{idx:05d} · {el:.0f}s</span></header>{"".join(body)}</article>')

page = open(os.path.join(R, 'scripts/trace_page_template.html')).read().replace('%%CARDS%%', '\n'.join(out)).replace('%%N%%', str(len(cards)))
open(os.path.join(R, 'outputs/think3d_traces.html'), 'w').write(page)
print('wrote', os.path.join(R, 'outputs/think3d_traces.html'), len(page)//1024, 'KB')

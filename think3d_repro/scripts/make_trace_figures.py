#!/usr/bin/env python3
"""Save reasoning-trace figures locally: one composite PNG per sample + raw assets.
Output: figures/traces/NN_<dataset>_<task>/  (inputs/, render_k.png, trace.txt)  and  figures/traces/NN_<dataset>_<task>.png"""
import json, os, re, html, shutil, textwrap, sys
from PIL import Image, ImageDraw, ImageFont
sys.argv = [sys.argv[0]]
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(R, 'scripts'))
# reuse extraction from the page generator (it builds `cards`)
src = open(os.path.join(R, 'scripts/make_trace_page.py')).read().split("def img_tag")[0]
ns = {'__file__': os.path.join(R, 'scripts/make_trace_page.py')}; exec(compile(src, 'make_trace_page', 'exec'), ns)
cards, SP = ns['cards'], ns['SP']

OUT = os.path.join(R, 'figures/traces'); os.makedirs(OUT, exist_ok=True)
def font(sz, bold=False):
    for f in (['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'] if bold else ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']):
        if os.path.exists(f): return ImageFont.truetype(f, sz)
    return ImageFont.load_default()
F, FB, FM = font(20), font(24, True), font(18)
W = 1600; PAD = 36; INK = (22, 32, 42); MUTED = (91, 107, 122); OK = (47, 122, 69); BAD = (179, 58, 58); CALL = (180, 86, 15); ACC = (14, 124, 123)

def strip_html(t):
    t = re.sub(r'<span class="ans">(.*?)</span>', r'[\1]', t)
    return html.unescape(re.sub(r'<br>', '\n', t))

def wrap(t, width=120, max_lines=None):
    lines = []
    for para in t.split('\n'):
        lines += textwrap.wrap(para, width) or ['']
    if max_lines and len(lines) > max_lines: lines = lines[:max_lines] + ['[...]']
    return lines

def load(p, maxw):
    im = Image.open(p if os.path.isabs(p) else os.path.join(SP, p)).convert('RGB')
    if im.width > maxw: im = im.resize((maxw, int(im.height * maxw / im.width)), Image.LANCZOS)
    return im

for k, (label, ds, idx, steps, ok, el) in enumerate(cards, 1):
    slug = f"{k:02d}_{re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')}"
    d = os.path.join(OUT, slug); os.makedirs(os.path.join(d, 'inputs'), exist_ok=True)
    blocks = []   # (kind, payload)
    txt = [f'{label}  ({ds} #{idx:05d})', '']
    for kind, name, imgs, content in steps:
        if kind == 'input':
            ims = []
            for i, p in enumerate(imgs):
                dst = os.path.join(d, 'inputs', f'cam{i+1}{os.path.splitext(p)[1]}'); shutil.copy(p if os.path.isabs(p) else os.path.join(SP, p), dst)
                ims.append(load(p, 360))
            q = html.unescape(content); txt += ['QUESTION: ' + q, '']
            blocks.append(('text', 'QUESTION', wrap(q, 110), INK)); blocks.append(('images', ims, [f'cam {i+1}' for i in range(len(ims))]))
        elif kind == 'model':
            t = strip_html(content); txt += [name.upper() + ':', t, '']
            blocks.append(('text', name.upper(), wrap(t, 120, 28), INK))
        elif kind == 'call':
            txt += ['TOOL CALL: pi3x_tool -> ' + content, '']
            blocks.append(('text', 'PI3X TOOL CALL', wrap('pi3x_tool -> ' + content, 120), CALL))
        elif kind == 'render':
            ims = []
            for j, p in enumerate(imgs):
                shutil.copy(p if os.path.isabs(p) else os.path.join(SP, p), os.path.join(d, f'render_{j+1}.png')); ims.append(load(p, 700))
            txt += [f'RENDERED 3D VIEW: {content}', '']
            blocks.append(('text', 'RENDERED 3D VIEW  (' + content + ')', [], ACC)); blocks.append(('images', ims, ['' for _ in ims]))
        elif kind == 'verdict':
            pred, gt, ok = content; txt += [f'VERDICT: {"CORRECT" if ok else "WRONG"}  model={pred}  ground_truth={gt}']
            blocks.append(('text', f'VERDICT: {"CORRECT" if ok else "WRONG"}   model answer {pred}   ground truth {gt}', [], OK if ok else BAD))
    open(os.path.join(d, 'trace.txt'), 'w').write('\n'.join(txt))
    # ---- compose ----
    H = PAD + 40
    for b in blocks:
        if b[0] == 'text': H += 30 + 26 * len(b[2]) + 12
        else: H += max(im.height for im in b[1]) + 44 if b[1] else 30
    H += PAD
    canvas = Image.new('RGB', (W, H), (247, 249, 251)); dr = ImageDraw.Draw(canvas)
    dr.rectangle([0, 0, 8, H], fill=OK if ok else BAD)
    y = PAD; dr.text((PAD, y), label + f'   ·   {ds} #{idx:05d}', font=FB, fill=INK); y += 40
    for b in blocks:
        if b[0] == 'text':
            _, title, lines, col = b
            dr.text((PAD, y), title, font=FM, fill=col); y += 30
            for ln in lines: dr.text((PAD, y), ln, font=F, fill=INK); y += 26
            y += 12
        else:
            ims, caps = b[1], b[2]; x = PAD; mh = max(im.height for im in ims) if ims else 0
            for im, cap in zip(ims, caps):
                if x + im.width > W - PAD: x = PAD; y += mh + 30
                if cap: dr.text((x, y), cap, font=FM, fill=MUTED)
                canvas.paste(im, (x, y + 22)); x += im.width + 14
            y += mh + 44
    canvas.save(os.path.join(OUT, slug + '.png'))
    print('saved', slug + '.png', f'{W}x{H}')
print('done ->', OUT)

"""Convert report/REPORT.md into a self-contained HTML page (images embedded as downscaled JPEG data URIs).
  python scripts/report_html.py [out.html]"""
import base64, io, os, re, sys
import markdown
from PIL import Image
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
md = open(os.path.join(R, "report/REPORT.md")).read()
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(R, "report/REPORT.html")
MAXW = int(os.environ.get("IMG_W", 1400)); Q = int(os.environ.get("IMG_Q", 70))
def embed(m):
    p = os.path.join(R, m.group(1))
    if not os.path.exists(p): return m.group(0)
    im = Image.open(p).convert("RGB")
    if im.width > MAXW: im = im.resize((MAXW, round(im.height * MAXW / im.width)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "JPEG", quality=Q, optimize=True)
    return f"![]({'data:image/jpeg;base64,' + base64.b64encode(b.getvalue()).decode()})"
md = re.sub(r"!\[\]\(([^)]+\.png)\)", embed, md)
body = markdown.markdown(md, extensions=["tables", "toc"], extension_configs={"toc": {"toc_depth": "2-3"}})
toc = markdown.Markdown(extensions=["toc"], extension_configs={"toc": {"toc_depth": "2-2"}}); toc.convert(re.sub(r"!\[\]\([^)]*\)", "", md)); toc_html = toc.toc
title = re.search(r"^# (.+)$", md, re.M).group(1).split(":")[0]
html = f"""<title>{title}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{ --bg:#fbfaf7; --panel:#f2f0ea; --ink:#1c2230; --muted:#5d6673; --rule:#d9dde3; --accent:#0f6e73; --accent-soft:#e3f0ef; --code:#eef1f4; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#14181e; --panel:#1c2229; --ink:#e6e9ee; --muted:#9aa4b1; --rule:#2d353f; --accent:#5ec4c8; --accent-soft:#1b2f31; --code:#1f262e; }} }}
:root[data-theme="dark"] {{ --bg:#14181e; --panel:#1c2229; --ink:#e6e9ee; --muted:#9aa4b1; --rule:#2d353f; --accent:#5ec4c8; --accent-soft:#1b2f31; --code:#1f262e; }}
body {{ background:var(--bg); color:var(--ink); font-family:"IBM Plex Sans",system-ui,sans-serif; font-size:15px; line-height:1.55; margin:0; }}
.wrap {{ max-width:1080px; margin:0 auto; padding:2.5rem 1.5rem 5rem; }}
h1,h2,h3,h4 {{ font-family:"Source Serif 4",Georgia,serif; font-weight:600; line-height:1.2; text-wrap:balance; }}
h1 {{ font-size:2.1rem; margin:0 0 .4rem; max-width:30ch; }} h2 {{ font-size:1.5rem; margin:3rem 0 .8rem; padding-top:1rem; border-top:1px solid var(--rule); }}
h3 {{ font-size:1.15rem; margin:2rem 0 .5rem; }} h4 {{ font-size:.85rem; font-family:"IBM Plex Sans",sans-serif; text-transform:uppercase; letter-spacing:.06em; color:var(--accent); margin:2rem 0 .4rem; }}
p, li {{ max-width:72ch; }} p > em:first-child:last-child {{ color:var(--muted); }}
a {{ color:var(--accent); }} a:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
nav.toc {{ background:var(--panel); border:1px solid var(--rule); border-radius:6px; padding:1rem 1.25rem; margin:1.5rem 0 2rem; font-size:.9rem; }}
nav.toc ul {{ margin:0; padding-left:1.1rem; columns:2; column-gap:2rem; }} nav.toc li {{ margin:.15rem 0; break-inside:avoid; }} nav.toc a {{ text-decoration:none; }} nav.toc a:hover {{ text-decoration:underline; }}
.tw {{ overflow-x:auto; margin:1rem 0 1.25rem; border:1px solid var(--rule); border-radius:6px; }}
table {{ border-collapse:collapse; font-size:.82rem; font-variant-numeric:tabular-nums; white-space:nowrap; min-width:100%; }}
th, td {{ padding:.35rem .6rem; border-bottom:1px solid var(--rule); text-align:left; }} th {{ background:var(--panel); font-weight:600; position:sticky; top:0; }}
td {{ font-family:"IBM Plex Mono","SFMono-Regular",monospace; font-size:.78rem; }} td:first-child, th:first-child {{ font-family:"IBM Plex Sans",sans-serif; font-size:.82rem; }}
tr:hover td {{ background:var(--accent-soft); }}
img {{ max-width:100%; display:block; margin:1.25rem 0 .4rem; border:1px solid var(--rule); border-radius:4px; background:#fff; }}
img + p em, p:has(img) + p > em {{ color:var(--muted); font-size:.85rem; }}
code {{ font-family:"IBM Plex Mono",monospace; font-size:.85em; background:var(--code); padding:.05em .3em; border-radius:3px; }}
blockquote {{ border-left:3px solid var(--accent); margin:1rem 0; padding:.2rem 1rem; color:var(--muted); }}
hr {{ border:0; border-top:1px solid var(--rule); margin:2rem 0; }}
</style>
<div class="wrap">
{re.sub(r"<table>", '<div class="tw"><table>', re.sub(r"</table>", "</table></div>", body.replace('<div class="toc">', '<nav class="toc" style="display:none">', 1)))}
</div>
"""
# put a real TOC after the h1 + intro line
html = html.replace("</h1>", "</h1>", 1)
first_p_end = html.find("</p>", html.find("</h1>")) + 4
html = html[:first_p_end] + "\n<nav class=\"toc\">" + toc_html.replace('<div class="toc">', "").replace("</div>", "") + "</nav>\n" + html[first_p_end:]
open(out, "w").write(html); print("wrote", out, f"{os.path.getsize(out)/1e6:.1f} MB")

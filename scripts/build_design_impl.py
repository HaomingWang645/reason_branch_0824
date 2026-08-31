"""Build ViewTree_Implementation_Design_Document.pdf (+ .html for the artifact page) from DESIGN_IMPLEMENTATION.md,
styled after ViewTree_Research_Design_Document.pdf (running header, footer page numbers)."""
import os, re, markdown, weasyprint
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
md = open(os.path.join(R, "DESIGN_IMPLEMENTATION.md")).read()
body = markdown.markdown(md, extensions=["tables"])
body = body.replace("[x]", '<span class="chk">&#9745;</span>').replace("[ ]", '<span class="chk">&#9744;</span>')
css = """
@page { size: A4; margin: 20mm 18mm 18mm 18mm;
  @top-left { content: "VIEWTREE  |  RESEARCH DESIGN — IMPLEMENTATION"; font-family: 'DejaVu Sans'; font-size: 7pt; letter-spacing: .12em; color: #666; }
  @bottom-right { content: "Implementation record  •  " counter(page); font-family: 'DejaVu Sans'; font-size: 7.5pt; color: #666; } }
body { font-family: 'DejaVu Serif', serif; font-size: 9.5pt; line-height: 1.42; color: #16181d; }
h1 { font-family: 'DejaVu Sans', sans-serif; font-size: 19pt; margin: 0 0 4pt; }
h1 + h1 { font-size: 26pt; color: #0f4c56; margin-bottom: 10pt; }
h2 { font-family: 'DejaVu Sans', sans-serif; font-size: 12.5pt; margin: 16pt 0 5pt; color: #0f4c56; border-bottom: .6pt solid #bbb; padding-bottom: 2pt; }
h3 { font-family: 'DejaVu Sans', sans-serif; font-size: 10.5pt; margin: 10pt 0 3pt; }
p { margin: 4pt 0; text-align: justify; } li { margin: 2pt 0; }
strong { font-family: 'DejaVu Sans', sans-serif; font-size: 9pt; }
table { border-collapse: collapse; font-size: 7.8pt; margin: 6pt 0; width: 100%; font-family: 'DejaVu Sans', sans-serif; }
th, td { border: .5pt solid #999; padding: 2.5pt 4pt; vertical-align: top; text-align: left; }
th { background: #e9eef0; } code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8pt; background: #f0f1f3; }
.chk { font-family: 'DejaVu Sans'; }
"""
weasyprint.HTML(string=body, base_url=R).write_pdf(os.path.join(R, "ViewTree_Implementation_Design_Document.pdf"), stylesheets=[weasyprint.CSS(string=css)])
title = "ViewTree Implementation Design Record"
html = f"""<title>{title}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono&display=swap">
<style>
:root {{ --bg:#fbfaf7; --panel:#f1efe9; --ink:#1c2230; --muted:#5d6673; --rule:#d9dde3; --accent:#0f6e73; --accent-soft:#e3f0ef; --code:#eef1f4; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#14181e; --panel:#1c2229; --ink:#e6e9ee; --muted:#9aa4b1; --rule:#2d353f; --accent:#5ec4c8; --accent-soft:#1b2f31; --code:#1f262e; }} }}
:root[data-theme="dark"] {{ --bg:#14181e; --panel:#1c2229; --ink:#e6e9ee; --muted:#9aa4b1; --rule:#2d353f; --accent:#5ec4c8; --accent-soft:#1b2f31; --code:#1f262e; }}
body {{ background:var(--bg); color:var(--ink); font-family:"Source Serif 4",Georgia,serif; font-size:16px; line-height:1.6; margin:0; }}
.wrap {{ max-width:920px; margin:0 auto; padding:2.5rem 1.5rem 5rem; }}
h1,h2,h3 {{ font-family:"IBM Plex Sans",system-ui,sans-serif; line-height:1.2; text-wrap:balance; }}
h1 {{ font-size:1.15rem; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); margin:0 0 .3rem; font-weight:500; }}
h1 + h1 {{ font-size:2.6rem; letter-spacing:0; text-transform:none; color:var(--accent); font-weight:600; margin:0 0 1rem; }}
h2 {{ font-size:1.35rem; margin:2.8rem 0 .7rem; padding-top:.9rem; border-top:1px solid var(--rule); font-weight:600; }}
h3 {{ font-size:1.02rem; margin:1.8rem 0 .4rem; font-weight:600; }}
p, li {{ max-width:74ch; }} strong {{ font-family:"IBM Plex Sans",sans-serif; font-size:.92em; }}
.tw {{ overflow-x:auto; margin:1rem 0 1.25rem; border:1px solid var(--rule); border-radius:6px; }}
table {{ border-collapse:collapse; font-family:"IBM Plex Sans",sans-serif; font-size:.8rem; font-variant-numeric:tabular-nums; min-width:100%; }}
th, td {{ padding:.4rem .6rem; border-bottom:1px solid var(--rule); text-align:left; vertical-align:top; }} th {{ background:var(--panel); }}
tr:hover td {{ background:var(--accent-soft); }}
code {{ font-family:"IBM Plex Mono",monospace; font-size:.82em; background:var(--code); padding:.05em .3em; border-radius:3px; }}
a {{ color:var(--accent); }} a:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
</style>
<div class="wrap">
{re.sub(r"<table>", '<div class="tw"><table>', body.replace("</table>", "</table></div>"))}
</div>"""
open(os.path.join(R, "report/DESIGN_IMPLEMENTATION.html"), "w").write(html)
print("wrote pdf+html")

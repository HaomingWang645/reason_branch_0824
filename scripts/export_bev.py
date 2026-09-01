"""Ceiling-free bird's-eye views for the motivation asset scenes -> <folder>/bev_no_ceiling.jpg
Points above eye level + 15% of room height (clamped below the ceiling slab) are removed
before rendering the top-down pose at 2x."""
import os, sys, cv2, json
import numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from viewtree.reconstruct import reconstruct
from viewtree.render import render, _frame, _lookat
from eval_video_bench import load_sti, load_vsti, read_frames
from eval_external import load_ost
from viewtree.data import load_questions, sample_frames

def bev(rec, out, scale=2):
    pts, up, a, b = _frame(rec)
    P = rec["points"]; Cc = rec["colors"]
    hp = pts @ up
    cams = rec["cam_centers"] if "cam_centers" in rec else None
    import viewtree.render as VR
    hc = (VR.cam_centers(rec) @ up) if hasattr(VR, "cam_centers") else None
    eye_h = hc.median() if hc is not None else torch.quantile(hp, 0.7)
    floor, ceil = torch.quantile(hp, 0.03), torch.quantile(hp, 0.97)
    cut = torch.minimum(eye_h + 0.15 * (ceil - floor), ceil - 0.12 * (ceil - floor))
    hfull = P @ up
    keep = hfull <= cut
    P2 = pts[hp <= cut]
    center = P2.mean(0); radius = (P2 - center).norm(dim=1).quantile(0.95)
    pose = _lookat(center + up * (1.8 * radius), center, a)
    H, W = rec["size"]; K = rec["intrinsics"][0].clone() * scale; K[2, 2] = 1.0
    img = render(P[keep], Cc[keep], pose, K, H * scale, W * scale, splat=4)
    im = (img.clamp(0, 1) * 255).byte().cpu().numpy()
    cv2.imwrite(out, cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(out, "kept", int(keep.sum()), "/", len(keep))

jobs = []
r = next(q for q in load_questions() if q["id"] == 2027)
jobs.append(("figures/motivation_kitchen_assets/bev_no_ceiling.jpg", lambda: sample_frames(r["video"], 32)))
it_s = {x["id"]: x for x in load_sti()}["sti_scene0025_00_4"]
jobs.append(("figures/motivation_assets/sti_scene0025_00/bev_no_ceiling.jpg", lambda: read_frames(it_s["video"], 16, it_s["t0"], it_s["t1"])))
it_v = {x["id"]: x for x in load_vsti()}["vsti_99"]
jobs.append(("figures/motivation_assets/vsti_scene0591_01/bev_no_ceiling.jpg", lambda: read_frames(it_v["video"], 32)))
def ost_imgs():
    for qid, imgs, prompt, gt, qtype in load_ost():
        if qid == "ost_142": return imgs
jobs.append(("figures/motivation_assets/ost_142/bev_no_ceiling.jpg", ost_imgs))
def mc_imgs():
    row = None
    for l in open("data/mindcube/data/raw/MindCube_tinybench.jsonl"):
        rr = json.loads(l)
        if len(rr.get("images", [])) == 4: row = rr; break
    def find_img(p):
        for root in ("data/mindcube/data", "data/mindcube", "data/mindcube/data/other_all_image"):
            q = os.path.join(root, p)
            if os.path.exists(q): return q
        raise FileNotFoundError(p)
    return [cv2.cvtColor(cv2.imread(find_img(p)), cv2.COLOR_BGR2RGB) for p in row["images"]]
jobs.append(("figures/motivation_assets/mindcube_among_group693_q1_5_2/bev_no_ceiling.jpg", mc_imgs))
for out, get in jobs:
    frames = get()
    rec = reconstruct(frames)
    bev(rec, out)
    del rec; torch.cuda.empty_cache()
print("DONE")

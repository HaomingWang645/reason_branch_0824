"""Per-benchmark motivation asset folders: input frames/views + pose-bank reconstruction
renders (1x and 2x). One representative explored sample per benchmark.
  figures/motivation_assets/{sti_...,vsti_...,ost_...,mindcube_...}/"""
import os, sys, json, cv2, glob
import numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from viewtree.reconstruct import reconstruct
from viewtree.render import render
from viewtree.posebank import build_pose_bank
from eval_video_bench import load_sti, load_vsti, read_frames
from eval_external import load_ost
OUT = "figures/motivation_assets"

def save_frames(frames, d):
    os.makedirs(d, exist_ok=True)
    for k, f in enumerate(frames): cv2.imwrite(f"{d}/frame_{k:02d}.jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])

def render_views(rec, base):
    H, W = rec["size"]; K = rec["intrinsics"][0].clone()
    bank, fwd, meta = build_pose_bank(rec, render_all=False)
    for tag, scale, splat in (("reconstruction_views", 1, 2), ("reconstruction_views_2x", 2, 4)):
        d = f"{base}/{tag}"; os.makedirs(d, exist_ok=True); n = 0
        K2 = K * scale; K2[2, 2] = 1.0
        for e in bank:
            img = render(rec["points"], rec["colors"], torch.tensor(e["extrinsic"], device=rec["points"].device), K2, H * scale, W * scale, splat=splat)
            cov = float((img.min(-1).values < 0.999).float().mean())
            if cov < 0.45: continue
            im = (img.clamp(0, 1) * 255).byte().cpu().numpy()
            name = "topdown" if e["kind"] == "topdown" else f"spot{e['pos']+1:02d}_dir{e['yaw']+1}"
            cv2.imwrite(f"{d}/render_{name}.jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92]); n += 1
        print(base.split("/")[-1], tag, n)

def uniform64(video, d):
    cap = cv2.VideoCapture(video); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = set(np.linspace(0, total - 1, 64).round().astype(int).tolist()); k = 0
    os.makedirs(d, exist_ok=True)
    for i in range(total):
        if not cap.grab(): break
        if i in idxs:
            ok, fr = cap.retrieve()
            if ok: cv2.imwrite(f"{d}/frame_{k:02d}.jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 92]); k += 1
    cap.release(); print(d, k)

# ---- STI: scene0025_00, question window 0-64.8 s, 16-frame memory (eval recipe) ----
it = {x["id"]: x for x in load_sti()}["sti_scene0025_00_4"]
base = f"{OUT}/sti_scene0025_00"; os.makedirs(base, exist_ok=True)
uniform64(it["video"], f"{base}/frames_64")
fr = read_frames(it["video"], 16, it["t0"], it["t1"])
render_views(reconstruct(fr), base)
json.dump({"id": it["id"], "question": it["prompt"], "gt": it["gt"]}, open(f"{base}/question.json", "w"), indent=1)

# ---- VSTI: scene0591_01, 32-frame memory ----
it = {x["id"]: x for x in load_vsti()}["vsti_99"]
base = f"{OUT}/vsti_scene0591_01"; os.makedirs(base, exist_ok=True)
uniform64(it["video"], f"{base}/frames_64")
fr = read_frames(it["video"], 32)
render_views(reconstruct(fr), base)
json.dump({"id": it["id"], "question": it["prompt"], "gt": it["gt"]}, open(f"{base}/question.json", "w"), indent=1)

# ---- OST: item ost_142, memory from its image history (<=12 images) ----
for qid, imgs, prompt, gt, qtype in load_ost():
    if qid == "ost_142": break
base = f"{OUT}/ost_142"; os.makedirs(base, exist_ok=True)
save_frames(imgs, f"{base}/history_images"); print(f"{base}/history_images", len(imgs))
render_views(reconstruct(imgs), base)
json.dump({"id": qid, "question": prompt, "gt": gt, "qtype": qtype}, open(f"{base}/question.json", "w"), indent=1)

# ---- MindCube: one 4-view tinybench item ----
row = None
for l in open("data/mindcube/data/raw/MindCube_tinybench.jsonl"):
    r = json.loads(l)
    if len(r.get("images", [])) == 4: row = r; break
def find_img(p):
    for root in ("data/mindcube/data", "data/mindcube", "data/mindcube/data/other_all_image"):
        q = os.path.join(root, p)
        if os.path.exists(q): return q
    raise FileNotFoundError(p)
imgs = [cv2.cvtColor(cv2.imread(find_img(p)), cv2.COLOR_BGR2RGB) for p in row["images"]]
base = f"{OUT}/mindcube_{row['id']}"; os.makedirs(base, exist_ok=True)
save_frames(imgs, f"{base}/input_views"); print(f"{base}/input_views", len(imgs))
render_views(reconstruct(imgs), base)
json.dump({"id": row["id"], "question": row["question"][:600], "gt": row.get("gt_answer", row.get("answer", ""))}, open(f"{base}/question.json", "w"), indent=1)
print("DONE")

"""ARKitScenes ground-truth scan renders for scene 47334117 (VSI trace #925 bedroom):
scan BEV (no ceiling) + eye-level views from the mesh, plus the standard asset set
(64 frames, VGGT pose-bank views, VGGT BEV) for side-by-side comparison."""
import os, sys, cv2
import numpy as np, torch, trimesh
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from viewtree.reconstruct import reconstruct
from viewtree.render import render, _lookat, _frame, cam_centers
from viewtree.posebank import build_pose_bank
from viewtree.data import sample_frames
BASE = "figures/motivation_assets/vsi_arkit_47334117"
VID = "data/videos/arkitscenes/47334117.mp4"
dev = "cuda"

# ---------- scan renders ----------
m = trimesh.load(f"{BASE}/scan/47334117_3dod_mesh.ply")
pts_np, fids = trimesh.sample.sample_surface(m, 10_000_000)
cols_np = m.visual.vertex_colors[m.faces[fids]].astype(np.float32)[..., :3].mean(1) / 255.0
P = torch.tensor(np.asarray(pts_np), dtype=torch.float32, device=dev)
C = torch.tensor(cols_np, dtype=torch.float32, device=dev)
zmin, zmax = P[:, 2].min().item(), P[:, 2].max().item()
up = torch.tensor([0., 0., 1.], device=dev)
H, W = 784, 1036
K = torch.tensor([[640., 0., W / 2], [0., 640., H / 2], [0., 0., 1.]], device=dev)
def shot(Ps, Cs, eye, tgt, upv, out):
    pose = _lookat(torch.tensor(eye, dtype=torch.float32, device=dev), torch.tensor(tgt, dtype=torch.float32, device=dev), torch.tensor(upv, dtype=torch.float32, device=dev))
    img = render(Ps, Cs, pose, K, H, W, splat=4)
    cv2.imwrite(out, cv2.cvtColor((img.clamp(0, 1) * 255).byte().cpu().numpy(), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92]); print(out)
keep = P[:, 2] <= zmin + 2.2
Pk, Ck = P[keep], C[keep]
c = Pk.mean(0).cpu().numpy(); ext = (P.max(0).values - P.min(0).values).cpu().numpy()
shot(Pk, Ck, [c[0], c[1], zmin + 0.95 * max(ext[0], ext[1])], [c[0], c[1], zmin], [1., 0., 0.], f"{BASE}/scan/scan_bev_no_ceiling.jpg")
# eye-level views from 4 positions inside the room toward the centre
for j, (dx, dy) in enumerate([(0.3, 0.3), (-0.3, 0.3), (-0.3, -0.3), (0.3, -0.3)]):
    eye = [c[0] + dx * ext[0], c[1] + dy * ext[1], zmin + 1.6]
    shot(P, C, eye, [c[0], c[1], zmin + 1.2], [0., 0., 1.], f"{BASE}/scan/scan_view_{j+1}.jpg")

# ---------- standard asset set (video + VGGT) ----------
import sys
if os.path.isdir(f"{BASE}/reconstruction_views"): print("skip vggt part"); sys.exit(0)
cap = cv2.VideoCapture(VID); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
idxs = set(np.linspace(0, total - 1, 64).round().astype(int).tolist()); k = 0
os.makedirs(f"{BASE}/frames_64", exist_ok=True)
for i in range(total):
    if not cap.grab(): break
    if i in idxs:
        ok, fr = cap.retrieve()
        if ok: cv2.imwrite(f"{BASE}/frames_64/frame_{k:02d}.jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 92]); k += 1
cap.release(); print("frames", k)
frames = sample_frames(VID, 32); rec = reconstruct(frames)
Hr, Wr = rec["size"]; Kr = rec["intrinsics"][0].clone()
bank, fwd, meta = build_pose_bank(rec, render_all=False)
for tag, scale, splat in (("reconstruction_views", 1, 2), ("reconstruction_views_2x", 2, 4)):
    d = f"{BASE}/{tag}"; os.makedirs(d, exist_ok=True); n = 0
    K2 = Kr * scale; K2[2, 2] = 1.0
    for e in bank:
        img = render(rec["points"], rec["colors"], torch.tensor(e["extrinsic"], device=rec["points"].device), K2, Hr * scale, Wr * scale, splat=splat)
        if float((img.min(-1).values < 0.999).float().mean()) < 0.45: continue
        name = "topdown" if e["kind"] == "topdown" else f"spot{e['pos']+1:02d}_dir{e['yaw']+1}"
        cv2.imwrite(f"{d}/render_{name}.jpg", cv2.cvtColor((img.clamp(0, 1) * 255).byte().cpu().numpy(), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92]); n += 1
    print(tag, n)
pts, upв, a, b = _frame(rec); hp = pts @ upв
hc = cam_centers(rec) @ upв; eye_h = hc.median()
floor, ceil = torch.quantile(hp, 0.03), torch.quantile(hp, 0.97)
cut = torch.minimum(eye_h + 0.15 * (ceil - floor), ceil - 0.12 * (ceil - floor))
kp = (rec["points"] @ upв) <= cut
P2 = pts[hp <= cut]; centre = P2.mean(0); radius = (P2 - centre).norm(dim=1).quantile(0.95)
pose = _lookat(centre + upв * (1.8 * radius), centre, a)
K2 = Kr * 2; K2[2, 2] = 1.0
img = render(rec["points"][kp], rec["colors"][kp], pose, K2, Hr * 2, Wr * 2, splat=4)
cv2.imwrite(f"{BASE}/bev_no_ceiling.jpg", cv2.cvtColor((img.clamp(0, 1) * 255).byte().cpu().numpy(), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
print("DONE")

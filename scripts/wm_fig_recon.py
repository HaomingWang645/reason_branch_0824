"""Figure 3 (world-model failure) assets, stage 1: VGGT reconstruction of the ARKit
bedroom (vsi_arkit_47334117, frames_64), per-frame pose decomposition, and a scan for
(current view A -> real frame B -> real frame C) chains whose relative movement is a
clean "turn + walk forward" step. Saves the recon tensor for stage 2 and a ceiling-free
BEV with the projection matrix so camera marks can be drawn on it.

Run: CUDA_VISIBLE_DEVICES=2 python scripts/wm_fig_recon.py
"""
import os, sys, json, cv2
import numpy as np, torch
sys.path.insert(0, '.')
from viewtree.reconstruct import reconstruct
from viewtree.render import render, _frame, _lookat, cam_centers

BASE = "figures/motivation_assets/vsi_arkit_47334117"
OUT = "figures/wm_failure_assets"
SCRATCH = os.environ.get("WM_SCRATCH", "/tmp/claude-1004/-home-haoming-reason-branch-0824/8c4c54d2-74a7-486b-bdec-a9f98888b186/scratchpad")
os.makedirs(OUT, exist_ok=True)

frames = [cv2.cvtColor(cv2.imread(f"{BASE}/frames_64/frame_{k:02d}.jpg"), cv2.COLOR_BGR2RGB) for k in range(64)]
rec = reconstruct(frames)
pts, up, a, b = _frame(rec)
cams = cam_centers(rec)
E = rec["extrinsics"]
R = E[:, :, :3]
fwd_w = torch.einsum('sij,j->si', R.transpose(1, 2), torch.tensor([0., 0., 1.], device=R.device))

hx, hy, hz = cams @ a, cams @ b, cams @ up
fx, fy, fz = fwd_w @ a, fwd_w @ b, fwd_w @ up
yaw = torch.atan2(fy, fx)
pitch = torch.asin(fz.clamp(-1, 1))

hp = pts @ up
floor, ceil = torch.quantile(hp, 0.03), torch.quantile(hp, 0.97)
room_h = float(ceil - floor)
scale_m = 2.5 / room_h  # assume a 2.5 m floor-to-ceiling room to express units in metres

deg = 180 / np.pi
S = len(frames)
rows = []
for i in range(S):
    for j in range(S):
        if i == j: continue
        dxy = torch.tensor([hx[j] - hx[i], hy[j] - hy[i]])
        d = float(dxy.norm())
        if d < 1e-6: continue
        bearing = float(torch.atan2(dxy[1], dxy[0]) - yaw[i])
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
        dyaw = float(yaw[j] - yaw[i]); dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
        rows.append(dict(i=i, j=j, d_m=d * scale_m, bearing_deg=bearing * deg, dyaw_deg=dyaw * deg,
                         pitch_i=float(pitch[i]) * deg, pitch_j=float(pitch[j]) * deg,
                         dh_m=float(hz[j] - hz[i]) * scale_m))

json.dump(rows, open(f"{OUT}/pairs.json", "w"))

def ok(r, dmin=0.4, dmax=2.4, ymin=20, ymax=100):
    return (abs(r["pitch_i"]) < 20 and abs(r["pitch_j"]) < 20 and abs(r["dh_m"]) < 0.45
            and dmin < r["d_m"] < dmax and ymin < abs(r["dyaw_deg"]) < ymax
            and abs(r["bearing_deg"] - r["dyaw_deg"]) < 25)

cand = [r for r in rows if ok(r)]
cand.sort(key=lambda r: abs(r["bearing_deg"] - r["dyaw_deg"]))
print(f"scale: {scale_m:.3f} m/unit  room_h {room_h:.2f}u  candidates {len(cand)}")
for r in cand[:40]:
    print(f"A={r['i']:02d} -> B={r['j']:02d}  turn {r['dyaw_deg']:+6.1f}deg  walk {r['d_m']:.2f}m  "
          f"bear {r['bearing_deg']:+6.1f}  pitch {r['pitch_i']:+5.1f}/{r['pitch_j']:+5.1f}  dh {r['dh_m']:+.2f}m")

# chains: A->B then B->C, second hop also clean (looser yaw: allow near-straight too)
def ok2(r):
    return (abs(r["pitch_j"]) < 14 and abs(r["dh_m"]) < 0.35 and 0.4 < r["d_m"] < 2.2
            and abs(r["dyaw_deg"]) < 95 and abs(r["bearing_deg"] - r["dyaw_deg"]) < 20)
by_i = {}
for r in rows: by_i.setdefault(r["i"], []).append(r)
print("\n--- chains A->B->C ---")
nch = 0
for r1 in cand[:25]:
    for r2 in by_i.get(r1["j"], []):
        if r2["j"] == r1["i"] or not ok2(r2): continue
        if abs(r2["dyaw_deg"]) < 15 and r2["d_m"] < 0.6: continue
        print(f"A={r1['i']:02d} B={r1['j']:02d} C={r2['j']:02d} | hop1 turn {r1['dyaw_deg']:+6.1f} walk {r1['d_m']:.2f}m"
              f" | hop2 turn {r2['dyaw_deg']:+6.1f} walk {r2['d_m']:.2f}m")
        nch += 1
        if nch >= 40: break
    if nch >= 40: break

# ceiling-free BEV + projection info for drawing camera marks
cut = torch.minimum(hz.median() + 0.15 * (ceil - floor), ceil - 0.12 * (ceil - floor))
P2 = pts[hp <= cut]
center = P2.mean(0); radius = (P2 - center).norm(dim=1).quantile(0.95)
pose = _lookat(center + up * (1.8 * radius), center, a)
H, W = rec["size"]; K = rec["intrinsics"][0].clone() * 2; K[2, 2] = 1.0
keep = (rec["points"] @ up) <= cut
img = render(rec["points"][keep], rec["colors"][keep], pose, K, H * 2, W * 2, splat=4)
im = (img.clamp(0, 1) * 255).byte().cpu().numpy()
cv2.imwrite(f"{OUT}/bev.jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])

def proj(p3):
    x = pose[:, :3] @ p3 + pose[:, 3]
    q = K @ x
    return [float(q[0] / q[2]), float(q[1] / q[2])]

meta = dict(scale_m=scale_m,
            cam_bev=[proj(cams[s]) for s in range(S)],
            fwd_bev=[proj(cams[s] + fwd_w[s] * 0.35 * radius / scale_m * scale_m) for s in range(S)],
            yaw=[float(v) * deg for v in yaw], pitch=[float(v) * deg for v in pitch],
            bev_size=[H * 2, W * 2])
json.dump(meta, open(f"{OUT}/pose_meta.json", "w"))
torch.save({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in rec.items()}, f"{SCRATCH}/wm_rec.pt")
print("saved bev + pose_meta + rec")

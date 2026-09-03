"""Figure 3 assets, stage 2: ViewTree renders at the requested poses (frames 25 and 23
of the ARKit bedroom) from the saved 64-frame reconstruction, plus a yaw-direction
sanity render at frame 9's pose rotated +/-86 degrees about the scene up axis.

Run: CUDA_VISIBLE_DEVICES=2 python scripts/wm_fig_render.py
"""
import os, sys, cv2
import numpy as np, torch
sys.path.insert(0, '.')
from viewtree.render import render, _frame

SCRATCH = os.environ.get("WM_SCRATCH", "/tmp/claude-1004/-home-haoming-reason-branch-0824/8c4c54d2-74a7-486b-bdec-a9f98888b186/scratchpad")
OUT = "figures/wm_failure_assets"
rec = torch.load(f"{SCRATCH}/wm_rec.pt", weights_only=False)
rec = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in rec.items()}
pts, up, a, b = _frame(rec)
H, W = rec["size"]
K2 = rec["intrinsics"][0].clone() * 2; K2[2, 2] = 1.0
P, C = rec["points"], rec["colors"]

def rend(pose34, out, splat=4):
    img = render(P, C, pose34.cuda(), K2, H * 2, W * 2, splat=splat)
    im = (img.clamp(0, 1) * 255).byte().cpu().numpy()
    cv2.imwrite(out, cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(out)

E = rec["extrinsics"]
for k in (25, 23):
    rend(E[k], f"{OUT}/render_{k}.jpg")

# yaw-direction sanity: rotate frame 9's camera about the up axis by +/-86 deg
def yawed(E9, deg):
    Rwc = E9[:, :3]; t = E9[:, 3]
    Cc = -Rwc.T @ t
    th = np.radians(deg)
    u = (up / up.norm()).cpu().numpy()
    Kx = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    Rot = torch.tensor(np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx), dtype=E9.dtype)
    Rwc2 = Rwc @ Rot.T
    t2 = -Rwc2 @ Cc
    return torch.cat([Rwc2, t2[:, None]], 1)

for d in (+86, -86):
    rend(yawed(E[9].cpu(), d), f"{SCRATCH}/yawcheck_{'p' if d>0 else 'm'}86.jpg", splat=4)

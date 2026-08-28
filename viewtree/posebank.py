"""Pose bank for multi-step view acquisition (ViewTree-D, DESIGN_DEPTH.md §1).

For one reconstruction: P standing positions farthest-point-sampled inside the
walked hull (eye height, wall clearance) x Y yaws (roll 0, pitch 10 deg down)
+ 1 top-down bird's-eye. Each entry: extrinsic (3,4), position index, yaw
index, coverage, validity flags. Camera actions (TURN_LEFT/RIGHT, FORWARD,
NEXT_SPOT, LOOK_AROUND, BIRD_EYE) are transitions between bank entries."""
import numpy as np
import torch
from .render import _frame, _lookat, cam_centers, render


@torch.no_grad()
def build_pose_bank(rec, num_pos=12, num_yaw=8, pitch_deg=10.0, clearance_frac=0.04,
                    min_coverage=0.45, grid=25, splat=2, render_all=True):
    from scipy.spatial import Delaunay
    pts, up, a, b = _frame(rec); dev = pts.device
    cams = cam_centers(rec)
    P2 = torch.stack([pts @ a, pts @ b], 1); hp = pts @ up
    C2 = torch.stack([cams @ a, cams @ b], 1); hc = cams @ up
    eye_h = hc.median(); floor, ceil = torch.quantile(hp, 0.03), torch.quantile(hp, 0.97)
    lo2, hi2 = torch.quantile(P2, 0.05, dim=0), torch.quantile(P2, 0.95, dim=0)
    center2 = (lo2 + hi2) / 2; diag = float((hi2 - lo2).norm()); clearance = clearance_frac * diag
    cl, ch = C2.min(0).values, C2.max(0).values; ext = (ch - cl).clamp(min=0.05 * diag); cl, ch = cl - 0.1 * ext, ch + 0.1 * ext
    gx = torch.linspace(float(cl[0]), float(ch[0]), grid, device=dev); gy = torch.linspace(float(cl[1]), float(ch[1]), grid, device=dev)
    G = torch.stack(torch.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
    try:
        hull = Delaunay(C2.cpu().numpy()); inside = torch.from_numpy(hull.find_simplex(G.cpu().numpy()) >= 0).to(dev)
    except Exception:
        inside = torch.ones(len(G), dtype=torch.bool, device=dev)
    band = (hp > eye_h - 0.25 * (ceil - floor)) & (hp < eye_h + 0.25 * (ceil - floor)); Pb = P2[band]
    if len(Pb) > 60_000: Pb = Pb[torch.randperm(len(Pb), device=dev)[:60_000]]
    dmin = torch.cdist(G, Pb).min(1).values if len(Pb) else torch.full((len(G),), 1e9, device=dev)
    valid = inside & (dmin > clearance)
    if valid.sum() < 3:
        order = torch.argsort(torch.where(inside, dmin, dmin - 1e6), descending=True); valid = torch.zeros_like(valid); valid[order[:8]] = True
    V = G[valid]
    sel = [int(torch.argmax((V - center2).norm(dim=1)))]
    while len(sel) < min(len(V), num_pos):
        d = torch.cdist(V, V[sel]).min(1).values; sel.append(int(torch.argmax(d)))
    H, W = rec["size"]; K = rec["intrinsics"][0]; p = np.deg2rad(pitch_deg)
    cell = float((ch - cl).max() / (grid - 1))
    bank, positions = [], []
    for pi, si in enumerate(sel):
        x, y = V[si]; eye = a * x + b * y + up * eye_h; positions.append([float(x), float(y)])
        tgt = a * center2[0] + b * center2[1] + up * eye_h; f0 = tgt - eye; f0 = f0 - up * (f0 @ up)
        if f0.norm() < 1e-6: f0 = a.clone()
        f0 = f0 / f0.norm(); r0 = torch.linalg.cross(f0, up); r0 = r0 / r0.norm()
        yaw0 = 0.0
        for yi in range(num_yaw):
            th = yaw0 + 2 * np.pi * yi / num_yaw
            fh = float(np.cos(th)) * f0 + float(np.sin(th)) * r0  # horizontal forward
            f = float(np.cos(p)) * fh - float(np.sin(p)) * up
            pose = _lookat(eye, eye + f, up)
            cov = None
            if render_all:
                img = render(rec["points"], rec["colors"], pose, K, H, W, splat=splat); cov = float((img.min(-1).values < 0.999).float().mean())
            bank.append(dict(idx=len(bank), pos=pi, yaw=yi, kind="eye", extrinsic=pose.cpu().tolist(), coverage=cov,
                             valid=(cov is None or cov >= min_coverage), eye=eye.cpu().tolist(), fwd=fh.cpu().tolist()))
    lo = torch.quantile(pts, 0.05, dim=0); hi = torch.quantile(pts, 0.95, dim=0); center = (lo + hi) / 2; radius = (hi - lo).norm() / 2
    pose = _lookat(center + up * (1.6 * radius), center, a)
    cov = float((render(rec["points"], rec["colors"], pose, K, H, W, splat=splat).min(-1).values < 0.999).float().mean()) if render_all else None
    bank.append(dict(idx=len(bank), pos=-1, yaw=-1, kind="topdown", extrinsic=pose.cpu().tolist(), coverage=cov, valid=True))
    # neighbour graph for FORWARD: nearest other position roughly in the forward direction
    Pm = np.array(positions); fwd_map = {}
    for e in bank:
        if e["kind"] != "eye": continue
        eye2 = np.array(e["eye"]); f2 = np.array(e["fwd"])
        ea = np.array([float(a @ torch.tensor(eye2, device=dev, dtype=a.dtype)), float(b @ torch.tensor(eye2, device=dev, dtype=a.dtype))])
        fa = np.array([float(a @ torch.tensor(f2, device=dev, dtype=a.dtype)), float(b @ torch.tensor(f2, device=dev, dtype=a.dtype))])
        best, bd = -1, 1e9
        for pj, pp in enumerate(Pm):
            if pj == e["pos"]: continue
            dvec = pp - ea; dist = np.linalg.norm(dvec)
            if dist < 1e-6: continue
            if dvec @ fa / dist > 0.5 and dist < bd: best, bd = pj, dist
        fwd_map[e["idx"]] = best
    meta = dict(num_pos=len(positions), num_yaw=num_yaw, positions=positions, eye_height_rel=float((eye_h - floor) / (ceil - floor + 1e-6)),
                cell=cell, size=[int(H), int(W)])
    return bank, fwd_map, meta


def transition(bank, fwd_map, idx, action, meta):
    """Apply a camera action to bank index idx -> new idx or None (invalid)."""
    e = bank[idx]; ny = meta["num_yaw"]
    if e["kind"] != "eye":
        return None  # bird's-eye is terminal
    at = lambda pos, yaw: next(x["idx"] for x in bank if x["pos"] == pos and x["yaw"] == yaw % ny)
    if action == "TURN_LEFT": j = at(e["pos"], e["yaw"] - 1)
    elif action == "TURN_RIGHT": j = at(e["pos"], e["yaw"] + 1)
    elif action == "LOOK_AROUND": j = at(e["pos"], e["yaw"] + ny // 2)
    elif action == "FORWARD":
        pj = fwd_map.get(idx, -1)
        if pj < 0: return None
        j = at(pj, e["yaw"])
    elif action == "NEXT_SPOT": j = at((e["pos"] + 1) % meta["num_pos"], 0)
    elif action == "BIRD_EYE": j = next(x["idx"] for x in bank if x["kind"] == "topdown")
    else: return None
    return j if bank[j]["valid"] else None

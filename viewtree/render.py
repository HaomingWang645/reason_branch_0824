"""GPU point-splat renderer + viewpoint generation.

Two proposers:
  overview_poses  - legacy: 4 elevated oblique views from OUTSIDE the room
                    envelope (dist 1.1 x radius, ~40 deg pitch) + top-down.
  human_poses     - constrained "person holding a camera": eye-height
                    positions inside the walkable region (convex hull of the
                    recorded camera centres), free-space clearance from walls,
                    roll = 0 (image horizontal parallel to the floor), mild
                    downward pitch, views with too little coverage discarded;
                    top-down bird's-eye kept as the final view.
Set VIEWTREE_POSES=human to make overview_poses dispatch to human_poses.
"""
import os
import numpy as np
import torch

BG = 1.0  # white background


@torch.no_grad()
def render(points, colors, extrinsic, intrinsic, H, W, splat=1, eps_rel=0.01):
    """z-buffered point splatting. points (N,3), colors (N,3) [0,1],
    extrinsic (3,4) world->cam, intrinsic (3,3). Returns (H,W,3) float."""
    dev = points.device
    R, t = extrinsic[:, :3], extrinsic[:, 3]
    Xc = points @ R.T + t
    z = Xc[:, 2]
    keep = z > 0.05
    Xc, z, cols = Xc[keep], z[keep], colors[keep]
    uv = Xc @ intrinsic.T
    u = (uv[:, 0] / z).round().long()
    v = (uv[:, 1] / z).round().long()

    depthbuf = torch.full((H * W,), float("inf"), device=dev)
    offs = range(-splat, splat + 1)
    idx_list, z_list, c_list = [], [], []
    for dy in offs:
        for dx in offs:
            uu, vv = u + dx, v + dy
            ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            idx = (vv[ok] * W + uu[ok])
            idx_list.append(idx)
            z_list.append(z[ok])
            c_list.append(cols[ok])
            depthbuf.scatter_reduce_(0, idx, z[ok], reduce="amin")
    img = torch.full((H * W, 3), BG, device=dev)
    idx = torch.cat(idx_list)
    zz = torch.cat(z_list)
    cc = torch.cat(c_list)
    near = zz <= depthbuf[idx] * (1 + eps_rel) + 0.01
    idx, zz, cc = idx[near], zz[near], cc[near]
    order = torch.argsort(zz, descending=True)  # nearest written last
    img[idx[order]] = cc[order]
    return img.view(H, W, 3)


def _lookat(eye, center, up):
    f = center - eye
    f = f / f.norm()
    r = torch.linalg.cross(f, up)
    r = r / (r.norm() + 1e-8)
    d = torch.linalg.cross(f, r)
    R = torch.stack([r, d, f])  # rows: right, down, forward (x right, y down, z fwd)
    t = -R @ eye
    return torch.cat([R, t[:, None]], 1)  # (3,4)


def _frame(rec):
    pts = rec["points"]
    if len(pts) > 300_000:
        pts = pts[torch.randperm(len(pts), device=pts.device)[:300_000]]
    up = -rec["extrinsics"][:, 1, :3].mean(0)
    up = up / up.norm()
    a = torch.linalg.cross(up, torch.tensor([1.0, 0, 0], device=up.device))
    if a.norm() < 0.1:
        a = torch.linalg.cross(up, torch.tensor([0, 1.0, 0], device=up.device))
    a = a / a.norm()
    b = torch.linalg.cross(up, a)
    return pts, up, a, b


def cam_centers(rec):
    ext = rec["extrinsics"]
    R, t = ext[:, :, :3], ext[:, :, 3]
    return -(R.transpose(1, 2) @ t[..., None])[..., 0]


@torch.no_grad()
def human_poses(rec, num_side=4, pitch_deg=10.0, clearance_frac=0.04,
                min_coverage=0.45, grid=25, splat=2):
    """Constrained viewpoints mimicking a person walking with a camera.
    Hard constraints (views violating them are discarded):
      inside   - eye position inside the convex hull of the recorded camera
                 trajectory (the region the person actually walked), at the
                 median recorded camera height (eye height);
      clear    - no reconstructed surface within clearance_frac x room diagonal
                 of the eye in the eye-height band (not inside a wall/object);
      level    - roll = 0: camera right-axis orthogonal to world up, pitch
                 fixed to a mild downward tilt (human-like);
      coverage - rendered view must paint >= min_coverage of the pixels.
    Positions are farthest-point-sampled from valid cells so the 4 views come
    from different sides of the room, each looking toward the room centre.
    The 5th view is the top-down bird's-eye (allowed 'at the end').
    Stores per-view metadata in rec["pose_meta"]."""
    from scipy.spatial import Delaunay
    pts, up, a, b = _frame(rec)
    dev = pts.device
    cams = cam_centers(rec)
    P2 = torch.stack([pts @ a, pts @ b], 1); hp = pts @ up
    C2 = torch.stack([cams @ a, cams @ b], 1); hc = cams @ up
    eye_h = hc.median()
    floor, ceil = torch.quantile(hp, 0.03), torch.quantile(hp, 0.97)
    lo2, hi2 = torch.quantile(P2, 0.05, dim=0), torch.quantile(P2, 0.95, dim=0)
    center2 = (lo2 + hi2) / 2
    diag = float((hi2 - lo2).norm())
    clearance = clearance_frac * diag
    # candidate grid over the camera-trajectory bbox
    cl, ch = C2.min(0).values, C2.max(0).values
    ext = (ch - cl).clamp(min=0.05 * diag)
    cl, ch = cl - 0.1 * ext, ch + 0.1 * ext
    gx = torch.linspace(float(cl[0]), float(ch[0]), grid, device=dev)
    gy = torch.linspace(float(cl[1]), float(ch[1]), grid, device=dev)
    G = torch.stack(torch.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
    # inside: convex hull of camera centres (fallback: bbox) 
    try:
        hull = Delaunay(C2.cpu().numpy())
        inside = torch.from_numpy(hull.find_simplex(G.cpu().numpy()) >= 0).to(dev)
    except Exception:
        inside = torch.ones(len(G), dtype=torch.bool, device=dev)
    # clear: min horizontal distance to surfaces in the eye-height band
    band = (hp > eye_h - 0.25 * (ceil - floor)) & (hp < eye_h + 0.25 * (ceil - floor))
    Pb = P2[band]
    if len(Pb) > 60_000:
        Pb = Pb[torch.randperm(len(Pb), device=dev)[:60_000]]
    dmin = torch.cdist(G, Pb).min(1).values if len(Pb) else torch.full((len(G),), 1e9, device=dev)
    valid = inside & (dmin > clearance)
    if valid.sum() < num_side:  # relax: keep the most-clear inside cells
        order = torch.argsort(torch.where(inside, dmin, dmin - 1e6), descending=True)
        valid = torch.zeros_like(valid); valid[order[:max(num_side, 8)]] = True
    V = G[valid]
    # farthest-point sampling, seeded at the cell farthest from the room centre
    sel = [int(torch.argmax((V - center2).norm(dim=1)))]
    while len(sel) < min(len(V), 4 * num_side):
        d = torch.cdist(V, V[sel]).min(1).values
        sel.append(int(torch.argmax(d)))
    H, W = rec["size"]; K = rec["intrinsics"][0]
    p = np.deg2rad(pitch_deg)
    poses, meta = [], []
    rejected = []
    for si in sel:
        x, y = V[si]
        eye = a * x + b * y + up * eye_h
        tgt = a * center2[0] + b * center2[1] + up * eye_h
        f = tgt - eye; f = f - up * (f @ up); 
        if f.norm() < 1e-6:
            f = a.clone()
        f = f / f.norm()
        f = float(np.cos(p)) * f - float(np.sin(p)) * up
        pose = _lookat(eye, eye + f, up)
        img = render(rec["points"], rec["colors"], pose, K, H, W, splat=splat)
        cov = float((img.min(-1).values < 0.999).float().mean())
        m = {"eye": eye.tolist(), "eye_height_rel": float((eye_h - floor) / (ceil - floor + 1e-6)),
             "inside_hull": True, "clearance": float(dmin[valid][si]), "clearance_req": clearance,
             "pitch_deg": pitch_deg, "roll_deg": float(torch.rad2deg(torch.asin((pose[0, :3] @ up).clamp(-1, 1)))),
             "coverage": cov, "kind": "human"}
        if cov >= min_coverage:
            poses.append(pose); meta.append(m)
        else:
            rejected.append((cov, pose, m))
        if len(poses) == num_side:
            break
    if len(poses) < num_side:  # fill from the best rejected (coverage-ranked)
        for cov, pose, m in sorted(rejected, key=lambda z: -z[0])[: num_side - len(poses)]:
            m["kind"] = "human_lowcov"; poses.append(pose); meta.append(m)
    # top-down bird's-eye (allowed as the final view)
    lo = torch.quantile(pts, 0.05, dim=0); hi = torch.quantile(pts, 0.95, dim=0)
    center = (lo + hi) / 2; radius = (hi - lo).norm() / 2
    eye = center + up * (1.6 * radius)
    poses.append(_lookat(eye, center, a))
    meta.append({"kind": "topdown"})
    rec["pose_meta"] = meta
    return poses


@torch.no_grad()
def pose_report(rec, poses):
    """Geometric audit of a pose list: inside walkable hull, clearance, pitch, roll."""
    from scipy.spatial import Delaunay
    pts, up, a, b = _frame(rec)
    cams = cam_centers(rec)
    C2 = torch.stack([cams @ a, cams @ b], 1)
    hp = pts @ up; floor, ceil = torch.quantile(hp, 0.03), torch.quantile(hp, 0.97)
    P2 = torch.stack([pts @ a, pts @ b], 1)
    lo2, hi2 = torch.quantile(P2, 0.05, dim=0), torch.quantile(P2, 0.95, dim=0)
    diag = float((hi2 - lo2).norm())
    try:
        hull = Delaunay(C2.cpu().numpy())
    except Exception:
        hull = None
    out = []
    for pose in poses:
        R, t = pose[:, :3], pose[:, 3]
        eye = -(R.T @ t)
        e2 = torch.stack([eye @ a, eye @ b]).cpu().numpy()
        h = float(eye @ up)
        fwd = R[2]; right = R[0]
        pitch = float(torch.rad2deg(torch.asin((-fwd @ up).clamp(-1, 1))))
        roll = float(torch.rad2deg(torch.asin((right @ up).clamp(-1, 1))))
        inside = bool(hull.find_simplex(e2[None])[0] >= 0) if hull is not None else True
        in_ext = bool((e2 >= lo2.cpu().numpy()).all() and (e2 <= hi2.cpu().numpy()).all())
        within_height = bool(floor <= h <= ceil)
        out.append({"inside_walked_hull": inside, "inside_room_extent": in_ext and within_height,
                    "height_rel": float((h - floor) / (ceil - floor + 1e-6)),
                    "dist_to_center_rel": float(np.linalg.norm(e2 - ((lo2 + hi2) / 2).cpu().numpy()) / (diag / 2)),
                    "pitch_deg": pitch, "roll_deg": roll})
    return out


@torch.no_grad()
def overview_poses(rec, num_side=4, elev=0.9, dist=1.1):
    if os.environ.get("VIEWTREE_POSES", "").lower() == "human":
        return human_poses(rec, num_side=num_side)
    return legacy_poses(rec, num_side=num_side, elev=elev, dist=dist)


@torch.no_grad()
def legacy_poses(rec, num_side=4, elev=0.9, dist=1.1):
    """Heuristic exploration viewpoints: 4 elevated oblique views + 1 top-down,
    aimed at the confident-point centroid. Returns list of (3,4) extrinsics."""
    pts = rec["points"]
    if len(pts) > 500_000:
        pts = pts[torch.randperm(len(pts), device=pts.device)[:500_000]]
    lo = torch.quantile(pts, 0.05, dim=0)
    hi = torch.quantile(pts, 0.95, dim=0)
    center = (lo + hi) / 2
    radius = (hi - lo).norm() / 2

    # world up: negative mean of camera down-axes (row 1 of R maps world->cam-y)
    up = -rec["extrinsics"][:, 1, :3].mean(0)
    up = up / up.norm()
    # two horizontal axes orthogonal to up
    a = torch.linalg.cross(up, torch.tensor([1.0, 0, 0], device=up.device))
    if a.norm() < 0.1:
        a = torch.linalg.cross(up, torch.tensor([0, 1.0, 0], device=up.device))
    a = a / a.norm()
    b = torch.linalg.cross(up, a)

    poses = []
    for k in range(num_side):
        ang = 2 * np.pi * k / num_side
        d = float(np.cos(ang)) * a + float(np.sin(ang)) * b
        eye = center + up * (elev * radius) + d * (dist * radius)
        poses.append(_lookat(eye, center, up))
    # top-down (use a as the up-hint to avoid degenerate cross product)
    eye = center + up * (1.6 * radius)
    poses.append(_lookat(eye, center, a))
    return poses

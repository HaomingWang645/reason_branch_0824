"""GPU point-splat renderer + heuristic overview viewpoint generation."""
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


@torch.no_grad()
def overview_poses(rec, num_side=4, elev=0.9, dist=1.1):
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

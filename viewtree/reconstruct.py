"""Frozen VGGT-1B reconstruction: frames -> world point cloud + cameras."""
import numpy as np
import torch
import torch.nn.functional as F

_MODEL = None


def _get_model(device):
    global _MODEL
    if _MODEL is None:
        from vggt.models.vggt import VGGT

        _MODEL = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    return _MODEL


def preprocess(frames, target_w=518):
    """uint8 RGB frames -> (S,3,H,W) float in [0,1], H,W multiples of 14."""
    ims = []
    for f in frames:
        t = torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
        _, h, w = t.shape
        new_h = int(round(h * target_w / w / 14) * 14)
        t = F.interpolate(t[None], size=(new_h, target_w), mode="bilinear", align_corners=False)[0]
        ims.append(t)
    return torch.stack(ims)


@torch.no_grad()
def reconstruct(frames, device="cuda", conf_percentile=30.0):
    """Returns dict with torch tensors on device:
    points (N,3) world, colors (N,3) in [0,1], extrinsics (S,3,4) world->cam,
    intrinsics (S,3,3), image size, and per-frame point maps for held-out eval.
    """
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    model = _get_model(device)
    images = preprocess(frames).to(device)
    S, _, H, W = images.shape
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(images[None])
    extri, intri = pose_encoding_to_extri_intri(pred["pose_enc"], (H, W))
    extri, intri = extri[0].float(), intri[0].float()  # (S,3,4),(S,3,3)
    depth = pred["depth"][0, ..., 0].float()  # (S,H,W)
    depth_conf = pred["depth_conf"][0].float()  # (S,H,W)

    # unproject depth to world points (camera convention: x_cam = R x_w + t)
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(u)
    pix = torch.stack([u + 0.5, v + 0.5, ones], -1)  # (H,W,3)
    Kinv = torch.linalg.inv(intri)  # (S,3,3)
    rays = torch.einsum("sij,hwj->shwi", Kinv, pix)  # (S,H,W,3)
    cam_pts = rays * depth[..., None]
    R = extri[:, :, :3]
    t = extri[:, :, 3]
    world = torch.einsum("sji,shwj->shwi", R, cam_pts - t[:, None, None, :])

    colors = images.permute(0, 2, 3, 1)  # (S,H,W,3)
    thresh = torch.quantile(depth_conf.flatten(), conf_percentile / 100.0)
    mask = (depth_conf > thresh) & (depth > 1e-4)

    return {
        "points": world[mask],
        "colors": colors[mask],
        "world_maps": world,          # (S,H,W,3) for per-frame subsets
        "color_maps": colors,
        "mask_maps": mask,
        "extrinsics": extri,
        "intrinsics": intri,
        "size": (H, W),
    }

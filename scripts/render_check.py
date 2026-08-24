"""Renderer viability check (design doc's biggest unvalidated assumption).

Protocol: 16 uniformly sampled eval frames are held out and NEVER contribute
points; a separate set of --num-frames source frames (uniform, half-step offset)
builds the point cloud. One VGGT pass runs on [eval + source] so all cameras
share a coordinate frame. We render at the odd 8 eval poses and compare with the
real frames (PSNR, covered-pixel PSNR, SSIM, coverage). Eval poses are identical
across --num-frames settings, so results are directly comparable.
"""
import argparse
import json
import os
import sys
import traceback

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viewtree.data import load_questions, sample_frames
from viewtree.reconstruct import reconstruct
from viewtree.render import overview_poses, render


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def ssim(a, b):
    # simple grayscale global SSIM on downsampled images
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    return ((2 * mu_a * mu_b + C1) * (2 * cov + C2)) / ((mu_a**2 + mu_b**2 + C1) * (va + vb + C2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-per-dataset", type=int, default=20)
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--splat", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exdir = os.path.join(repo, "results", "render_examples")
    os.makedirs(exdir, exist_ok=True)

    rows = load_questions()
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], {})[r["scene_name"]] = r["video"]
    fout = open(args.out, "w")
    for ds, scenes in sorted(by_ds.items()):
        names = sorted(scenes)[: args.scenes_per_dataset]
        for si, name in enumerate(names):
            try:
                N = args.num_frames
                NE = 16
                allf = sample_frames(scenes[name], 257)  # fixed dense pool
                total = len(allf)
                eval_idx = np.linspace(0, total - 1, NE).round().astype(int)
                half = (total - 1) / (2 * N)
                src_idx = (np.linspace(0, total - 1 - 2 * half, N) + half).round().astype(int)
                src_idx = np.array([i + 1 if i in set(eval_idx) else i for i in src_idx])
                frames = [allf[i] for i in eval_idx] + [allf[i] for i in src_idx]
                rec = reconstruct(frames)
                H, W = rec["size"]
                K = rec["intrinsics"][0]
                src = torch.arange(NE, NE + N)
                m = rec["mask_maps"][src]
                pts = rec["world_maps"][src][m]
                cols = rec["color_maps"][src][m]
                res = {"dataset": ds, "scene": name, "views": []}
                for j in range(1, NE, 2):
                    img = render(pts, cols, rec["extrinsics"][j], rec["intrinsics"][j],
                                 H, W, splat=args.splat)
                    img8 = (img.clamp(0, 1) * 255).byte().cpu().numpy()
                    real = (rec["color_maps"][j].clamp(0, 1) * 255).byte().cpu().numpy()
                    covmask = (img8 != 255).any(-1)
                    cov = float(covmask.mean())
                    mpsnr = psnr(img8[covmask], real[covmask]) if covmask.any() else 0.0
                    res["views"].append({
                        "frame": j, "psnr": psnr(img8, real), "masked_psnr": mpsnr,
                        "ssim": float(ssim(img8, real)), "coverage": cov,
                    })
                    if si == 0 and j == 7:
                        cv2.imwrite(f"{exdir}/{ds}_{name}_heldout_render.png",
                                    cv2.cvtColor(img8, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(f"{exdir}/{ds}_{name}_heldout_real.png",
                                    cv2.cvtColor(real, cv2.COLOR_RGB2BGR))
                # qualitative overview renders from the full cloud
                if si < 3:
                    for vi, pose in enumerate(overview_poses(rec)):
                        img = render(rec["points"], rec["colors"], pose, K, H, W,
                                     splat=args.splat)
                        img8 = (img.clamp(0, 1) * 255).byte().cpu().numpy()
                        cv2.imwrite(f"{exdir}/{ds}_{name}_overview{vi}.png",
                                    cv2.cvtColor(img8, cv2.COLOR_RGB2BGR))
                fout.write(json.dumps(res) + "\n")
                fout.flush()
                print(f"{ds} {si+1}/{len(names)} {name} ok", flush=True)
                del rec
                torch.cuda.empty_cache()
            except Exception:
                traceback.print_exc()
    fout.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

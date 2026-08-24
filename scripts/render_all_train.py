"""Pre-render top-down views for all MindCube train items (GRPO needs them)."""
import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MC_ROOT = os.path.join(REPO, "data", "mindcube", "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    import torch
    from viewtree.reconstruct import reconstruct
    from viewtree.render import overview_poses, render

    rows = [json.loads(l) for l in
            open(os.path.join(MC_ROOT, "raw", "MindCube_train.jsonl"))]
    rows = rows[args.shard::args.num_shards]
    outdir = os.path.join(REPO, "data", "mindcube_renders")
    os.makedirs(outdir, exist_ok=True)
    done = 0
    for i, item in enumerate(rows):
        path = os.path.join(outdir, f"{item['id']}.png")
        if os.path.exists(path):
            continue
        try:
            imgs = [cv2.cvtColor(cv2.imread(os.path.join(MC_ROOT, p)),
                                 cv2.COLOR_BGR2RGB) for p in item["images"]]
            rec = reconstruct(imgs)
            H, W = rec["size"]
            pose = overview_poses(rec)[-1]
            img = render(rec["points"], rec["colors"], pose, rec["intrinsics"][0],
                         H, W, splat=2)
            cv2.imwrite(path, cv2.cvtColor(
                (img.clamp(0, 1) * 255).byte().cpu().numpy(), cv2.COLOR_RGB2BGR))
            del rec
            torch.cuda.empty_cache()
            done += 1
            if done % 100 == 0:
                print(f"[render shard {args.shard}] {done} rendered "
                      f"({i+1}/{len(rows)} seen)", flush=True)
        except Exception as e:
            print("failed", item["id"], repr(e)[:80], flush=True)
    print(f"DONE rendered {done}", flush=True)


if __name__ == "__main__":
    main()

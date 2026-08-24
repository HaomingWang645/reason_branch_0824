"""Pre-render top-down views needed by render-dependent SFT answer examples."""
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MC_ROOT = os.path.join(REPO, "data", "mindcube", "data")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO, "data", "sft_data.jsonl"))
    args = ap.parse_args()
    import torch
    from viewtree.reconstruct import reconstruct
    from viewtree.render import overview_poses, render

    train = {json.loads(l)["id"]: json.loads(l)
             for l in open(os.path.join(MC_ROOT, "raw", "MindCube_train.jsonl"))}
    need = set()
    for l in open(args.data):
        r = json.loads(l)
        if r["render"]:
            need.add(r["render"])
    outdir = os.path.join(REPO, "data", "mindcube_renders")
    os.makedirs(outdir, exist_ok=True)
    done = 0
    for rid in sorted(need):
        path = os.path.join(outdir, f"{rid}.png")
        if os.path.exists(path):
            continue
        try:
            item = train[rid]
            imgs = [cv2.cvtColor(cv2.imread(os.path.join(MC_ROOT, p)), cv2.COLOR_BGR2RGB)
                    for p in item["images"]]
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
        except Exception as e:
            print(rid, "failed:", e)
    print(f"rendered {done}/{len(need)}")


if __name__ == "__main__":
    main()

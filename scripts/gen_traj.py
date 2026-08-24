"""Stage I data generation (design doc §6.2, Eq. 5-6) on MindCube train.

For each item, the teacher VLM answers from a ladder of evidence states:
  s1   = view 1 only                      (STOP after 0 moves)
  s2   = views 1-2                        (1 MOVE)
  s3   = views 1-3                        (2 MOVEs)
  s4   = all views                        (3 MOVEs)
  s4r  = all views + top-down VGGT render (MOVEs + RENDER/FUSE)
Each state stores the teacher's answer, mean token log-prob, and correctness.
Control labels (which action is best from each state, Eq. 6) are derived
offline from these outcome ladders; the same ladders provide fusion training
sets (complementary / redundant / distractor combinations, §6.5).
"""
import argparse
import json
import os
import re
import sys
import time
import traceback

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viewtree.tree import answer_logprob
from viewtree.vlm import QwenVL

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MC_ROOT = os.path.join(REPO, "data", "mindcube", "data")

PRE = "These images show a scene photographed from different viewpoints.\n"
PRE_R = ("The first {k} images show a scene photographed from different viewpoints. "
         "The final image is a top-down view rendered from a 3D reconstruction "
         "(it may contain holes).\n")
SUFFIX = "\nAnswer with the option's letter from the given choices directly."


def letter(text):
    m = re.search(r"\b([A-F])\b", text.strip())
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-32B-Instruct")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(MC_ROOT, "raw", "MindCube_train.jsonl"))]
    rows = rows[args.shard::args.num_shards]
    if args.limit:
        rows = rows[:args.limit]

    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    rows = [r for r in rows if r["id"] not in done]

    import torch
    from viewtree.reconstruct import reconstruct
    from viewtree.render import overview_poses, render

    vlm = QwenVL(args.model)
    fout = open(args.out, "a")
    t0 = time.time()
    for ri, r in enumerate(rows):
        try:
            imgs = [cv2.cvtColor(cv2.imread(os.path.join(MC_ROOT, p)), cv2.COLOR_BGR2RGB)
                    for p in r["images"]]
            q = r["question"] + SUFFIX
            n = len(imgs)
            states = {}
            for k in range(1, n + 1):
                pred, lp = answer_logprob(vlm, imgs[:k], PRE + q, max_new_tokens=8)
                states[f"s{k}"] = {"pred": pred, "lp": lp,
                                   "correct": letter(pred) == r["gt_answer"]}
            # rendered top-down from all views
            try:
                rec = reconstruct(imgs)
                H, W = rec["size"]
                pose = overview_poses(rec)[-1]
                img = render(rec["points"], rec["colors"], pose,
                             rec["intrinsics"][0], H, W, splat=2)
                td = (img.clamp(0, 1) * 255).byte().cpu().numpy()
                pred, lp = answer_logprob(vlm, imgs + [td],
                                          PRE_R.format(k=n) + q, max_new_tokens=8)
                states[f"s{n}r"] = {"pred": pred, "lp": lp,
                                    "correct": letter(pred) == r["gt_answer"]}
                del rec
                torch.cuda.empty_cache()
            except Exception:
                states[f"s{n}r"] = {"pred": "", "lp": -99.0, "correct": False,
                                    "error": "render_failed"}
            fout.write(json.dumps({
                "id": r["id"], "gt": r["gt_answer"], "n_views": n,
                "category": os.path.dirname(r["images"][0]).split("/")[1]
                if "/" in r["images"][0] else "", "states": states,
            }) + "\n")
            if (ri + 1) % 25 == 0:
                fout.flush()
                el = (time.time() - t0) / 60
                print(f"[traj shard {args.shard}] {ri+1}/{len(rows)} items, "
                      f"{el:.1f} min", flush=True)
        except Exception:
            traceback.print_exc()
    fout.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

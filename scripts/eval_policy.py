"""Greedy control-policy rollout on MindCube tinybench: follow STOP/MOVE/RENDER
decisions, answer where the policy stops, record accuracy and views used.
The doc's H2/H6-style efficiency readout: accuracy vs acquisition cost."""
import argparse
import json
import os
import re
import sys

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
CONTROL_PROMPT = (
    "{pre}Question: {q}\n"
    "You may either answer now or acquire more evidence. Reply with exactly one "
    "word: STOP if the current views are sufficient to answer correctly, MOVE to "
    "view the scene from another side, or RENDER to inspect a reconstructed "
    "top-down view."
)
SUFFIX = "\nAnswer with the option's letter from the given choices directly."


def letter(t):
    m = re.search(r"\b([A-F])\b", t.strip())
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--split", default="MindCube_tinybench")
    args = ap.parse_args()

    import torch
    from viewtree.reconstruct import reconstruct
    from viewtree.render import overview_poses, render

    rows = [json.loads(l) for l in
            open(os.path.join(MC_ROOT, "raw", args.split + ".jsonl"))]
    rows = rows[args.shard::args.num_shards]
    vlm = QwenVL("Qwen/Qwen2.5-VL-7B-Instruct", adapter=args.adapter)
    fout = open(args.out, "w")
    for ri, r in enumerate(rows):
        try:
            imgs = [cv2.cvtColor(cv2.imread(os.path.join(MC_ROOT, p)),
                                 cv2.COLOR_BGR2RGB) for p in r["images"]]
            n = len(imgs)
            k, used_render = 1, False
            actions = []
            while True:
                act, _ = answer_logprob(
                    vlm, imgs[:k], CONTROL_PROMPT.format(pre=PRE, q=r["question"]),
                    max_new_tokens=4)
                act = act.upper()
                if "MOVE" in act and k < n:
                    actions.append("MOVE")
                    k += 1
                    continue
                if "RENDER" in act:
                    actions.append("RENDER")
                    used_render = True
                else:
                    actions.append("STOP")
                break
            if used_render:
                rec = reconstruct(imgs)
                H, W = rec["size"]
                pose = overview_poses(rec)[-1]
                img = render(rec["points"], rec["colors"], pose,
                             rec["intrinsics"][0], H, W, splat=2)
                td = (img.clamp(0, 1) * 255).byte().cpu().numpy()
                pred, _ = answer_logprob(vlm, imgs + [td],
                                         PRE_R.format(k=n) + r["question"] + SUFFIX,
                                         max_new_tokens=8)
                views = n + 1
                del rec
                torch.cuda.empty_cache()
            else:
                pred, _ = answer_logprob(vlm, imgs[:k],
                                         PRE + r["question"] + SUFFIX,
                                         max_new_tokens=8)
                views = k
            fout.write(json.dumps({
                "id": r["id"], "views": views, "actions": actions,
                "correct": letter(pred) == r["gt_answer"],
            }) + "\n")
            if (ri + 1) % 50 == 0:
                fout.flush()
                print(f"{ri+1}/{len(rows)}", flush=True)
        except Exception as e:
            print("skip", r["id"], repr(e)[:80], flush=True)
    fout.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

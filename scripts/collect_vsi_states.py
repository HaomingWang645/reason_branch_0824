"""Collect VSI-domain tree states (features + correctness labels) for
confidence-head domain adaptation.

Scene split: the 288 VSI-Bench scenes sorted; EVEN-indexed scenes are
head-training scenes (collected here), ODD-indexed are reserved for evaluation.
For each question on a head-train scene: direct state, 5 branch states, and the
top-2-fused state, each with the 3584-d last-token feature and the state's own
answer correctness. Uses the same 32-frame/splat-2 recipe as tree inference.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viewtree.data import load_questions, sample_frames
from viewtree.reconstruct import reconstruct
from viewtree.render import overview_poses, render
from viewtree.score import score_row
from viewtree.tree import BRANCH_PRE, FUSE_PRE, VIEW_DESCS, answer_logprob, build_q
from viewtree.vlm import QwenVL

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def head_train_scenes():
    rows = load_questions()
    scenes = sorted({(r["dataset"], r["scene_name"]) for r in rows})
    return scenes[0::2]  # even-indexed: head-train; odd: eval


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--feats-out", required=True)
    args = ap.parse_args()
    os.makedirs(args.feats_out, exist_ok=True)

    rows = load_questions()
    scenes = set(head_train_scenes()[args.shard::args.num_shards])
    by_scene = {}
    for r in rows:
        k = (r["dataset"], r["scene_name"])
        if k in scenes:
            by_scene.setdefault(k, []).append(r)

    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass

    vlm = QwenVL("Qwen/Qwen2.5-VL-7B-Instruct",
                 adapter=os.path.join(REPO, "checkpoints", "sft_lora_v2"))
    fout = open(args.out, "a")
    import time
    t0 = time.time()
    for si, (key, qrows) in enumerate(sorted(by_scene.items())):
        qrows = [r for r in qrows if r["id"] not in done]
        if not qrows:
            continue
        try:
            frames = sample_frames(qrows[0]["video"], 32)
            rec = reconstruct(frames)
            H, W = rec["size"]
            K = rec["intrinsics"][0]
            views = []
            for pose in overview_poses(rec)[:5]:
                img = render(rec["points"], rec["colors"], pose, K, H, W, splat=2)
                views.append((img.clamp(0, 1) * 255).byte().cpu().numpy())
            del rec
            torch.cuda.empty_cache()
        except Exception as e:
            print("scene failed", key, e, flush=True)
            continue
        base = [frames[i] for i in np.linspace(0, 31, 8).round().astype(int)]
        for r in qrows:
            try:
                qtext = build_q(r)
                feats, states = {}, {}
                pred, _, ft = answer_logprob(
                    vlm, base, "These are frames of a video.\n" + qtext,
                    want_feature=True)
                feats["direct"] = ft.half()
                states["direct"] = {"pred": pred, "score": score_row(r, pred)}
                confs = []
                for vi, v in enumerate(views):
                    pre = BRANCH_PRE.format(k=len(base), desc=VIEW_DESCS[vi])
                    pred, lp, ft = answer_logprob(vlm, base + [v], pre + qtext,
                                                  want_feature=True)
                    feats[f"branch{vi}"] = ft.half()
                    states[f"branch{vi}"] = {"pred": pred,
                                             "score": score_row(r, pred)}
                    confs.append(lp)
                kept = sorted(range(5), key=lambda i: -confs[i])[:2]
                kept_views = [views[i] for i in kept]
                descs = ", ".join(VIEW_DESCS[i] for i in kept)
                pre = FUSE_PRE.format(k=len(base), m=2, descs=descs)
                pred, _, ft = answer_logprob(vlm, base + kept_views, pre + qtext,
                                             want_feature=True)
                feats["fused"] = ft.half()
                states["fused"] = {"pred": pred, "score": score_row(r, pred)}
                torch.save(feats, os.path.join(args.feats_out, f"{r['id']}.pt"))
                fout.write(json.dumps({"id": r["id"], "states": states,
                                       "question_type": r["question_type"]}) + "\n")
            except Exception as e:
                print("q failed", r["id"], e, flush=True)
        fout.flush()
        print(f"[collect shard {args.shard}] scene {si+1}/{len(by_scene)} "
              f"{(time.time()-t0)/60:.1f} min", flush=True)
    fout.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

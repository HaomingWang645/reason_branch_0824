"""Evaluate one condition on a shard of VSI-Bench (sharded by scene).

Conditions:
  current   - last video frame only
  frames16  - 16 uniformly sampled frames
  memory    - 12 frames + 5 rendered overview views from frozen VGGT reconstruction
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viewtree.data import load_questions, sample_frames
from viewtree.score import score_row
from viewtree.vlm import QwenVL, build_prompt

FRAMES_PRE = "These are frames of a video.\n"
MEM_PRE = (
    "The first {k} images are frames of a video captured while moving through a room. "
    "The last {m} images are novel overview views of the SAME room, rendered from a 3D "
    "reconstruction of the video (they may contain holes or distortion; the final one "
    "is a top-down view).\n"
)


def get_memory_views(scene_key, frames, args, splat=1, cache_dir=None):
    """Render (and cache) overview views for a scene."""
    import torch
    from viewtree.reconstruct import reconstruct
    from viewtree.render import overview_poses, render

    cache = os.path.join(cache_dir or args.render_cache, scene_key)
    paths = [os.path.join(cache, f"view{i}.png") for i in range(5)]
    if all(os.path.exists(p) for p in paths):
        import cv2
        return [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths]
    rec = reconstruct(frames)
    H, W = rec["size"]
    K = rec["intrinsics"][0].clone()
    views = []
    os.makedirs(cache, exist_ok=True)
    for i, pose in enumerate(overview_poses(rec)):
        img = render(rec["points"], rec["colors"], pose, K, H, W, splat=splat)
        img8 = (img.clamp(0, 1) * 255).byte().cpu().numpy()
        views.append(img8)
        import cv2
        cv2.imwrite(paths[i], cv2.cvtColor(img8, cv2.COLOR_RGB2BGR))
    del rec
    torch.cuda.empty_cache()
    return views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True,
                    choices=["current", "frames16", "frames12", "memory",
                             "renders_only", "memory32"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit-scenes", type=int, default=0)
    ap.add_argument("--render-cache", default=None)
    ap.add_argument("--parity", choices=["all", "even", "odd"], default="all")
    args = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.render_cache is None:
        args.render_cache = os.path.join(repo, "data", "renders")

    rows = load_questions()
    scenes = sorted({(r["dataset"], r["scene_name"]) for r in rows})
    if args.parity == "even":
        scenes = scenes[0::2]
    elif args.parity == "odd":
        scenes = scenes[1::2]
    scenes = scenes[args.shard::args.num_shards]
    if args.limit_scenes:
        scenes = scenes[:args.limit_scenes]
    scene_set = set(scenes)
    by_scene = {}
    for r in rows:
        k = (r["dataset"], r["scene_name"])
        if k in scene_set:
            by_scene.setdefault(k, []).append(r)

    done_ids = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try:
                done_ids.add(json.loads(l)["id"])
            except Exception:
                pass

    vlm = QwenVL(args.model, adapter=args.adapter)
    fout = open(args.out, "a")
    t0 = time.time()
    n_done = 0
    for si, (key, qrows) in enumerate(sorted(by_scene.items())):
        qrows = [r for r in qrows if r["id"] not in done_ids]
        if not qrows:
            continue
        scene_key = f"{key[0]}_{key[1]}"
        try:
            n_src = 32 if args.condition == "memory32" else 16
            frames = sample_frames(qrows[0]["video"], n_src)
            if args.condition == "current":
                images, pre = [frames[-1]], ""
            elif args.condition == "frames16":
                images, pre = frames, FRAMES_PRE
            elif args.condition == "frames12":
                idx = np.linspace(0, 15, 12).round().astype(int)
                images, pre = [frames[i] for i in idx], FRAMES_PRE
            elif args.condition == "renders_only":
                views = get_memory_views(scene_key, frames, args)
                images = views
                pre = ("These are views of a room rendered from a 3D reconstruction "
                       "(they may contain holes or distortion; the final one is a "
                       "top-down view).\n")
            elif args.condition == "memory32":
                repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                views = get_memory_views(scene_key, frames, args, splat=2,
                                         cache_dir=os.path.join(repo, "data", "renders32"))
                sel = [frames[i] for i in np.linspace(0, 31, 12).round().astype(int)]
                images = sel + views
                pre = MEM_PRE.format(k=len(sel), m=len(views))
            else:
                views = get_memory_views(scene_key, frames, args)
                sel = [frames[i] for i in np.linspace(0, 15, 12).round().astype(int)]
                images = sel + views
                pre = MEM_PRE.format(k=len(sel), m=len(views))
        except Exception:
            traceback.print_exc()
            for r in qrows:
                fout.write(json.dumps({"id": r["id"], "pred": "", "score": 0.0,
                                       "error": "scene_failed"}) + "\n")
            fout.flush()
            continue
        for r in qrows:
            try:
                pred = vlm.ask(images, build_prompt(r, pre))
            except Exception:
                traceback.print_exc()
                pred = ""
            s = score_row(r, pred)
            fout.write(json.dumps({"id": r["id"], "pred": pred, "score": s,
                                   "question_type": r["question_type"]}) + "\n")
            n_done += 1
        fout.flush()
        el = time.time() - t0
        print(f"[{args.condition} shard {args.shard}] scene {si+1}/{len(by_scene)} "
              f"({scene_key}) done, {n_done} q, {el/60:.1f} min", flush=True)
    fout.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

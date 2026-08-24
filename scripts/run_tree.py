"""End-to-end ViewTree-lite evaluation on VSI-Bench (sharded by scene)."""
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
from viewtree.tree import run_tree
from viewtree.vlm import QwenVL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--conf-head", default=None)
    ap.add_argument("--num-frames", type=int, default=32)
    ap.add_argument("--keep-k", type=int, default=2)
    ap.add_argument("--limit-scenes", type=int, default=0)
    args = ap.parse_args()

    rows = load_questions()
    scenes = sorted({(r["dataset"], r["scene_name"]) for r in rows})
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

    import torch
    from viewtree.reconstruct import reconstruct

    vlm = QwenVL(args.model, adapter=args.adapter)
    conf = None
    if args.conf_head:
        from viewtree.tree import load_conf_head
        conf = load_conf_head(args.conf_head)
    fout = open(args.out, "a")
    t0 = time.time()
    for si, (key, qrows) in enumerate(sorted(by_scene.items())):
        qrows = [r for r in qrows if r["id"] not in done_ids]
        if not qrows:
            continue
        try:
            frames = sample_frames(qrows[0]["video"], args.num_frames)
            rec = reconstruct(frames)
        except Exception:
            traceback.print_exc()
            for r in qrows:
                fout.write(json.dumps({"id": r["id"], "pred": "", "score": 0.0,
                                       "error": "scene_failed"}) + "\n")
            fout.flush()
            continue
        for r in qrows:
            try:
                pred, trace = run_tree(vlm, r, frames, rec, keep_k=args.keep_k, conf=conf)
            except Exception:
                traceback.print_exc()
                pred, trace = "", {"mode": "error"}
            s = score_row(r, pred)
            fout.write(json.dumps({
                "id": r["id"], "pred": pred, "score": s,
                "question_type": r["question_type"], "mode": trace.get("mode"),
                "gate": trace.get("gate"),
            }) + "\n")
        fout.flush()
        del rec
        torch.cuda.empty_cache()
        print(f"[tree shard {args.shard}] scene {si+1}/{len(by_scene)} "
              f"({key[0]}_{key[1]}) {(time.time()-t0)/60:.1f} min", flush=True)
    fout.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

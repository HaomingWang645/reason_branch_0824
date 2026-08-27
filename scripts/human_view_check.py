"""Audit legacy vs human-constrained viewpoints on VSI odd-half scenes:
geometry (inside walked hull / room extent, height, pitch, roll) and render
coverage; save side-by-side example figures. Usage:
  python scripts/human_view_check.py --scenes 30 --out results/human_views
"""
import argparse, json, os, sys
import cv2, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viewtree.data import load_questions, sample_frames
from viewtree.reconstruct import reconstruct
from viewtree.render import human_poses, legacy_poses, pose_report, render

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scenes", type=int, default=30); ap.add_argument("--out", required=True); ap.add_argument("--examples", type=int, default=6)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rows = load_questions(); scenes = sorted({(r["dataset"], r["scene_name"]) for r in rows})[1::2]
    vid = {(r["dataset"], r["scene_name"]): r["video"] for r in rows}
    rng = np.random.default_rng(0); pick = [scenes[i] for i in rng.choice(len(scenes), a.scenes, replace=False)]
    fout = open(os.path.join(a.out, "audit.jsonl"), "w")
    for si, key in enumerate(pick):
        frames = sample_frames(vid[key], 32); rec = reconstruct(frames); H, W = rec["size"]; K = rec["intrinsics"][0]
        for kind, fn in [("legacy", legacy_poses), ("human", human_poses)]:
            poses = fn(rec); rep = pose_report(rec, poses)
            for vi, (pose, r) in enumerate(zip(poses, rep)):
                img = render(rec["points"], rec["colors"], pose, K, H, W, splat=2)
                r.update(scene=f"{key[0]}/{key[1]}", kind=kind, view=vi, coverage=float((img.min(-1).values < 0.999).float().mean()))
                if kind == "human": r["proposer_kind"] = rec["pose_meta"][vi]["kind"]
                fout.write(json.dumps(r) + "\n")
                if si < a.examples:
                    cv2.imwrite(os.path.join(a.out, f"{si}_{kind}_view{vi}.jpg"), cv2.cvtColor((img.clamp(0, 1) * 255).byte().cpu().numpy(), cv2.COLOR_RGB2BGR))
        if si < a.examples:
            for i in range(4): cv2.imwrite(os.path.join(a.out, f"{si}_frame{i}.jpg"), cv2.cvtColor(frames[i * 10], cv2.COLOR_RGB2BGR))
        fout.flush(); print(si + 1, key, flush=True); del rec; torch.cuda.empty_cache()
    print("DONE")

if __name__ == "__main__":
    main()

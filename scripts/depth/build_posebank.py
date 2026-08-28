"""Phase 0: reconstruct each training video once and render its pose bank.
Output per scene: data/posebank/<src>/<scene>/{bank.json, frames_XX.jpg (8 ctx frames), view_NNN.jpg}.
  python scripts/depth/build_posebank.py --manifest data/train3r/manifest.jsonl --shard 0 --num-shards 8"""
import argparse, json, os, sys, time
import cv2, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from viewtree.data import sample_frames
from viewtree.reconstruct import reconstruct
from viewtree.render import render
from viewtree.posebank import build_pose_bank
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", required=True); ap.add_argument("--out", default=os.path.join(REPO, "data", "posebank"))
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1); ap.add_argument("--frames", type=int, default=32); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    scenes = sorted({(json.loads(l)["source"], json.loads(l)["scene"], json.loads(l)["video"]) for l in open(a.manifest)})[a.shard::a.num_shards]
    if a.limit: scenes = scenes[: a.limit]
    t0 = time.time(); n = 0
    for src, scene, video in scenes:
        od = os.path.join(a.out, src, scene)
        if os.path.exists(os.path.join(od, "bank.json")): continue
        try:
            frames = sample_frames(video, a.frames); rec = reconstruct(frames); H, W = rec["size"]; K = rec["intrinsics"][0]
            bank, fwd_map, meta = build_pose_bank(rec)
            os.makedirs(od, exist_ok=True)
            for i, fi in enumerate(np.linspace(0, len(frames) - 1, 8).round().astype(int)):
                cv2.imwrite(os.path.join(od, f"frame_{i:02d}.jpg"), cv2.cvtColor(frames[fi], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
            for e in bank:
                img = render(rec["points"], rec["colors"], torch.tensor(e["extrinsic"], device=rec["points"].device), K, H, W, splat=2)
                cv2.imwrite(os.path.join(od, f"view_{e['idx']:03d}.jpg"), cv2.cvtColor((img.clamp(0, 1) * 255).byte().cpu().numpy(), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            json.dump(dict(bank=bank, fwd_map={str(k): v for k, v in fwd_map.items()}, meta=meta, video=video, n_frames=a.frames), open(os.path.join(od, "bank.json"), "w"))
            del rec; torch.cuda.empty_cache(); n += 1
            if n % 10 == 0: print(f"[s{a.shard}] {n} scenes {(time.time()-t0)/60:.1f} min", flush=True)
        except Exception as e:
            print("skip", src, scene, repr(e)[:120], flush=True); torch.cuda.empty_cache()
    print("DONE", flush=True)

if __name__ == "__main__":
    main()

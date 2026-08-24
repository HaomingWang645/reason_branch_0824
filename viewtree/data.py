"""VSI-Bench loading and frame sampling."""
import json
import os

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VSI_SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--nyu-visionx--VSI-Bench/snapshots/"
    "d7cb1a3960b79dd3e20d4990b83005e96e1bcd9d"
)
VIDEO_ROOT = os.path.join(REPO, "data", "videos")

NUMERICAL_TYPES = {
    "object_counting",
    "object_size_estimation",
    "object_abs_distance",
    "room_size_estimation",
}


def load_questions():
    rows = [json.loads(l) for l in open(os.path.join(VSI_SNAP, "test.jsonl"))]
    for r in rows:
        r["video"] = os.path.join(VIDEO_ROOT, r["dataset"], f"{r['scene_name']}.mp4")
    return rows


def sample_frames(video_path, num_frames):
    """Uniformly sample RGB frames (H, W, 3) uint8 from a video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"cannot read {video_path}")
    idxs = np.linspace(0, total - 1, num_frames).round().astype(int)
    frames, want = [], set(idxs.tolist())
    order = sorted(want)
    pos = 0
    for i in range(total):
        ok = cap.grab()
        if not ok:
            break
        if pos < len(order) and i == order[pos]:
            ok, im = cap.retrieve()
            if not ok:
                break
            frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            pos += 1
        if pos >= len(order):
            break
    cap.release()
    if len(frames) < len(order):
        raise RuntimeError(f"only decoded {len(frames)}/{len(order)} frames of {video_path}")
    # replicate frames for duplicate indices (very short videos)
    lookup = {i: f for i, f in zip(order, frames)}
    return [lookup[i] for i in idxs.tolist()]

"""Capture full reasoning traces (frames, branch views, confidences, decisions)
for a few VSI-Bench questions, for visualization."""
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viewtree.data import load_questions, sample_frames
from viewtree.reconstruct import reconstruct
from viewtree.render import overview_poses, render
from viewtree.score import score_row
from viewtree.tree import (BRANCH_PRE, FUSE_PRE, ROOT_GATE, VIEW_DESCS,
                           answer_logprob, build_q, load_conf_head)
from viewtree.vlm import QwenVL

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "traces")
IDS = [int(x) for x in sys.argv[1:]] or [1313, 5148, 1257, 1352, 1299, 5136]


def save(img, name):
    cv2.imwrite(os.path.join(OUT, name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    meta = {r["id"]: r for r in load_questions()}
    vlm = QwenVL("Qwen/Qwen2.5-VL-7B-Instruct",
                 adapter=os.path.join(REPO, "checkpoints", "sft_lora_v2"))
    conf = load_conf_head(os.path.join(REPO, "checkpoints", "conf_head.pt"))

    def sc_answer(images, prompt):
        pred, lp, ft = answer_logprob(vlm, images, prompt, want_feature=True)
        return pred, conf(ft)

    traces = []
    for qid in IDS:
        row = meta[qid]
        frames = sample_frames(row["video"], 32)
        rec = reconstruct(frames)
        H, W = rec["size"]
        K = rec["intrinsics"][0]
        qtext = build_q(row)
        base = [frames[i] for i in np.linspace(0, 31, 8).round().astype(int)]
        t = {"id": qid, "question": row["question"], "options": row["options"],
             "gt": row["ground_truth"], "qtype": row["question_type"],
             "scene": f"{row['dataset']}/{row['scene_name']}"}
        for i, f in enumerate(base[:4]):
            save(f, f"{qid}_frame{i}.jpg")
        gate, _ = answer_logprob(vlm, base, ROOT_GATE.format(q=row["question"]),
                                 max_new_tokens=4)
        t["gate"] = gate.strip()
        dpred, dconf = sc_answer(base, "These are frames of a video.\n" + qtext)
        t["direct"] = {"pred": dpred, "conf": round(dconf, 3)}
        poses = overview_poses(rec)[:5]
        views, branches = [], []
        for vi, pose in enumerate(poses):
            img = render(rec["points"], rec["colors"], pose, K, H, W, splat=2)
            v8 = (img.clamp(0, 1) * 255).byte().cpu().numpy()
            views.append(v8)
            save(v8, f"{qid}_view{vi}.jpg")
            pre = BRANCH_PRE.format(k=len(base), desc=VIEW_DESCS[vi])
            bpred, bconf = sc_answer(base + [v8], pre + qtext)
            branches.append({"view": vi, "desc": VIEW_DESCS[vi],
                             "pred": bpred, "conf": round(bconf, 3)})
        t["branches"] = branches
        order = sorted(range(5), key=lambda i: -branches[i]["conf"])
        kept = order[:2]
        t["kept"] = kept
        preds = {branches[i]["pred"].strip() for i in kept}
        if len(preds) == 1 and branches[kept[0]]["conf"] > dconf:
            t["mode"] = "branch_consensus"
            t["final"] = branches[kept[0]]["pred"]
        else:
            kept_views = [views[branches[i]["view"]] for i in kept]
            descs = ", ".join(VIEW_DESCS[branches[i]["view"]] for i in kept)
            pre = FUSE_PRE.format(k=len(base), m=2, descs=descs)
            fpred, fconf = sc_answer(base + kept_views, pre + qtext)
            t["fuse"] = {"pred": fpred, "conf": round(fconf, 3)}
            if dconf > fconf and dconf > max(branches[i]["conf"] for i in kept):
                t["mode"] = "fused_fallback_direct"
                t["final"] = dpred
            else:
                t["mode"] = "fused"
                t["final"] = fpred
        t["score"] = score_row(row, t["final"])
        traces.append(t)
        print(qid, t["mode"], "final=", t["final"], "gt=", t["gt"],
              "score=", t["score"], flush=True)
        del rec
        torch.cuda.empty_cache()
    old = []
    tj = os.path.join(OUT, "traces.json")
    if os.path.exists(tj):
        old = [t for t in json.load(open(tj)) if t["id"] not in {x["id"] for x in traces}]
    json.dump(old + traces, open(tj, "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()

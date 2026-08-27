"""Capture full reasoning traces (frames, gate, branch views, head confidences,
decisions) for a few VSI-Bench questions with a chosen adapter + confidence
head, following run_tree's exact decision rule (gate YES -> direct; branch
consensus; fuse; fallback). Branches are rendered even when the gate stops
early so the figure can show what the policy chose NOT to look at.
  python scripts/trace_examples_v2.py --adapter CK --conf-head H --tag d10k IDS...
"""
import argparse, json, os, sys
import cv2, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viewtree.data import load_questions, sample_frames
from viewtree.reconstruct import reconstruct
from viewtree.render import overview_poses, render
from viewtree.score import score_row
from viewtree.tree import (BRANCH_PRE, FUSE_PRE, ROOT_GATE, VIEW_DESCS,
                           answer_logprob, build_q, load_conf_head)
from viewtree.vlm import QwenVL
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True); ap.add_argument("--conf-head", required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("ids", type=int, nargs="+")
    a = ap.parse_args()
    OUT = os.path.join(REPO, "results", "traces", a.tag); os.makedirs(OUT, exist_ok=True)
    save = lambda img, n: cv2.imwrite(os.path.join(OUT, n), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    meta = {r["id"]: r for r in load_questions()}
    vlm = QwenVL("Qwen/Qwen2.5-VL-7B-Instruct", adapter=a.adapter)
    conf = load_conf_head(a.conf_head)
    def sc(images, prompt):
        pred, lp, ft = answer_logprob(vlm, images, prompt, want_feature=True)
        return pred, float(conf(ft))
    traces = []
    for qid in a.ids:
        row = meta[qid]; frames = sample_frames(row["video"], 32); rec = reconstruct(frames)
        H, W = rec["size"]; K = rec["intrinsics"][0]; qtext = build_q(row)
        base = [frames[i] for i in np.linspace(0, 31, 8).round().astype(int)]
        t = {"id": qid, "question": row["question"], "options": row["options"], "gt": row["ground_truth"],
             "qtype": row["question_type"], "scene": f"{row['dataset']}/{row['scene_name']}"}
        for i, f in enumerate(base[:4]): save(f, f"{qid}_frame{i}.jpg")
        gate, _ = answer_logprob(vlm, base, ROOT_GATE.format(q=row["question"]), max_new_tokens=4)
        t["gate"] = gate.strip()
        dpred, dconf = sc(base, "These are frames of a video.\n" + qtext)
        t["direct"] = {"pred": dpred, "conf": round(dconf, 3)}
        views, branches = [], []
        for vi, pose in enumerate(overview_poses(rec)[:5]):
            v8 = (render(rec["points"], rec["colors"], pose, K, H, W, splat=2).clamp(0, 1) * 255).byte().cpu().numpy()
            views.append(v8); save(v8, f"{qid}_view{vi}.jpg")
            bpred, bconf = sc(base + [v8], BRANCH_PRE.format(k=len(base), desc=VIEW_DESCS[vi]) + qtext)
            branches.append({"view": vi, "desc": VIEW_DESCS[vi], "pred": bpred, "conf": round(bconf, 3)})
        t["branches"] = branches
        order = sorted(range(5), key=lambda i: -branches[i]["conf"]); kept = order[:2]; t["kept"] = kept
        kept_views = [views[i] for i in kept]
        fpred, fconf = sc(base + kept_views, FUSE_PRE.format(k=len(base), m=2, descs=", ".join(VIEW_DESCS[i] for i in kept)) + qtext)
        t["fuse"] = {"pred": fpred, "conf": round(fconf, 3)}
        preds = {branches[i]["pred"].strip() for i in kept}
        if "YES" in gate.upper():
            t["mode"], t["final"], t["executed"] = "direct", dpred, "gate=YES: answered from video frames; branches shown were NOT executed"
        elif len(preds) == 1 and branches[kept[0]]["conf"] > dconf:
            t["mode"], t["final"], t["executed"] = "branch_consensus", branches[kept[0]]["pred"], "kept branches agree and beat direct: early stop, fusion NOT executed"
        elif dconf > fconf and dconf > max(branches[i]["conf"] for i in kept):
            t["mode"], t["final"], t["executed"] = "fused_fallback_direct", dpred, "fused, but head ranks direct above fused and branches: fall back to direct"
        else:
            t["mode"], t["final"], t["executed"] = "fused", fpred, "fused answer from 2 kept views selected"
        t["score"] = score_row(row, t["final"]); traces.append(t)
        print(qid, t["gate"], t["mode"], "final=", t["final"], "gt=", t["gt"], "score=", t["score"], flush=True)
        del rec; torch.cuda.empty_cache()
    json.dump(traces, open(os.path.join(OUT, "traces.json"), "w"), indent=1); print("DONE")

if __name__ == "__main__":
    main()

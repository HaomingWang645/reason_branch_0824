"""Run the depth-1 reasoning tree on OST-Bench samples (image history ->
VGGT reconstruction -> constrained branch views -> prune/fuse/arbitrate) and
save traces in the same schema as trace_examples_v2 so tree_diagram.py can
draw them.  python scripts/trace_ost.py --adapter CK --conf-head H --tag ost --n 6"""
import argparse, json, os, re, sys
import cv2, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_external import load_ost, MC_SUFFIX
from viewtree.reconstruct import reconstruct
from viewtree.render import overview_poses, render
from viewtree.tree import BRANCH_PRE, FUSE_PRE, ROOT_GATE, VIEW_DESCS, answer_logprob, load_conf_head
from viewtree.vlm import QwenVL
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def letter(t):
    m = re.search(r"\b([A-H])\b", t.strip()); return m.group(1) if m else ""

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True); ap.add_argument("--conf-head", required=True)
    ap.add_argument("--tag", default="ost"); ap.add_argument("--n", type=int, default=6); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--ids", nargs="*", default=None)
    a = ap.parse_args()
    OUT = os.path.join(REPO, "results", "traces", a.tag); os.makedirs(OUT, exist_ok=True)
    save = lambda img, n: cv2.imwrite(os.path.join(OUT, n), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    items = [it for it in load_ost() if len(it[1]) >= 6]
    rng = np.random.default_rng(a.seed)
    by_type = {}
    for it in items: by_type.setdefault(it[4], []).append(it)
    picks = []
    if a.ids:
        byid = {it[0]: it for it in load_ost()}; picks = [byid[i] for i in a.ids]
    while len(picks) < a.n:
        for t in sorted(by_type):
            if len(picks) < a.n: picks.append(by_type[t][int(rng.integers(len(by_type[t])))])
    vlm = QwenVL("Qwen/Qwen2.5-VL-7B-Instruct", adapter=a.adapter); conf = load_conf_head(a.conf_head)
    def sc(images, prompt):
        pred, lp, ft = answer_logprob(vlm, images, prompt, want_feature=True, max_new_tokens=8)
        return pred, float(conf(ft))
    traces = []
    for qid, imgs, prompt, gt, qtype in picks:
        rec = reconstruct(imgs); H, W = rec["size"]; K = rec["intrinsics"][0]
        base = [imgs[i] for i in np.linspace(0, len(imgs) - 1, min(8, len(imgs))).round().astype(int)]
        qtext = prompt.split("\n")[1] if prompt.startswith("These images") else prompt.split("\n")[0]
        t = {"id": qid, "question": qtext, "options": prompt, "gt": gt, "qtype": qtype, "scene": "OST-Bench"}
        for i, f in enumerate(base[:4]): save(f, f"{qid}_frame{i}.jpg")
        gate, _ = answer_logprob(vlm, base, ROOT_GATE.format(q=qtext), max_new_tokens=4); t["gate"] = gate.strip()
        dpred, dconf = sc(base, prompt); t["direct"] = {"pred": letter(dpred) or dpred, "conf": round(dconf, 3)}
        views, branches = [], []
        for vi, pose in enumerate(overview_poses(rec)[:5]):
            v8 = (render(rec["points"], rec["colors"], pose, K, H, W, splat=2).clamp(0, 1) * 255).byte().cpu().numpy()
            views.append(v8); save(v8, f"{qid}_view{vi}.jpg")
            bpred, bconf = sc(base + [v8], BRANCH_PRE.format(k=len(base), desc=VIEW_DESCS[vi]) + prompt)
            branches.append({"view": vi, "desc": VIEW_DESCS[vi], "pred": letter(bpred) or bpred, "conf": round(bconf, 3)})
        t["branches"] = branches
        order = sorted(range(5), key=lambda i: -branches[i]["conf"]); kept = order[:2]; t["kept"] = kept
        fpred, fconf = sc(base + [views[i] for i in kept], FUSE_PRE.format(k=len(base), m=2, descs=", ".join(VIEW_DESCS[i] for i in kept)) + prompt)
        t["fuse"] = {"pred": letter(fpred) or fpred, "conf": round(fconf, 3)}
        preds = {branches[i]["pred"] for i in kept}
        if "YES" in gate.upper(): t["mode"], t["final"], t["executed"] = "direct", t["direct"]["pred"], "gate=YES: answered from the observed images; branches NOT executed"
        elif len(preds) == 1 and branches[kept[0]]["conf"] > dconf: t["mode"], t["final"], t["executed"] = "branch_consensus", branches[kept[0]]["pred"], "kept branches agree and beat direct: early stop"
        elif dconf > fconf and dconf > max(branches[i]["conf"] for i in kept): t["mode"], t["final"], t["executed"] = "fused_fallback_direct", t["direct"]["pred"], "head ranks direct above fused and branches: fall back"
        else: t["mode"], t["final"], t["executed"] = "fused", t["fuse"]["pred"], "fused answer from 2 kept views"
        t["score"] = float(t["final"] == gt); traces.append(t)
        print(qid, qtype, t["gate"], t["mode"], "final", t["final"], "gt", gt, "score", t["score"], flush=True)
        del rec; torch.cuda.empty_cache()
    json.dump(traces, open(os.path.join(OUT, "traces.json"), "w"), indent=1); print("DONE")

if __name__ == "__main__":
    main()

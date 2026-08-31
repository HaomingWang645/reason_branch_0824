"""Trace ViewTree-D beam search on selected VSI questions: records every explored state
(path, answer, value), the beam/kept sets per level, and saves the rendered views + context
frames to figures/treeD_trace/<id>/.  Same logic as run_tree_d.py.
  python scripts/depth/trace_tree_d.py --adapter CK --value-head H --ids results/depth/trace_ids.json --out results/depth/treeD_trace.jsonl"""
import argparse, json, os, sys, time
import cv2, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viewtree.data import load_questions, sample_frames
from viewtree.reconstruct import reconstruct
from viewtree.render import render
from viewtree.posebank import build_pose_bank, transition
from viewtree.score import score_row
from viewtree.tree import ROOT_GATE, answer_logprob, build_q, load_conf_head
from viewtree.vlm import QwenVL
from build_phase1 import ACTIONS, PRE, WALK, describe
from train_sft_c import CTRL
from run_tree_d import action_scores
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", required=True); ap.add_argument("--value-head", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ids", required=True); ap.add_argument("--figdir", default=os.path.join(REPO, "figures/treeD_trace"))
    ap.add_argument("--beam", type=int, default=3); ap.add_argument("--keep", type=int, default=2); ap.add_argument("--depth", type=int, default=3); ap.add_argument("--num-frames", type=int, default=32)
    a = ap.parse_args()
    want = json.load(open(a.ids)); want = {i for v in want.values() for i in v} if isinstance(want, dict) else set(want)
    rows = [r for r in load_questions() if r["id"] in want]
    by_scene = {}
    for r in rows: by_scene.setdefault((r["dataset"], r["scene_name"]), []).append(r)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out): done.add(json.loads(l)["id"])
    vlm = QwenVL("Qwen/Qwen2.5-VL-7B-Instruct", adapter=a.adapter); head = load_conf_head(a.value_head)
    fout = open(a.out, "a"); t0 = time.time()
    for key, qrows in by_scene.items():
        qrows = [r for r in qrows if r["id"] not in done]
        if not qrows: continue
        frames = sample_frames(qrows[0]["video"], a.num_frames); rec = reconstruct(frames); H, W = rec["size"]; K = rec["intrinsics"][0]
        bank, fwd, meta = build_pose_bank(rec, render_all=False)
        for e in bank: e["valid"] = True
        base = [frames[i] for i in np.linspace(0, len(frames) - 1, 8).round().astype(int)]
        rcache = {}
        def view(i):
            if i not in rcache:
                img = render(rec["points"], rec["colors"], torch.tensor(bank[i]["extrinsic"], device=rec["points"].device), K, H, W, splat=2)
                cov = float((img.min(-1).values < 0.999).float().mean()); bank[i]["valid"] = cov >= 0.45
                rcache[i] = (img.clamp(0, 1) * 255).byte().cpu().numpy()
            return rcache[i]
        for r in qrows:
            fd = os.path.join(a.figdir, str(r["id"])); os.makedirs(fd, exist_ok=True)
            for j, im in enumerate(base): cv2.imwrite(os.path.join(fd, f"frame{j}.jpg"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            qtext = build_q(r); calls = 0; states = []; levels = []
            def answer(path):
                nonlocal calls; calls += 1
                renders = [view(i) for _, i in path]
                prompt = PRE.format(k=8) + (WALK.format(m=len(renders), steps=describe(bank, path)) if renders else "") + qtext
                pred, lp, ft = answer_logprob(vlm, base + renders, prompt, max_new_tokens=12, want_feature=True); return pred, float(head(ft))
            gate, _ = answer_logprob(vlm, base, ROOT_GATE.format(q=r["question"]), max_new_tokens=4); calls += 1
            dpred, dconf = answer([])
            trace = {"gate": gate.strip(), "direct": dpred.strip(), "dconf": round(dconf, 3)}
            if "YES" in gate.upper():
                final, mode, best_path = dpred, "direct", []
            else:
                beam = [[("start at spot", e["idx"])] for e in bank if e["kind"] == "eye" and e["yaw"] == 0][: a.beam]
                scored = {}; final, mode, best_path = None, None, None
                for d in range(1, a.depth + 1):
                    for path in beam:
                        key2 = tuple(i for _, i in path)
                        if key2 in scored: continue
                        if not all(bank[i]["valid"] for _, i in path if view(i) is not None): scored[key2] = (None, -1.0, path); continue
                        p, c = answer(path); scored[key2] = (p, c, path)
                    level = sorted([scored[tuple(i for _, i in p)] for p in beam], key=lambda z: -z[1])
                    kept = [z for z in level if z[0] is not None][: a.keep]
                    lev = {"depth": d, "beam": [{"path": [(x, int(i)) for x, i in p], "answer": scored[tuple(i for _, i in p)][0], "value": round(scored[tuple(i for _, i in p)][1], 3),
                                                "kept": any(z[2] is p for z in kept), "valid": scored[tuple(i for _, i in p)][0] is not None} for p in beam], "proposals": []}
                    levels.append(lev)
                    if not kept: break
                    if len(kept) == a.keep and len({z[0].strip() for z in kept}) == 1 and kept[0][1] > dconf:
                        final, mode, best_path = kept[0][0], f"consensus_d{d}", kept[0][2]; break
                    if d == a.depth: break
                    nxt = []
                    for _, _, path in kept:
                        cur = path[-1][1]
                        if bank[cur]["kind"] == "topdown": continue
                        valid = [x for x in ACTIONS if transition(bank, fwd, cur, x, meta) is not None and (x != "BIRD_EYE" or d == a.depth - 1)]
                        if not valid: continue
                        renders = [view(i) for _, i in path]
                        prompt = CTRL.format(pre=PRE.format(k=8), walk=WALK.format(m=len(renders), steps=describe(bank, path)), q=r["question"], valid=", ".join(["STOP"] + valid))
                        sc = action_scores(vlm, base + renders, prompt, ["STOP"] + valid); calls += 1
                        top = sorted(valid, key=lambda x: -sc[x])[: a.beam]
                        lev["proposals"].append({"from": [(x, int(i)) for x, i in path], "scores": {k: round(v, 2) for k, v in sc.items()}, "top": top})
                        for act in top:
                            j = transition(bank, fwd, cur, act, meta)
                            if j is not None and j not in [i for _, i in path]: nxt.append(path + [(act, j)])
                    if not nxt: break
                    beam = nxt
                if final is None:
                    best = max(scored.values(), key=lambda z: z[1]) if scored else (None, -1, [])
                    if best[0] is not None and best[1] > dconf: final, mode, best_path = best[0], "best_state", best[2]
                    else: final, mode, best_path = dpred, "fallback_direct", []
            for i in rcache: cv2.imwrite(os.path.join(fd, f"view{i}.jpg"), cv2.cvtColor(rcache[i], cv2.COLOR_RGB2BGR))
            desc = {int(i): (describe(bank, [("", i)]).split("-> ")[1]) for i in rcache}
            trace.update(mode=mode, path=[(act, int(i)) for act, i in (best_path or [])], calls=calls, depth=len(best_path or []), levels=levels, views=desc,
                         question=r["question"], gt=str(r.get("answer", r.get("ground_truth", ""))), question_type=r["question_type"], scene=r["scene_name"])
            s = score_row(r, final)
            fout.write(json.dumps({"id": r["id"], "pred": final.strip(), "score": s, **trace}) + "\n"); fout.flush()
            print(f"id {r['id']} {mode} depth {len(best_path or [])} score {s} calls {calls}", flush=True)
        del rec; torch.cuda.empty_cache(); print(f"scene {key} done {(time.time()-t0)/60:.1f} min", flush=True)
    fout.close(); print("DONE", flush=True)

if __name__ == "__main__":
    main()

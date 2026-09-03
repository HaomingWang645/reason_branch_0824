"""ViewTree-D inference: beam search over camera walks on VSI-Bench (odd half by default).
Test-time pose bank is built lazily (poses computed once, renders only for visited entries).
At each level the controller proposes actions (top-b by logit over the valid set), the answerer
scores each new state with the value head; keep k; stop on consensus; final arbitration vs direct.
  python scripts/depth/run_tree_d.py --adapter CK --value-head H --parity odd --shard 0 --num-shards 4 --out results/depth/vsi_s0.jsonl"""
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
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@torch.no_grad()
def action_scores(vlm, images, prompt, valid):
    """Log-prob of each valid action's first token given the control prompt."""
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    content = [{"type": "image", "image": Image.fromarray(im), "min_pixels": 224 * 224, "max_pixels": 448 * 448} for im in images]
    content.append({"type": "text", "text": prompt}); messages = [{"role": "user", "content": content}]
    text = vlm.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True); image_inputs, _ = process_vision_info(messages)
    inp = vlm.processor(text=[text], images=image_inputs, return_tensors="pt").to(vlm.device)
    logits = vlm.model(**inp).logits[0, -1].float().log_softmax(-1)
    tok = vlm.processor.tokenizer; out = {}
    for v in valid:
        ids = tok.encode(v, add_special_tokens=False); out[v] = float(logits[ids[0]])
    return out

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", required=True); ap.add_argument("--value-head", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--parity", choices=["all", "even", "odd"], default="odd"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--beam", type=int, default=3); ap.add_argument("--keep", type=int, default=2); ap.add_argument("--depth", type=int, default=3); ap.add_argument("--limit-scenes", type=int, default=0)
    ap.add_argument("--num-frames", type=int, default=32)
    ap.add_argument("--policy", choices=["ctrl", "random"], default="ctrl")
    a = ap.parse_args()
    rows = load_questions(); scenes = sorted({(r["dataset"], r["scene_name"]) for r in rows})
    if a.parity == "even": scenes = scenes[0::2]
    elif a.parity == "odd": scenes = scenes[1::2]
    scenes = scenes[a.shard::a.num_shards]
    if a.limit_scenes: scenes = scenes[: a.limit_scenes]
    by_scene = {}
    for r in rows:
        k = (r["dataset"], r["scene_name"])
        if k in set(scenes): by_scene.setdefault(k, []).append(r)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try: done.add(json.loads(l)["id"])
            except Exception: pass
    vlm = QwenVL("Qwen/Qwen2.5-VL-7B-Instruct", adapter=a.adapter); head = load_conf_head(a.value_head)
    fout = open(a.out, "a"); t0 = time.time(); nq = 0
    for key in scenes:
        qrows = [r for r in by_scene.get(key, []) if r["id"] not in done]
        if not qrows: continue
        try:
            frames = sample_frames(qrows[0]["video"], a.num_frames); rec = reconstruct(frames); H, W = rec["size"]; K = rec["intrinsics"][0]
            bank, fwd, meta = build_pose_bank(rec, render_all=False)
            for e in bank: e["valid"] = True  # coverage checked lazily on render
            base = [frames[i] for i in np.linspace(0, len(frames) - 1, 8).round().astype(int)]
            rcache = {}
            def view(i):
                if i not in rcache:
                    img = render(rec["points"], rec["colors"], torch.tensor(bank[i]["extrinsic"], device=rec["points"].device), K, H, W, splat=2)
                    cov = float((img.min(-1).values < 0.999).float().mean()); bank[i]["valid"] = cov >= 0.45
                    rcache[i] = (img.clamp(0, 1) * 255).byte().cpu().numpy()
                return rcache[i]
        except Exception as ex:
            for r in qrows: fout.write(json.dumps({"id": r["id"], "pred": "", "score": 0.0, "question_type": r["question_type"], "error": "scene_failed"}) + "\n")
            fout.flush(); print("scene failed", key, repr(ex)[:100], flush=True); continue
        for r in qrows:
            try:
                qtext = build_q(r); calls = 0
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
                    starts = [e["idx"] for e in bank if e["kind"] == "eye"]
                    # level-0 proposals: controller chooses START -> we expand to the first a.beam start spots (facing centre, yaw 0)
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
                            if a.policy == "random":
                                import random as _rnd
                                _rnd.seed(int(r["id"]) * 100 + d)
                                top = _rnd.sample(valid, min(a.beam, len(valid)))
                            else:
                                renders = [view(i) for _, i in path]
                                prompt = CTRL.format(pre=PRE.format(k=8), walk=WALK.format(m=len(renders), steps=describe(bank, path)), q=r["question"], valid=", ".join(["STOP"] + valid))
                                sc = action_scores(vlm, base + renders, prompt, ["STOP"] + valid); calls += 1
                                top = sorted(valid, key=lambda x: -sc[x])[: a.beam]
                            for act in top:
                                j = transition(bank, fwd, cur, act, meta)
                                if j is not None and j not in [i for _, i in path]: nxt.append(path + [(act, j)])
                        if not nxt: break
                        beam = nxt
                    if final is None:
                        # fuse the best kept paths' final answers by value: take best-valued state; arbitrate vs direct
                        best = max(scored.values(), key=lambda z: z[1]) if scored else (None, -1, [])
                        if best[0] is not None and best[1] > dconf: final, mode, best_path = best[0], "best_state", best[2]
                        else: final, mode, best_path = dpred, "fallback_direct", []
                trace.update(mode=mode, path=[(act, int(i)) for act, i in (best_path or [])], calls=calls, depth=len(best_path or []))
                s = score_row(r, final)
                fout.write(json.dumps({"id": r["id"], "pred": final.strip(), "score": s, "question_type": r["question_type"], **trace}) + "\n"); nq += 1
            except Exception as ex:
                print("q failed", r["id"], repr(ex)[:120], flush=True); torch.cuda.empty_cache()
        fout.flush(); del rec; torch.cuda.empty_cache(); print(f"[s{a.shard}] scene {key} done, {nq} q, {(time.time()-t0)/60:.1f} min", flush=True)
    fout.close(); print("DONE", flush=True)

if __name__ == "__main__":
    main()

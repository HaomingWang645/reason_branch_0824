"""ViewTree-D beam search on STI-Bench / VSTI-Bench (new corpus-trained controller, no retraining).
Same beam core as run_tree_d.py; loaders/frames/scoring reused from eval_video_bench.
  python scripts/depth/run_tree_d_bench.py --bench vsti --adapter checkpoints/depth/sft_c \
    --value-head checkpoints/depth/value_head.pt --shard 0 --num-shards 4 --out results/depth/vsti_treeD_s0.jsonl"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from viewtree.reconstruct import reconstruct
from viewtree.render import render
from viewtree.posebank import build_pose_bank, transition
from viewtree.tree import ROOT_GATE, answer_logprob, load_conf_head
from viewtree.vlm import QwenVL
from build_phase1 import ACTIONS, PRE, WALK, describe
from train_sft_c import CTRL
from run_tree_d import action_scores
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts"))
from eval_video_bench import FRAMES_PRE, head_train_scenes, load_sti, load_vsti, read_frames, score

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["sti", "vsti"], required=True); ap.add_argument("--adapter", required=True); ap.add_argument("--value-head", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--beam", type=int, default=3); ap.add_argument("--keep", type=int, default=2); ap.add_argument("--depth", type=int, default=3)
    a = ap.parse_args()
    items = load_sti() if a.bench == "sti" else load_vsti()
    vids = sorted({it["video"] for it in items})[a.shard::a.num_shards]; vs = set(vids)
    items = [it for it in items if it["video"] in vs]
    if a.limit: items = items[: a.limit]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try: done.add(json.loads(l)["id"])
            except Exception: pass
    items = [it for it in items if it["id"] not in done]
    htrain = head_train_scenes()
    vlm = QwenVL("Qwen/Qwen2.5-VL-7B-Instruct", adapter=a.adapter); head = load_conf_head(a.value_head)
    fout = open(a.out, "a"); t0 = time.time(); nq = 0; cache = {"key": None}
    for it in items:
        try:
            key = (it["video"], it["t0"], it["t1"], it["n"])
            if cache["key"] != key:
                frames = read_frames(it["video"], it["n"], it["t0"], it["t1"])
                try:
                    rec = reconstruct(frames); H, W = rec["size"]; K = rec["intrinsics"][0]
                    bank, fwd, meta = build_pose_bank(rec, render_all=False)
                    for e in bank: e["valid"] = True
                except Exception as ex:
                    rec = None; print("recon failed", it["scene"], repr(ex)[:80], flush=True)
                cache = {"key": key, "frames": frames, "rec": rec, "bank": bank if rec else None, "fwd": fwd if rec else None, "meta": meta if rec else None, "rcache": {}}
            frames = cache["frames"]; rec = cache["rec"]; bank = cache["bank"]; fwd = cache["fwd"]; meta = cache["meta"]; rcache = cache["rcache"]
            def view(i):
                if i not in rcache:
                    H, W = rec["size"]; K = rec["intrinsics"][0]
                    img = render(rec["points"], rec["colors"], torch.tensor(bank[i]["extrinsic"], device=rec["points"].device), K, H, W, splat=2)
                    cov = float((img.min(-1).values < 0.999).float().mean()); bank[i]["valid"] = cov >= 0.45
                    rcache[i] = (img.clamp(0, 1) * 255).byte().cpu().numpy()
                return rcache[i]
            prompt = it["prompt"]; k = len(frames); calls = 0
            qtext = prompt.split("\n")[0] if not prompt.startswith("Object description") else prompt.split("\n")[1]
            out = dict(id=it["id"], qtype=it["qtype"], source=it["source"], scene=it["scene"], gt=it["gt"], numeric=it["numeric"], clean=it["scene"] not in htrain)
            def answer(path):
                nonlocal calls; calls += 1
                renders = [view(i) for _, i in path]
                p = PRE.format(k=k) + (WALK.format(m=len(renders), steps=describe(bank, path)) if renders else "") + prompt
                pred, lp, ft = answer_logprob(vlm, frames + renders, p, max_new_tokens=12, want_feature=True); return pred, float(head(ft))
            gate, _ = answer_logprob(vlm, frames, ROOT_GATE.format(q=qtext), max_new_tokens=4); calls += 1
            dpred, dconf = answer([])
            out.update(gate=gate.strip(), direct=dpred.strip(), dconf=round(dconf, 3))
            if rec is None or "YES" in gate.upper():
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
                        p = CTRL.format(pre=PRE.format(k=k), walk=WALK.format(m=len(renders), steps=describe(bank, path)), q=qtext, valid=", ".join(["STOP"] + valid))
                        scs = action_scores(vlm, frames + renders, p, ["STOP"] + valid); calls += 1
                        top = sorted(valid, key=lambda x: -scs[x])[: a.beam]
                        for act in top:
                            j = transition(bank, fwd, cur, act, meta)
                            if j is not None and j not in [i for _, i in path]: nxt.append(path + [(act, j)])
                    if not nxt: break
                    beam = nxt
                if final is None:
                    best = max(scored.values(), key=lambda z: z[1]) if scored else (None, -1, [])
                    if best[0] is not None and best[1] > dconf: final, mode, best_path = best[0], "best_state", best[2]
                    else: final, mode, best_path = dpred, "fallback_direct", []
            out.update(pred=final.strip(), mode=mode, path=[(act, int(i)) for act, i in (best_path or [])], calls=calls, depth=len(best_path or []))
            out["score"] = score(it, out["pred"]); fout.write(json.dumps(out) + "\n"); nq += 1
            if nq % 25 == 0: fout.flush(); print(f"[{a.bench} treeD s{a.shard}] {nq} q {(time.time()-t0)/60:.1f} min", flush=True)
        except Exception as e:
            print("skip", it["id"], repr(e)[:120], flush=True); torch.cuda.empty_cache()
    fout.close(); print("DONE", flush=True)

if __name__ == "__main__":
    main()

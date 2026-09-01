"""ViewTree-D beam search on OST-Bench (corpus-trained controller; image history ->
reconstruction -> pose bank -> beam over camera walks; no retraining)."""
import argparse, json, os, re, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts"))
from eval_external import load_ost
from viewtree.reconstruct import reconstruct
from viewtree.render import render
from viewtree.posebank import build_pose_bank, transition
from viewtree.tree import ROOT_GATE, answer_logprob, load_conf_head
from viewtree.vlm import QwenVL
from build_phase1 import ACTIONS, PRE, WALK, describe
from train_sft_c import CTRL
from run_tree_d import action_scores

def letter(t):
    m = re.search(r"\b([A-H])\b", t.strip()); return m.group(1) if m else ""

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True); ap.add_argument("--value-head", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--beam", type=int, default=3); ap.add_argument("--keep", type=int, default=2); ap.add_argument("--depth", type=int, default=3)
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try: done.add(json.loads(l)["id"])
            except Exception: pass
    vlm = QwenVL("Qwen/Qwen2.5-VL-7B-Instruct", adapter=a.adapter); head = load_conf_head(a.value_head)
    fout = open(a.out, "a"); n = 0; t0 = time.time()
    for i, (qid, imgs, prompt, gt, qtype) in enumerate(load_ost()):
        if i % a.num_shards != a.shard or qid in done: continue
        try:
            base = [imgs[j] for j in np.linspace(0, len(imgs) - 1, min(8, len(imgs))).round().astype(int)]
            k = len(base); calls = 0
            qtext = prompt.split("\n")[1]
            out = {"id": qid, "qtype": qtype, "gt": gt}
            gate, _ = answer_logprob(vlm, base, ROOT_GATE.format(q=qtext), max_new_tokens=4); calls += 1
            rec = None; bank = fwd = meta = None; rcache = {}
            def view(idx):
                if idx not in rcache:
                    H, W = rec["size"]; K = rec["intrinsics"][0]
                    img = render(rec["points"], rec["colors"], torch.tensor(bank[idx]["extrinsic"], device=rec["points"].device), K, H, W, splat=2)
                    cov = float((img.min(-1).values < 0.999).float().mean()); bank[idx]["valid"] = cov >= 0.45
                    rcache[idx] = (img.clamp(0, 1) * 255).byte().cpu().numpy()
                return rcache[idx]
            def answer(path):
                nonlocal calls; calls += 1
                renders = [view(j) for _, j in path]
                p = PRE.format(k=k) + (WALK.format(m=len(renders), steps=describe(bank, path)) if renders else "") + prompt
                pred, lp, ft = answer_logprob(vlm, base + renders, p, max_new_tokens=8, want_feature=True); return letter(pred), float(head(ft))
            dpred, dconf = answer([])
            out.update(gate=gate.strip(), direct=dpred, dconf=round(dconf, 3))
            if "YES" in gate.upper():
                final, mode, best_path = dpred, "direct", []
            else:
                try:
                    rec = reconstruct(imgs)
                    bank, fwd, meta = build_pose_bank(rec, render_all=False)
                    for e in bank: e["valid"] = True
                except Exception as ex:
                    rec = None; print("recon failed", qid, repr(ex)[:80], flush=True)
                if rec is None:
                    final, mode, best_path = dpred, "direct", []
                else:
                    beam = [[("start at spot", e["idx"])] for e in bank if e["kind"] == "eye" and e["yaw"] == 0][: a.beam]
                    scored = {}; final, mode, best_path = None, None, None
                    for d in range(1, a.depth + 1):
                        for path in beam:
                            key2 = tuple(j for _, j in path)
                            if key2 in scored: continue
                            if not all(bank[j]["valid"] for _, j in path if view(j) is not None): scored[key2] = (None, -1.0, path); continue
                            p, c = answer(path); scored[key2] = (p, c, path)
                        level = sorted([scored[tuple(j for _, j in p)] for p in beam], key=lambda z: -z[1])
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
                            renders = [view(j) for _, j in path]
                            p = CTRL.format(pre=PRE.format(k=k), walk=WALK.format(m=len(renders), steps=describe(bank, path)), q=qtext, valid=", ".join(["STOP"] + valid))
                            scs = action_scores(vlm, base + renders, p, ["STOP"] + valid); calls += 1
                            top = sorted(valid, key=lambda x: -scs[x])[: a.beam]
                            for act in top:
                                j2 = transition(bank, fwd, cur, act, meta)
                                if j2 is not None and j2 not in [j for _, j in path]: nxt.append(path + [(act, j2)])
                        if not nxt: break
                        beam = nxt
                    if final is None:
                        best = max(scored.values(), key=lambda z: z[1]) if scored else (None, -1, [])
                        if best[0] is not None and best[1] > dconf: final, mode, best_path = best[0], "best_state", best[2]
                        else: final, mode, best_path = dpred, "fallback_direct", []
                    del rec; torch.cuda.empty_cache()
            out.update(final=final, mode=mode, path=[(act, int(j)) for act, j in (best_path or [])], calls=calls, depth=len(best_path or []), correct=final == gt)
            fout.write(json.dumps(out) + "\n"); n += 1
            if n % 25 == 0: fout.flush(); print(f"[ost treeD s{a.shard}] {n} items {(time.time()-t0)/60:.1f} min", flush=True)
        except Exception as e:
            print("skip", qid, repr(e)[:120], flush=True); torch.cuda.empty_cache()
    fout.close(); print("DONE", flush=True)

if __name__ == "__main__":
    main()

"""Phase 2: oracle walks.  With a frozen answerer (LoRA adapter), run a bounded
beam search over each scene's pose bank for each QA and record, per visited
state, the answer, its correctness and an answer margin (logprob of the
answer tokens).  The oracle walk = shortest path whose state is correct with
the largest margin (STOP at depth 0 if the direct answer already is).
Outputs one json line per QA: {id, scene, direct, states:[...], oracle:[(action, idx)], labels:[(state_path, action)]}
Features of every state are saved for the value head.
  python scripts/depth/oracle_walks.py --adapter CK --shard 0 --num-shards 8 --out data/train3r/oracle_s0.jsonl --feats data/train3r/feats_oracle"""
import argparse, json, os, sys, random, time
import cv2, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts", "depth"))
from viewtree.posebank import transition
from viewtree.tree import answer_logprob
from viewtree.score import parse_number
from viewtree.vlm import QwenVL
from build_phase1 import ACTIONS, PRE, WALK, describe
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def correct(r, pred):
    if r["numeric"]:
        p = parse_number(pred); g = parse_number(r["a"])
        if p is None or g is None: return 0.0
        rel = abs(p - g) / max(abs(g), 1e-8); return float(np.mean(rel < (1.0 - np.arange(0.5, 1.0, 0.05))))
    import re
    m = re.search(r"\b([A-F])\b", pred.strip()); return float(bool(m) and m.group(1) == r["a"].strip())

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", required=True); ap.add_argument("--qa", default=os.path.join(REPO, "data/train3r/qa_all.jsonl"))
    ap.add_argument("--bank", default=os.path.join(REPO, "data/posebank")); ap.add_argument("--out", required=True); ap.add_argument("--feats", default=None)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1); ap.add_argument("--per-scene", type=int, default=40)
    ap.add_argument("--beam", type=int, default=3); ap.add_argument("--depth", type=int, default=3); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); rng = random.Random(a.shard)
    import collections; by_scene = collections.defaultdict(list)
    for l in open(a.qa):
        r = json.loads(l); by_scene[r["scene"]].append(r)
    scenes = sorted(s for s in by_scene if os.path.exists(os.path.join(a.bank, "scannet", s, "bank.json")) or os.path.exists(os.path.join(a.bank, "scannetppv2", s, "bank.json")))[a.shard::a.num_shards]
    if a.limit: scenes = scenes[: a.limit]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try: done.add(json.loads(l)["id"])
            except Exception: pass
    if a.feats: os.makedirs(a.feats, exist_ok=True)
    vlm = QwenVL("Qwen/Qwen2.5-VL-7B-Instruct", adapter=a.adapter); fout = open(a.out, "a"); t0 = time.time(); n = 0
    imcache = {}
    def img(p):
        if p not in imcache: imcache[p] = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        return imcache[p]
    for scene in scenes:
        src = "scannet" if os.path.exists(os.path.join(a.bank, "scannet", scene)) else "scannetppv2"; sd = os.path.join(a.bank, src, scene)
        b = json.load(open(os.path.join(sd, "bank.json"))); bank = b["bank"]; fwd = {int(k): v for k, v in b["fwd_map"].items()}; meta = b["meta"]
        imcache.clear(); frames = [img(os.path.join(sd, f"frame_{i:02d}.jpg")) for i in range(8)]
        qs = by_scene[scene]; rng.shuffle(qs)
        for qi, r in enumerate(qs[: a.per_scene]):
            qid = f"{scene}::{qi}"
            if qid in done: continue
            try:
                def state_answer(path):
                    renders = [img(os.path.join(sd, f"view_{idx:03d}.jpg")) for _, idx in path]
                    prompt = PRE.format(k=8) + (WALK.format(m=len(renders), steps=describe(bank, path)) if renders else "") + r["q"]
                    pred, lp, ft = answer_logprob(vlm, frames + renders, prompt, max_new_tokens=12, want_feature=True)
                    return pred, float(lp), ft
                states = {}
                pred, lp, ft = state_answer([]); c = correct(r, pred); states[()] = dict(pred=pred, lp=lp, correct=c)
                feats = {(): ft.half().cpu()}
                starts = [e["idx"] for e in bank if e["kind"] == "eye" and e["valid"]]; rng.shuffle(starts)
                beam = [(("start at spot", s),) for s in starts[: a.beam]]
                for d in range(1, a.depth + 1):
                    scored = []
                    for path in beam:
                        if path in states: continue
                        pred, lp, ft = state_answer(list(path)); c = correct(r, pred); states[path] = dict(pred=pred, lp=lp, correct=c); feats[path] = ft.half().cpu()
                    for path in beam:
                        scored.append((states[path]["correct"], states[path]["lp"], path))
                    scored.sort(key=lambda x: (-x[0], -x[1]))
                    if d == a.depth: break
                    nxt = []
                    for _, _, path in scored[: a.beam]:
                        cur = path[-1][1]
                        if bank[cur]["kind"] == "topdown": continue
                        acts = ACTIONS[:]; rng.shuffle(acts)
                        for act in acts:
                            j = transition(bank, fwd, cur, act, meta)
                            if j is not None and j not in [p[1] for p in path]: nxt.append(path + ((act, j),))
                    if not nxt: break
                    beam = nxt[: a.beam * 3]
                # oracle: shortest correct path with max margin; ties -> shorter
                cands = [(len(p), -s["lp"], p) for p, s in states.items() if s["correct"] >= 0.5]
                if not cands: oracle = None
                else: cands.sort(); oracle = cands[0][2]
                has_oracle = oracle is not None
                labels = []
                if has_oracle:
                    for k in range(len(oracle)): labels.append((list(oracle[:k]), oracle[k][0] if k > 0 else "START", oracle[k][1]))
                    labels.append((list(oracle), "STOP", None))
                out = dict(id=qid, scene=scene, src=src, qtype=r["qtype"], numeric=r["numeric"], a=r["a"], q=r["q"], oracle=(list(oracle) if has_oracle else None),
                           direct_correct=states[()]["correct"], best_correct=max(s["correct"] for s in states.values()),
                           states=[dict(path=list(p), **s) for p, s in states.items()], labels=labels)
                fout.write(json.dumps(out) + "\n"); n += 1
                if a.feats: torch.save({json.dumps(list(p)): f for p, f in feats.items()}, os.path.join(a.feats, qid.replace("::", "_") + ".pt"))
                if n % 20 == 0: fout.flush(); print(f"[s{a.shard}] {n} qa {(time.time()-t0)/60:.1f} min", flush=True)
            except Exception as e:
                print("skip", qid, repr(e)[:120], flush=True); torch.cuda.empty_cache()
    fout.close(); print("DONE", flush=True)

if __name__ == "__main__":
    main()

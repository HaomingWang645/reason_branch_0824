"""Phase 1 data (SFT-A answerer): for each QA on a scene with a pose bank,
sample a random valid walk of depth 0..3 in the bank and emit one example:
  images = 8 context frames + walk renders, prompt = walk description + question,
  target = answer.  Also emits control examples for Phase 2 warm start? No —
Phase 1 is answer-only.  python scripts/depth/build_phase1.py --per-scene 60 --out data/train3r/phase1.jsonl"""
import argparse, json, os, random, sys, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from viewtree.posebank import transition
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ACTIONS = ["TURN_LEFT", "TURN_RIGHT", "FORWARD", "NEXT_SPOT", "LOOK_AROUND", "BIRD_EYE"]
PRE = "The first {k} images are frames of a video captured while walking through a room.\n"
WALK = ("The next {m} images were rendered from a 3D reconstruction of the SAME room as a camera walk taken by a person "
        "(holes possible): {steps}.\n")
TOPDOWN = "top-down bird's-eye view"

def describe(bank, path):
    out = []
    for j, (act, idx) in enumerate(path):
        e = bank[idx]
        if e["kind"] == "topdown": out.append(f"step {j+1}: {act.lower().replace('_', ' ')} -> {TOPDOWN}")
        else: out.append(f"step {j+1}: {act.lower().replace('_', ' ')} -> eye-level view from standing spot {e['pos']+1} facing direction {e['yaw']+1} of {8}")
    return "; ".join(out)

def random_walk(bank, fwd, meta, depth, rng):
    starts = [e["idx"] for e in bank if e["kind"] == "eye" and e["valid"]]
    if not starts: return None
    cur = rng.choice(starts); path = [("start at spot", cur)]
    for _ in range(depth - 1):
        acts = ACTIONS[:] ; rng.shuffle(acts); moved = False
        for act in acts:
            if act == "BIRD_EYE" and _ < depth - 2: continue
            j = transition(bank, fwd, cur, act, meta)
            if j is not None and j not in [p[1] for p in path]: path.append((act, j)); cur = j; moved = True; break
        if not moved or bank[cur]["kind"] == "topdown": break
    return path

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--qa", default=os.path.join(REPO, "data/train3r/qa_all.jsonl")); ap.add_argument("--bank", default=os.path.join(REPO, "data/posebank"))
    ap.add_argument("--per-scene", type=int, default=60); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rng = random.Random(a.seed)
    by_scene = collections.defaultdict(list)
    for l in open(a.qa):
        r = json.loads(l); by_scene[r["scene"]].append(r)
    n = 0; depth_hist = collections.Counter(); fout = open(a.out, "w")
    for src in os.listdir(a.bank):
        for scene in os.listdir(os.path.join(a.bank, src)):
            bp = os.path.join(a.bank, src, scene, "bank.json")
            if not os.path.exists(bp) or scene not in by_scene: continue
            b = json.load(open(bp)); bank = b["bank"]; fwd = {int(k): v for k, v in b["fwd_map"].items()}; meta = b["meta"]
            frames = [os.path.join(a.bank, src, scene, f"frame_{i:02d}.jpg") for i in range(8)]
            qs = by_scene[scene]; rng.shuffle(qs)
            for r in qs[: a.per_scene]:
                depth = rng.choice([0, 1, 1, 2, 2, 3])
                path = random_walk(bank, fwd, meta, depth, rng) if depth else []
                if path is None: continue
                renders = [os.path.join(a.bank, src, scene, f"view_{idx:03d}.jpg") for _, idx in path]
                prompt = PRE.format(k=8) + (WALK.format(m=len(renders), steps=describe(bank, path)) if renders else "") + r["q"]
                fout.write(json.dumps(dict(id=f"{scene}::{n}", kind="answer", images=frames, render=None, renders=renders, prompt=prompt, target=r["a"],
                                           qtype=r["qtype"], src=r["src"], depth=len(renders), numeric=r["numeric"])) + "\n")
                n += 1; depth_hist[len(renders)] += 1
    fout.close(); print("examples", n, "depth hist", dict(depth_hist))

if __name__ == "__main__":
    main()

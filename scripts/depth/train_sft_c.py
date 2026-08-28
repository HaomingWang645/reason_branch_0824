"""Phase 2b: build controller-imitation examples (SFT-C) from oracle walks and
train them mixed with Phase-1 answer examples.  Control prompt at each oracle
state lists the valid actions (hard mask from the bank) and the target is the
oracle action; the answer examples keep the answerer from forgetting.
  python scripts/depth/train_sft_c.py build --out data/train3r/phase2.jsonl
  then: torchrun ... scripts/train_sft.py --data data/train3r/phase2.jsonl --out checkpoints/depth/sft_c"""
import argparse, glob, json, os, random, sys, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viewtree.posebank import transition
from build_phase1 import ACTIONS, PRE, WALK, describe
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CTRL = ("{pre}{walk}Question: {q}\nYou may answer now or move the camera to gather more evidence. Valid moves from here: {valid}. "
        "Reply with exactly one token: STOP to answer now, or one of the valid moves.")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["build"]); ap.add_argument("--oracle", default=os.path.join(REPO, "data/train3r/oracle_s*.jsonl"))
    ap.add_argument("--phase1", default=os.path.join(REPO, "data/train3r/phase1.jsonl")); ap.add_argument("--bank", default=os.path.join(REPO, "data/posebank"))
    ap.add_argument("--out", required=True); ap.add_argument("--answer-ratio", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rng = random.Random(a.seed); banks = {}
    def bank_of(src, scene):
        k = (src, scene)
        if k not in banks:
            b = json.load(open(os.path.join(a.bank, src, scene, "bank.json"))); banks[k] = (b["bank"], {int(x): v for x, v in b["fwd_map"].items()}, b["meta"])
        return banks[k]
    out = []; stats = collections.Counter()
    for f in sorted(glob.glob(a.oracle)):
        for l in open(f):
            r = json.loads(l)
            if r["oracle"] is None or not r["labels"]: stats["no_oracle"] += 1; continue
            bank, fwd, meta = bank_of(r["src"], r["scene"]); sd = os.path.join(a.bank, r["src"], r["scene"])
            frames = [os.path.join(sd, f"frame_{i:02d}.jpg") for i in range(8)]
            for path, act, _ in r["labels"]:
                path = [tuple(p) for p in path]
                if act == "START":  # first acquisition: which spot to start at is not a camera move; label as NEXT_SPOT-equivalent
                    act = "START"
                cur = path[-1][1] if path else None
                valid = ["STOP"] + ([x for x in ACTIONS if transition(bank, fwd, cur, x, meta) is not None] if cur is not None else ["START"])
                if act not in valid: stats["act_not_valid"] += 1; continue
                renders = [os.path.join(sd, f"view_{idx:03d}.jpg") for _, idx in path]
                prompt = CTRL.format(pre=PRE.format(k=8), walk=(WALK.format(m=len(renders), steps=describe(bank, path)) if renders else ""), q=r["q"], valid=", ".join(valid))
                out.append(dict(id=f"{r['id']}::ctrl{len(path)}", kind="control", images=frames, render=None, renders=renders, prompt=prompt, target=act, depth=len(renders)))
                stats[f"ctrl_{act}"] += 1
    ans = [json.loads(l) for l in open(a.phase1)]; rng.shuffle(ans); ans = ans[: int(len(out) * a.answer_ratio)]
    allx = out + ans; rng.shuffle(allx)
    with open(a.out, "w") as f:
        for x in allx: f.write(json.dumps(x) + "\n")
    print("control", len(out), "answer", len(ans), dict(stats))

if __name__ == "__main__":
    main()

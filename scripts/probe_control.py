"""Probe STOP/MOVE/RENDER control decisions on MindCube tinybench states."""
import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viewtree.tree import answer_logprob
from viewtree.vlm import QwenVL

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MC_ROOT = os.path.join(REPO, "data", "mindcube", "data")

PRE = "These images show a scene photographed from different viewpoints.\n"
CONTROL_PROMPT = (
    "{pre}Question: {q}\n"
    "You may either answer now or acquire more evidence. Reply with exactly one "
    "word: STOP if the current views are sufficient to answer correctly, MOVE to "
    "view the scene from another side, or RENDER to inspect a reconstructed "
    "top-down view."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            open(os.path.join(MC_ROOT, "raw", "MindCube_tinybench.jsonl"))]
    if args.limit:
        rows = rows[:args.limit]
    vlm = QwenVL("Qwen/Qwen2.5-VL-7B-Instruct", adapter=args.adapter)
    fout = open(args.out, "w")
    for ri, r in enumerate(rows):
        try:
            imgs = [cv2.cvtColor(cv2.imread(os.path.join(MC_ROOT, p)), cv2.COLOR_BGR2RGB)
                    for p in r["images"]]
            acts = {}
            for k in range(1, len(imgs) + 1):
                pred, _ = answer_logprob(
                    vlm, imgs[:k],
                    CONTROL_PROMPT.format(pre=PRE, q=r["question"]),
                    max_new_tokens=4)
                acts[f"s{k}"] = pred.strip().upper()[:8]
            fout.write(json.dumps({"id": r["id"], "n_views": len(imgs),
                                   "actions": acts}) + "\n")
            if (ri + 1) % 50 == 0:
                fout.flush()
                print(f"{ri+1}/{len(rows)}", flush=True)
        except Exception as e:
            print("skip", r["id"], e, flush=True)
    fout.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

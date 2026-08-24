"""Build the Stage I SFT dataset from teacher outcome ladders (doc Eq. 6).

Two example families:
  control: given views[:k] + question -> best action (STOP / MOVE / RENDER),
           derived from the teacher ladder's eventual-correctness pattern.
           STOP is preferred on ties (cheapest). All-wrong ladders are skipped
           (no defensible label).
  answer:  given the evidence state at the ladder's first-correct point ->
           ground-truth letter (answers supervised by GT, never teacher text,
           because the teacher is weak on this domain).
Output: sft_data.jsonl with {id, kind, images, prompt, target}.
"""
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MC_ROOT = os.path.join(REPO, "data", "mindcube", "data")

CONTROL_PROMPT = (
    "{pre}Question: {q}\n"
    "You may either answer now or acquire more evidence. Reply with exactly one "
    "word: STOP if the current views are sufficient to answer correctly, MOVE to "
    "view the scene from another side, or RENDER to inspect a reconstructed "
    "top-down view."
)
PRE = "These images show a scene photographed from different viewpoints.\n"
PRE_R = ("The first {k} images show a scene photographed from different viewpoints. "
         "The final image is a top-down view rendered from a 3D reconstruction "
         "(it may contain holes).\n")
SUFFIX = "\nAnswer with the option's letter from the given choices directly."


def main():
    train = {json.loads(l)["id"]: json.loads(l)
             for l in open(os.path.join(MC_ROOT, "raw", "MindCube_train.jsonl"))}
    out = open(os.path.join(REPO, "data", "sft_data.jsonl"), "w")
    n_ctrl, n_ans, skipped = 0, 0, 0
    for f in sorted(glob.glob(os.path.join(REPO, "results", "traj_s*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            item = train.get(r["id"])
            if item is None:
                continue
            n = r["n_views"]
            s = r["states"]
            order = [f"s{k}" for k in range(1, n + 1)] + [f"s{n}r"]
            order = [k for k in order if k in s]
            corr = [bool(s[k]["correct"]) for k in order]
            if not any(corr):
                skipped += 1
                continue
            first_ok = corr.index(True)
            # control examples: one per prefix state up to first_ok
            for k in range(min(first_ok + 1, n)):  # states s1..s_{first_ok+1}
                state = order[k]
                depth = k + 1
                future_ok = any(corr[k + 1:])
                if corr[k]:
                    action = "STOP"
                elif not future_ok:
                    break
                elif k == n - 1:  # at s_n, only render remains
                    action = "RENDER"
                else:
                    action = "MOVE"
                out.write(json.dumps({
                    "id": f"{r['id']}::ctrl::{state}", "kind": "control",
                    "images": item["images"][:depth],
                    "render": None,
                    "prompt": CONTROL_PROMPT.format(pre=PRE, q=item["question"]),
                    "target": action,
                }) + "\n")
                n_ctrl += 1
                if corr[k]:
                    break
            # answer example at the first-correct state (GT supervised)
            state = order[first_ok]
            uses_render = state.endswith("r")
            depth = n if uses_render else first_ok + 1
            pre = PRE_R.format(k=n) if uses_render else PRE
            out.write(json.dumps({
                "id": f"{r['id']}::ans::{state}", "kind": "answer",
                "images": item["images"][:depth],
                "render": r["id"] if uses_render else None,
                "prompt": pre + item["question"] + SUFFIX,
                "target": r["gt"],
            }) + "\n")
            n_ans += 1
    out.close()
    print(f"control={n_ctrl} answer={n_ans} skipped_all_wrong={skipped}")


if __name__ == "__main__":
    main()

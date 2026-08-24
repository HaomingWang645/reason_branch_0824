"""Stage III / SFT v2 dataset from ON-POLICY ladders (doc §6.5).

Families:
  control    - STOP/MOVE/RENDER from outcome patterns (as v1, on-policy)
  answer     - GT letter at the first-correct state
  fusion     - complementary sets: s1 wrong but full evidence correct ->
               all views (+render if that is what fixed it), GT target
  redundant  - s1 correct AND s4 correct -> full-evidence answer (extra views
               must not break the answer); sampled at 30%
Render boost: control RENDER examples arise whenever s4 wrong -> s4r correct.
Output: data/sft_data_v2.jsonl (same schema as v1).
"""
import glob
import json
import os
import random

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
    rng = random.Random(0)
    train = {json.loads(l)["id"]: json.loads(l)
             for l in open(os.path.join(MC_ROOT, "raw", "MindCube_train.jsonl"))}
    out = open(os.path.join(REPO, "data", "sft_data_v2.jsonl"), "w")
    counts = {"control": 0, "answer": 0, "fusion": 0, "redundant": 0}
    ctrl_targets = {"STOP": 0, "MOVE": 0, "RENDER": 0}
    for f in sorted(glob.glob(os.path.join(REPO, "results", "traj_sft_s*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            item = train.get(r["id"])
            if item is None:
                continue
            n = r["n_views"]
            s = r["states"]
            order = [k for k in [f"s{i}" for i in range(1, n + 1)] + [f"s{n}r"]
                     if k in s]
            corr = [bool(s[k]["correct"]) for k in order]
            if not any(corr):
                continue
            first_ok = corr.index(True)

            def emit(kind, images, render, prompt, target, tag):
                out.write(json.dumps({
                    "id": f"{r['id']}::{tag}", "kind": kind, "images": images,
                    "render": render, "prompt": prompt, "target": target,
                }) + "\n")
                counts[kind] += 1

            # control ladder up to first correct state
            for k in range(min(first_ok + 1, n)):
                depth = k + 1
                future_ok = any(corr[k + 1:])
                if corr[k]:
                    action = "STOP"
                elif not future_ok:
                    break
                elif k == n - 1:
                    action = "RENDER"
                else:
                    action = "MOVE"
                emit("control", item["images"][:depth], None,
                     CONTROL_PROMPT.format(pre=PRE, q=item["question"]),
                     action, f"ctrl{depth}")
                ctrl_targets[action] += 1
                if corr[k]:
                    break

            # answer at first-correct state
            st = order[first_ok]
            uses_r = st.endswith("r")
            depth = n if uses_r else first_ok + 1
            emit("answer", item["images"][:depth], r["id"] if uses_r else None,
                 (PRE_R.format(k=n) if uses_r else PRE) + item["question"] + SUFFIX,
                 r["gt"], f"ans{st}")

            # fusion: s1 wrong, full evidence correct
            s1_ok = corr[0]
            s4_ok = f"s{n}" in s and s[f"s{n}"]["correct"]
            s4r_ok = f"s{n}r" in s and s[f"s{n}r"]["correct"]
            if not s1_ok and (s4_ok or s4r_ok) and first_ok > 0:
                use_r = s4r_ok and not s4_ok
                emit("fusion", item["images"], r["id"] if use_r else None,
                     (PRE_R.format(k=n) if use_r else PRE) + item["question"] + SUFFIX,
                     r["gt"], "fuse")

            # redundant robustness (sampled)
            if s1_ok and s4_ok and rng.random() < 0.3:
                emit("redundant", item["images"], None,
                     PRE + item["question"] + SUFFIX, r["gt"], "red")
    out.close()
    print("counts:", counts)
    print("control targets:", ctrl_targets)


if __name__ == "__main__":
    main()

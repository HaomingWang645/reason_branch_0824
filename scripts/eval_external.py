"""Unified evaluation over external spatial benchmarks (paired base vs adapters).

Benches: viewspatial, ost, omnispatial, blink_mv, blink_spatial, blink_depth,
blink_loc, blink_count. Each yields (id, images, prompt, gt_letter). All MC,
letter exact-match. OST uses the cumulative image history per scan (latest 12).
"""
import argparse
import ast
import glob
import io
import json
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viewtree.vlm import QwenVL

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(REPO, "data", "external")
MC_SUFFIX = "Answer with the option's letter from the given choices directly."
LET = "ABCDEFGH"


def imread(p):
    im = cv2.imread(p)
    if im is None:
        raise RuntimeError(f"missing image {p}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def load_viewspatial():
    rows = json.load(open(f"{EXT}/viewspatial/ViewSpatial-Bench.json"))
    for i, r in enumerate(rows):
        try:
            imgs = [imread(os.path.join(EXT, "viewspatial",
                                        p.replace("ViewSpatial-Bench/", "")))
                    for p in r["image_path"]]
        except RuntimeError:
            continue
        prompt = f"{r['question']}\n{r['choices']}\n{MC_SUFFIX}"
        yield f"vs_{i}", imgs, prompt, r["answer"].strip()[0], r["question_type"]


def load_ost():
    rows = json.load(open(f"{EXT}/ost/OST_bench.json"))
    hist = {}
    order = {}
    for r in rows:
        order.setdefault(r["scan_id"], []).append(r)
    for scan, rs in order.items():
        rs.sort(key=lambda r: r["turn_id"])
        imgs_so_far = []
        for r in rs:
            new = ast.literal_eval(r["new_observations"])
            imgs_so_far = imgs_so_far + list(new)
            hist[(scan, r["turn_id"])] = list(imgs_so_far)
    for i, r in enumerate(rows):
        paths = hist[(r["scan_id"], r["turn_id"])][-12:]
        try:
            imgs = [imread(os.path.join(EXT, "ost", "image_upload", p))
                    for p in paths]
        except RuntimeError:
            continue
        opts = ast.literal_eval(r["option"]) if isinstance(r["option"], str) else r["option"]
        if not opts or r["answer"] not in opts:
            continue
        gt = LET[opts.index(r["answer"])]
        optstr = "\n".join(f"{LET[j]}. {o}" for j, o in enumerate(opts))
        prompt = ("These images were taken in chronological order while exploring "
                  f"a room.\n{r['origin_question']}\n{optstr}\n{MC_SUFFIX}")
        yield f"ost_{i}", imgs, prompt, gt, r["type"].split("-")[0]


def load_omnispatial():
    rows = json.load(open(f"{EXT}/omnispatial/OmniSpatial-test/data.json"))
    for r in rows:
        img_id = r["id"].split("_")[0]
        cands = glob.glob(f"{EXT}/omnispatial/OmniSpatial-test/{r['task_type']}/{img_id}.*")
        if not cands:
            continue
        try:
            imgs = [imread(cands[0])]
        except RuntimeError:
            continue
        optstr = "\n".join(f"{LET[j]}. {o}" for j, o in enumerate(r["options"]))
        prompt = f"{r['question']}\n{optstr}\n{MC_SUFFIX}"
        yield f"omni_{r['id']}", imgs, prompt, LET[r["answer"]], r["task_type"]


def load_blink(sub):
    import pandas as pd
    from PIL import Image
    f = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--BLINK-Benchmark--BLINK/snapshots/"
        f"*/{sub}/val-*.parquet"))[0]
    df = pd.read_parquet(f)
    for _, r in df.iterrows():
        imgs = []
        for c in ["image_1", "image_2", "image_3", "image_4"]:
            v = r.get(c)
            if v is not None and isinstance(v, dict) and v.get("bytes"):
                imgs.append(np.array(Image.open(io.BytesIO(v["bytes"])).convert("RGB")))
        opts = list(r["choices"])
        optstr = "\n".join(f"{LET[j]}. {o}" for j, o in enumerate(opts))
        gt = re.sub(r"[()]", "", r["answer"]).strip()
        prompt = f"{r['question']}\n{optstr}\n{MC_SUFFIX}"
        yield f"blink_{sub}_{r['idx']}", imgs, prompt, gt, sub


LOADERS = {
    "viewspatial": load_viewspatial,
    "ost": load_ost,
    "omnispatial": load_omnispatial,
    "blink_mv": lambda: load_blink("Multi-view_Reasoning"),
    "blink_spatial": lambda: load_blink("Spatial_Relation"),
    "blink_depth": lambda: load_blink("Relative_Depth"),
    "blink_loc": lambda: load_blink("Object_Localization"),
    "blink_count": lambda: load_blink("Counting"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=sorted(LOADERS))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    vlm = QwenVL("Qwen/Qwen2.5-VL-7B-Instruct", adapter=args.adapter)
    fout = open(args.out, "a")
    n = 0
    for i, (qid, imgs, prompt, gt, qtype) in enumerate(LOADERS[args.bench]()):
        if i % args.num_shards != args.shard or qid in done:
            continue
        try:
            pred = vlm.ask(imgs, prompt, max_new_tokens=8)
            m = re.search(r"\b([A-H])\b", pred.strip())
            correct = bool(m) and m.group(1) == gt
        except Exception as e:
            print("skip", qid, repr(e)[:80], flush=True)
            continue
        fout.write(json.dumps({"id": qid, "pred": pred, "gt": gt,
                               "correct": correct, "qtype": qtype}) + "\n")
        n += 1
        if n % 100 == 0:
            fout.flush()
            print(f"[{args.bench} s{args.shard}] {n}", flush=True)
    fout.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

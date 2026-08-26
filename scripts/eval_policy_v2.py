"""Greedy rollout of a design-variant policy (ladder/select, optional CoT,
horizon cap). Records accuracy, views used, generated answer tokens."""
import argparse
import json
import os
import re
import sys

import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_grpo_v2 import (CTRL_LADDER, CTRL_SELECT, PRE, PRE_R, SUFFIX,
                           SUFFIX_COT, build_inputs, letter_of)
from viewtree.vlm import QwenVL

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MC_ROOT = os.path.join(REPO, "data", "mindcube", "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--mode", choices=["ladder", "select"], default="ladder")
    ap.add_argument("--cot", action="store_true")
    ap.add_argument("--max-depth", type=int, default=99)
    ap.add_argument("--split", default="MindCube_tinybench")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from viewtree.reconstruct import reconstruct
    from viewtree.render import overview_poses, render

    rows = [json.loads(l) for l in open(os.path.join(MC_ROOT, "raw", args.split + ".jsonl"))]
    vlm = QwenVL("Qwen/Qwen2.5-VL-7B-Instruct", adapter=args.adapter)
    model, tok = vlm.model, vlm.processor.tokenizer
    suffix = SUFFIX_COT if args.cot else SUFFIX
    fout = open(args.out, "w")
    with torch.no_grad():
        for ri, r in enumerate(rows):
            try:
                imgs = [cv2.cvtColor(cv2.imread(os.path.join(MC_ROOT, p)), cv2.COLOR_BGR2RGB)
                        for p in r["images"]]
                n, q = len(imgs), r["question"]
                seen, acts = [0], []
                while True:
                    unseen = [i for i in range(n) if i not in seen]
                    depth_ok = len(seen) < min(n, args.max_depth)
                    prompt = (CTRL_LADDER.format(pre=PRE, q=q) if args.mode == "ladder"
                              else CTRL_SELECT.format(pre=PRE, q=q, seen=[i + 1 for i in seen],
                                                      unseen=[i + 1 for i in unseen] or "none"))
                    inp = build_inputs(vlm, [imgs[i] for i in seen], prompt)
                    out = model.generate(**inp, max_new_tokens=4, do_sample=False,
                                         pad_token_id=tok.eos_token_id)
                    txt = vlm.processor.decode(out[0, inp.input_ids.shape[1]:],
                                               skip_special_tokens=True).upper()
                    if "RENDER" in txt:
                        acts.append("RENDER"); use_render = True; break
                    if "MOVE" in txt and depth_ok and unseen:
                        if args.mode == "select":
                            m = re.search(r"MOVE\s*(\d)", txt)
                            k = int(m.group(1)) - 1 if m else unseen[0]
                            k = k if k in unseen else unseen[0]
                        else:
                            k = unseen[0]
                        acts.append(f"MOVE{k+1}"); seen.append(k); continue
                    acts.append("STOP"); use_render = False; break
                ims = [imgs[i] for i in seen]
                pre = PRE
                if use_render:
                    rec = reconstruct(imgs)
                    H, W = rec["size"]
                    img = render(rec["points"], rec["colors"], overview_poses(rec)[-1],
                                 rec["intrinsics"][0], H, W, splat=2)
                    ims = ims + [(img.clamp(0, 1) * 255).byte().cpu().numpy()]
                    pre = PRE_R.format(k=len(seen))
                    del rec; torch.cuda.empty_cache()
                inp = build_inputs(vlm, ims, pre + q + suffix)
                out = model.generate(**inp, max_new_tokens=64 if args.cot else 6,
                                     do_sample=False, pad_token_id=tok.eos_token_id)
                gen = out[0, inp.input_ids.shape[1]:]
                pred = vlm.processor.decode(gen, skip_special_tokens=True)
                fout.write(json.dumps({
                    "id": r["id"], "views": len(seen) + int(use_render), "actions": acts,
                    "ntok": int(gen.shape[0]), "correct": letter_of(pred, cot=args.cot) == r["gt_answer"],
                }) + "\n")
                if (ri + 1) % 100 == 0:
                    fout.flush(); print(f"{ri+1}/{len(rows)}", flush=True)
            except Exception as e:
                print("skip", r["id"], repr(e)[:80], flush=True)
    fout.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

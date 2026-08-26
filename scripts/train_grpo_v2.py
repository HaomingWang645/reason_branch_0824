"""GRPO design-space trainer (single GPU per variant).

Reasoning-path designs:
  --mode ladder   : sequential STOP/MOVE/RENDER (views in fixed order)
  --mode select   : branching by learned view selection — MOVE k picks WHICH
                    unseen view to acquire next (order-free)
  --cot           : answer with a brief rationale then "Answer: X"; policy
                    gradient over rationale+answer tokens (reasoning-length axis)
  --max-depth K   : horizon cap on views acquired
  --cost/--view-budget : efficiency pressure (dual variable lambda)
  --len-pen       : per-token penalty on generated answer tokens (cot)
"""
import argparse
import json
import os
import random
import re
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viewtree.vlm import QwenVL

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MC_ROOT = os.path.join(REPO, "data", "mindcube", "data")
RENDER_DIR = os.path.join(REPO, "data", "mindcube_renders")

PRE = "These images show a scene photographed from different viewpoints.\n"
PRE_R = ("The first {k} images show a scene photographed from different viewpoints. "
         "The final image is a top-down view rendered from a 3D reconstruction "
         "(it may contain holes).\n")
CTRL_LADDER = (
    "{pre}Question: {q}\n"
    "You may either answer now or acquire more evidence. Reply with exactly one "
    "word: STOP if the current views are sufficient to answer correctly, MOVE to "
    "view the scene from another side, or RENDER to inspect a reconstructed "
    "top-down view.")
CTRL_SELECT = (
    "{pre}Question: {q}\n"
    "You have seen views {seen}. Unseen views: {unseen}. Reply with exactly one "
    "of: STOP (answer now), MOVE k (acquire unseen view k), or RENDER (inspect "
    "a reconstructed top-down view).")
SUFFIX = "\nAnswer with the option's letter from the given choices directly."
SUFFIX_COT = ("\nThink briefly in one or two sentences about the spatial layout, "
              "then finish with 'Answer: X' where X is the option letter.")


def build_inputs(vlm, images, prompt):
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    content = [{"type": "image", "image": Image.fromarray(im),
                "min_pixels": 224 * 224, "max_pixels": 448 * 448} for im in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = vlm.processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    return vlm.processor(text=[text], images=image_inputs,
                         return_tensors="pt").to(vlm.device)


def letter_of(text, cot=False):
    if cot:
        m = re.search(r"Answer:\s*\(?([A-F])\b", text)
        if m:
            return m.group(1)
    ms = re.findall(r"\b([A-F])\b", text.strip())
    return ms[-1] if ms else None


def pg_loss(model, inp, gen_ids, weight, device):
    """Policy-gradient loss on generated token ids appended to inp."""
    ids = gen_ids.view(1, -1).to(device)
    full = torch.cat([inp.input_ids, ids], 1)
    labels = torch.full_like(full, -100)
    labels[:, inp.input_ids.shape[1]:] = ids
    out = model(input_ids=full, attention_mask=torch.ones_like(full),
                pixel_values=inp.pixel_values, image_grid_thw=inp.image_grid_thw,
                labels=labels)
    return weight * out.loss * ids.shape[1]  # sum-logprob scaling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ladder", "select"], default="ladder")
    ap.add_argument("--cot", action="store_true")
    ap.add_argument("--max-depth", type=int, default=99)
    ap.add_argument("--items", type=int, default=1000)
    ap.add_argument("--group", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--accum-items", type=int, default=8)
    ap.add_argument("--view-budget", type=float, default=2.3)
    ap.add_argument("--cost", type=float, default=0.05)
    ap.add_argument("--len-pen", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda"

    vlm = QwenVL("Qwen/Qwen2.5-VL-7B-Instruct", device=device,
                 adapter=os.path.join(REPO, "checkpoints", "sft_lora_v2"))
    model = vlm.model
    for n, p in model.named_parameters():
        p.requires_grad = "lora" in n
    model.train()
    tok = vlm.processor.tokenizer

    rows = [json.loads(l) for l in
            open(os.path.join(MC_ROOT, "raw", "MindCube_train.jsonl"))]
    random.Random(1).shuffle(rows)
    rows = [r for r in rows
            if os.path.exists(os.path.join(RENDER_DIR, f"{r['id']}.png"))][: args.items]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr)
    lam, eta = 0.0, 0.02
    stats = {"acc": [], "views": [], "toks": []}
    n_steps = 0
    for idx, r in enumerate(rows):
        try:
            imgs = [cv2.cvtColor(cv2.imread(os.path.join(MC_ROOT, p)), cv2.COLOR_BGR2RGB)
                    for p in r["images"]]
            td = cv2.cvtColor(cv2.imread(os.path.join(RENDER_DIR, f"{r['id']}.png")),
                              cv2.COLOR_BGR2RGB)
            n, gt, q = len(imgs), r["gt_answer"], r["question"]
            suffix = SUFFIX_COT if args.cot else SUFFIX
            greedy_cache = {}

            def sample_answer(seen, use_render):
                """Returns (correct, gen_ids or None, inp, n_tokens)."""
                key = (tuple(seen), use_render)
                ims = [imgs[i] for i in seen] + ([td] if use_render else [])
                pre = PRE_R.format(k=len(seen)) if use_render else PRE
                inp = build_inputs(vlm, ims, pre + q + suffix)
                if not args.cot:
                    if key not in greedy_cache:
                        with torch.no_grad():
                            out = model.generate(**inp, max_new_tokens=6, do_sample=False,
                                                 pad_token_id=tok.eos_token_id)
                        pred = vlm.processor.decode(out[0, inp.input_ids.shape[1]:],
                                                    skip_special_tokens=True)
                        greedy_cache[key] = letter_of(pred) == gt
                    return greedy_cache[key], None, None, 0
                with torch.no_grad():
                    out = model.generate(**inp, max_new_tokens=64, do_sample=True,
                                         temperature=1.0, pad_token_id=tok.eos_token_id)
                gen = out[0, inp.input_ids.shape[1]:]
                pred = vlm.processor.decode(gen, skip_special_tokens=True)
                return letter_of(pred, cot=True) == gt, gen, inp, int(gen.shape[0])

            rollouts = []
            for g in range(args.group):
                seen, decisions = [0], []
                with torch.no_grad():
                    while True:
                        unseen = [i for i in range(n) if i not in seen]
                        depth_ok = len(seen) < min(n, args.max_depth)
                        if args.mode == "ladder":
                            prompt = CTRL_LADDER.format(pre=PRE, q=q)
                        else:
                            prompt = CTRL_SELECT.format(
                                pre=PRE, q=q, seen=[i + 1 for i in seen],
                                unseen=[i + 1 for i in unseen] or "none")
                        inp = build_inputs(vlm, [imgs[i] for i in seen], prompt)
                        out = model.generate(**inp, max_new_tokens=4, do_sample=True,
                                             temperature=1.0, pad_token_id=tok.eos_token_id)
                        gen = out[0, inp.input_ids.shape[1]:]
                        txt = vlm.processor.decode(gen, skip_special_tokens=True).upper()
                        act = "STOP"
                        if "RENDER" in txt:
                            act = "RENDER"
                        elif "MOVE" in txt and depth_ok and unseen:
                            act = "MOVE"
                        # normalise the token target to the canonical action string
                        if act == "MOVE":
                            if args.mode == "select":
                                m = re.search(r"MOVE\s*(\d)", txt)
                                k = int(m.group(1)) - 1 if m else unseen[0]
                                k = k if k in unseen else unseen[0]
                                tgt = f"MOVE {k + 1}"
                            else:
                                k, tgt = unseen[0], "MOVE"
                        else:
                            tgt = act
                        decisions.append((list(seen), prompt, tgt))
                        if act == "MOVE":
                            seen.append(k)
                            continue
                        break
                use_render = act == "RENDER"
                correct, ans_gen, ans_inp, ntok = sample_answer(seen, use_render)
                views = len(seen) + (1 if use_render else 0)
                reward = (float(correct) - lam * max(0, views - 1) * args.cost
                          - args.len_pen * ntok)
                rollouts.append(dict(decisions=decisions, reward=reward, correct=correct,
                                     views=views, ans_gen=ans_gen, ans_inp=ans_inp,
                                     ntok=ntok))
            rs = np.array([x["reward"] for x in rollouts])
            adv = (rs - rs.mean()) / (rs.std() + 1e-6)
            stats["acc"] += [x["correct"] for x in rollouts]
            stats["views"] += [x["views"] for x in rollouts]
            stats["toks"] += [x["ntok"] for x in rollouts]
            if rs.std() > 1e-6:
                for x, a in zip(rollouts, adv):
                    if abs(a) < 1e-4:
                        continue
                    w = float(a) / args.group
                    for seen, prompt, tgt in x["decisions"]:
                        inp = build_inputs(vlm, [imgs[i] for i in seen], prompt)
                        ids = torch.tensor(tok.encode(tgt, add_special_tokens=False))
                        (pg_loss(model, inp, ids, w, device) / 4).backward()
                    if args.cot and x["ans_gen"] is not None and x["ntok"] > 0:
                        (pg_loss(model, x["ans_inp"], x["ans_gen"], w, device)
                         / max(x["ntok"], 1)).backward()
        except Exception as e:
            print("item failed", r["id"], repr(e)[:100], flush=True)
        if (idx + 1) % args.accum_items == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            n_steps += 1
            mv = float(np.mean(stats["views"][-200:] or [0]))
            lam = max(0.0, lam + eta * (mv - args.view_budget))
            if n_steps % 5 == 0:
                print(f"idx {idx+1} acc {np.mean(stats['acc'][-200:] or [0]):.3f} "
                      f"views {mv:.2f} toks {np.mean(stats['toks'][-200:] or [0]):.1f} "
                      f"lam {lam:.3f}", flush=True)
            if n_steps % 25 == 0:
                model.save_pretrained(args.out)
    model.save_pretrained(args.out)
    print("SAVED", args.out, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

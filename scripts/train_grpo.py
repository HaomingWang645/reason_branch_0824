"""Stage IV (scaled): GRPO on the view-control policy (doc §6.6).

Episode = MindCube train item. At state k (views 1..k) the policy samples
STOP / MOVE / RENDER from the control prompt (temperature 1.0). STOP answers
greedily from the current state; MOVE advances; RENDER answers from
all-views + top-down render. Reward = correctness − λ·extra_views, with the
dual variable λ updated toward a mean-views budget. Group-relative advantage
over G rollouts per item; policy gradient applied to the sampled control
tokens only (answers stay greedy / GT-supervised from SFT). LoRA continues
from the Stage III adapter.
"""
import argparse
import json
import os
import random
import sys

import cv2
import numpy as np

# On Exclusive_Process GPUs every rank must only ever see its own device;
# pin before torch initializes CUDA.
_lr = os.environ.get("LOCAL_RANK")
if _lr is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = _lr

import torch
import torch.distributed as dist
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viewtree.vlm import QwenVL

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MC_ROOT = os.path.join(REPO, "data", "mindcube", "data")
RENDER_DIR = os.path.join(REPO, "data", "mindcube_renders")

PRE = "These images show a scene photographed from different viewpoints.\n"
PRE_R = ("The first {k} images show a scene photographed from different viewpoints. "
         "The final image is a top-down view rendered from a 3D reconstruction "
         "(it may contain holes).\n")
CONTROL_PROMPT = (
    "{pre}Question: {q}\n"
    "You may either answer now or acquire more evidence. Reply with exactly one "
    "word: STOP if the current views are sufficient to answer correctly, MOVE to "
    "view the scene from another side, or RENDER to inspect a reconstructed "
    "top-down view."
)
SUFFIX = "\nAnswer with the option's letter from the given choices directly."


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
    return vlm.processor(text=[text], images=image_inputs, return_tensors="pt"
                         ).to(vlm.device)


def letter_of(text):
    import re
    m = re.search(r"\b([A-F])\b", text.strip())
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=4000)
    ap.add_argument("--group", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--accum-items", type=int, default=8)
    ap.add_argument("--view-budget", type=float, default=2.3)
    ap.add_argument("--cost", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(REPO, "checkpoints", "grpo_lora"))
    args = ap.parse_args()

    ddp = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    device = "cuda:0"  # each rank sees only its own GPU (CVD pinned above)
    torch.cuda.set_device(0)
    if ddp:
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))

    vlm = QwenVL("Qwen/Qwen2.5-VL-7B-Instruct", device=device,
                 adapter=os.path.join(REPO, "checkpoints", "sft_lora_v2"))
    model = vlm.model  # PeftModel; LoRA params trainable
    for n, p in model.named_parameters():
        p.requires_grad = "lora" in n
    model.train()
    tok = vlm.processor.tokenizer
    ACTION_IDS = {a: tok.encode(a, add_special_tokens=False) for a in
                  ["STOP", "MOVE", "RENDER"]}

    rows = [json.loads(l) for l in
            open(os.path.join(MC_ROOT, "raw", "MindCube_train.jsonl"))]
    random.Random(1).shuffle(rows)
    rows = [r for r in rows
            if os.path.exists(os.path.join(RENDER_DIR, f"{r['id']}.png"))]
    per = args.items // world
    rows = rows[: args.items][rank::world][:per]  # equal length on every rank

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr)
    lam, eta = 0.0, 0.02
    stats = {"acc": [], "views": [], "loss": 0.0}
    n_steps = 0
    for idx, r in enumerate(rows):
        try:
            imgs = [cv2.cvtColor(cv2.imread(os.path.join(MC_ROOT, p)),
                                 cv2.COLOR_BGR2RGB) for p in r["images"]]
            td = cv2.cvtColor(cv2.imread(os.path.join(RENDER_DIR, f"{r['id']}.png")),
                              cv2.COLOR_BGR2RGB)
            n = len(imgs)
            gt = r["gt_answer"]
            q = r["question"]

            # cached greedy answers per answer-state
            ans_cache = {}

            @torch.no_grad()
            def answer_from(state):  # state: int k, or "render"
                if state in ans_cache:
                    return ans_cache[state]
                if state == "render":
                    inp = build_inputs(vlm, imgs + [td],
                                       PRE_R.format(k=n) + q + SUFFIX)
                else:
                    inp = build_inputs(vlm, imgs[:state], PRE + q + SUFFIX)
                out = model.generate(**inp, max_new_tokens=6, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
                pred = vlm.processor.decode(out[0, inp.input_ids.shape[1]:],
                                            skip_special_tokens=True)
                ans_cache[state] = letter_of(pred) == gt
                return ans_cache[state]

            # G rollouts (no-grad sampling)
            rollouts = []
            with torch.no_grad():
                ctl_inputs = {}
                for g in range(args.group):
                    k, decisions = 1, []
                    while True:
                        if k not in ctl_inputs:
                            ctl_inputs[k] = build_inputs(
                                vlm, imgs[:k], CONTROL_PROMPT.format(pre=PRE, q=q))
                        inp = ctl_inputs[k]
                        out = model.generate(**inp, max_new_tokens=3,
                                             do_sample=True, temperature=1.0,
                                             pad_token_id=tok.eos_token_id)
                        txt = vlm.processor.decode(
                            out[0, inp.input_ids.shape[1]:],
                            skip_special_tokens=True).upper()
                        if "MOVE" in txt and k < n:
                            act = "MOVE"
                        elif "RENDER" in txt:
                            act = "RENDER"
                        else:
                            act = "STOP"
                        decisions.append((k, act))
                        if act == "MOVE":
                            k += 1
                        elif act == "RENDER":
                            correct = answer_from("render")
                            views = k + 1
                            break
                        else:
                            correct = answer_from(k)
                            views = k
                            break
                    reward = float(correct) - lam * max(0, views - 1) * args.cost
                    rollouts.append({"decisions": decisions, "reward": reward,
                                     "correct": correct, "views": views})
            rs = np.array([x["reward"] for x in rollouts])
            adv = (rs - rs.mean()) / (rs.std() + 1e-6)
            stats["acc"] += [x["correct"] for x in rollouts]
            stats["views"] += [x["views"] for x in rollouts]

            # policy-gradient re-forward on sampled control tokens
            loss_total = 0.0
            pairs = [] if abs(rs.std()) < 1e-6 else list(zip(rollouts, adv))
            for x, a in pairs:
                if abs(a) < 1e-4:
                    continue
                for k, act in x["decisions"]:
                    inp = build_inputs(vlm, imgs[:k],
                                       CONTROL_PROMPT.format(pre=PRE, q=q))
                    ids = torch.tensor([ACTION_IDS[act]], device=device)
                    full = torch.cat([inp.input_ids, ids], 1)
                    labels = torch.full_like(full, -100)
                    labels[:, inp.input_ids.shape[1]:] = ids
                    am = torch.ones_like(full)
                    out = model(input_ids=full, attention_mask=am,
                                pixel_values=inp.pixel_values,
                                image_grid_thw=inp.image_grid_thw, labels=labels)
                    loss = float(a) * out.loss / (args.group * 2)
                    loss.backward()
                    loss_total += loss.item()
            stats["loss"] += loss_total
        except Exception as e:
            print("item failed", r["id"], repr(e)[:120], flush=True)
        # sync boundary keyed to loop index — identical on every rank
        if (idx + 1) % args.accum_items == 0:
            if ddp:
                for p in model.parameters():
                    if p.requires_grad:
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            n_steps += 1
            mv = float(np.mean(stats["views"][-200:] or [0]))
            lam = max(0.0, lam + eta * (mv - args.view_budget))
            if rank == 0 and n_steps % 5 == 0:
                print(f"idx {idx+1} acc {np.mean(stats['acc'][-200:] or [0]):.3f} "
                      f"views {mv:.2f} lam {lam:.3f} "
                      f"loss {stats['loss']:.4f}", flush=True)
            stats["loss"] = 0.0
            if rank == 0 and n_steps % 25 == 0:
                model.save_pretrained(args.out)
    if rank == 0:
        model.save_pretrained(args.out)
        print("SAVED", args.out, flush=True)
    if ddp:
        dist.destroy_process_group()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

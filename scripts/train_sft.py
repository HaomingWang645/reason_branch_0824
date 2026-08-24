"""Stage I SFT: LoRA fine-tune Qwen2.5-VL-7B on control + answer examples.

Single- or multi-GPU (torchrun) data-parallel; vision tower frozen, LoRA on the
language model. Loss only on target tokens.
"""
import argparse
import json
import os
import random
import sys

import cv2
import torch
import torch.distributed as dist
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MC_ROOT = os.path.join(REPO, "data", "mindcube", "data")


def load_example(r, processor, max_pixels=448 * 448):
    from qwen_vl_utils import process_vision_info

    content = []
    for p in r["images"]:
        content.append({"type": "image",
                        "image": Image.open(os.path.join(MC_ROOT, p)).convert("RGB"),
                        "min_pixels": 224 * 224, "max_pixels": max_pixels})
    if r["render"]:
        rp = os.path.join(REPO, "data", "mindcube_renders", f"{r['render']}.png")
        content.append({"type": "image", "image": Image.open(rp).convert("RGB"),
                        "min_pixels": 224 * 224, "max_pixels": max_pixels})
    content.append({"type": "text", "text": r["prompt"]})
    messages = [{"role": "user", "content": content}]
    prompt_text = processor.apply_chat_template(messages, tokenize=False,
                                                add_generation_prompt=True)
    full_text = prompt_text + r["target"] + "<|im_end|>"
    image_inputs, _ = process_vision_info(messages)
    enc = processor(text=[full_text], images=image_inputs, return_tensors="pt")
    prompt_enc = processor(text=[prompt_text], images=image_inputs, return_tensors="pt")
    labels = enc.input_ids.clone()
    labels[:, : prompt_enc.input_ids.shape[1]] = -100
    enc["labels"] = labels
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO, "data", "sft_data.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--out", default=os.path.join(REPO, "checkpoints", "sft_lora"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=0)
    args = ap.parse_args()

    ddp = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"

    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    model.visual.requires_grad_(False)
    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)
    if rank == 0:
        model.print_trainable_parameters()
    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    rows = [json.loads(l) for l in open(args.data)]
    random.Random(0).shuffle(rows)
    rows = rows[rank::world]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    total_steps = (len(rows) * args.epochs) // args.accum
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1))

    step, accum_loss = 0, 0.0
    os.makedirs(args.out, exist_ok=True)
    for ep in range(args.epochs):
        for i, r in enumerate(rows):
            try:
                enc = load_example(r, processor).to(device)
            except Exception as e:
                if rank == 0:
                    print("skip", r["id"], e, flush=True)
                continue
            out = model(**enc)
            loss = out.loss / args.accum
            loss.backward()
            accum_loss += loss.item()
            if (i + 1) % args.accum == 0:
                if ddp:  # manual grad sync (no DDP wrapper; PEFT-friendly)
                    for p in model.parameters():
                        if p.requires_grad and p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if rank == 0 and step % 10 == 0:
                    print(f"ep{ep} step {step}/{total_steps} loss {accum_loss:.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
                accum_loss = 0.0
                if rank == 0 and step % 100 == 0:
                    model.save_pretrained(args.out)
                if args.max_steps and step >= args.max_steps:
                    break
        if args.max_steps and step >= args.max_steps:
            break
    if rank == 0:
        model.save_pretrained(args.out)
        print("SAVED", args.out, flush=True)
    if ddp:
        dist.destroy_process_group()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

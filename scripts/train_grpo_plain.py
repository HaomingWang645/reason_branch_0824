"""Baseline: SFT + GRPO on standard benchmark data, no world memory.
Policy answers from ALL given views (no control, no render). G sampled answers
per item, reward = correctness, group-relative advantage, policy gradient on
the sampled answer tokens. LoRA continues from --init adapter (plain SFT).
DDP: torchrun with DSE_GPUS=<comma list> for per-rank GPU pinning."""
import argparse, json, os, random, re, sys
import cv2, numpy as np
_lr = os.environ.get("LOCAL_RANK")
if _lr is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
    _g = os.environ.get("DSE_GPUS"); os.environ["CUDA_VISIBLE_DEVICES"] = (_g.split(",")[int(_lr)] if _g else _lr)
import torch, torch.distributed as dist
from datetime import timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viewtree.vlm import QwenVL
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); MC_ROOT = os.path.join(REPO, "data", "mindcube", "data")
PRE = "These images show a scene photographed from different viewpoints.\n"
SUFFIX = "\nAnswer with the option's letter from the given choices directly."

def build_inputs(vlm, images, prompt):
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    content = [{"type": "image", "image": Image.fromarray(im), "min_pixels": 224 * 224, "max_pixels": 448 * 448} for im in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = vlm.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    return vlm.processor(text=[text], images=image_inputs, return_tensors="pt").to(vlm.device)

def letter_of(t):
    m = re.findall(r"\b([A-F])\b", t.strip()); return m[-1] if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True); ap.add_argument("--out", required=True); ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--items", type=int, default=9995); ap.add_argument("--group", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-6); ap.add_argument("--accum-items", type=int, default=8)
    a = ap.parse_args()
    ddp = "RANK" in os.environ; rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    if ddp: torch.cuda.set_device(0); dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    vlm = QwenVL(a.model, device="cuda", adapter=a.init); model = vlm.model
    for n, p in model.named_parameters(): p.requires_grad = "lora" in n
    model.train(); tok = vlm.processor.tokenizer
    rows = [json.loads(l) for l in open(os.path.join(MC_ROOT, "raw", "MindCube_train.jsonl"))]
    random.Random(1).shuffle(rows); rows = rows[: a.items]; per = len(rows) // world; rows = rows[rank::world][:per]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    acc_hist = []; n_steps = 0
    for idx, r in enumerate(rows):
        try:
            imgs = [cv2.cvtColor(cv2.imread(os.path.join(MC_ROOT, p)), cv2.COLOR_BGR2RGB) for p in r["images"]]
            inp = build_inputs(vlm, imgs, PRE + r["question"] + SUFFIX); L = inp.input_ids.shape[1]
            gens, rewards = [], []
            with torch.no_grad():
                for g in range(a.group):
                    out = model.generate(**inp, max_new_tokens=4, do_sample=True, temperature=1.0, pad_token_id=tok.eos_token_id)
                    gen = out[0, L:]; txt = vlm.processor.decode(gen, skip_special_tokens=True)
                    gens.append(gen); rewards.append(float(letter_of(txt) == r["gt_answer"]))
            rs = np.array(rewards); acc_hist += rewards
            if rs.std() > 1e-6:
                adv = (rs - rs.mean()) / (rs.std() + 1e-6)
                for gen, ad in zip(gens, adv):
                    if abs(ad) < 1e-4 or len(gen) == 0: continue
                    full = torch.cat([inp.input_ids, gen[None]], 1); labels = torch.full_like(full, -100); labels[:, L:] = gen[None]
                    out = model(input_ids=full, attention_mask=torch.ones_like(full), pixel_values=inp.pixel_values, image_grid_thw=inp.image_grid_thw, labels=labels)
                    (float(ad) / a.group * out.loss).backward()
        except Exception as e:
            print("item failed", r["id"], repr(e)[:100], flush=True)
        if (idx + 1) % a.accum_items == 0:
            if ddp:
                for p in model.parameters():
                    if p.requires_grad:
                        if p.grad is None: p.grad = torch.zeros_like(p)
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad(set_to_none=True); n_steps += 1
            if n_steps % 5 == 0 and rank == 0: print(f"idx {idx+1} acc {np.mean(acc_hist[-300:]):.3f}", flush=True)
            if n_steps % 25 == 0 and rank == 0: model.save_pretrained(a.out)
    if rank == 0: model.save_pretrained(a.out); print("SAVED", a.out, flush=True)
    if ddp: dist.destroy_process_group()
    print("DONE", flush=True)

if __name__ == "__main__":
    main()

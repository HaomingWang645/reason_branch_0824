"""Phase 3: GRPO over whole camera walks (ViewTree-D).  From SFT-C, sample G
walks per question: at each state the policy emits STOP or a valid camera
move (masked: invalid moves are removed from the candidate set at sampling;
proposing one is penalised), renders come from the pose bank, the final answer
is generated at STOP.  Reward = correctness - lam*steps - 0.1*[masked proposal]
+ 0.05*[answer became correct after a step]; lam is a dual variable toward a
mean-steps budget.  Policy gradient on action tokens and final answer tokens.
DDP via torchrun + DSE_GPUS."""
import argparse, json, os, random, re, sys
import cv2, numpy as np
_lr = os.environ.get("LOCAL_RANK")
if _lr is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
    _g = os.environ.get("DSE_GPUS"); os.environ["CUDA_VISIBLE_DEVICES"] = (_g.split(",")[int(_lr)] if _g else _lr)
import torch, torch.distributed as dist
from datetime import timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viewtree.posebank import transition
from viewtree.vlm import QwenVL
from viewtree.score import parse_number
from build_phase1 import ACTIONS, PRE, WALK, describe
from train_sft_c import CTRL
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def build_inputs(vlm, images, prompt):
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    content = [{"type": "image", "image": Image.fromarray(im), "min_pixels": 224 * 224, "max_pixels": 448 * 448} for im in images]
    content.append({"type": "text", "text": prompt}); messages = [{"role": "user", "content": content}]
    text = vlm.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True); image_inputs, _ = process_vision_info(messages)
    return vlm.processor(text=[text], images=image_inputs, return_tensors="pt").to(vlm.device)

def correct(r, pred):
    if r["numeric"]:
        p = parse_number(pred); g = parse_number(r["a"])
        if p is None or g is None: return 0.0
        rel = abs(p - g) / max(abs(g), 1e-8); return float(np.mean(rel < (1.0 - np.arange(0.5, 1.0, 0.05))))
    m = re.search(r"\b([A-F])\b", pred.strip()); return float(bool(m) and m.group(1) == r["a"].strip())

def pg_loss(model, inp, gen_ids, weight):
    full = torch.cat([inp.input_ids, gen_ids[None]], 1); labels = torch.full_like(full, -100); labels[:, inp.input_ids.shape[1]:] = gen_ids[None]
    out = model(input_ids=full, attention_mask=torch.ones_like(full), pixel_values=inp.pixel_values, image_grid_thw=inp.image_grid_thw, labels=labels)
    return weight * out.loss

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--init", required=True); ap.add_argument("--out", required=True); ap.add_argument("--qa", default=os.path.join(REPO, "data/train3r/qa_all.jsonl"))
    ap.add_argument("--bank", default=os.path.join(REPO, "data/posebank")); ap.add_argument("--items", type=int, default=120000); ap.add_argument("--per-scene", type=int, default=80)
    ap.add_argument("--group", type=int, default=6); ap.add_argument("--lr", type=float, default=2e-6); ap.add_argument("--accum-items", type=int, default=8)
    ap.add_argument("--step-budget", type=float, default=1.2); ap.add_argument("--cost", type=float, default=0.2); ap.add_argument("--max-depth", type=int, default=3); ap.add_argument("--curriculum", type=float, default=0.3); ap.add_argument("--skip-frac", type=float, default=0.0)
    a = ap.parse_args()
    ddp = "RANK" in os.environ; rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    if ddp: torch.cuda.set_device(0); dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    vlm = QwenVL("Qwen/Qwen2.5-VL-7B-Instruct", device="cuda", adapter=a.init); model = vlm.model
    for n_, p in model.named_parameters(): p.requires_grad = "lora" in n_
    model.train(); tok = vlm.processor.tokenizer
    import collections; by_scene = collections.defaultdict(list)
    for l in open(a.qa):
        r = json.loads(l); by_scene[r["scene"]].append(r)
    rng = random.Random(1); items = []
    for scene in sorted(by_scene):
        src = "scannet" if os.path.exists(os.path.join(a.bank, "scannet", scene, "bank.json")) else ("scannetppv2" if os.path.exists(os.path.join(a.bank, "scannetppv2", scene, "bank.json")) else None)
        if src is None: continue
        qs = by_scene[scene][:]; rng.shuffle(qs); items += [(src, r) for r in qs[: a.per_scene]]
    rng.shuffle(items); items = items[: a.items]; items = items[int(len(items) * a.skip_frac):]; per = len(items) // world; items = items[rank::world][:per]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr); lam, eta = 0.0, 0.02
    stats = {"acc": [], "steps": [], "masked": []}; n_steps = 0; banks = {}
    for idx, (src, r) in enumerate(items):
        try:
            sd = os.path.join(a.bank, src, r["scene"])
            if sd not in banks:
                b = json.load(open(os.path.join(sd, "bank.json"))); banks = {sd: (b["bank"], {int(k): v for k, v in b["fwd_map"].items()}, b["meta"])}
            bank, fwd, meta = banks[sd]
            frames = [cv2.cvtColor(cv2.imread(os.path.join(sd, f"frame_{i:02d}.jpg")), cv2.COLOR_BGR2RGB) for i in range(8)]
            imgc = {}
            def view(i):
                if i not in imgc: imgc[i] = cv2.cvtColor(cv2.imread(os.path.join(sd, f"view_{i:03d}.jpg")), cv2.COLOR_BGR2RGB)
                return imgc[i]
            maxd = 2 if idx < a.curriculum * len(items) else a.max_depth
            starts = [e["idx"] for e in bank if e["kind"] == "eye" and e["valid"]]
            rollouts = []
            with torch.no_grad():
                for g in range(a.group):
                    path, decisions, masked, prev_correct, improved = [], [], 0, None, 0
                    while True:
                        cur = path[-1][1] if path else None
                        valid = ["STOP"] + ([x for x in ACTIONS if transition(bank, fwd, cur, x, meta) is not None] if cur is not None else ["START"])
                        if len(path) >= maxd: valid = ["STOP"]
                        renders = [view(i) for _, i in path]
                        prompt = CTRL.format(pre=PRE.format(k=8), walk=(WALK.format(m=len(renders), steps=describe(bank, path)) if renders else ""), q=r["q"], valid=", ".join(valid))
                        inp = build_inputs(vlm, frames + renders, prompt); L = inp.input_ids.shape[1]
                        out = model.generate(**inp, max_new_tokens=4, do_sample=True, temperature=1.0, pad_token_id=tok.eos_token_id)
                        gen = out[0, L:]; txt = vlm.processor.decode(gen, skip_special_tokens=True).strip().upper().replace(" ", "_")
                        act = next((v for v in valid if txt.startswith(v)), None)
                        if act is None: masked += 1; act = "STOP"  # proposed an invalid/masked move -> penalised, treated as STOP
                        decisions.append((list(path), act, gen.clone()))
                        if act == "STOP": break
                        if act == "START": j = rng.choice(starts)
                        else: j = transition(bank, fwd, cur, act, meta)
                        path = path + [(act, j)]
                        if bank[j]["kind"] == "topdown": 
                            decisions.append((list(path), "STOP", None)); break
                    renders = [view(i) for _, i in path]
                    aprompt = PRE.format(k=8) + (WALK.format(m=len(renders), steps=describe(bank, path)) if renders else "") + r["q"]
                    ainp = build_inputs(vlm, frames + renders, aprompt); L = ainp.input_ids.shape[1]
                    out = model.generate(**ainp, max_new_tokens=12, do_sample=True, temperature=0.7, pad_token_id=tok.eos_token_id)
                    agen = out[0, L:]; c = correct(r, vlm.processor.decode(agen, skip_special_tokens=True))
                    reward = c - lam * len(path) * a.cost - 0.1 * masked
                    rollouts.append(dict(decisions=decisions, path=path, reward=reward, correct=c, agen=agen, ainp=ainp, masked=masked))
            rs = np.array([x["reward"] for x in rollouts]); stats["acc"] += [x["correct"] for x in rollouts]; stats["steps"] += [len(x["path"]) for x in rollouts]; stats["masked"] += [x["masked"] for x in rollouts]
            if rs.std() > 1e-6:
                adv = (rs - rs.mean()) / (rs.std() + 1e-6)
                for x, ad in zip(rollouts, adv):
                    if abs(ad) < 1e-4: continue
                    w = float(ad) / a.group
                    for path, act, gen in x["decisions"]:
                        if gen is None: continue
                        renders = [view(i) for _, i in path]
                        valid = ["STOP"] + ([xx for xx in ACTIONS if transition(bank, fwd, path[-1][1], xx, meta) is not None] if path else ["START"])
                        prompt = CTRL.format(pre=PRE.format(k=8), walk=(WALK.format(m=len(renders), steps=describe(bank, path)) if renders else ""), q=r["q"], valid=", ".join(valid))
                        inp = build_inputs(vlm, frames + renders, prompt); (pg_loss(model, inp, gen, w) / 4).backward()
                    (pg_loss(model, x["ainp"], x["agen"], w) / 2).backward()
        except Exception as e:
            print("item failed", r.get("scene"), repr(e)[:120], flush=True); torch.cuda.empty_cache()
        if (idx + 1) % a.accum_items == 0:
            if ddp:
                for p in model.parameters():
                    if p.requires_grad:
                        if p.grad is None: p.grad = torch.zeros_like(p)
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); opt.step(); opt.zero_grad(set_to_none=True); n_steps += 1
            ms = float(np.mean(stats["steps"][-300:] or [0])); lam = max(0.0, lam + eta * (ms - a.step_budget))
            if n_steps % 5 == 0 and rank == 0: print(f"idx {idx+1} acc {np.mean(stats['acc'][-300:]):.3f} steps {ms:.2f} masked {np.mean(stats['masked'][-300:]):.2f} lam {lam:.3f}", flush=True)
            if n_steps % 25 == 0 and rank == 0: model.save_pretrained(a.out)
    if rank == 0: model.save_pretrained(a.out); print("SAVED", a.out, flush=True)
    if ddp: dist.destroy_process_group()
    print("DONE", flush=True)

if __name__ == "__main__":
    main()

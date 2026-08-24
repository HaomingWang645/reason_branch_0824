"""ViewTree-lite: training-free end-to-end tree inference.

Implements the full ViewTree inference loop (design doc §5.9) with zero trained
components, standing in for the learned controller/confidence head:
  - root step: VLM answers from frames OR requests exploration (STOP-vs-MOVE)
  - branch: B candidate overview viewpoints rendered from the VGGT point cloud
  - confidence: mean answer-token log-probability per branch (the doc's H4
    baseline "token confidence"), used for Top-K pruning
  - fuse: retained branches' views presented jointly, pose-tagged, for the
    final answer
Budget: max B candidate branches, keep K, depth 1 (branch -> fuse).
"""
import numpy as np
import torch

from .render import overview_poses, render
from .vlm import MC_SUFFIX, NUM_SUFFIX


ROOT_GATE = (
    "Question: {q}\n"
    "Can you answer this question confidently from these video frames alone? "
    "Reply with exactly one word: YES or EXPLORE."
)

BRANCH_PRE = (
    "The first {k} images are frames of a video captured while moving through a room. "
    "The final image is a novel view of the SAME room rendered from a 3D "
    "reconstruction (it may contain holes). Rendered view description: {desc}\n"
)

FUSE_PRE = (
    "The first {k} images are frames of a video captured while moving through a room. "
    "The remaining {m} images are novel views of the SAME room rendered from a 3D "
    "reconstruction (holes possible): {descs}.\n"
)

VIEW_DESCS = [
    "elevated view from side 1", "elevated view from side 2",
    "elevated view from side 3", "elevated view from side 4",
    "top-down overhead view",
]


@torch.no_grad()
def state_feature(vlm, inputs):
    """Last-layer hidden state of the final prompt token (confidence feature)."""
    out = vlm.model(**inputs, output_hidden_states=True)
    return out.hidden_states[-1][0, -1].float().cpu()


@torch.no_grad()
def answer_logprob(vlm, images, prompt, max_new_tokens=32, want_feature=False):
    """Greedy answer + mean token log-prob (training-free confidence)."""
    from qwen_vl_utils import process_vision_info
    from PIL import Image

    content = []
    for im in images:
        if isinstance(im, np.ndarray):
            im = Image.fromarray(im)
        content.append({"type": "image", "image": im,
                        "min_pixels": 224 * 224, "max_pixels": vlm.max_pixels})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = vlm.processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = vlm.processor(text=[text], images=image_inputs, padding=True,
                           return_tensors="pt").to(vlm.device)
    out = vlm.model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        output_scores=True, return_dict_in_generate=True,
        pad_token_id=vlm.processor.tokenizer.eos_token_id,
    )
    seq = out.sequences[0, inputs.input_ids.shape[1]:]
    lps = []
    for tok, score in zip(seq, out.scores):
        lp = torch.log_softmax(score[0].float(), -1)[tok]
        lps.append(lp.item())
        if tok == vlm.processor.tokenizer.eos_token_id:
            break
    pred = vlm.processor.decode(seq, skip_special_tokens=True)
    mean_lp = float(np.mean(lps)) if lps else -99.0
    if want_feature:
        return pred, mean_lp, state_feature(vlm, inputs)
    return pred, mean_lp


def build_q(row):
    if row["options"]:
        return row["question"] + "\n" + "\n".join(row["options"]) + "\n" + MC_SUFFIX
    return row["question"] + "\n" + NUM_SUFFIX


@torch.no_grad()
def run_tree(vlm, row, frames, rec, keep_k=2, num_branches=5, splat=2):
    """Returns (pred, trace)."""
    trace = {"branches": []}
    qtext = build_q(row)
    base = [frames[i] for i in np.linspace(0, len(frames) - 1, 8).round().astype(int)]

    # 1. root gate: STOP vs explore
    gate, _ = answer_logprob(vlm, base, ROOT_GATE.format(q=row["question"]),
                             max_new_tokens=4)
    trace["gate"] = gate.strip()
    direct_pred, direct_lp = answer_logprob(
        vlm, base, "These are frames of a video.\n" + qtext)
    if "YES" in gate.upper():
        trace["mode"] = "direct"
        return direct_pred, trace

    # 2. branch: render candidate views (cheap heuristic proposals)
    H, W = rec["size"]
    K = rec["intrinsics"][0]
    poses = overview_poses(rec)[:num_branches]
    views = []
    for pose in poses:
        img = render(rec["points"], rec["colors"], pose, K, H, W, splat=splat)
        views.append((img.clamp(0, 1) * 255).byte().cpu().numpy())

    # 3. per-branch answer + token confidence
    scored = []
    for vi, v in enumerate(views):
        pre = BRANCH_PRE.format(k=len(base), desc=VIEW_DESCS[vi])
        pred, lp = answer_logprob(vlm, base + [v], pre + qtext)
        scored.append({"view": vi, "pred": pred, "logprob": lp})
    trace["branches"] = scored

    # 4. prune to top-K by confidence; disagreement-aware keep
    order = sorted(range(len(scored)), key=lambda i: -scored[i]["logprob"])
    kept = order[:keep_k]
    preds = {scored[i]["pred"].strip() for i in kept}
    if len(preds) == 1 and scored[kept[0]]["logprob"] > direct_lp:
        # consensus among retained branches -> early stop with that answer
        trace["mode"] = "branch_consensus"
        return scored[kept[0]]["pred"], trace

    # 5. fuse retained evidence
    kept_views = [views[scored[i]["view"]] for i in kept]
    descs = ", ".join(VIEW_DESCS[scored[i]["view"]] for i in kept)
    pre = FUSE_PRE.format(k=len(base), m=len(kept_views), descs=descs)
    fuse_pred, fuse_lp = answer_logprob(vlm, base + kept_views, pre + qtext)
    trace["mode"] = "fused"
    trace["fuse_logprob"] = fuse_lp
    # final selection: best confidence among fuse and direct
    if direct_lp > fuse_lp and direct_lp > max(scored[i]["logprob"] for i in kept):
        trace["mode"] = "fused_fallback_direct"
        return direct_pred, trace
    return fuse_pred, trace

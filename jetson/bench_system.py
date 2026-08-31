"""System benchmark (latency / memory / energy) for ViewTree on Jetson AGX Orin.

W0-style bring-up measurement per DESIGN_MOBILE_JETSON.md: the unmodified torch
pipeline (bf16, sdpa), random input data, un-finetuned base model. Accuracy is
NOT measured — only system cost, with call shapes (image counts, token counts,
reasoning depth) exactly matching the server implementation:

  frames16   1 call/question: 16 frames, decode 32          (scripts/run_eval.py)
  memory32   scene: VGGT(32)+5 renders; 1 call: 17 imgs     (scripts/run_eval.py)
  tree1      scene: VGGT(32); direct route = 2 calls,
             full route = 8 calls (gate/direct/5 branch/fuse) (viewtree/tree.py)
  vtd        scene: VGGT(32)+pose bank; beam walk to forced
             depth 0..3, worst-case no-consensus              (scripts/depth/run_tree_d.py)

Decode lengths are forced (min_new_tokens=max_new_tokens) so random inputs
cannot shorten reasoning. The value-head feature is read from the prefill's
hidden states inside generate() (the mobile plan's backend contract) instead of
the server path's second full forward; that overhead is measured separately.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

# Jetson compat: NV torch 2.5.0a0 lacks SDPA's enable_gqa kwarg, but
# transformers' version check treats 2.5.0a0 as >= 2.5. Force the repeat_kv
# fallback path (numerically identical, slightly more memory).
try:
    import transformers.integrations.sdpa_attention as _sdpa_mod
    _sdpa_mod.use_gqa_in_sdpa = lambda *a, **k: False
except Exception:
    pass

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "depth"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from telemetry import Telemetry

from viewtree.render import render, overview_poses  # noqa: E402
from viewtree.posebank import build_pose_bank, transition  # noqa: E402
from viewtree.tree import ROOT_GATE, BRANCH_PRE, FUSE_PRE, VIEW_DESCS, build_q  # noqa: E402
from build_phase1 import ACTIONS, PRE, WALK, describe  # noqa: E402
from train_sft_c import CTRL  # noqa: E402

QUESTION = ("How many chairs are in this room?\nA. 1\nB. 2\nC. 3\nD. 4\n"
            "Answer with the option's letter from the given choices directly.")
ROW = {"question": "How many chairs are in this room?",
       "options": ["A. 1", "B. 2", "C. 3", "D. 4"]}

RNG = np.random.default_rng(0)


def make_frames(n, h=1080, w=1920):
    """Random context frames: smooth low-frequency noise (more realistic token
    stats than white noise; identical compute either way)."""
    small = RNG.integers(0, 256, size=(n, h // 8, w // 8, 3), dtype=np.uint8)
    import cv2
    return [cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR) for s in small]


def cuda_mem_gb():
    return torch.cuda.max_memory_allocated() / 1e9


def last_win(tel):
    """Stats for the most recently closed telemetry window only."""
    d = tel._window_stats(*tel.windows[-1])
    d.pop("label", None)  # avoid collision with the record's own label
    return d


class Bench:
    def __init__(self, tel, out_path):
        self.tel = tel
        self.records = []
        self.out_path = out_path

    def log(self, **kw):
        self.records.append(kw)
        with open(self.out_path, "w") as f:
            json.dump(self.records, f, indent=1)

    # ---- timed primitive: one VLM call ---------------------------------
    @torch.no_grad()
    def vlm_call(self, vlm, images, prompt, decode_tokens, want_feature=False,
                 prefill_only=False, label="call"):
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
        t0 = time.monotonic()
        text = vlm.processor.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = vlm.processor(text=[text], images=image_inputs, padding=True,
                               return_tensors="pt").to(vlm.device)
        torch.cuda.synchronize()
        t1 = time.monotonic()  # preprocessing done
        n_prompt = int(inputs.input_ids.shape[1])
        feature = None
        if prefill_only:
            logits = vlm.model(**inputs).logits[0, -1]
            torch.cuda.synchronize()
            t2 = t3 = time.monotonic()
            n_dec = 0
        else:
            out = vlm.model.generate(
                **inputs, max_new_tokens=decode_tokens,
                min_new_tokens=decode_tokens, do_sample=False,
                output_hidden_states=want_feature, return_dict_in_generate=True,
                pad_token_id=vlm.processor.tokenizer.eos_token_id)
            torch.cuda.synchronize()
            t3 = time.monotonic()
            t2 = None  # prefill/decode split not observable inside generate
            n_dec = int(out.sequences.shape[1] - n_prompt)
            if want_feature:
                feature = out.hidden_states[0][-1][0, -1].float()
        dt = time.monotonic() - t0
        rec = dict(label=label, n_images=len(images),
                   n_prompt_tokens=n_prompt, n_decode_tokens=n_dec,
                   preproc_s=t1 - t0, total_s=dt,
                   model_s=t3 - t1)
        return rec, feature

    def run_head(self, head, feature):
        t0 = time.monotonic()
        with torch.no_grad():
            v = float(torch.sigmoid(head(feature[None]).squeeze()))
        return v, time.monotonic() - t0


def load_models():
    from viewtree.vlm import QwenVL
    t0 = time.monotonic()
    vlm = QwenVL("Qwen/Qwen2.5-VL-7B-Instruct")
    t_vlm = time.monotonic() - t0
    t0 = time.monotonic()
    from viewtree.reconstruct import _get_model
    _get_model("cuda")
    t_vggt = time.monotonic() - t0
    head = torch.nn.Sequential(
        torch.nn.Linear(3584, 512), torch.nn.GELU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(512, 1)).to("cuda").float().eval()
    return vlm, head, t_vlm, t_vggt


def scene_setup(bench, frames, n_frames, tag):
    """VGGT reconstruction, timed; returns rec."""
    from viewtree.reconstruct import reconstruct
    tel = bench.tel
    torch.cuda.reset_peak_memory_stats()
    with tel.window(f"recon_{tag}"):
        rec = reconstruct(frames[:n_frames])
        torch.cuda.synchronize()
    bench.log(kind="phase", label=f"recon_{tag}", n_frames=n_frames,
              n_points=int(rec["points"].shape[0]),
              cuda_peak_gb=cuda_mem_gb(), **last_win(tel))
    # drop per-frame maps (dead weight for inference; mobile plan drops them)
    for k in ("world_maps", "color_maps", "mask_maps"):
        rec.pop(k, None)
    torch.cuda.empty_cache()
    return rec


def q_direct_route(bench, vlm, head, base, decode, label):
    """gate + direct answer (the 71%-frequency path in both tree methods)."""
    recs = []
    r, _ = bench.vlm_call(vlm, base, ROOT_GATE.format(q=ROW["question"]), 4,
                          label="gate")
    recs.append(r)
    r, ft = bench.vlm_call(vlm, base, "These are frames of a video.\n" + QUESTION,
                           decode, want_feature=True, label="direct")
    recs.append(r)
    _, ht = bench.run_head(head, ft)
    return recs, ht


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "jetson", "results",
                                                  "bench_raw.json"))
    ap.add_argument("--nq", type=int, default=3, help="questions per config")
    ap.add_argument("--micro-reps", type=int, default=3)
    ap.add_argument("--skip-micro", action="store_true")
    ap.add_argument("--methods", default="frames16,memory32,tree1,vtd")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    tel = Telemetry(hz=20)
    idle_w = tel.baseline_idle_w(5.0)
    tel.start()
    bench = Bench(tel, a.out)
    bench.log(kind="meta", idle_total_w=idle_w, powermode="MAXN",
              device="Jetson AGX Orin 64GB", ts=time.strftime("%F %T"))

    # ---- load models ----
    torch.cuda.reset_peak_memory_stats()
    with tel.window("load_models"):
        vlm, head, t_vlm, t_vggt = load_models()
    bench.log(kind="phase", label="load_models", vlm_load_s=t_vlm,
              vggt_load_s=t_vggt, cuda_peak_gb=cuda_mem_gb(),
              **last_win(tel))

    frames = make_frames(32)
    base = [frames[i] for i in np.linspace(0, 31, 8).round().astype(int)]

    # ---- warmup ----
    for _ in range(2):
        bench.vlm_call(vlm, base[:2], "warmup " + QUESTION, 4, label="warmup")

    methods = a.methods.split(",")

    # ---- scene setup (shared across tree methods) ----
    rec = scene_setup(bench, frames, 32, "32f")
    H, W = rec["size"]; K = rec["intrinsics"][0]

    with tel.window("posebank_build"):
        bank, fwd, meta = build_pose_bank(rec, render_all=False)
        for e in bank:
            e["valid"] = True
    bench.log(kind="phase", label="posebank_build", n_entries=len(bank),
              **last_win(tel))

    rcache = {}
    render_times = []

    def view(i):
        if i not in rcache:
            t0 = time.monotonic()
            img = render(rec["points"], rec["colors"],
                         torch.tensor(bank[i]["extrinsic"],
                                      device=rec["points"].device),
                         K, H, W, splat=2)
            out = (img.clamp(0, 1) * 255).byte().cpu().numpy()
            torch.cuda.synchronize()
            render_times.append(time.monotonic() - t0)
            rcache[i] = out
        return rcache[i]

    # 5 human_poses renders for memory32 / tree1
    with tel.window("renders_5"):
        poses = overview_poses(rec)[:5]
        views5 = []
        for pose in poses:
            img = render(rec["points"], rec["colors"], pose, K, H, W, splat=2)
            views5.append((img.clamp(0, 1) * 255).byte().cpu().numpy())
        torch.cuda.synchronize()
    bench.log(kind="phase", label="renders_5", **last_win(tel))

    # ---- microbenchmarks: VLM call shapes ----
    if not a.skip_micro:
        shapes = [  # (n_frames_from_base, n_renders, decode, prefill_only, label)
            (8, 0, 4, False, "gate_8i_d4"),
            (8, 0, 12, False, "ans_8i_d12"),
            (8, 0, 32, False, "ans_8i_d32"),
            (9, 1, 12, False, "ans_9i_d12"),
            (10, 2, 12, False, "ans_10i_d12"),
            (11, 3, 12, False, "ans_11i_d12"),
            (9, 1, 0, True, "ctrl_9i_prefill"),
            (16, 0, 32, False, "ans_16i_d32"),
            (17, 5, 32, False, "ans_17i_d32"),
        ]
        for nf, nr, dec, pf, label in shapes:
            imgs = (base * 2)[:nf - nr] + views5[:nr]
            for rep in range(a.micro_reps):
                with tel.window(f"micro_{label}_{rep}"):
                    r, _ = bench.vlm_call(vlm, imgs, QUESTION, dec,
                                          prefill_only=pf, label=label)
                bench.log(kind="micro", rep=rep, **r, **last_win(tel))
        # naive server path overhead: extra full forward for the feature
        for rep in range(a.micro_reps):
            with tel.window(f"micro_feat_fwd_{rep}"):
                r, _ = bench.vlm_call(vlm, base, QUESTION, 0, prefill_only=True,
                                      label="feature_extra_forward")
            bench.log(kind="micro", rep=rep, **r, **last_win(tel))

    # ---- end-to-end per question ----
    if "frames16" in methods:
        imgs16 = (frames * 1)[::2][:16]
        for q in range(a.nq):
            with tel.window(f"q_frames16_{q}"):
                r, _ = bench.vlm_call(vlm, imgs16,
                                      "These are frames of a video.\n" + QUESTION,
                                      32, label="frames16")
            bench.log(kind="question", method="frames16", q=q, calls=1,
                      **last_win(tel))

    if "memory32" in methods:
        for q in range(a.nq):
            with tel.window(f"q_memory32_{q}"):
                imgs = (frames[::3][:12]) + views5
                r, _ = bench.vlm_call(vlm, imgs,
                                      FUSE_PRE.format(k=12, m=5,
                                                      descs=", ".join(VIEW_DESCS))
                                      + QUESTION, 32, label="memory32")
            bench.log(kind="question", method="memory32", q=q, calls=1,
                      **last_win(tel))

    if "tree1" in methods:
        # route A: gated direct (2 calls)
        for q in range(a.nq):
            with tel.window(f"q_tree1_direct_{q}"):
                recs, _ = q_direct_route(bench, vlm, head, base, 32, "tree1")
            bench.log(kind="question", method="tree1_direct", q=q,
                      calls=len(recs), **last_win(tel))
        # route B: full tree (8 calls: gate, direct, 5 branch, fuse)
        for q in range(a.nq):
            with tel.window(f"q_tree1_full_{q}"):
                recs, _ = q_direct_route(bench, vlm, head, base, 32, "tree1")
                for vi, v in enumerate(views5):
                    pre = BRANCH_PRE.format(k=8, desc=VIEW_DESCS[vi])
                    r, ft = bench.vlm_call(vlm, base + [v], pre + QUESTION, 32,
                                           want_feature=True, label="branch")
                    recs.append(r)
                    bench.run_head(head, ft)
                pre = FUSE_PRE.format(k=8, m=2,
                                      descs=", ".join(VIEW_DESCS[:2]))
                r, ft = bench.vlm_call(vlm, base + views5[:2], pre + QUESTION,
                                       32, want_feature=True, label="fuse")
                recs.append(r)
                bench.run_head(head, ft)
            bench.log(kind="question", method="tree1_full", q=q,
                      calls=len(recs), **last_win(tel))

    if "vtd" in methods:
        # beam walk, forced to reach target depth (no consensus stop),
        # worst-case call counts; renders cached across questions of the scene
        # (matches run_tree_d.py rcache) — first question pays cold renders.
        def vtd_question(depth_target, q):
            calls = []
            with tel.window(f"q_vtd_d{depth_target}_{q}"):
                recs, _ = q_direct_route(bench, vlm, head, base, 12,
                                         f"vtd_d{depth_target}")
                calls += recs
                if depth_target >= 1:
                    beam = [[("start at spot", e["idx"])] for e in bank
                            if e["kind"] == "eye" and e["yaw"] == 0][:3]
                    scored = {}
                    for d in range(1, depth_target + 1):
                        for path in beam:
                            key2 = tuple(i for _, i in path)
                            if key2 in scored:
                                continue
                            renders = [view(i) for _, i in path]
                            prompt = (PRE.format(k=8)
                                      + WALK.format(m=len(renders),
                                                    steps=describe(bank, path))
                                      + QUESTION)
                            r, ft = bench.vlm_call(vlm, base + renders, prompt,
                                                   12, want_feature=True,
                                                   label=f"ans_lvl{d}")
                            calls.append(r)
                            v, _ = bench.run_head(head, ft)
                            scored[key2] = (v, path)
                        level = sorted(
                            (scored[tuple(i for _, i in p)] for p in beam),
                            key=lambda z: -z[0])
                        kept = level[:2]
                        if d == depth_target:
                            break
                        nxt = []
                        for _, path in kept:
                            cur = path[-1][1]
                            if bank[cur]["kind"] == "topdown":
                                continue
                            valid = [x for x in ACTIONS
                                     if transition(bank, fwd, cur, x, meta)
                                     is not None
                                     and (x != "BIRD_EYE" or d == depth_target - 1)]
                            if not valid:
                                continue
                            renders = [view(i) for _, i in path]
                            prompt = CTRL.format(
                                pre=PRE.format(k=8),
                                walk=WALK.format(m=len(renders),
                                                 steps=describe(bank, path)),
                                q=ROW["question"],
                                valid=", ".join(["STOP"] + valid))
                            r, sc = bench.vlm_call(vlm, base + renders, prompt,
                                                   0, prefill_only=True,
                                                   label=f"ctrl_lvl{d}")
                            calls.append(r)
                            rng2 = np.random.default_rng(d * 100 + q)
                            top = list(rng2.permutation(valid))[:3]
                            for act in top:
                                j = transition(bank, fwd, cur, act, meta)
                                if j is not None and j not in [i for _, i in path]:
                                    nxt.append(path + [(act, j)])
                        if not nxt:
                            break
                        beam = nxt
            n_dec = sum(c["n_decode_tokens"] for c in calls)
            n_pt = sum(c["n_prompt_tokens"] for c in calls)
            bench.log(kind="question", method=f"vtd_d{depth_target}", q=q,
                      calls=len(calls), decode_tokens=n_dec, prompt_tokens=n_pt,
                      **last_win(tel))

        for depth_target in (0, 1, 2, 3):
            for q in range(a.nq if depth_target < 2 else max(2, a.nq - 1)):
                vtd_question(depth_target, q)

    if render_times:
        bench.log(kind="phase", label="render_per_view_lazy",
                  n=len(render_times),
                  mean_s=float(np.mean(render_times)),
                  p95_s=float(np.percentile(render_times, 95)))

    tel.stop()
    bench.log(kind="meta_end", ts=time.strftime("%F %T"),
              cuda_peak_gb_overall=cuda_mem_gb())
    print("DONE", a.out)


if __name__ == "__main__":
    main()

# ViewTree — Execution Decisions Beyond the Design Document

Everything in this file is **added by the implementation effort on 2026-08-24** and is
not specified in `ViewTree_Research_Design_Document.pdf`. The design doc left five
blocker categories open; per instruction, the mobile stack (blocker 4) is **ignored**
— all experiments run on server GPUs.

## 0. Hardware actually used

8× NVIDIA H100 (6× NVL 96GB, 2× PCIe 80GB), 1.5 TB RAM, 256 CPU cores, single node.

## 1. Model choices (design doc §3.1 left these open)

| Role | Choice | Rationale |
|---|---|---|
| VLM controller ("lightweight VLM") | **Qwen2.5-VL-7B-Instruct** | Already cached locally; strong spatial baseline in VSI-Bench literature; Qwen2.5-VL-3B can be swapped in later for the true mobile-class model. |
| Teacher VLM (Stage I SFT, later) | **Qwen2.5-VL-32B-Instruct** (cached) | Same family → same prompt/token conventions for distillation. |
| Reconstruction backbone | **facebook/VGGT-1B, frozen** | As prescribed by the doc (§5.2); already cached. |
| Confidence head (later stages) | Linear head on last-layer hidden state of a designated token | Doc §5.5; simplest instantiation first. |

## 2. Renderer choice (design doc §3.1 "Renderer" was a named box)

**Custom pure-PyTorch GPU point-splat renderer** (`viewtree/render.py`):
z-buffered splatting (3×3 pixel splats, two-pass scatter-reduce depth test),
white background, ~2.4 M points per 16-frame scene, renders a 518-wide view in
milliseconds on H100. No third-party graphics dependency (no Open3D/PyTorch3D/
nvdiffrast), so the same code path can later be ported to mobile.

Geometry source: VGGT depth-map branch + estimated cameras, unprojected to world
points (the depth+camera route is more accurate than VGGT's direct point-map head,
per the VGGT paper). Points below the 30th percentile of VGGT depth-confidence are
dropped (doc's "reliability metadata" made concrete).

**Viability check protocol (added):** one VGGT pass on 16 uniform frames; build the
cloud from the 8 even-indexed frames only; render at the 8 odd-indexed held-out
camera poses; compare to the real frames (PSNR, global SSIM, pixel coverage).
This measures *novel-view* quality, not self-reprojection. Script:
`scripts/render_check.py`. Run on 60 scenes (20 per source dataset).

## 3. Benchmark + first experiment (design doc §8.2, §9.2 check #1)

- Primary benchmark: **VSI-Bench** (full test split: 5,130 questions, 288 scenes;
  videos from ScanNet / ScanNet++ / ARKitScenes, all local).
- Scoring: official protocol — exact-match accuracy for multiple-choice;
  Mean Relative Accuracy (mean over thresholds θ ∈ {0.50…0.95, step 0.05} of
  1[relative error < 1−θ]) for the 4 numerical question types. Headline number =
  mean over the 10 question types.
- Prompts: VSI-Bench standard suffixes ("Answer with the option's letter…",
  "Do not respond with anything other than a single number!"), greedy decoding,
  max 32 new tokens, per-image cap 448² pixels (min 224²).

### Conditions (RQ1 instantiation — added; the doc only names them abstractly)

| Condition | Input to VLM | Design-doc role |
|---|---|---|
| `current` | last video frame only | Current-view baseline (§8.4) |
| `frames16` | 16 uniformly sampled frames | History-frame baseline (§8.4) |
| `memory` | 12 uniform frames + **5 rendered overview views** (4 elevated oblique at 45°-spaced azimuths + 1 top-down, aimed at the confident-point centroid; world-up estimated as −mean camera down-axis) | Explicit-memory condition, M0-level: no learned controller yet — a *fixed heuristic view policy* stands in for the controller to test whether reconstructed novel views add information at all |

All conditions share the same VLM, decoding, and scoring (doc §8.5 requirement).
Evaluation is paired: every condition answers all 5,130 questions.

### GPU allocation for the first run (all 8 H100s)

- GPU 0: `current` (1 shard)
- GPU 1–2: `frames16` (2 shards, sharded by scene)
- GPU 3–6: `memory` (4 shards; VGGT reconstruction runs once per scene, renders cached to `data/renders/`)
- GPU 7: `render_check.py` (60 scenes), then free for reruns

## 4. Quantities the doc left blank

- Frames per scene fed to VGGT: **16** at width 518 (VGGT native).
- Rendered views per scene: **5**; render resolution = VGGT frame resolution.
- Render-check scenes: **60**; held-out views per scene: **8**.
- First-run compute estimate: ≤ ~3 GPU-hours per condition shard (measured, see results).

## 5. Deferred checks resolved or noted

- **Dataset license note:** VSI-Bench is distributed under Apache-2.0; underlying
  ScanNet/ScanNet++/ARKitScenes have their own research-use terms — reconstruction
  and rendering here are internal research use, consistent with those terms; no
  rendered data will be redistributed.
- **Scene-level splits:** VSI-Bench is evaluation-only here (no training yet), so no
  leakage concern in this phase. When SFT/RL data generation starts, splits will be
  made at scene level per doc §6.1.

## 6. Repository layout (added)

```
viewtree/           # library: data, reconstruct, render, vlm, score
scripts/run_eval.py     # condition × shard evaluation driver (resumable)
scripts/render_check.py # renderer viability protocol
data/videos/            # extracted VSI-Bench videos (not committed)
data/renders/           # per-scene cached overview renders (not committed)
results/                # jsonl predictions + metrics + example renders
```

## 7. Results

### 7.1 RQ1 — explicit-memory views vs frame prompting (design doc §9.2 check #1): PASS

Qwen2.5-VL-7B, full VSI-Bench (5,130 questions), paired. Score = accuracy (MC) /
MRA (numerical). Headline = mean over the 10 question types.

| condition | mean of types | scene-bootstrap 95% CI |
|---|---|---|
| current (1 frame) | 0.266 | [0.252, 0.280] |
| frames16 | 0.311 | [0.297, 0.325] |
| **memory (12 frames + 5 rendered views)** | **0.333** | **[0.320, 0.347]** |

memory − frames16 = **+2.2 pts, 95% CI [+0.9, +3.6]** (paired scene bootstrap,
B=2000) → reconstructed novel views add information beyond raw frames.

Controls (full 5,130-question paired run):

| condition | mean of types | 95% CI | Δ vs frames16 |
|---|---|---|---|
| frames12 (same frame count as memory) | 0.316 | [0.303, 0.329] | +0.5 [−0.3, +1.4] |
| renders_only (5 rendered views, no frames) | 0.301 | [0.287, 0.316] | −1.0 [−2.7, +0.7] |

Two conclusions the doc needs: (a) the memory gain is **not a frame-count
effect** — frames12 ≈ frames16, so the +1.7 pts of memory over frames12 comes
from the renders; (b) **5 rendered views alone ≈ 16 real frames overall**, and
the two carry *complementary* information: renders_only is far better on
object_rel_direction_hard (0.343 vs 0.217) and object_abs_distance (0.166 vs
0.087) — allocentric/metric structure — while frames are better on
room_size/object_size (appearance-scale). The combined condition wins overall,
which is exactly the complementary-evidence premise behind branch-and-fuse (H3).

Per-type deltas (memory − frames16): object_counting **+9.5**, room_size **+3.4**,
rel_direction easy/medium/hard **+8.8/+1.4/+3.0**, route_planning +2.1,
obj_appearance_order −0.3, object_rel_distance −1.0, object_abs_distance +0.7,
object_size_estimation **−5.0** (splat rendering distorts apparent object scale —
motivates doc §5.7's warning that renders can mislead; a FUSE-trained model or
size-question gating should recover this).

### 7.2 Renderer viability (blocker #2): VIABLE, with a measured recipe

Held-out novel-view protocol (fixed 16 eval poses excluded from the cloud; only
source-frame count varies; 30 scenes):

| source frames | splat | coverage | covered-pixel PSNR | overall PSNR |
|---|---|---|---|---|
| 16 | 1 | 0.654 | 16.1 dB | 10.0 dB |
| 32 | 1 | 0.816 | 16.6 dB | 12.0 dB |
| 48 | 1 | 0.869 | 16.5 dB | 12.7 dB |
| 16 | 2 | 0.678 | 15.9 dB | 10.2 dB |
| 32 | 2 | 0.833 | 16.4 dB | 12.2 dB |

**Answer to "can sampling rate fix it": largely yes.** Coverage holes are the
failure mode, not wrong colors (covered-pixel PSNR is stable ~16–17 dB).
16→32 frames closes most of the gap; 32→48 has diminishing returns. The
residual ~13–18% is never-observed geometry, which no sampling rate can fix —
the render-validity guard (doc §5.5) remains necessary. **Adopted recipe:
32 source frames, splat 2.**

### 7.3 End-to-end training-free system (ViewTree-lite)

Full inference loop (gate → branch 5 views → token-confidence Top-2 prune with
consensus early-stop → pose-tagged fusion), zero trained components; ~3.3 s per
question on H100 (~2.5× the static memory condition). Full 5,130-question run:

| condition | mean of types | 95% CI | Δ vs frames16 |
|---|---|---|---|
| tree (end-to-end, training-free) | 0.331 | [0.317, 0.345] | **+2.0 [+0.6, +3.3]** |
| memory (static, for reference) | 0.333 | [0.320, 0.347] | +2.2 [+0.9, +3.6] |

**Reading:** the untrained tree matches (does not yet beat) static all-evidence
prompting overall, but with a different per-type profile that is highly
informative for the trained stages:

- It **fixes the object-size regression** (0.347 vs memory's 0.284 — above even
  frames16's 0.334): pruning + direct-fallback avoids misleading renders.
- It **loses the counting gain** (0.255 vs memory's 0.340): consensus early-stop
  answers from one render instead of fusing all five.
- Per-mode accuracy: direct 0.364 (n=325), branch_consensus 0.316 (n=2395),
  fallback-direct 0.315 (n=1258), **fused 0.268 (n=1152 — the weakest path)**.

The weak fused mode + weak token-confidence routing are precisely the doc's
predictions H3/H4: untrained fusion hurts and answer-token probability is a poor
branch score. This is the empirical justification for Stage II (rollout-trained
confidence head) and Stage III (fusion training). An oracle per-question-type
routing between memory and tree would already reach ≈0.35.

### 7.4 memory32 (better renderer recipe): no accuracy gain — negative result

memory32 (32-frame reconstruction, splat-2 renders) = 0.331 [0.315, 0.346] ≈
memory (0.333). **Render coverage is not the accuracy bottleneck at this
operating point**; the 16-frame recipe is kept for QA (half the reconstruction
cost), and the 32-frame recipe matters only if later stages show render-limited
failures.

### 7.5 Stage I teacher-ladder audit (doc §6.8): teacher is weak on cross-view integration

10k-item outcome ladders (Qwen2.5-VL-32B on MindCube train; state = growing
evidence): accuracy *decreases* with more views — s1 0.412, s2 0.385, s3 0.380,
s4 0.371, s4+render 0.385 (4-choice, chance 0.25). "MOVE harmful" (correct→
wrong when adding views: 1061) outnumbers "MOVE needed" (808); the top-down
render is net-positive at s4 (+350/−258). ~32% of ladders are all-wrong.

**Consequences adopted:** (a) SFT control labels come from ladder outcome
patterns only; (b) answer supervision uses ground truth, never teacher text;
(c) all-wrong ladders are excluded from control supervision; (d) the student
(7B) ladder on MindCube tinybench is being measured as the pre-SFT baseline.

## 8. Stage I training setup (added)

LoRA (r=16, α=32, LM-only, vision tower frozen) on Qwen2.5-VL-7B; loss on
target tokens only; examples = control decisions (STOP/MOVE/RENDER) + GT-letter
answers at the ladder's first-correct state; bf16, grad-accum 16, cosine LR
1e-4, 1 epoch, multi-GPU data parallel. Scripts: `build_sft.py`,
`render_train_views.py`, `train_sft.py`; post-SFT eval reuses `gen_traj.py
--adapter` on tinybench.

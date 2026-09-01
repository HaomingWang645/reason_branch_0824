# ViewTree-D: multi-step (depth > 1) view acquisition — experiment design

*Drafted 2026-08-28. Status: **completed 2026-08-31** — all phases run (Phase 4
cross-domain hardening dropped: Phase 3 GRPO collapsed to STOP and added nothing
over SFT-C). Final numbers in RESULTS.md §8; deployed controller = SFT-C + value
head + beam (0.530 on VSI held-out, +2.1 vs the data-matched baseline).*

## 0. Why a new design is needed

The current best system (RESULTS §6c) is a **depth-1** tree: five
constrained candidate views are rendered once, scored by a confidence head,
pruned to two and fused. It cannot (a) look *again* after seeing a render,
(b) move *locally* toward the thing the question is about, or (c) learn to
stop from experience — the stop/continue decision is a calibrated head, not a
policy. On MindCube the sequential ladder policy *did* learn multi-step
acquisition (up to 4 moves, RL-trained), but its action space was "next view
in a fixed order" and its `RENDER` action was never used. ViewTree-D makes the
camera itself the action space, keeps the human-camera constraints as a hard
action mask, and trains the controller end-to-end in phases on a corpus one
to two orders of magnitude larger than MindCube.

## 1. Reasoning path = a walk in the memory

State s_t = (question, K context frames, renders r_1..r_t seen so far, current
camera pose p_t). One step = one **camera action** followed by one render:

| action | effect on pose | validity (hard mask) |
|---|---|---|
| `TURN_LEFT` / `TURN_RIGHT` | yaw ± 45°, same position | always valid (roll stays 0, pitch 10° down) |
| `FORWARD` | step one walkable-grid cell along the view direction | target cell inside walked hull, clearance ≥ 4 % diag, render coverage ≥ 45 % |
| `NEXT_SPOT` | jump to the next farthest-point standing position, facing room centre | same validity as above |
| `LOOK_AROUND` | +180° yaw at same spot | always valid |
| `BIRD_EYE` | top-down view (allowed only as the *last* acquisition) | valid once; forces STOP after |
| `STOP` | answer from frames + all acquired renders | always |

Depth cap D = 3 acquisitions (≤ 3 renders + optional bird's-eye). Invalid
actions are masked at sampling time (route 1); an RL penalty on *proposing*
a masked action is added as an auxiliary reward (route 2) so the policy
internalises the constraint instead of relying on the mask.

**Inference = beam search over walks.** Branching b = 3 (top-3 actions by
policy logit) at each level, keep k = 2 by the value head, early stop when the
two kept paths agree with margin, final arbitration between fused-path answer
and direct answer exactly as today. Budget: ≤ 12 VLM calls/question (depth-1
tree: ≤ 8). The gate stays (answer directly if sufficient).

**Pre-rendered memory.** For every training scene the reconstruction is built
once (32 frames) and a **pose bank** is rendered offline: 12 standing
positions (FPS in the walked hull) × 8 yaws + top-down = 97 renders/scene.
Every walk state is a subset of bank renders, so SFT and RL never render
online — the only per-step cost is the VLM call. At test time the same bank
is rendered lazily (only poses the beam visits).

## 2. Training corpus (large scale, combined; all scene-disjoint from every evaluation set)

| source | items | scenes | what it adds |
|---|---|---|---|
| VLM-3R `vsibench_train` (ScanNet + ScanNet++ train + route-plan) | ~300k VSI-style QA (10 types incl. numeric) | ~2,000 | VSI-type spatial questions on real walkthrough videos |
| VLM-3R `vstibench_train` | ~150k camera/object spatio-temporal QA | same videos | camera-motion / temporal questions (VSTI-type) |
| VSI-590K subset whose videos we hold (ScanNet, ScanNet++ v2, ARKitScenes train) | ~200k | ~3,500 | more question templates, ARKitScenes domain |
| MindCube train | 10k multi-view MC | 3.3k groups | few-view (2–4 image) regime, rotation/among questions |
| **held out, never trained on** | VSI-Bench (all 288 scenes), VSTI-Bench (ScanNet val), STI-Bench videos, OST-Bench scenes, MindCube tiny/rest | | scene-level exclusion enforced by id lists in `data/train3r/exclude_scenes.json` |

Videos: `Journey9ni/vlm3r_videos` (3,537 videos: ARKitScenes 1,480, ScanNet
1,201, ScanNet++ 856; 54 GB) — downloading now. Target after de-duplication
and balancing: **≈ 400k QA on ≈ 3,500 scenes** (40× MindCube). Numeric
answers are trained as text and scored by MRA as in VSI-Bench.

**Corpus as assembled (2026-08-28).** The gated `vlm3r_videos` mirror could not be
downloaded; videos come from VSI-590K's own ScanNet (1,513) and ScanNet++ v2
(856) tarballs instead, ARKitScenes deferred. After joining QA to held videos
and excluding every evaluation scene (348 ScanNet scenes; 73,680 QA dropped):

| source | QA kept | scenes |
|---|---|---|
| VLM-3R `vsibench_train` (VSI-type, ScanNet + ScanNet++) | 192,056 | |
| VLM-3R `vstibench_train` (camera-motion / temporal) | 94,110 | |
| VSI-590K ScanNet + ScanNet++ v2 | 207,497 | |
| **total** | **493,663** (176,356 numeric) | **1,709** |

`data/train3r/qa_all.jsonl`, `data/train3r/manifest.jsonl`. MindCube (10k,
image sets) is added in Phase 1 with a bank built from its ≤ 4 views.
Pose bank measured on a ScanNet scene: 97 entries, 71 pass the 45 %
coverage mask, eye height 0.80 of room height.

## 3. Phases

**Phase 0 — memory + pose bank (GPU, ~1 day on 8 GPUs).** VGGT on 32 frames
per video, walkable hull + validity mask, 97 renders per scene cached as JPEG,
plus per-render coverage and the camera pose. ≈ 3,500 scenes × 97 renders.

**Phase 1 — SFT-A, answerer (≈ 6 h).** LoRA on Qwen2.5-VL-7B (fresh, r = 16).
Inputs: 8 context frames + 0…3 bank renders (pose-tagged with the walk that
produced them) + question → answer. Sampling of the render sets: uniform over
depth 0–3, with 30 % *purposeful* sets (renders whose pose faces the
objects named in the question — we have the object positions from the QA
generators) and 70 % random walks, so the answerer learns to use good views
and to ignore useless ones. ≈ 600k examples, 1 epoch, 8-GPU DDP.

**Phase 2 — oracle walks + SFT-C, controller imitation (≈ 12 h).** With SFT-A
frozen, run a bounded search over the pose bank for each question (beam 3,
depth ≤ 3): the **oracle walk** is the shortest walk whose fused answer is
correct with the largest answer-logprob margin; if the direct answer is
already correct with margin, the oracle is `STOP` at depth 0. Labels: at each
state of the oracle walk the chosen action; at off-path states sampled from
the search, `STOP` if correct-with-margin else the action leading toward the
oracle. ≈ 300k action examples + the answer examples of Phase 1 re-mixed
(so the controller does not forget answering). Also train the **value head**
(state → P(correct)), initialised from head v2-human, on all search states.

**Phase 3 — RL-1, GRPO on walks, in-domain (≈ 1.5 days).** Full walks sampled
from the policy (temperature 1, masked actions removed), G = 6 walks per
question, reward
`r = 1[correct] − λ·(steps) − 0.1·1[proposed a masked action] + 0.05·1[answer changed to correct after a step]`,
λ a dual variable driving mean steps toward a budget of 1.2 (curriculum:
depth cap 2 for the first 30 % of items, then 3). Policy gradient on the
action tokens *and* the final answer tokens (unlike the depth-1 GRPO, which
touched only control tokens): 120k questions, 8-GPU DDP, sampled across
sources in proportion to a fixed mixture (VSI-type 50 %, VSTI-type 25 %,
VSI-590K 15 %, MindCube 10 %).

**Phase 4 — RL-2, cross-domain hardening (≈ 0.5 day).** Same objective on a
mixture that emphasises the weakest tasks from the Phase-3 held-in
validation (per-type accuracy on a 5 % train-scene validation split), plus
random frame-count / resolution perturbation for robustness.

**Ablations (same eval, 1 GPU each):** depth cap 1 vs 2 vs 3; beam b = 1
(greedy walk) vs 3; no mask (route 2 only) vs mask (route 1 only) vs both;
Phase-2-only (no RL); Phase-1 answerer inside today's depth-1 tree
(isolates "better answerer" from "deeper search").

## 4. Evaluation

Primary: VSI-Bench held-out odd half (paired with every system in RESULTS
§1c/§6c). Secondary: VSTI-Bench (clean subset), STI-Bench, OST-Bench (full
tree), MindCube tiny/rest. Baselines: depth-1 tree (current best), static
memory, SFT-plain, SFT+GRPO-plain, zero-shot, and — because the corpus is
new — a **no-memory SFT+GRPO on the same 400k corpus** (frames only), so
the effect of depth is separated from the effect of data. Report per task,
paired scene-bootstrap CIs, and cost (VLM calls, renders, mean depth) per
question. Success criterion: ≥ +2 pts mean-of-types over the depth-1 tree on
VSI held-out with ≤ 1.5× its VLM calls, and no regression on OST.

## 5. Compute plan (8 × H100)

| phase | GPU-hours | wall (8 GPUs) |
|---|---|---|
| 0 memory + pose bank | ~150 | ~1 day |
| 1 SFT-A | ~50 | 6 h |
| 2 oracle search + SFT-C + value head | ~100 | 12 h |
| 3 RL-1 | ~280 | 1.5 days |
| 4 RL-2 | ~90 | 0.5 day |
| evaluation + ablations | ~120 | 0.6 day |
| **total** | **~800** | **≈ 4.5 days** |

## 6. Risks and pre-registered fallbacks

- *Reconstruction fails on some training videos* (mixed resolution, motion
  blur): skip scene; corpus is large enough.
- *Oracle walks mostly STOP at depth 0* (answerer already right): then depth
  brings nothing on that item — the corpus for Phase 2 is filtered to items
  where depth ≥ 1 changes the outcome (expected 30–40 %).
- *RL collapses to always-STOP or always-max-depth*: the dual λ and the
  "answer improved" shaping term are there for this; monitor mean depth and
  per-type accuracy every 500 items.
- *Deeper is not better*: the ablation grid answers this directly; a null
  result is reported as such.

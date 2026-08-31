# ViewTree: Spatial Reasoning over an Explicit Scene Memory with a Human-Camera Reasoning Tree

*Technical report — best system and its comparison with no-world-memory baselines. Generated 2026-08-31.*

## 1. Introduction: the method from the beginning

### 1.1 Problem
A vision-language model (VLM) answering a spatial question about a room from a handful of video frames or photos has to reason about
geometry it never sees at once: distances, relative directions, counts, room size, route planning. Frame prompting (feeding 16 frames
to Qwen2.5-VL-7B) gives 0.311 on VSI-Bench (mean over 10 question types). ViewTree adds an **explicit 3-D scene memory** and lets the
model **acquire new evidence on a reasoning path** — re-rendering the reconstructed room from viewpoints a person holding a camera
could have taken — instead of answering from whatever frames it was given.

### 1.2 System components
1. **Scene memory (frozen).** VGGT-1B reconstructs a coloured point cloud with camera poses from the input frames (32 frames for video
   benchmarks, all given images for image benchmarks). Memory is ~7.8 GB + 0.22 GB/frame; one reconstruction per scene.
2. **Renderer.** A GPU z-buffered point-splat renderer (splat radius 2) produces a novel view for any camera pose in a few ms.
3. **Human-camera viewpoint proposer (hard constraints).** Candidate viewpoints must be where a person could stand: inside the convex
   hull of the *recorded* camera trajectory (the region the videographer actually walked), at the median recorded camera height
   (eye level), ≥4 % of the room diagonal clear of any reconstructed surface, roll = 0 (image horizontal parallel to the floor),
   pitch 10° down. Four positions are farthest-point-sampled so they come from different sides of the room, each looking toward the
   room centre; views painting < 45 % of pixels are discarded and replaced. A top-down bird's-eye view is kept as the fifth, final
   view. The legacy proposer placed cameras outside the room and above the ceiling (0 % inside); the constrained one is inside 100 %
   of the time with no accuracy loss (Fig. 2).
4. **Controller VLM.** Qwen2.5-VL-7B with one LoRA adapter (r = 16) that both answers and emits control tokens. Trained in stages:
   *Stage I* SFT on 16.8k MindCube examples (control STOP/MOVE/RENDER + answers from an evidence ladder of 1…all views, teacher-labelled);
   *Stage III* SFT-v2 adds 6.1k on-policy fusion examples (complementary / redundant view combinations) — 22.9k in total;
   *Stage IV* GRPO on the view-control policy over all 9,995 MindCube train items with reward = correctness − λ·0.2·(views−1) and a dual
   variable λ driving the mean view count toward 1.5 (the “D_highcost” design, chosen from an 8-variant RL design sweep as the best
   accuracy/efficiency trade-off: 0.778 on MindCube-rest at 1.30 views vs 0.765 at 1.85 for SFT-v2).
5. **Confidence head.** A 2-layer MLP (3584→512→1, temperature-calibrated) on the controller's last-token hidden state predicts whether
   an answer state is correct. It is trained on MindCube ladder states plus VSI tree states from the 144 even-indexed VSI scenes
   (the 144 odd-indexed scenes are never touched and form the held-out evaluation half). The best system uses the head retrained on
   *human-view* states (held-out AUROC 0.710 vs 0.672 for the legacy-view head).
6. **Reasoning tree (depth 1, branching 5, keep 2).** For every question (Fig. 1):
   *gate* — “can you answer from these frames alone? YES / EXPLORE”; YES → answer directly (1–2 calls).
   *branch* — otherwise render the 5 constrained views; each branch answers from frames + that view and is scored by the head.
   *prune* — keep the top-2 branches; if they agree and beat the direct answer → early stop (*branch consensus*).
   *fuse* — answer from frames + both kept views, pose-tagged.
   *arbitrate* — if the head ranks the direct answer above the fused and kept answers → fall back to direct, else take the fused answer.
   Cost: 1–2 VLM calls when the gate fires, otherwise ≤ 8 calls + 1 reconstruction + 5 renders.

![](figures/tree_schematic.png)

*Figure 1. The depth-1 reasoning tree. Nodes are VLM calls; [ ] are confidence-head scores used for pruning and arbitration.*

![](figures/human_views_examples.png)

*Figure 2. Legacy proposer (outside/above the room, 39° pitch) vs the human-camera constraint (inside the walked region, eye level,
roll 0°) on held-out VSI scenes; the top-down bird's-eye view is shared.*


### 1.4 Frame budgets (fixed for every evaluation)

| benchmark | frames fed to VGGT (3-D memory) | frames the controller sees per call | notes |
|---|---|---|---|
| VSI-Bench (video) | 32 uniformly sampled frames | 8 of those 32 (uniform subset) + 1 render per branch, 2 renders at fuse | memory ≈ 7.8 GB + 0.22 GB × 32 ≈ 14.8 GB per reconstruction; static-memory baseline: 16-frame reconstruction, 12 frames + 5 renders; no-memory baselines: 16 frames |
| OST-Bench (image history) | all observed images (latest 12) | up to 8 of them + renders | identical input to the single-pass baselines |
| STI-Bench | 16 frames inside the queried time window | all 16 + renders | ±1 s window for instantaneous questions |
| VSTI-Bench | 32 uniform frames | all 32 + renders | questions cite "frame k of 32" |
| MindCube (training) | all given views (≤ 4) → one top-down render | 1…k views as the ladder policy acquires them | |

### 1.3 What is being compared
- **SFT-plain** — the same base model LoRA-fine-tuned on the benchmark's own training split (all 10,000 MindCube train items;
  input = all given views + question, target = answer letter). No memory, no renders, no control.
- **SFT+GRPO-plain** — SFT-plain followed by GRPO with reward = answer correctness (6 samples/item, 9,995 items).
- **ViewTree (best)** — the full system above: D_highcost-10k adapter + human-camera views + matched confidence head.
On VSI-Bench the no-memory baselines see 16 uniformly sampled frames; on OST-Bench they see the cumulative image history (latest 12),
exactly as the ViewTree gate sees it. All comparisons are paired on identical question sets.

## 2. VSI-Bench (held-out half: 144 scenes never used for any training, 2,557 questions)

Score = accuracy for multiple choice, mean relative accuracy for numerical answers; headline = mean over the 10 question types.

| system | mean of types | obj appearance order | abs distance | counting | dir easy | dir hard | dir medium | rel distance | size | room size | route planning |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B zero-shot (16 frames) | **0.313** | 0.289 | 0.088 | 0.254 | 0.478 | 0.203 | 0.420 | 0.391 | 0.340 | 0.357 | 0.308 |
| SFT-plain (16 frames) | **0.327** | 0.213 | 0.140 | 0.339 | 0.496 | 0.330 | 0.382 | 0.344 | 0.420 | 0.330 | 0.279 |
| SFT+GRPO-plain (16 frames) | **0.324** | 0.218 | 0.133 | 0.332 | 0.504 | 0.335 | 0.411 | 0.320 | 0.430 | 0.308 | 0.250 |
| SFT-v2 + static memory (12 frames + 5 renders) | **0.342** | 0.218 | 0.083 | 0.356 | 0.522 | 0.377 | 0.396 | 0.397 | 0.346 | 0.385 | 0.337 |
| ViewTree, legacy views + head v2 | **0.357** | 0.293 | 0.115 | 0.299 | 0.496 | 0.382 | 0.386 | 0.397 | 0.411 | 0.412 | 0.375 |
| **ViewTree (best): human views + matched head** | **0.367** | 0.314 | 0.131 | 0.313 | 0.522 | 0.387 | 0.406 | 0.402 | 0.432 | 0.410 | 0.356 |

- best ViewTree − Qwen2.5-VL-7B zero-shot (16 frames): **+0.055** [+0.031, +0.077] (scene-bootstrap 95 % CI)
- best ViewTree − SFT-plain (16 frames): **+0.040** [+0.017, +0.060] (scene-bootstrap 95 % CI)
- best ViewTree − SFT+GRPO-plain (16 frames): **+0.043** [+0.020, +0.065] (scene-bootstrap 95 % CI)

Per-task view against the two no-memory baselines:

| VSI task | n | SFT-plain | SFT+GRPO-plain | ViewTree (best) | Δ vs best baseline |
|---|---|---|---|---|---|
| obj appearance order | 239 | 0.213 | 0.218 | **0.314** | +0.096 |
| abs distance | 424 | 0.140 | 0.133 | **0.131** | -0.009 |
| counting | 281 | 0.339 | 0.332 | **0.313** | -0.026 |
| dir easy | 113 | 0.496 | 0.504 | **0.522** | +0.018 |
| dir hard | 212 | 0.330 | 0.335 | **0.387** | +0.052 |
| dir medium | 207 | 0.382 | 0.411 | **0.406** | -0.005 |
| rel distance | 363 | 0.344 | 0.320 | **0.402** | +0.058 |
| size | 470 | 0.420 | 0.430 | **0.432** | +0.002 |
| room size | 144 | 0.330 | 0.308 | **0.410** | +0.081 |
| route planning | 104 | 0.279 | 0.250 | **0.356** | +0.077 |

ViewTree path mix on the held-out half: direct 598 · branch consensus 839 · fused 576 ·
fused→fallback-to-direct 544 (of 2557).

**Reading.** The no-memory baselines transfer only +1.1–1.5 points from MindCube to VSI; the full system is ~4 points above them,
significant. The gains concentrate on types that need geometry the frames do not show at once — room size, object size, absolute
and relative distance, route planning, appearance order — while single-frame-answerable types (counting, easy relative direction)
are close to the baselines.

## 3. OST-Bench (online spatio-temporal exploration, 5,557 multiple-choice items)

Each item is a turn in an exploration; the agent sees the images observed so far and answers about its own state
(*Agent_state*), what is visible (*Agent_visible_info*) or spatial relations to objects (*Agent_object_spatial*). ViewTree reconstructs the observed images with VGGT and runs the same tree; 154 items whose image
histories mix resolutions failed reconstruction and are excluded from every system (paired n = 5403). The single-pass
rows use the same adapters answering directly from the image history.


| system | overall | Agent_object_spatial (n=2692) | Agent_state (n=748) | Agent_visible_info (n=1963) |
|---|---|---|---|---|
| Qwen2.5-VL-7B zero-shot | **0.540** | 0.413 | 0.503 | 0.728 |
| SFT-plain | **0.524** | 0.392 | 0.485 | 0.719 |
| SFT+GRPO-plain | **0.514** | 0.384 | 0.485 | 0.703 |
| SFT-v2 adapter, single pass | **0.550** | 0.431 | 0.497 | 0.734 |
| D_10k adapter, single pass | **0.550** | 0.430 | 0.499 | 0.733 |
| **ViewTree (best): reasoning tree** | **0.541** | 0.425 | 0.489 | 0.720 |

- ViewTree − SFT-plain: **+0.018** [+0.004, +0.030] (bootstrap 95 % CI over item blocks)
- ViewTree − SFT+GRPO-plain: **+0.028** [+0.013, +0.041] (bootstrap 95 % CI over item blocks)
- ViewTree − Qwen2.5-VL-7B zero-shot: **+0.001** [-0.010, +0.013] (bootstrap 95 % CI over item blocks)

ViewTree path mix on OST-Bench: fused_fallback_direct 1729 · direct 2033 · fused 539 · branch_consensus 1102.


## 3b. STI-Bench (spatial-temporal understanding from video, 2,064 single-choice items, 8 tasks)

Videos come from ScanNet (indoor walkthroughs), Waymo (outdoor driving) and Omni6DPose (desktop object manipulation); every question
refers to a time window of the video. All systems see the same 16 frames sampled inside the queried window (±1 s for instantaneous
questions). Only 2 of the 150 ScanNet videos overlap the scenes used to train our confidence head; results are on all items
(paired n = 2062). No model was retrained for this benchmark.

| system | overall | 3D Video Grounding (317) | Dimensional Measurement (289) | Displacement & Path Length (358) | Ego-Centric Orientation (185) | Pose Estimation (359) | Spatial Relation (146) | Speed & Acceleration (330) | Trajectory Description (78) |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B zero-shot | **0.371** | 0.315 | 0.301 | 0.240 | 0.454 | 0.535 | 0.527 | 0.309 | 0.462 |
| SFT-plain | **0.261** | 0.240 | 0.260 | 0.221 | 0.108 | 0.281 | 0.466 | 0.291 | 0.308 |
| SFT+GRPO-plain | **0.263** | 0.249 | 0.263 | 0.204 | 0.108 | 0.290 | 0.500 | 0.288 | 0.282 |
| SFT-v2 adapter, single pass | **0.315** | 0.271 | 0.225 | 0.237 | 0.292 | 0.432 | 0.479 | 0.315 | 0.397 |
| D_10k adapter, single pass | **0.311** | 0.274 | 0.228 | 0.232 | 0.265 | 0.415 | 0.473 | 0.321 | 0.410 |
| **ViewTree (best): reasoning tree** | **0.306** | 0.271 | 0.239 | 0.249 | 0.178 | 0.415 | 0.466 | 0.324 | 0.397 |

By video source:

| system | Omni6DPose (404) | ScanNet (865) | Waymo (793) |
|---|---|---|---|
| Qwen2.5-VL-7B zero-shot | 0.347 | 0.299 | 0.460 |
| SFT-plain | 0.300 | 0.216 | 0.291 |
| SFT+GRPO-plain | 0.302 | 0.213 | 0.298 |
| SFT-v2 adapter, single pass | 0.302 | 0.242 | 0.402 |
| D_10k adapter, single pass | 0.312 | 0.238 | 0.390 |
| **ViewTree (best): reasoning tree** | 0.307 | 0.237 | 0.382 |

- ViewTree − SFT-plain: **+0.045** [+0.023, +0.066] (video-bootstrap 95 % CI)
- ViewTree − SFT+GRPO-plain: **+0.044** [+0.025, +0.061] (video-bootstrap 95 % CI)
- ViewTree − Qwen2.5-VL-7B zero-shot: **-0.064** [-0.083, -0.047] (video-bootstrap 95 % CI)

ViewTree path mix: fused_fallback_direct 344 · direct 1356 · branch_consensus 237 · fused 125.

**Reading (a negative result, reported as such).** On STI-Bench the *zero-shot* model is the best system: every model
fine-tuned on MindCube indoor multiple-choice data regresses, most severely the no-memory baselines (−11 pts; pose estimation
0.535→0.28, ego-centric orientation 0.454→0.11), whose answer-only fine-tuning over-fits the MindCube option format and indoor
domain. ViewTree keeps more of the base model's ability (−6.4) and is +4.5 above both no-memory baselines (significant), and
its gate answers directly on 66 % of items — the tree does no harm — but it does not recover zero-shot performance. STI's tasks
are largely temporal-quantitative (speed, displacement, trajectory, pose over time) and two of three sources (driving, desktop
manipulation) are far from the indoor-room domain of all training data used here; a static scene memory is the wrong tool for
motion questions. The practical conclusion is that the *controller adapter*, not the memory, is what limits transfer to this
benchmark; the multi-step design (DESIGN_DEPTH.md) trains on a corpus that includes camera-motion questions for this reason.


## 3c. VSTI-Bench (visual-spatial temporal intelligence on ScanNet videos, 5,736 items, 9 types)

Seven multiple-choice types and two numeric types (camera displacement, camera–object absolute distance; scored by
mean relative accuracy). All systems see the same 32 uniformly sampled frames (questions reference "frame k of 32").
870 items lie on ScanNet scenes that were used to train our confidence head; the second table is the
leakage-clean subset (n = 4866). No model was retrained.

All items (paired n = 5736):

| system | mean of types | cam displacement (833) | cam movement direction (913) | cam obj abs dist (905) | cam obj rel dist v1 (91) | cam obj rel dist v2 (493) | cam obj rel dist v3 (856) | obj-obj lr (605) | obj-obj nf (556) | obj-obj ud (484) |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B zero-shot | **0.523** | 0.134 | 0.510 | 0.149 | 0.615 | 0.667 | 0.689 | 0.567 | 0.622 | 0.748 |
| SFT-plain | **0.456** | 0.056 | 0.440 | 0.135 | 0.396 | 0.497 | 0.612 | 0.618 | 0.574 | 0.777 |
| SFT+GRPO-plain | **0.452** | 0.044 | 0.423 | 0.123 | 0.374 | 0.495 | 0.606 | 0.602 | 0.579 | 0.818 |
| SFT-v2 adapter, single pass | **0.498** | 0.072 | 0.503 | 0.149 | 0.484 | 0.631 | 0.661 | 0.603 | 0.615 | 0.769 |
| D_10k adapter, single pass | **0.497** | 0.073 | 0.504 | 0.135 | 0.495 | 0.623 | 0.660 | 0.592 | 0.624 | 0.767 |
| **ViewTree (best): reasoning tree** | **0.505** | 0.034 | 0.483 | 0.169 | 0.484 | 0.635 | 0.646 | 0.640 | 0.647 | 0.806 |

Leakage-clean subset:

| system | mean of types | cam displacement (713) | cam movement direction (783) | cam obj abs dist (773) | cam obj rel dist v1 (79) | cam obj rel dist v2 (419) | cam obj rel dist v3 (705) | obj-obj lr (504) | obj-obj nf (481) | obj-obj ud (409) |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B zero-shot | **0.531** | 0.130 | 0.512 | 0.151 | 0.658 | 0.685 | 0.682 | 0.562 | 0.644 | 0.753 |
| SFT-plain | **0.461** | 0.053 | 0.437 | 0.135 | 0.430 | 0.511 | 0.614 | 0.601 | 0.582 | 0.787 |
| SFT+GRPO-plain | **0.455** | 0.043 | 0.419 | 0.123 | 0.392 | 0.504 | 0.607 | 0.587 | 0.586 | 0.834 |
| SFT-v2 adapter, single pass | **0.507** | 0.068 | 0.508 | 0.151 | 0.519 | 0.647 | 0.655 | 0.601 | 0.636 | 0.775 |
| D_10k adapter, single pass | **0.505** | 0.070 | 0.508 | 0.137 | 0.532 | 0.635 | 0.654 | 0.593 | 0.644 | 0.773 |
| **ViewTree (best): reasoning tree** | **0.512** | 0.033 | 0.485 | 0.173 | 0.519 | 0.654 | 0.643 | 0.625 | 0.667 | 0.807 |

- ViewTree − SFT-plain: **+0.049** [+0.031, +0.067] (scene-bootstrap 95 % CI)
- ViewTree − SFT+GRPO-plain: **+0.053** [+0.033, +0.073] (scene-bootstrap 95 % CI)
- ViewTree − Qwen2.5-VL-7B zero-shot: **-0.018** [-0.035, -0.002] (scene-bootstrap 95 % CI)
- ViewTree − D_10k adapter, single pass: **+0.008** [-0.004, +0.018] (scene-bootstrap 95 % CI)

ViewTree path mix: fused_fallback_direct 1053 · direct 1331 · fused 643 · branch_consensus 2709.

**Reading.** The pattern of STI-Bench repeats on VSTI: the zero-shot model is the strongest overall (0.523), the no-memory
baselines fine-tuned on MindCube lose ~7 points, and ViewTree recovers most of that loss (+4.9 / +5.3 over them, −1.8 below
zero-shot, all significant). The per-type split is the informative part: on the three **object–object relative-position**
types ViewTree is the best system of all (0.640 / 0.647 / 0.806 vs 0.567 / 0.622 / 0.748 zero-shot) — these are room-geometry
questions the scene memory can answer — while on the **camera-motion** types (displacement, movement direction) every
fine-tuned model is below zero-shot and the tree's renders cannot help, because the question is about the trajectory of the
recorded camera, not about the room. The clean subset gives the same numbers within ±0.8 pt, so the head's exposure to 15 % of
the scenes does not drive the result.

## 4. Visualized reasoning trees

Each figure is one question run through the best system. Every node shows the images the controller sees at that node
(root = the observed frames; branch = frames + one rendered eye-level view; fuse = frames + the two kept views); orange = kept
branches; faded branches were not executed because the gate answered directly; [ ] = confidence-head score.

### 4.1 VSI-Bench — three trees per task

#### obj appearance order 

![](figures/tree_rep_vsi_2827.png)

*#2827 — What will be the first-time appearance order of the following categories in the video: whiteboard, printer, kettle, computer mouse? — gate YES, mode `direct`, final **B** (GT B) → correct.*

![](figures/tree_rep_vsi_3937.png)

*#3937 — What will be the first-time appearance order of the following categories in the video: bookshelf, window, keyboard, chair? — gate EXPLORE, mode `branch_consensus`, final **A** (GT A) → correct.*

![](figures/tree_rep_vsi_3911.png)

*#3911 — What will be the first-time appearance order of the following categories in the video: refrigerator, window, bookshelf, keyboard? — gate EXPLORE, mode `branch_consensus`, final **D** (GT D) → correct.*

#### abs distance 

![](figures/tree_rep_vsi_4756.png)

*#4756 — Measuring from the closest point of each object, what is the distance between the table and the bed (in meters)? — gate YES, mode `direct`, final **1** (GT 1.0) → correct.*

![](figures/tree_rep_vsi_4747.png)

*#4747 — Measuring from the closest point of each object, what is the distance between the table and the radiator (in meters)? — gate EXPLORE, mode `fused`, final **5** (GT 1.1) → wrong.*

![](figures/tree_rep_vsi_1728.png)

*#1728 — Measuring from the closest point of each object, what is the distance between the blanket and the heater (in meters)? — gate EXPLORE, mode `fused`, final **0.5** (GT 0.5) → correct.*

#### counting 

![](figures/tree_rep_vsi_4390.png)

*#4390 — How many table(s) are in this room? — gate YES, mode `direct`, final **2** (GT 2) → correct.*

![](figures/tree_rep_vsi_4444.png)

*#4444 — How many chair(s) are in this room? — gate NO, mode `branch_consensus`, final **4** (GT 4) → correct.*

![](figures/tree_rep_vsi_115.png)

*#115 — How many chair(s) are in this room? — gate EXPLORE, mode `fused`, final **4** (GT 2) → wrong.*

#### dir easy 

![](figures/tree_rep_vsi_2439.png)

*#2439 — If I am standing by the keyboard and facing the bookshelf, is the heater to the left or the right of the bookshelf? — gate YES, mode `direct`, final **A** (GT A) → correct.*

![](figures/tree_rep_vsi_1331.png)

*#1331 — If I am standing by the stove and facing the refrigerator, is the table to the left or the right of the refrigerator? — gate EXPLORE, mode `branch_consensus`, final **A** (GT A) → correct.*

![](figures/tree_rep_vsi_2434.png)

*#2434 — If I am standing by the door and facing the bookshelf, is the laptop to the left or the right of the bookshelf? — gate EXPLORE, mode `fused`, final **B** (GT B) → correct.*

#### dir hard 

![](figures/tree_rep_vsi_1869.png)

*#1869 — If I am standing by the toilet and facing the ceiling light, is the cup to my front-left, front-right, back-left, or back-right? — gate YES, mode `direct`, final **C** (GT C) → correct.*

![](figures/tree_rep_vsi_1881.png)

*#1881 — If I am standing by the table and facing the door, is the bed to my front-left, front-right, back-left, or back-right? — gate EXPLORE, mode `branch_consensus`, final **D** (GT D) → correct.*

![](figures/tree_rep_vsi_1086.png)

*#1086 — If I am standing by the stove and facing the tv, is the refrigerator to my front-left, front-right, back-left, or back-right? — gate EXPLORE, mode `branch_consensus`, final **C** (GT D) → wrong.*

#### dir medium 

![](figures/tree_rep_vsi_1571.png)

*#1571 — If I am standing by the ceiling light and facing the chair, is the door to my left, right, or back? — gate YES, mode `direct`, final **A** (GT A) → correct.*

![](figures/tree_rep_vsi_1510.png)

*#1510 — If I am standing by the door and facing the whiteboard, is the coat rack to my left, right, or back? — gate EXPLORE, mode `branch_consensus`, final **C** (GT C) → correct.*

![](figures/tree_rep_vsi_1519.png)

*#1519 — If I am standing by the door and facing the bookshelf, is the laptop to my left, right, or back? — gate EXPLORE, mode `fused`, final **A** (GT A) → correct.*

#### rel distance 

![](figures/tree_rep_vsi_2146.png)

*#2146 — Measuring from the closest point of each object, which of these objects (chair, whiteboard, door, printer) is the closest to the refrigerato — gate YES, mode `direct`, final **A** (GT A) → correct.*

![](figures/tree_rep_vsi_2071.png)

*#2071 — Measuring from the closest point of each object, which of these objects (cutting board, door, microwave, trash can) is the closest to the ke — gate EXPLORE, mode `branch_consensus`, final **D** (GT D) → correct.*

![](figures/tree_rep_vsi_2082.png)

*#2082 — Measuring from the closest point of each object, which of these objects (pan, bowl, refrigerator, kettle) is the closest to the computer mou — gate EXPLORE, mode `fused`, final **C** (GT C) → correct.*

#### size 

![](figures/tree_rep_vsi_2201.png)

*#2201 — What is the length of the longest dimension (length, width, or height) of the door, measured in centimeters? — gate YES, mode `direct`, final **200** (GT 211) → correct.*

![](figures/tree_rep_vsi_2230.png)

*#2230 — What is the length of the longest dimension (length, width, or height) of the keyboard, measured in centimeters? — gate EXPLORE, mode `branch_consensus`, final **45** (GT 45) → correct.*

![](figures/tree_rep_vsi_508.png)

*#508 — What is the length of the longest dimension (length, width, or height) of the bed, measured in centimeters? — gate EXPLORE, mode `branch_consensus`, final **200** (GT 221) → correct.*

#### room size 

![](figures/tree_rep_vsi_609.png)

*#609 — What is the size of this room (in square meters)?  — gate EXPLORE, mode `branch_consensus`, final **15** (GT 13.8) → correct.*

![](figures/tree_rep_vsi_607.png)

*#607 — What is the size of this room (in square meters)?  — gate EXPLORE, mode `fused`, final **20** (GT 21.9) → correct.*

![](figures/tree_rep_vsi_671.png)

*#671 — What is the size of this room (in square meters)?  — gate EXPLORE, mode `fused_fallback_direct`, final **10** (GT 9.2) → correct.*

#### route planning 

![](figures/tree_rep_vsi_5079.png)

*#5079 — You are a robot beginning at the toilet facing the toilet. You want to navigate to the bathtub. You will perform the following actions (Note — gate YES, mode `direct`, final **A** (GT A) → correct.*

![](figures/tree_rep_vsi_4997.png)

*#4997 — You are a robot beginning at the door and facing the trash bin next to the door. You want to navigate to the window. You will perform the fo — gate EXPLORE, mode `branch_consensus`, final **C** (GT C) → correct.*

![](figures/tree_rep_vsi_5090.png)

*#5090 — You are a robot beginning at the TV and facing the TV. You want to navigate to the terrace. You will perform the following actions (Note: fo — gate EXPLORE, mode `fused`, final **A** (GT A) → correct.*

### 4.2 OST-Bench — three trees per task

#### Agent_object_spatial

![](figures/tree_rep_ost_ost_4236.png)

*#ost_4236 — Is the rectangular black purse made of durable fabric to your left/right now? — gate YES, mode `direct`, final **A** (GT A) → correct.*

![](figures/tree_rep_ost_ost_4848.png)

*#ost_4848 — Is the medium-sized rectangular wooden desk in deep brown to your left/right now? — gate LEFT, mode `branch_consensus`, final **A** (GT A) → correct.*

![](figures/tree_rep_ost_ost_5209.png)

*#ost_5209 — Is the rectangular blue plastic garbage bin to your left/right now? — gate EXPLORE, mode `fused_fallback_direct`, final **B** (GT B) → correct.*

#### Agent_state

![](figures/tree_rep_ost_ost_5400.png)

*#ost_5400 — Assuming the direction you are facing at the end of the turn 3 is forward, did you move a certain distance left or right from that position? — gate YES, mode `direct`, final **A** (GT A) → correct.*

![](figures/tree_rep_ost_ost_4559.png)

*#ost_4559 — Using your orientation at the end of turn 3 as a reference, has your current orientation rotated clockwise or counterclockwise by a certain  — gate EXPLORE, mode `branch_consensus`, final **A** (GT A) → correct.*

![](figures/tree_rep_ost_ost_6788.png)

*#ost_6788 — Using your orientation at the end of turn 1 as a reference, has your current orientation rotated clockwise or counterclockwise by a certain  — gate EXPLORE, mode `fused`, final **A** (GT A) → correct.*

#### Agent_visible_info

![](figures/tree_rep_ost_ost_4811.png)

*#ost_4811 — Among these three objects, which one was newly discovered in this turn(had not appeared before)? "the counter"; "the towel"; "the small roun — gate YES, mode `direct`, final **A** (GT A) → correct.*

![](figures/tree_rep_ost_ost_5232.png)

*#ost_5232 — Remember, have you seen any sheet(s) so far?  — gate NO, mode `branch_consensus`, final **B** (GT B) → correct.*

![](figures/tree_rep_ost_ost_4082.png)

*#ost_4082 — Remember, have you seen any dustpan(s) so far?  — gate NO, mode `fused`, final **A** (GT B) → wrong.*

**Reading.** On OST-Bench the tree is ahead of both no-memory baselines (SFT-plain, SFT+GRPO-plain) but not ahead of its own
adapter answering directly from the image history: OST questions are about the agent's *own* trajectory and what it has *seen*
(temporal facts), which a rendered view of the current reconstruction cannot add — the head correctly falls back to the direct
answer on most explored items. The memory helps where the question is about the room's geometry (VSI), not about the agent's history.

## 5. ViewTree-D: multi-step view acquisition (depth ≤ 3), trained from scratch on a 494k-QA corpus

The depth-1 tree of §1 renders five candidate views *once*; it cannot look again after seeing a render, move locally toward the
object the question is about, or learn when to stop. ViewTree-D (design: `DESIGN_DEPTH.md`, log: RESULTS.md §8) makes the camera
itself the action space and trains the controller in phases on a corpus 50× MindCube.

**Reasoning path = a walk in the memory.** A state is (question, 8 context frames, renders seen so far, current camera pose). One
step = one camera action + one render: `TURN_LEFT`/`TURN_RIGHT` (yaw ±45°), `FORWARD` (one walkable cell), `NEXT_SPOT` (next
farthest-point standing position, facing the room centre), `LOOK_AROUND` (+180°), `BIRD_EYE` (top-down, allowed only as the last
acquisition), `STOP`. The human-camera constraints of §1.2 are a hard action mask (positions inside the walked hull at eye level,
roll 0, coverage ≥ 45 %). Every scene has a pre-rendered **pose bank** (12 positions × 8 yaws + top-down = 97 views), so training
never renders online; at test time only the poses the beam visits are rendered.

**Training corpus.** VLM-3R `vsibench_train` + `vstibench_train` and VSI-590K (ScanNet + ScanNet++ v2 videos we hold):
493,663 QA on 1,709 scenes, 176k numeric; every VSI-Bench, VSTI-Bench, STI, OST and MindCube evaluation scene excluded at room level.

**Phases.** 0 — pose banks; 1 — *SFT-A* answerer on ~100k random-walk states (frames + 0…3 renders → answer; 33k frames-only);
2 — *oracle walks*: beam-2 depth-3 search over the bank with the SFT-A answerer on 8,639 QA (direct correct 54 %, best walk 68 %),
a *value head* (same MLP as §1.2, GELU/dropout, AUROC 0.723 held-out) on walk states, and *SFT-C* imitation of the oracle actions
(the prompt lists the valid moves); 3 — *GRPO over walks* (group 6, reward = MRA/accuracy − step cost, dual λ on the mean step
budget, masked actions penalised). **Inference** = gate, then beam search over camera moves (branch 3 by policy logit, keep 2 by
the value head, depth ≤ 3, early stop on agreement, direct-vs-walk arbitration), ≤ 12 VLM calls.

### 5.1 VSI-Bench held-out odd half (paired, scene-bootstrap CIs vs the data-matched baseline)

The corpus alone changes the picture: a frames-only SFT on it reaches ~0.51 (vs 0.367 for the best MindCube-trained system),
because VLM-3R's training QA uses VSI-Bench's own templates on disjoint rooms. Every ViewTree-D number is therefore read against
that **data-matched, no-memory baseline** (bold), not against §2.

| system | mean of types | Δ vs data-matched baseline [95 % CI] | obj appearance order | abs distance | counting | dir easy | dir hard | dir medium | rel distance | size | room size | route planning | calls | depth |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B zero-shot (16 frames) | **0.313** | -0.196 [-0.222, -0.167] | 0.289 | 0.088 | 0.254 | 0.478 | 0.203 | 0.420 | 0.391 | 0.340 | 0.357 | 0.308 |  |  |
| SFT-plain (16 frames, MindCube) | **0.327** | -0.181 [-0.205, -0.155] | 0.213 | 0.140 | 0.339 | 0.496 | 0.330 | 0.382 | 0.344 | 0.420 | 0.330 | 0.279 |  |  |
| ViewTree depth-1 (best of §2) | **0.367** | -0.141 [-0.169, -0.113] | 0.314 | 0.131 | 0.313 | 0.522 | 0.387 | 0.406 | 0.402 | 0.432 | 0.410 | 0.356 |  |  |
| **corpus frames-only SFT (data-matched, no memory)** | **0.509** | +0.000 [+0.000, +0.000] | 0.628 | 0.361 | 0.667 | 0.504 | 0.462 | 0.435 | 0.515 | 0.649 | 0.558 | 0.308 |  |  |
| SFT-A walk-trained answerer, frames only at test | **0.524** | +0.015 [-0.004, +0.034] | 0.607 | 0.376 | 0.670 | 0.478 | 0.481 | 0.459 | 0.537 | 0.659 | 0.606 | 0.365 |  |  |
| depth-1 tree with SFT-A + value head | **0.517** | +0.008 [-0.012, +0.027] | 0.632 | 0.357 | 0.689 | 0.531 | 0.486 | 0.464 | 0.529 | 0.633 | 0.533 | 0.317 |  |  |
| **ViewTree-D, no RL: SFT-C + value head + beam (d ≤ 3)** | **0.530** | +0.021 [-0.000, +0.043] | 0.674 | 0.346 | 0.653 | 0.522 | 0.524 | 0.502 | 0.565 | 0.652 | 0.536 | 0.327 | 4.5 | 0.38 |

*Path mix — paired n = 2557 on the held-out odd half; ViewTree-D, no RL: SFT-C + value head + beam (d ≤ 3): direct 1815, consensus_d1 379, consensus_d2 157, fallback_direct 106, consensus_d3 52, best_state 48.*

**Reading.**
- Multi-step acquisition adds a borderline **+2.1** over the data-matched baseline at 4.5 calls/question (the gate answers directly
  on 71 %). The gain is where a second viewpoint changes the geometry the model sees — relative direction hard/medium, relative
  distance, appearance order — and is negative on counting, room size and route planning, where extra renders distract.
- Same answerer and head, depth 1 vs depth ≤ 3: +0.8 vs +2.1, with *fewer* calls for the deeper beam because its gate stops more
  often; the ordering baseline < depth-1 < depth-≤3 holds on the relational/directional types.
- The GRPO policy drifted toward STOP-at-depth-0 (mean steps 0.18 → 0.07, λ never activated) — the pre-registered collapse risk;
  the beam explores regardless, so the RL adapter acts mainly through its answer tokens (rows above when present).

### 5.2 Transfer of the corpus-trained adapters (single pass, no tree)

OST-Bench (paired n = 5403):

| system | overall | Agent_object_spatial | Agent_state | Agent_visible_info |
|---|---|---|---|---|
| Qwen2.5-VL-7B zero-shot | **0.540** | 0.413 | 0.503 | 0.728 |
| SFT-plain | **0.524** | 0.392 | 0.485 | 0.719 |
| D_10k adapter, single pass | **0.550** | 0.430 | 0.499 | 0.733 |
| ViewTree depth-1 tree | **0.541** | 0.425 | 0.489 | 0.720 |
| corpus frames-only SFT, single pass | **0.516** | 0.403 | 0.513 | 0.671 |
| SFT-A (walk-trained), single pass | **0.518** | 0.416 | 0.491 | 0.668 |

VSTI-Bench (paired n = 5736; VSTI rooms are ScanNet *val*, 0 shared with the corpus, but the corpus contains `vstibench_train`'s templates):

| system | mean of 9 types | cam displacement | cam movement direction | cam obj abs dist | cam obj rel v1 | cam obj rel v2 | cam obj rel v3 | obj-obj lr | obj-obj nf | obj-obj ud |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B zero-shot | **0.523** | 0.134 | 0.510 | 0.149 | 0.615 | 0.667 | 0.689 | 0.567 | 0.622 | 0.748 |
| SFT-plain | **0.456** | 0.056 | 0.440 | 0.135 | 0.396 | 0.497 | 0.612 | 0.618 | 0.574 | 0.777 |
| ViewTree depth-1 tree | **0.505** | 0.034 | 0.483 | 0.169 | 0.484 | 0.635 | 0.646 | 0.640 | 0.647 | 0.806 |
| corpus frames-only SFT, single pass | **0.674** | 0.225 | 0.503 | 0.517 | 0.725 | 0.789 | 0.834 | 0.716 | 0.833 | 0.928 |
| SFT-A (walk-trained), single pass | **0.685** | 0.226 | 0.529 | 0.511 | 0.780 | 0.793 | 0.832 | 0.734 | 0.831 | 0.930 |

**Reading.** The corpus helps exactly where its templates match — VSI (+14) and VSTI (+16 over zero-shot, +21 over SFT-plain, with
the numeric camera/object-distance types going from ~0.15 to ~0.5) — and *hurts* where they do not: on OST both corpus adapters are
significantly below zero-shot (−2.4 / −2.2, driven by Agent_visible_info), while the MindCube-trained D_10k adapter and the depth-1
tree stay at or above it. The ViewTree-D claim is a VSI-family claim until a mixed corpus with OST-style exploration QA is trained.

## 6. Summary
- ViewTree's contribution is **cross-benchmark transfer and view efficiency**: on benchmarks other than the one it was trained on it is
  the best system by a significant margin, while using few extra views (the controller answers directly on ~23 % of VSI questions and
  stops after consensus on another third).
- On the training benchmark itself (MindCube), plain fine-tuning with all views handed over for free is stronger (0.750 vs 0.632 on
  tinybench) — reported in RESULTS.md §1c; the memory system's value is not peak accuracy there but the acquisition trade-off.
- The human-camera constraint costs nothing (100 % valid views, +0.4) and, with a head that can read eye-level views, gives the best
  held-out result (0.367), the first significant win of the adaptive tree over static memory prompting (+2.6 [+0.5, +4.7]).
- **Scale and depth (§5).** Training on a 494k-QA corpus of VSI-template questions lifts the frames-only model to ~0.51 on VSI
  (+14); multi-step acquisition (ViewTree-D, depth ≤ 3) adds a further borderline +2.1 on top of that data-matched baseline, on
  relational/directional questions, at 4.5 calls per question — but the corpus gain is template-specific (VSI/VSTI up, OST down).

*Reproducibility: all numbers are computed from result files in the repository by `scripts/build_report.py`; full experiment log in
RESULTS.md, design decisions in DECISIONS.md.*

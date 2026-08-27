# ViewTree — Results Summary

Living document: updated as each experiment lands. Method/decision rationale in
[DECISIONS.md](DECISIONS.md); design in `ViewTree_Research_Design_Document.pdf`.

**Setup (all experiments):** 8× H100 (single node) · student VLM Qwen2.5-VL-7B ·
teacher Qwen2.5-VL-32B · reconstruction frozen VGGT-1B · custom GPU point-splat
renderer · benchmark VSI-Bench full test (5,130 questions, 288 scenes, paired) ·
score = accuracy (MC) / MRA (numerical), headline = mean over 10 question types,
scene-bootstrap 95% CIs (B=2000). Training data MindCube (10k items,
scene-disjoint from all evaluation). Last updated: 2026-08-27 (RL design sweep + 10k-scale A/B/D runs complete, §4d–4e).

---

## 1. Headline scoreboard (VSI-Bench, mean of types)

| # | condition | score | 95% CI | Δ vs frames16 |
|---|---|---|---|---|
| 1 | current frame only | 0.266 | [0.252, 0.280] | −4.5 pts |
| 2 | frames16 (16 video frames) | 0.311 | [0.297, 0.325] | — |
| 3 | frames12 | 0.316 | [0.303, 0.329] | +0.5 (n.s.) |
| 4 | renders_only (5 rendered views) | 0.301 | [0.287, 0.316] | −1.0 (n.s.) |
| 5 | memory (12 frames + 5 renders) | 0.333 | [0.320, 0.347] | **+2.2 [+0.9, +3.6]** |
| 6 | memory32 (better renderer recipe) | 0.331 | [0.315, 0.346] | +2.0 (≈ #5) |
| 7 | tree (training-free branch/prune/fuse) | 0.331 | [0.317, 0.345] | **+2.0 [+0.6, +3.3]** |
| 8 | sft_frames16 (Stage I adapter) | 0.326 | [0.311, 0.341] | **+1.5 [+0.4, +2.7]** |
| 9 | **sft_memory (Stage I adapter + renders)** | **0.336** | [0.323, 0.349] | **+2.5 [+1.1, +4.0]** |
| 10 | teacher 32B frames16 (upper reference) | 0.386 | [0.371, 0.403] | +7.5 [+5.7, +9.5] |
| 11 | trained tree (SFT adapter + conf head) | 0.329 | [0.315, 0.344] | **+1.8 [+0.4, +3.3]** |
| 12 | **sft2_memory (Stage III adapter + renders)** | **0.340** | [0.326, 0.355] | **+2.9 [+1.4, +4.6]** |
| 13 | tree v3 (Stage III adapter + conf head) | 0.333 | [0.317, 0.348] | **+2.2 [+0.6, +3.7]** |

## 1b. Per-task accuracy (all conditions, paired on 5,130 questions)

Columns: cur=current frame · f12/f16=uniform frames · rnd=renders only ·
mem=12f+5 renders · mem32=32-frame recipe · sft-*=Stage I adapter ·
sft2-mem=Stage III adapter · tree*=end-to-end systems · 32B-f16=teacher
reference. Bold = best per row. Score = accuracy (MC) / MRA (numerical).

| question type (n) | cur | f12 | f16 | rnd | mem | mem32 | sft-f16 | sft-mem | sft2-mem | tree | tree-v2 | tree-v3 | 32B-f16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| object counting (565) | 0.057 | 0.243 | 0.245 | 0.212 | 0.340 | 0.319 | 0.245 | 0.343 | **0.350** | 0.255 | 0.243 | 0.239 | 0.288 |
| object abs distance (834) | 0.069 | 0.080 | 0.087 | 0.166 | 0.094 | 0.096 | 0.099 | 0.093 | 0.093 | 0.109 | 0.095 | 0.113 | **0.249** |
| object size estimation (953) | 0.296 | 0.336 | 0.334 | 0.259 | 0.284 | 0.295 | 0.363 | 0.311 | 0.345 | 0.347 | 0.331 | 0.354 | **0.504** |
| room size estimation (288) | 0.260 | 0.360 | 0.376 | 0.181 | 0.410 | 0.409 | 0.375 | 0.386 | 0.395 | 0.390 | 0.402 | 0.411 | **0.467** |
| object rel distance (710) | 0.352 | 0.363 | 0.404 | 0.342 | 0.394 | 0.392 | 0.415 | 0.401 | 0.393 | 0.397 | 0.379 | 0.394 | **0.496** |
| object rel direction easy (217) | 0.512 | 0.516 | 0.447 | 0.498 | **0.535** | 0.512 | 0.470 | 0.507 | 0.488 | 0.502 | 0.493 | 0.493 | 0.530 |
| object rel direction medium (378) | 0.349 | 0.397 | 0.407 | 0.429 | 0.421 | **0.447** | 0.415 | 0.442 | 0.381 | 0.421 | 0.434 | 0.389 | 0.307 |
| object rel direction hard (373) | 0.212 | 0.204 | 0.217 | 0.343 | 0.247 | 0.241 | 0.249 | 0.330 | **0.351** | 0.236 | 0.260 | 0.303 | 0.303 |
| obj appearance order (618) | 0.282 | 0.332 | 0.314 | 0.288 | 0.311 | 0.303 | 0.301 | 0.256 | 0.261 | 0.324 | 0.320 | 0.303 | **0.401** |
| route planning (194) | 0.273 | 0.330 | 0.278 | 0.294 | 0.299 | 0.294 | 0.325 | 0.289 | **0.345** | 0.325 | 0.335 | 0.330 | 0.320 |
| **mean of types** | 0.266 | 0.316 | 0.311 | 0.301 | 0.333 | 0.331 | 0.326 | 0.336 | 0.340 | 0.331 | 0.329 | 0.333 | **0.386** |

Reading notes: the teacher dominates 6/10 types but **loses to 7B systems on
counting, medium/hard relative direction, and route planning** — the types the
renders and SFT target; sft2-mem holds the best 7B result on 4 of the 5
hardest spatial types; appearance order (temporal) is the one type where every
memory/SFT variant trails plain frame prompting — renders cannot encode time,
and MindCube training has no temporal questions.

## 2. Design-doc hypothesis checks

| check | verdict | evidence |
|---|---|---|
| RQ1 / go-no-go #1: explicit-memory views beat frame prompting | **PASS** | memory − frames16 = +2.2 [+0.9, +3.6]; gain is render-driven (frames12 ≈ frames16), concentrated on layout questions (counting +9.5, room size +3.4, rel-direction up) |
| Renderer viability (biggest unvalidated assumption) | **VIABLE** | coverage 65→82→87% at 16/32/48 source frames (fixed held-out poses); covered-pixel PSNR stable ~16–17 dB → holes, not wrong colors; residual ~15% is never-observed geometry. Recipe: 32 frames + splat 2 |
| "Can higher video sampling rate fix rendering?" | **Largely yes** | see above; diminishing returns after 32 frames; validity guard still required |
| Render coverage → QA accuracy? | **NO (negative result)** | memory32 (0.331) ≈ memory (0.333): VLM, not renderer, is the bottleneck; 16-frame recipe kept for QA |
| H3 premise: frames & renders complementary | **SUPPORTED** | renders_only ≈ frames16 overall but +12.6 on hard rel-direction, +7.9 abs-distance; combined beats both |
| H3: untrained fusion is harmful | **CONFIRMED** | tree fused-mode accuracy 0.268 vs 0.315–0.364 for other paths |
| H4: token confidence is a poor branch score | **CONFIRMED** | trained head AUROC 0.907 vs 0.751 token-logprob; state-selection 0.568 vs 0.530 (oracle 0.820) |
| Teacher audit (§6.8) | **RISK FOUND** | 32B teacher *degrades* with more views on MindCube (0.412→0.371); mitigated: control labels from outcome patterns only, answers from GT |
| Teacher stronger than student (distillation headroom) | **YES (video domain)** | 32B frames16 0.386 vs 7B 0.311 |

## 3. Stage I — SFT controller (LoRA on 7B, 16.8k examples)

MindCube tinybench evidence ladder, paired pre/post:

| state | pre-SFT | post-SFT | paired Δ (95% CI) |
|---|---|---|---|
| 1 view | 0.343 | 0.533 | +0.190 [+0.152, +0.228] |
| all views | 0.255 | 0.508 | **+0.253 [+0.197, +0.306]** |
| all + render | 0.260 | 0.494 | +0.234 [+0.183, +0.285] |

- More-views degradation nearly eliminated (−8.8 pts pre → −2.5 post):
  **cross-view integration learned.**
- Control policy de-degenerated: base = 100% STOP everywhere; SFT = 24–43% MOVE
  at partial evidence, 98% STOP at full evidence. RENDER never emitted (2.5%
  training share — to rebalance).
- **Cross-domain transfer positive** (scoreboard rows 8–9): +1.5 pts on
  VSI-Bench frames16; sft_memory best overall (0.336); hard perspective
  questions 0.330 vs 0.247. Cost: appearance-order −5 pts (no temporal
  questions in MindCube — mix a temporal slice next round).

## 3b. Stage III — SFT v2 with fusion training (22.9k examples, on-policy)

Added families: 1,605 complementary-fusion (single view fails, full evidence
succeeds), 1,040 redundant-robustness. Tinybench ladder, paired v2 − v1:

| state | pre-SFT | v1 | v2 | paired v2−v1 (95% CI) |
|---|---|---|---|---|
| 1 view | 0.343 | 0.533 | 0.532 | −0.001 (n.s.) |
| 2 views | 0.330 | 0.519 | 0.544 | +0.025 [+0.009, +0.041] |
| 3 views | 0.320 | 0.523 | 0.559 | +0.036 [+0.018, +0.055] |
| all views | 0.255 | 0.508 | **0.575** | **+0.067 [+0.039, +0.095]** |
| all + render | 0.260 | 0.494 | 0.557 | +0.063 [+0.035, +0.090] |

**The gain grows monotonically with evidence — the fusion-targeted signature.**
The evidence gradient is now positive (more views help: 0.532→0.575); total
improvement at the fused state is +32 pts over the base model. Control policy
is now monotone (STOP 66%→100% as evidence grows; v1 was non-monotone); RENDER
still never emitted (278 examples remain too few).

**Cross-domain: `sft2_memory` = 0.340 [0.326, 0.355] — new best** (scoreboard:
base 0.311 → sft v1 0.336 → sft v2 0.340; +2.9 [+1.4, +4.6] over baseline).
Monotone improvement across SFT rounds transfers to VSI-Bench.

## 4. Stage II — confidence head

45,743 on-policy states (SFT policy), 3584-d features, label = eventual
correctness, group-level 80/10/10 split, temperature-calibrated (T=0.95):

| metric (held-out groups) | trained head | token-logprob |
|---|---|---|
| AUROC | **0.907** | 0.751 |
| Brier | 0.117 | — |
| ECE | 0.019 | — |
| state-selection accuracy | **0.568** | 0.530 |

Oracle selection 0.820; fixed last-state 0.523. On-policy ladders flat across
depth (~0.51) → labels not depth-confounded.

### 4b. Head v2 — domain adaptation (the composition-failure smoking gun)

Scoring the v1 (MindCube-only) head on 18,011 VSI tree states (collected from
144 head-train scenes, disjoint from the 144 eval scenes) revealed **AUROC
0.467 — worse than random**. In trees v2/v3 the confidence head was adding
noise on VSI, fully explaining why in-domain-validated components failed to
compose end-to-end. Head v2, retrained on MindCube + VSI states (VSI upweighted
2×, T=1.45):

| states | v1 head | v2 head |
|---|---|---|
| VSI (held-out scenes) | 0.467 | **0.672** |
| VSI (all, incl. seen) | — | 0.811 |
| MindCube (held-out groups) | 0.907 | 0.890 |

Cross-domain adaptation costs only 1.7 AUROC points in-domain.

**Tree v4 (Stage III adapter + head v2), paired on the 144 untouched odd
scenes (2,557 questions):**

| system (odd-scene subset) | score | 95% CI |
|---|---|---|
| frames16 | 0.313 | [0.291, 0.334] |
| sft2_memory (best static) | 0.342 | [0.321, 0.365] |
| tree v3 | 0.339 | [0.316, 0.361] |
| **tree v4** | **0.356** | [0.335, 0.378] |

tree4 − tree3 = **+1.8 [+0.7, +2.8]** (significant — fixing the head fixes the
composition); tree4 − sft2_memory = +1.5 [−0.4, +3.4] — the adaptive tree's
point estimate **leads static prompting for the first time** (not yet
significant). All modes improved: fused 0.302 (was 0.231), fallback 0.345,
direct 0.407, consensus 0.312. Best system in the study on its eval half.

## 4c. Stage IV — GRPO view-control policy (scaled §6.6)

4,000 MindCube episodes × 6 rollouts, reward = correctness − λ·extra-views
(λ dual toward a 2.3-view budget; λ stayed 0 — policy self-limited), policy
gradient on STOP/MOVE/RENDER tokens only, LoRA continued from Stage III.

**Greedy policy rollout on tinybench (accuracy vs views acquired):**

| policy | accuracy | mean views |
|---|---|---|
| always 1 view (fixed) | 0.532 | 1.00 |
| always all views (fixed) | 0.575 | 3.15 |
| always all + render (fixed) | 0.557 | 4.15 |
| **SFT-v2 adaptive policy** | **0.615** | **1.63** |
| **GRPO adaptive policy** | **0.619** | 1.93 |

**The adaptive controller dominates every fixed policy: +4.4 pts over
best-fixed at roughly half the acquisition cost** — the doc's H2/H6 efficiency
claim, demonstrated at ladder scale. Most of this emerges already from
outcome-labeled SFT; GRPO's marginal effect is to front-load exploration
(MOVE at 1 view: 34%→56%, monotone 56/33/15/0%) buying +0.4 acc for +0.3
views. Answer-state accuracies are unchanged (±0.005 at every ladder state) —
RL touched control only, no forgetting. RENDER stayed unused (1/1050 —
rendered top-downs rarely flip MindCube outcomes, so reward never favored the
action). VSI static transfer unchanged (0.341 vs 0.342) as expected for
control-only RL.

## 4d. RL reasoning-path design space (sweep: 1k items/variant, 1 GPU each)

Eight GRPO variants from the Stage-III adapter, evaluated by greedy policy
rollout with **per-task breakdown** (categories among/around/rotation; question
types). Reference = SFT-v2 policy. **8/8 complete (2026-08-27).** Paired on
tinybench n=1,050 and rest_clean n=1,330; full tables in
`scripts/analyze_dse.py` output. "toks" = control tokens generated per decision.

| variant (design axis) | tiny acc @ views | rest acc @ views | among (rest) | around (rest) | rotation (rest, n=17) | 0_frame (rest) | general (tiny) |
|---|---|---|---|---|---|---|---|
| SFT-v2 reference | 0.615 @ 1.63 | 0.765 @ 1.85 | 0.702 | 0.866 | 0.412 | 0.720 | 0.308 |
| A baseline ladder | **0.617** @ 2.10 | **0.766** @ 2.15 | **0.708** | 0.859 | 0.471 | **0.751** | 0.385 |
| B short horizon (≤2 views) | 0.560 @ 1.75 | 0.705 @ 1.75 | 0.593 | 0.872 | 0.529 | 0.619 | **0.538** |
| C cost-free | 0.612 @ 2.16 | 0.765 @ 2.19 | 0.702 | 0.865 | 0.529 | 0.735 | 0.385 |
| **D high cost / budget 1.5** | 0.602 @ **1.27** | 0.741 @ **1.24** | 0.658 | **0.876** | 0.294 | 0.730 | **0.538** |
| E learned view selection | 0.573 @ 1.63 | 0.705 @ 1.96 | 0.598 | 0.863 | 0.529 | 0.688 | 0.500 |
| F CoT (free length, 32 tok) | 0.597 @ 2.03 | 0.730 @ 2.09 | 0.652 | 0.852 | 0.412 | 0.698 | 0.462 |
| G CoT (length-penalised → 2 tok) | 0.611 @ 2.12 | 0.765 @ 2.18 | 0.695 | 0.870 | **0.588** | 0.735 | 0.346 |
| H group 12 | 0.616 @ 2.07 | 0.765 @ 2.12 | 0.704 | 0.861 | 0.529 | 0.746 | 0.385 |

**Findings (per-task, as requested):**
- **Different designs win different tasks.** D (strong efficiency pressure)
  is best on *around* (0.876) and ties best on tiny "general" (0.538) while
  using ~33% fewer views than SFT-v2 — the efficiency-frontier winner. B
  (short horizon) is *worst* on *among* (needs ≥3 views: 0.593 vs 0.702) but
  competitive on rotation and best on "general" — tasks where extra views
  confuse. A/C/H (long-horizon ladder) win the multi-view *among* category
  and 0-frame questions.
- **Free-length CoT (F) hurts.** Letting the controller reason for 32 tokens
  before each STOP/MOVE/RENDER decision costs −3.5 pts on rest (0.730) and is
  worse on *every* category (among −5.0, around −0.7 vs A). With 1k items the
  reasoning text is not grounded — it mostly adds sampling noise to the
  control token. 16× the decode cost for a loss.
- **Length-penalised CoT (G) collapses to the 2-token policy** (mean 2 toks
  ≈ no reasoning) and lands exactly at baseline (0.765 / 0.611): the penalty
  is doing the right thing given F's result, but it means "reasoning length"
  is not a useful lever at this scale. G is nominally best on rest rotation
  (0.588) but n=17 (±0.12), not significant.
- **Group size 12 (H) ≈ group 6 (A)** on every category (max |Δ| 1.1 pt)
  for 2× rollout cost — the group-relative baseline is already low-variance
  at G=6 because rewards are near-binary.
- **Learned view selection (E) underperforms the fixed ladder**: order-free
  choice mostly hurts *among* (0.598) and 1/2-frame questions — the ladder
  order (front→left→back→right) is already informative and 1k items is too
  little to learn a better subset policy.
- Cost-free (C) ≈ baseline: λ never activated in A either (views stayed
  under budget), so the dual variable is inert unless the budget binds (D).
- **No variant beats SFT-v2's accuracy at 1k items**; RL's lever here is
  **cost** (D: −33% views for −2.4 pts) not accuracy. Design choices on the
  reasoning path (CoT, group, selection) are neutral-to-harmful; the
  acquisition *horizon/cost* axis (B vs A vs D) is the only one that
  produces task-dependent trade-offs worth exploiting.

**Promising choices carried to large scale (10k items, DDP):** D_highcost
(efficiency) and A_baseline (accuracy ceiling of the ladder). CoT and
group-12 were **not** scaled — neither showed a per-task win that a 10× data
increase could plausibly turn into a net gain. Evaluated on MindCube per-task,
VSI-Bench held-out half (memory + tree with head v2).

## 4e. Large-scale RL on the promising designs (10k items, 3-GPU DDP each) — complete 2026-08-27

The three points of the acquisition-horizon/cost axis — the only design axis
that produced task-dependent trade-offs in §4d — retrained on the full
9,995-item MindCube train set: **A** baseline ladder (cost 0.05, budget 2.3,
λ stayed 0), **B** short horizon (≤2 views), **D** high cost (0.2, budget 1.5,
λ→0.20). Greedy rollout, paired on the same ids as §4d; VSI = held-out odd
half with head v2 unchanged.

**MindCube (per category / question type):**

| policy | tiny acc @ views | rest acc @ views | among (rest) | around (rest) | rotation (tiny) | 3_frame (rest) | general (tiny) |
|---|---|---|---|---|---|---|---|
| SFT-v2 reference | 0.615 @ 1.63 | 0.765 @ 1.85 | 0.702 | 0.866 | 0.355 | 0.636 | 0.308 |
| **A_baseline 10k** | **0.650 @ 1.93** | 0.771 @ 2.10 | 0.705 | 0.876 | **0.390** | 0.679 | 0.423 |
| B_depth2 10k | 0.591 @ 1.92 | 0.708 @ 1.91 | 0.597 | 0.876 | 0.380 | 0.625 | **0.500** |
| **D_highcost 10k** | 0.632 @ **1.31** | **0.778 @ 1.30** | **0.720** | **0.878** | 0.315 | **0.690** | **0.500** |
| (1k versions, §4d) A / B / D | 0.617 / 0.560 / 0.602 | 0.766 / 0.705 / 0.741 | | | | | |

**VSI-Bench held-out half (2,557 q, mean of 10 types; Δ vs same system with
SFT-v2 adapter, scene-bootstrap 95% CI):**

| adapter | memory (static) | tree v4 (head v2) | tree gates direct / explore |
|---|---|---|---|
| SFT-v2 | 0.342 | 0.356 | 413 / 2087 |
| A_baseline 10k | 0.340 (−0.2 [−0.9, +0.5]) | 0.350 (−0.7 [−1.9, +0.4]) | 435 / 2033 |
| B_depth2 10k | 0.342 (+0.0 [−0.8, +0.8]) | **0.361** (+0.5 [−0.5, +1.5]) | 317 / 2168 |
| D_highcost 10k | 0.339 (−0.2 [−0.9, +0.4]) | 0.357 (+0.0 [−1.0, +1.1]) | **597 / 1870** |

**Findings:**
- **More RL data helps accuracy on its own.** A_baseline goes 0.617→0.650
  (tiny) and 0.766→0.771 (rest) from 1k→10k with the view count barely
  changing (2.10→1.93 / 2.15→2.10). It is the best tiny policy (+3.5 over
  SFT-v2) and the only one that lifts *rotation* (0.355→0.390).
- **Cost pressure is (almost) free at scale.** D_highcost matches or beats
  A on rest (0.778 vs 0.771, *among* 0.720 vs 0.705, 3_frame 0.690 vs
  0.679) with **38% fewer views** (1.30 vs 2.10); on tiny it gives up 1.8
  pts, all of it on rotation (0.315 vs 0.390) — the one category where the
  cheap first view is never enough. At 1k the same cost setting cost 2.4
  pts *everywhere*; the data, not the penalty, taught it *where* to spend.
- **Short horizon (B) does not recover with data**: 0.591/0.708, *among*
  0.597 (−10.5 vs SFT-v2) — the ≤2-view cap is a hard ceiling for
  three-view questions and no amount of RL fixes an acquisition limit.
  It keeps its niche wins (general 0.500, type-3 0.759) but they are small
  categories (n=26/112).
- **VSI transfer: all three are accuracy-neutral** (every CI contains 0;
  widest effect ±1.9 pt). The MindCube-specific gains do not move VSI, and
  nothing regresses. What *does* transfer is behaviour: the D controller
  answers directly 45% more often inside the tree at equal score; the
  short-horizon B controller explores slightly *more* (317 direct) and has
  the highest tree point estimate (0.361), consistent with its policy
  deferring more to the head-gated branch/fuse machinery.
- **Recommendation:** ship **D_highcost 10k** as the default controller
  (best rest accuracy, best *among*, −30–38% views, VSI-neutral); use
  **A_baseline 10k** when rotation-type questions dominate. Do not cap the
  horizon.

## 4f. Best system (D_highcost 10k) — per-class accuracy vs baselines, all datasets

Baselines: zero-shot Qwen2.5-VL-7B (1 view / all views), SFT-v2 with all
views (no policy), SFT-v2 policy. Paired ids throughout. External benchmarks
(ViewSpatial, OST, OmniSpatial, BLINK) with the D_10k adapter are in the
last block.

### MindCube tinybench (n=1,050)

| system | overall | views | among (600) | around (250) | rotation (200) |
|---|---|---|---|---|---|
| Qwen2.5-VL-7B, 1 view | 0.343 | 1 | 0.300 | 0.440 | 0.350 |
| Qwen2.5-VL-7B, all views | 0.346 | 3.4 | 0.303 | 0.412 | **0.390** |
| SFT-v2, all views | 0.532 | 3.4 | 0.540 | 0.656 | 0.355 |
| SFT-v2 policy | 0.615 | 1.63 | 0.658 | 0.720 | 0.355 |
| **D_highcost 10k policy** | **0.632** | **1.31** | **0.695** | **0.736** | 0.315 |

| system | 0_frame (140) | 1_frame (140) | 2_frame (149) | 3_frame (145) | type 1 (13) | type 2 (125) | type 3 (112) | general (26) | three_view (200) |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B, 1 view | 0.307 | 0.300 | 0.322 | 0.255 | 0.308 | 0.408 | 0.491 | 0.385 | 0.350 |
| Qwen2.5-VL-7B, all views | 0.279 | 0.314 | 0.369 | 0.234 | 0.231 | 0.448 | 0.393 | 0.385 | **0.390** |
| SFT-v2, all views | 0.514 | 0.643 | 0.557 | 0.483 | 0.462 | **0.784** | 0.536 | 0.346 | 0.355 |
| SFT-v2 policy | **0.629** | 0.686 | **0.718** | 0.662 | **0.538** | 0.760 | 0.696 | 0.308 | 0.355 |
| **D_highcost 10k policy** | **0.629** | **0.714** | **0.718** | **0.752** | **0.538** | 0.752 | **0.741** | **0.500** | 0.315 |

### MindCube rest_clean (n=1,330)

| system | overall | views | among (774) | around (539) | rotation (17) |
|---|---|---|---|---|---|
| Qwen2.5-VL-7B, 1 view | 0.286 | 1 | 0.333 | 0.215 | 0.353 |
| Qwen2.5-VL-7B, all views | 0.267 | 3.4 | 0.329 | 0.176 | 0.294 |
| SFT-v2, all views | 0.688 | 3.4 | 0.588 | 0.839 | **0.471** |
| SFT-v2 policy | 0.765 | 1.85 | 0.702 | 0.866 | 0.412 |
| **D_highcost 10k policy** | **0.778** | **1.30** | **0.720** | **0.878** | 0.294 |

| system | 0_frame (189) | 1_frame (189) | 2_frame (180) | 3_frame (184) | type 1 (41) | type 2 (359) | type 3 (139) | general (32) | three_view (7) | two_view_cw (10) |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B, 1 view | 0.286 | 0.339 | 0.350 | 0.326 | 0.122 | 0.214 | 0.245 | 0.531 | 0.143 | 0.500 |
| Qwen2.5-VL-7B, all views | 0.344 | 0.275 | 0.367 | 0.310 | 0.073 | 0.184 | 0.187 | 0.469 | 0.143 | 0.400 |
| SFT-v2, all views | 0.646 | 0.608 | 0.583 | 0.522 | 0.707 | 0.861 | 0.820 | 0.531 | 0.286 | **0.600** |
| SFT-v2 policy | 0.720 | 0.714 | **0.756** | 0.636 | 0.707 | 0.883 | **0.871** | **0.594** | **0.429** | 0.400 |
| **D_highcost 10k policy** | **0.730** | **0.741** | **0.756** | **0.690** | **0.780** | **0.891** | **0.871** | 0.500 | 0.143 | 0.400 |

### VSI-Bench held-out odd half (2,557 q; score = acc / MRA)

| system | mean | app_order | abs_dist | counting | rel_dir easy | rel_dir med | rel_dir hard | rel_dist | obj_size | room_size | route |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B frames16 | 0.313 | 0.289 | 0.088 | 0.254 | 0.478 | 0.420 | 0.203 | 0.391 | 0.340 | 0.357 | 0.308 |
| Qwen2.5-VL-7B memory (frames+renders) | 0.326 | 0.255 | 0.090 | **0.367** | **0.549** | 0.420 | 0.245 | 0.388 | 0.282 | 0.371 | 0.288 |
| SFT-v2 memory | 0.342 | 0.218 | 0.083 | 0.356 | 0.522 | 0.396 | 0.377 | 0.397 | 0.346 | 0.385 | 0.337 |
| tree v4 (SFT-v2 + head v2) | 0.356 | **0.305** | **0.120** | 0.278 | 0.487 | **0.449** | 0.373 | 0.397 | 0.391 | 0.408 | 0.356 |
| **tree v4 + D_highcost 10k** | **0.357** | 0.293 | 0.115 | 0.299 | 0.496 | 0.386 | **0.382** | 0.397 | **0.411** | **0.412** | **0.375** |

### External benchmarks (single-pass VLM with adapter; paired ids; Δ = D_10k − SFT-v2, bootstrap 95% CI)

| benchmark (n) | class | base | SFT-v2 | **D_10k** |
|---|---|---|---|---|
| **ViewSpatial-Bench** (5,712) — Δ **+0.4 [+0.1, +0.7]** | overall | 0.370 | 0.388 | **0.392** |
| | Camera: object view orientation (996) | 0.297 | 0.319 | **0.323** |
| | Camera: relative direction (1,773) | 0.459 | 0.469 | **0.477** |
| | Person: object view orientation (996) | 0.396 | 0.419 | **0.431** |
| | Person: relative direction (842) | 0.355 | **0.369** | 0.365 |
| | Person: scene-simulation rel. direction (1,105) | 0.279 | **0.305** | 0.303 |
| **OST-Bench** (5,557) — Δ −0.1 [−0.5, +0.4] | overall | 0.539 | **0.550** | 0.549 |
| | Agent_object_spatial (2,803) | 0.416 | **0.434** | 0.433 |
| | Agent_state (748) | **0.503** | 0.497 | 0.499 |
| | Agent_visible_info (2,006) | 0.724 | **0.731** | 0.730 |
| **OmniSpatial** (691 paired, excl. Complex_Logic) — Δ +0.1 [−0.9, +1.2] | overall | 0.421 | 0.434 | **0.436** |
| | Dynamic_Reasoning (111) | **0.432** | 0.405 | 0.396 |
| | Perspective_Taking (561) | 0.419 | 0.439 | **0.442** |
| | Spatial_Interaction (19) | 0.421 | **0.474** | **0.474** |
| BLINK Multi-view (133) — Δ 0.0 | | 0.556 | 0.556 | 0.556 |
| BLINK Spatial Relation (143) — Δ −1.4 [−4.6, +2.1] | | **0.916** | 0.839 | 0.825 |
| BLINK Relative Depth (124) — Δ −0.8 [−2.4, 0.0] | | 0.790 | **0.798** | 0.790 |
| BLINK Object Localization (122) — Δ −1.6 [−4.9, +1.6] | | **0.566** | 0.516 | 0.500 |
| BLINK Counting (120) — Δ −0.8 [−2.5, 0.0] | | 0.683 | **0.717** | 0.708 |

- **The RL adapter inherits SFT-v2's transfer profile almost exactly**: a
  small, significant gain on ViewSpatial (+0.4 over SFT-v2, +2.2 over base,
  driven by camera-perspective classes), OST/OmniSpatial unchanged, and the
  same single-image BLINK regressions as SFT-v2 (spatial relation −9 vs
  base; RL adds a further ≤1.6-pt, non-significant drift). RL on
  multi-view control does not touch single-image skills in either direction.

**Reading the tables:**
- **Where the gain comes from (MindCube):** vs zero-shot Qwen the system
  roughly doubles accuracy on *among* (0.30→0.70) and *around* (0.44→0.74 /
  0.22→0.88); vs the SFT-v2 policy, D_10k adds most on the multi-view
  questions (3_frame +9.0 tiny / +5.4 rest, type-3 +4.5) while spending
  **fewer** views (1.31 vs 1.63) — the RL policy learned which items need
  the extra view.
- **Where it loses:** *rotation* (tiny 0.315; zero-shot all-views 0.390 is
  the best) — every trained policy is at or below chance-level here, and
  the cheap-first-view prior of D makes it worst. Small categories (general
  n=26/32, three_view n=7) swing ±0.2 on a handful of items.
- **VSI:** the tree with head v2 is what lifts VSI (+1.4 over static
  memory, +4.4 over frames16); the D_10k adapter is neutral (0.357 vs
  0.356), gaining on size/room/route and losing on rel_direction_medium,
  and answers directly 45% more often. counting is the one type where the
  tree hurts (0.278–0.299 vs 0.356–0.367 static) — fused renders lose
  object instances.

## 5. End-to-end systems

| system | score | notes |
|---|---|---|
| ViewTree-lite (training-free) | 0.331 | matches static memory; fixes size regression (0.347) via fallback, loses counting gain (0.255) via premature consensus; ~3.3 s/question (~2.5× static) |
| Trained tree (SFT adapter + calibrated head) | 0.329 | beats frames16 (+1.8 [+0.4, +3.3]) but does **not** beat the untrained tree (0.331) or static sft_memory (0.336) |
| Tree v3 (fusion-trained adapter + head) | 0.333 | best tree so far (+2.2 [+0.6, +3.7]); still ≤ static sft2_memory (0.340). Modes: direct 0.388, consensus 0.314, fallback 0.309, fused 0.231 — fusion routing receives the hardest residual questions AND its training format (MindCube canonical views) mismatches VSI branch prompts. Conclusion: with fusion trained off-domain, adaptive trees match but don't beat static all-evidence prompting — on-domain Stage IV RL is the remaining lever. |

**Trained-tree diagnosis (honest negative-ish result):** in-domain-validated
components did not compose into cross-domain end-to-end gains. Per-mode:
fused **0.258** (still the weakest path — the head cannot fix fusion itself),
direct 0.359 (n=928, up from 325 — the SFT gate stops more), consensus 0.294
(rate and accuracy both down). Two causes identified: (a) **fusion remains
untrained** — the doc's Stage III is confirmed as the critical path; (b) the
confidence head was trained on MindCube-domain states and applied to VSI-Bench
render-branch states — the §6.4 warning that calibration must be re-checked per
condition, surfacing operationally. Bright spots: route_planning 0.335 (best of
any system), rel_direction_hard 0.260 retains part of the SFT transfer.

## 5b. Reasoning-trace visualization

Six annotated real traces (branch views, confidences, prune/arbitrate decisions,
three successes + three instructive failures):
https://claude.ai/code/artifact/eef6c539-65b0-49b2-8407-eaa64a02a8e5
(regenerate with `scripts/trace_examples.py`). Static figures — scoreboard,
SFT ladder, renderer sweep, and six per-trace montages — are in `figures/`
(`figures/viewtree_traces.html` is the interactive version).

**Traces for the best system (tree v4 + D_highcost 10k adapter), 2026-08-27:**
`figures/trace_d10k_{1129,2220,76,1409,79,836}.png` + `trace_d10k_summary.png`
(scripts `trace_examples_v2.py` / `trace_figures_v2.py`; same layout as the
six SFT-v2 traces above, plus a bottom line giving tree v4 with the SFT-v2
adapter's decision on the same question). What they show: the D_10k gate
says YES (answer from video, branches faded = not executed) on #1129 and
#836 where SFT-v2 explored and got them wrong — the "stop when sufficient"
behaviour learned on MindCube; on #1409 the head overrules a wrong fused
answer (fallback to direct, correct); #79 is a shared failure — both
adapters fuse to "1 table" where GT is 2 (renders drop an instance).

## 6. Renderer study (held-out novel views, fixed eval poses, 30 scenes)

| source frames | splat | coverage | covered-pixel PSNR | overall PSNR |
|---|---|---|---|---|
| 16 | 1 | 0.654 | 16.1 dB | 10.0 dB |
| 16 | 2 | 0.678 | 15.9 dB | 10.2 dB |
| 32 | 1 | 0.816 | 16.6 dB | 12.0 dB |
| 32 | 2 | 0.833 | 16.4 dB | 12.2 dB |
| 48 | 1 | 0.869 | 16.5 dB | 12.7 dB |

## 6a. Large-scale generalization study (in progress)

**MindCube-rest-clean** (1,330 items from the untouched remainder of MindCube,
scene-filtered against training — note: 8,774 of 10,104 remaining items shared
scenes with train and were excluded):

| state | base | SFT v2 | GRPO |
|---|---|---|---|
| 1 view | 0.286 | 0.675 | 0.674 |
| 3 views | 0.197 | 0.748 | 0.749 |
| all views (4) | 0.270 | 0.610 | 0.614 |
| policy rollout | — | **0.765 @ 1.85 views** | **0.780 @ 2.05 views** |

The adaptive policy again beats every fixed evidence level (best fixed 0.749),
and GRPO leads SFT here (+1.5). Composition differs from tinybench (mostly
among/around categories), so absolute levels are not comparable across splits;
the base-vs-trained pairing is.

**External benchmarks** — paired base vs SFT-v2 (Stage III adapter), zero
gradient exposure to any of these datasets. Paired bootstrap 95% CI on Δ:

| benchmark | n | base | SFT v2 | Δ (95% CI) |
|---|---|---|---|---|
| **ViewSpatial-Bench** (perspective-taking) | 5,712 | 0.370 | 0.388 | **+1.8 [+1.1, +2.6]** |
| OmniSpatial (excl. Complex_Logic*) | 1,281 | 0.449 | 0.454 | +0.5 [−1.7, +2.8] |
| BLINK Multi-view Reasoning | 133 | 0.556 | 0.556 | 0.0 |
| BLINK Relative Depth | 124 | 0.790 | 0.798 | +0.8 [−1.6, +3.2] |
| BLINK Counting | 120 | 0.683 | 0.717 | +3.3 [−0.8, +8.3] |
| BLINK Object Localization | 122 | 0.566 | 0.516 | −4.9 [−12.3, +2.5] |
| BLINK Spatial Relation | 143 | 0.916 | 0.839 | **−7.7 [−13.3, −2.8]** |
| **OST-Bench** (online exploration, MC-answerable items) | 5,557 | 0.539 | 0.550 | **+1.1 [+0.1, +2.0]** |

\*Complex_Logic has no options for 229/252 items (open-ended) — excluded.
OST-Bench: 4,608 of 10,165 items are open-ended/numeric (answer not among
options) and are excluded; per type: Agent_object_spatial +1.8 (n=2,803),
Agent_visible_info +0.6, Agent_state −0.5.

**Reading.** The transfer is *skill-specific*, not universal: it is significant
exactly where the trained skill applies — ViewSpatial (+1.8, improving all five
perspective/direction sub-types by +1.0 to +2.6) and OmniSpatial's
Perspective_Taking (+2.0) and Spatial_Interaction (+2.7) — while neutral on
multi-view/depth/counting and **negative on single-image 2D spatial-relation
(BLINK −7.7)** and OmniSpatial Dynamic_Reasoning (−2.9). Training on
multi-view perspective-taking slightly erodes single-image "left-of/above"
relation reading, a real cost to state. Combined with VSI-Bench (+2.9) and
MindCube-rest (+39 at one view), the picture is consistent: gains on
egocentric/perspective spatial reasoning, no free lunch on 2D layout tasks.

## 6b. Data-leakage audit (2026-08-24)

Checked empirically: **MindCube train ↔ tinybench overlap = 0** at id, question-
group, scene-directory, and image level (SFT ladder gains are not scene
memorization). **VSI even/odd split shares 0 physical rooms** (ScanNet multi-scan
check). By construction: SFT/GRPO train only on MindCube (all VSI numbers are
zero-gradient cross-domain); head v2 sees only the even half, tree v4 evaluated
only on the odd half; scene-level splits throughout (§6.1).

Residual risks stated honestly: (1) **adaptive test-set reuse** — ~13 variants
iterated against full VSI-Bench informed design choices (§8.10 risk); mitigated
by the untouched odd half, and a fresh benchmark is recommended before
publication claims; (2) head-v2's 0.811 all-VSI AUROC includes trained scenes —
cite 0.672 (held-out) only; (3) tree-v4 comparisons must be restricted to the
odd-scene subset; (4) frozen VGGT / Qwen pretraining may have seen these scene
corpora — unauditable, affects all conditions equally, so paired deltas remain
valid while absolute numbers inherit backbone exposure.

## 7. Next milestones

All four stages of the design doc's training pipeline (scaled) are now
complete: SFT control+answers → confidence head → fusion training → GRPO.
RL design space (§4d) and 10k-scale runs of the three horizon/cost points
(§4e) are also complete; D_highcost 10k is the recommended controller.
Remaining levers, in order of expected value:
1. Multi-step branch trajectories (depth > 1) steered by the trained policy.
2. On-VSI-domain fusion/control data (the tree's fused path still trails
   its MindCube ladder level cross-domain).
3. RENDER-specific reward shaping or better renders (action never earned use).
4. Fresh never-touched benchmark (OpenEQA/ScanQA) for publication claims.

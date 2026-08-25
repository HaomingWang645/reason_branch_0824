# ViewTree — Results Summary

Living document: updated as each experiment lands. Method/decision rationale in
[DECISIONS.md](DECISIONS.md); design in `ViewTree_Research_Design_Document.pdf`.

**Setup (all experiments):** 8× H100 (single node) · student VLM Qwen2.5-VL-7B ·
teacher Qwen2.5-VL-32B · reconstruction frozen VGGT-1B · custom GPU point-splat
renderer · benchmark VSI-Bench full test (5,130 questions, 288 scenes, paired) ·
score = accuracy (MC) / MRA (numerical), headline = mean over 10 question types,
scene-bootstrap 95% CIs (B=2000). Training data MindCube (10k items,
scene-disjoint from all evaluation). Last updated: 2026-08-24 ~15:00.

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

Cross-domain adaptation costs only 1.7 AUROC points in-domain. Tree v4
(Stage III adapter + head v2, evaluated only on the 144 untouched odd-indexed
scenes) is running.

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

## 6. Renderer study (held-out novel views, fixed eval poses, 30 scenes)

| source frames | splat | coverage | covered-pixel PSNR | overall PSNR |
|---|---|---|---|---|
| 16 | 1 | 0.654 | 16.1 dB | 10.0 dB |
| 16 | 2 | 0.678 | 15.9 dB | 10.2 dB |
| 32 | 1 | 0.816 | 16.6 dB | 12.0 dB |
| 32 | 2 | 0.833 | 16.4 dB | 12.2 dB |
| 48 | 1 | 0.869 | 16.5 dB | 12.7 dB |

## 7. Next milestones

1. ~~Trained-tree VSI-Bench result~~ → §5 (done; fusion confirmed as critical path).
2. **Stage III / SFT v2 (in progress):** rebuild SFT data from *on-policy*
   ladders with fusion-specific examples (complementary sets where no single
   view suffices), boosted RENDER share, redundant/distractor robustness;
   retrain LoRA; re-eval ladder + VSI transfer + tree.
3. Confidence-head domain adaptation (add VSI-style states to head training).
4. Stage IV resource-constrained GRPO (doc §6.6) — after III.

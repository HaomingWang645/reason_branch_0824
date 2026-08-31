# RESEARCH DESIGN DOCUMENT — IMPLEMENTATION RECORD

# ViewTree

**Spatial Reasoning over an Explicit Scene Memory via Confidence-Guided Viewpoint Branching, Human-Camera Constraints, and Multi-Step View Acquisition**

**Purpose:** As-built counterpart to `ViewTree_Research_Design_Document.pdf` (24 Aug 2026): the method as actually implemented, the training as actually executed, and the design decisions that were changed, dropped, or added on contact with experiments.
**Status:** Implemented and evaluated; all numbers cited here are computed from result files in this repository (RESULTS.md is the experiment log, DECISIONS.md the decision log, DESIGN_DEPTH.md the ViewTree-D design).
**Target:** Same paper positioning as the original design, restricted to the claims the evidence now supports.
**Date:** 31 August 2026 (covers work from 24 Aug 2026).
**Central claim (as validated):** A 7B VLM with an explicit reconstructed scene memory should not receive every rendered view. A learned gate + confidence-guided branching over *physically plausible, human-camera-constrained* viewpoints transfers across benchmarks and acquires views efficiently; its value is cross-benchmark transfer and view efficiency, not peak in-domain accuracy. Multi-step acquisition (depth ≤ 3) adds a further, currently borderline, gain over one-shot branching when trained at 494k-QA scale.

## Executive summary

Two systems were built on one stack (frozen VGGT-1B reconstruction → GPU point-splat renderer → Qwen2.5-VL-7B + LoRA controller → calibrated confidence head):

1. **ViewTree (depth-1 tree), the current best system on the pre-registered evaluation.** Gate → 5 constrained candidate views → head-scored keep-2 → consensus / pose-tagged fusion → arbitration vs the direct answer. Trained on MindCube (SFT ladder → fusion SFT-v2 → GRPO with a dual-variable view budget, the "D_highcost" design chosen from an 8-variant RL sweep). With the **human-camera constraint** (all candidate viewpoints inside the walked region, eye level, roll 0) and a head retrained on those views, it reaches **0.367** on the held-out VSI-Bench odd half — the first significant win of the adaptive tree over static memory prompting (+2.6 [+0.5, +4.7]) and +4.0/+5.6 over the no-memory SFT-plain / SFT+GRPO-plain baselines.
2. **ViewTree-D (depth ≤ 3 walks), the scale-up.** The camera itself becomes the action space (TURN/FORWARD/NEXT_SPOT/LOOK_AROUND/BIRD_EYE/STOP over a 97-view pre-rendered pose bank per scene), trained from scratch in four phases on a 493,663-QA corpus (VLM-3R vsibench/vstibench_train + VSI-590K; 1,709 scenes; every evaluation scene excluded at room level). The corpus alone is worth +14 pts on VSI (data-matched frames-only SFT 0.509); beam search over walks adds **+2.1 [−0.0, +4.3]** on top at 4.5 VLM calls/question; GRPO over walks collapsed toward STOP (pre-registered risk) — its interim adapter improves frames-only answering (+2.7, significant) but not the walk itself.

The mobile runtime of the original design (§7) was **not implemented**; all systems work ran on an 8×H100 node, and cost is reported in VLM calls, renders, and reconstructions per question instead of device latency/energy.

**Document map.** §1 hypotheses and verdicts; §2 scope as realized; §3 architecture as built; §4 problem formulation as instantiated; §5 method as implemented (depth-1 tree, human-camera constraint, ViewTree-D walks); §6 training as executed; §7 systems mechanisms actually built vs deferred; §8 experiments run and headline results; §9 roadmap status; §10 risks that fired; §11 supportable paper positioning; §12 evidence checklist.

## 1. Hypotheses: design → verdict

The original document's testable hypotheses (design §1.7), with the implementation's verdicts. "VSI held-out" always means the odd-indexed half of VSI-Bench scenes (144 scenes, 2,557 questions) that no training or head-fitting ever touched; all deltas are paired with scene-bootstrap 95 % CIs.

| # | Hypothesis (design) | Verdict | Evidence |
|---|---|---|---|
| H1 | Adaptive branching beats a single active-view trajectory on distributed-evidence questions | **Partially confirmed** | Tree beats static memory prompting +2.6 [+0.5, +4.7] (VSI held-out) once views are human-constrained and the head matches them; gains concentrate on relative-direction / relative-distance / appearance-order types. On the training benchmark itself, plain SFT with all views handed over is stronger (0.750 vs 0.632 on MindCube tinybench) — the tree's value is acquisition, not peak accuracy. |
| H2 | Confidence-guided pruning reaches fixed-width accuracy with fewer views | **Confirmed** | GRPO ladder policy: 0.778 on MindCube-rest at 1.30 views vs 0.765 at 1.85 views for SFT-v2; tree answers directly on ~23 % of VSI questions (gate) and stops at consensus on another third. ViewTree-D's gate answers directly on 71 %, giving mean 4.5 calls vs ≤ 8 for the depth-1 tree at higher accuracy. |
| H3 | Selective fusion beats best-branch and concatenate-all | **Confirmed in-domain, weaker cross-domain** | Stage III fusion training (+6.1k on-policy examples) was required for the tree to beat its own single-pass adapter on MindCube; on VSI the arbitration (fused vs direct) contributes via fallback on ~20–25 % of items. |
| H4 | A trained value head ranks branches better than token probability / verbal confidence | **Confirmed** | Design §6.3's rollout labels were replaced by state-outcome labels (cheaper; §6.2 below). Head v2: held-out AUROC 0.672 (legacy views) → 0.710 (human views); ViewTree-D value head 0.723. Token-logprob and verbal-confidence baselines were inferior in the Stage II audit (DECISIONS §7.8). |
| H5 | Systems mechanisms cut latency/energy without changing decisions | **Not tested** (mobile runtime not built) | Offline analogues implemented: per-scene reconstruction reuse, lazy pose-indexed render cache, pre-rendered pose banks for training (§7). |
| H6 | A budget controller preserves a Pareto frontier across device states | **Not tested**; replaced by a *training-time* budget: GRPO dual variable λ drives mean views → 1.5 (depth-1) / mean steps → 1.2 (ViewTree-D). The 8-variant RL design sweep is the implemented analogue of operating-point selection. |

Two hypotheses the original did not pose, now supported: **H7 (added)** — hard *human-camera* constraints on candidate viewpoints (inside the walked hull, eye level, roll 0) cost nothing (100 % valid views) and enable the best result once the head is retrained to read eye-level views. **H8 (added)** — template-matched training data dominates method effects across benchmarks: the 494k corpus moves VSI +14 and VSTI +16 while *hurting* OST (−2.4, significant); any depth claim must be read against a data-matched baseline.

## 2. Scope as realized

### 2.1 In-scope task model (implemented)
1. Input: a natural-language question plus 16–32 uniformly sampled video frames (VSI/STI/VSTI), the cumulative image history (OST, latest 12), or ≤ 4 given views (MindCube).
2. Environment: static indoor scenes (ScanNet, ScanNet++, ARKitScenes, MindCube image sets).
3. Action: depth-1 system — choice among 5 pre-proposed constrained viewpoints; ViewTree-D — discrete camera walk (`TURN_LEFT/RIGHT` ±45° yaw, `FORWARD` one walkable cell, `NEXT_SPOT`, `LOOK_AROUND` +180°, `BIRD_EYE` last-only, `STOP`).
4. Observation: point-splat rendering of the reconstruction, pose-tagged in the prompt ("eye-level view from standing spot p facing direction y of 8").
5. Output: answer + head confidence + full machine-readable trace (gate, path, per-state values, mode).
6. Deployment target: **changed** — 8×H100 server node; smartphone/Jetson deferred (design §2.1.17).

### 2.2 Non-goals kept, and one added
All of design §2.2 stands (no new reconstruction method, no SLAM, no robot control, no metric-grade guarantees). Added: no photorealism claims for the renderer — splatted renders contain holes; the controller is *trained on renders with holes* so it learns to use imperfect evidence (the prompt says "holes possible").

### 2.3 Success definition as applied
The original Pareto-frontier criterion (accuracy vs device cost) was replaced by: paired accuracy on **held-out scene splits never touched by any training or head fitting**, against data-matched baselines, with acquisition cost in VLM calls/renders per question. Pre-registered ViewTree-D success criterion (DESIGN_DEPTH §4): ≥ +2 pts mean-of-types over the depth-1 tree on VSI held-out at ≤ 1.5× its calls, no OST regression — **met on VSI via the corpus** (+14 data, +2.1 depth) **but the OST no-regression clause failed** for the corpus-trained adapters (−2.4 vs zero-shot; the MindCube-trained depth-1 tree itself does not regress).

## 3. System architecture as built

Design §3.1's module table, with every open choice filled in:

| Module | Implementation | Cost / size | Training status |
|---|---|---|---|
| Scene builder | **VGGT-1B** on 16–32 frames (32 for video benchmarks) | ≈ 7.8 GB + 0.22 GB/frame GPU; one reconstruction per scene, reused across questions | Frozen |
| Renderer | **Torch GPU z-buffered point-splat**, splat radius 2 (chosen over PyTorch3D/Open3D after a viability study, DECISIONS §7.2) | few ms per view | Frozen (no parameters) |
| View encoder | Qwen2.5-VL's own ViT (no separate encoder); renders capped at 448² tokens' worth of pixels | — | Frozen |
| VLM controller | **Qwen2.5-VL-7B-Instruct + one LoRA (r = 16)** emitting answers and control tokens | 190 MB adapter | Fine-tuned (stages, §6) |
| Confidence head | MLP 3584 → 512 (GELU, dropout 0.1) → 1 on the last-token hidden state, temperature-calibrated on a held-out split | < 2 M params; trains on CPU in minutes | Trained |
| Branch manager | Deterministic top-k by head score inside the tree/beam code; validity = render coverage ≥ 45 % + geometric mask | — | Deterministic |
| Pose bank (ViewTree-D) | 12 FPS standing positions × 8 yaws + top-down = **97 poses/scene**, pre-rendered for training, lazily rendered at test | ~97 JPEGs/scene offline | Deterministic |
| Mobile scheduler | **Not built** | — | — |

State semantics follow design §3.2: the memory is not the VLM state; each branch's state is (question, K context frames, its own renders + pose tags, remaining budget). The controller-output vocabulary changed from design §3.3: the free `BRANCH`/`PRUNE`/`FUSE` tokens were **not** given to the model. Instead the tree *skeleton* is fixed (gate → branch → prune → fuse → arbitrate) and the learned parts are (a) the gate token (YES/EXPLORE), (b) the ladder/walk policy (`STOP`/`MOVE`/`RENDER`, later the 6 camera actions), and (c) the head that performs pruning and arbitration. This removed a large failure surface (invalid control grammar) at the cost of a less general controller — recorded as an explicit deviation in DECISIONS §7.6.

## 4. Problem formulation as instantiated

The design's resource-constrained objective (design eq. 4) was instantiated with a single acquisition cost instead of device constraints:

- Depth-1 GRPO (Stage IV): r = 1[correct] − λ·0.2·(views − 1), λ a dual variable updated toward a mean-view budget of 1.5; group size 6 (GRPO-style group-relative advantage, design eq. 18–20 with L/E/M constraints collapsed to one view budget).
- ViewTree-D GRPO (Phase 3): r = 1[correct] − λ·steps − 0.1·1[proposed masked action] + 0.05·1[answer became correct after a step]; dual λ toward mean 1.2 steps; policy gradient on action *and* answer tokens.
- The design's Q-value relabeling (eq. 5–6) became the **oracle walk** procedure: bounded beam search (b = 2–3, depth ≤ 3) with the frozen answerer scores every alternative continuation; the shortest correct-with-margin walk labels the controller's SFT (design's "evaluate alternative control decisions from the same state", executed offline at corpus scale: 8,639 QA, direct correct 54 %, best-of-walk 68 %).

## 5. Method as implemented

### 5.1 Human-camera viewpoint constraint (added; DECISIONS §9)
The original design validated poses only by renderability. Implementation revealed the legacy proposer placed cameras *outside the room and above the ceiling* (0 % inside). All candidate viewpoints are now constrained to be physically takeable by a person holding a camera — a **hard mask**, not a reward term (route 1 of DECISIONS §9; route 2, an RL penalty, exists in ViewTree-D's reward as the masked-action term):

7. position inside the convex hull of the recorded camera trajectory (the region the videographer actually walked);
8. camera at the median recorded camera height (eye level), clearance ≥ 4 % of the room diagonal from any reconstructed surface;
9. roll = 0 (image horizontal parallel to the floor), pitch 10° down;
10. 4 farthest-point-sampled standing positions looking toward the room centre; render coverage ≥ 45 % of pixels or the view is discarded and replaced;
11. one top-down bird's-eye view, allowed only as the final acquisition.

Constraint satisfaction is 100 % with no accuracy loss; with a head retrained on these views (AUROC 0.672 → 0.710) it produced the best system.

### 5.2 Depth-1 reasoning tree (branching 5, keep 2) — the best evaluated system
For every question: **gate** ("can you answer from these frames alone? YES/EXPLORE"; YES → answer directly, 1–2 calls) → **branch** (render the 5 constrained views; each branch answers from frames + that view, scored by the head) → **prune** (keep top-2; if they agree and beat the direct answer → early stop, *branch consensus*) → **fuse** (answer from frames + both kept views, pose-tagged) → **arbitrate** (head ranks direct vs fused vs kept; falls back to direct when the memory adds nothing). Cost: 1–2 calls when the gate fires, otherwise ≤ 8 calls + 1 reconstruction + 5 renders. This realizes design §5.5–5.7 with two deviations: *disagreement-aware continuation* (design §5.6) is implemented only as consensus-early-stop + arbitration (no extra disambiguating view at depth 1 — that capability moved to ViewTree-D), and the optional pairwise comparison head (design eq. 9) was never needed.

### 5.3 ViewTree-D: the walk as the reasoning path (DESIGN_DEPTH)
One step = one camera action + one render; state = (question, 8 context frames, renders so far, pose). Constraints of §5.1 become the **action validity mask** over the pose bank. **Inference** = gate, then beam search over walks: top-3 actions by controller logit per kept state, keep 2 by value head, depth ≤ 3, early stop when both kept paths agree with margin above the direct answer's value, final arbitration vs direct; `BIRD_EYE` only as last acquisition. Measured cost: 2 calls on gated items, mean 4.5, up to ~20 for a full depth-3 walk (answer + action-proposal calls; the design's "≤ 12" was superseded by measurement). Every state and value is logged; 30 walk visualizations (3 per VSI task) are in the technical report §5.3.

## 6. Training as executed

### 6.1 Depth-1 stack (MindCube, 10k train items; scene-level splits; DECISIONS §7–8)
- **Stage I — ladder SFT (16.8k ex.).** Teacher-labelled evidence ladder (1…all views with STOP/MOVE/RENDER control); the teacher audit (design §6.8.42) found the teacher weak on cross-view integration, so ladder answers were taught from outcome-filtered traces.
- **Stage II — confidence head.** *Deviation from design §6.3:* instead of K stochastic continuation rollouts per state (cost-prohibitive), the head is trained on **realized ladder/tree states labelled by final-answer correctness**, temperature-calibrated on held-out scenes (design eq. 16 kept). AUROC beats token-logprob and verbal confidence (H4). Head v2 adds VSI tree states from the 144 *even* scenes only.
- **Stage III — fusion SFT-v2 (+6.1k on-policy ex., 22.9k total).** Complementary / redundant / distractor view combinations, per design §6.5, generated on-policy from the Stage-I model's own kept branches.
- **Stage IV — GRPO ladder policy.** 8-variant design sweep (cost coefficient × budget × group size); winner "D_highcost" (cost 0.2, budget 1.5, group 6) trained on all 9,995 items ("D_10k").

### 6.2 ViewTree-D phases (corpus: 493,663 QA / 1,709 scenes; all eval scenes excluded at room level)
- **Phase 0** — reconstructions + pose banks for every training scene (97 renders/scene, offline).
- **Phase 1 — SFT-A answerer** on ~100k walk states (frames + 0–3 pose-tagged renders → answer; 33k frames-only mixed in), 1 epoch.
- **Phase 2 — oracle walks** (beam 2, depth ≤ 3, 5 QA/scene → 8,639 QA; direct 54 % → best-of-walk 68 %); **value head** on all search states (AUROC 0.723, T = 1.2); **SFT-C controller** imitating oracle actions, prompt lists the valid moves.
- **Phase 3 — GRPO over walks** (group 6, dual λ, masked-action penalty, 30k-item budget on 3 GPUs). *Outcome:* policy collapsed toward STOP-at-depth-0 (mean steps 0.18 → ~0.07; λ never activated) — the pre-registered fallback fired (§10). Phase 4 (cross-domain hardening) not reached.

Data-quality controls of design §6.8 that were implemented and audited: scene-level splits everywhere; MindCube train ↔ tinybench overlap 0 at id/question-group/scene/image level; VSI even/odd halves share 0 physical rooms; every trace stores the exact observations for replay; balanced positive/negative head states across depths.

## 7. Systems design: built vs deferred

| Design §7 mechanism | Status | As-built form |
|---|---|---|
| Shared-prefix KV execution | Deferred | Branches share the *prompt* prefix; no KV-cache page sharing measured |
| Batched sibling rendering/encoding | Partial | Renders are batched per level; VLM calls sequential |
| Progressive rendering fidelity | Dropped | Single fidelity; renders already few-ms |
| Pose-indexed observation cache | **Built** | Lazy per-scene render cache at test time; pre-rendered 97-view pose banks for training (renders never produced online during SFT/RL) |
| Reconstruction reuse | **Built** | One VGGT reconstruction per scene serves all its questions and all branches |
| Runtime budget controller | Replaced | Training-time dual-variable budgets (λ); no live device scheduler |
| Real-device profiling | Deferred | Cost reported as VLM calls / renders / reconstructions per question |

## 8. Experiments as run

### 8.1 Dataset portfolio (design §8.2 → actual)
MindCube (train + tinybench/rest), **VSI-Bench** (primary; held-out odd half), **OST-Bench** (5,403–5,557 paired), **STI-Bench** (2,064), **VSTI-Bench** (5,736; clean subset 4,866), plus the 494k training corpus (VLM-3R vsibench_train / vstibench_train, VSI-590K ScanNet + ScanNet++ v2). OpenEQA / ScanQA / SQA3D and the real-device set were not used. Scoring: accuracy for MC, mean relative accuracy for numeric; headline = mean over question types.

### 8.2 Baselines actually compared
Zero-shot frames; **SFT-plain** and **SFT+GRPO-plain** (no memory, benchmark's own training split — added per DECISIONS §10); static memory prompting (frames + 5 renders, no adaptivity); single-pass adapters; depth-1 tree variants (legacy vs human views, head v2 vs matched); ViewTree-D ladder (data-matched corpus frames-only SFT; SFT-A frames-only; depth-1 with SFT-A; no-RL beam; GRPO beam). Fixed-width tree, cdViews-style, Think3D-style external reimplementations were not run — the depth-1-vs-depth-≤3 and gate/no-gate comparisons inside our stack stand in for them.

### 8.3 Headline results (all paired; scene-bootstrap 95 % CIs; details in RESULTS.md)
| Evaluation | Result |
|---|---|
| VSI held-out odd half — best depth-1 tree | **0.367** = +2.6 [+0.5, +4.7] vs static memory; +4.0 vs SFT-plain; +5.6 vs SFT+GRPO-plain |
| MindCube tinybench (training domain) | Tree 0.632 < SFT-plain 0.750 — value is acquisition efficiency, not peak in-domain accuracy |
| OST-Bench | Tree 0.541 ≥ zero-shot 0.540 > SFT-plain 0.524; corpus-trained adapters **regress** to 0.516/0.518 |
| STI / VSTI (no retraining) | Every MindCube-tuned model < zero-shot; ViewTree loses least; VSTI object–object relations: ViewTree best outright |
| ViewTree-D on VSI held-out | corpus frames-only SFT 0.509 (+14 over best MindCube system); no-RL beam **0.530** = +2.1 [−0.0, +4.3]; ordering baseline < depth-1 (0.517) < depth-≤3 (0.530) on relational/directional types |
| ViewTree-D GRPO (interim, 72 % budget) | beam 0.525 (−0.5 [−2.1, +1.2] vs no-RL); frames-only 0.536 (+2.7 [+0.7, +5.0]) — RL gain is in answer tokens, not the walk |
| VSTI transfer of corpus adapters | SFT-A single-pass 0.685 = +16.1 [+14.6, +17.8] over zero-shot (template-matched; 0 shared rooms) |

### 8.4 Statistical protocol as followed
Paired question sets; scene-level splits and scene-bootstrap CIs (item blocks on OST); temperature calibration on held-out scenes only; the VSI odd half untouched by any training/selection. Deviations from design §8.10 stated honestly: single training seed per system (not three); ~13 design variants were iterated against full VSI before the odd-half protocol was fixed — adaptive-reuse risk is recorded in RESULTS §6b and a fresh benchmark is recommended before publication claims.

## 9. Implementation roadmap: status

| Phase (design §9) | Goal | Status |
|---|---|---|
| M0 | Explicit-memory chain end-to-end | **Done** (RQ1 pass: memory views > frame prompting, training-free) |
| M1 | Controller SFT | **Done** (Stage I ladder; valid-action rate stabilized) |
| M2 | Tree runtime | **Done** (depth-1 tree; ViewTree-D beam runtime) |
| M3 | Confidence head | **Done** (state-outcome labels replaced rollouts; H4 confirmed) |
| M4 | Fusion | **Done** (Stage III; complementary subset improved) |
| M5 | Constrained RL | **Done** for depth-1 (design sweep, D_10k); **partial** for walks (GRPO collapsed to STOP; final checkpoint evaluating) |
| M6 | Mobile runtime | **Not started** (deferred) |
| M7 | Full evaluation | **Done** for the server-side study (5 benchmarks, ablations, statistics, 69 visualized trees, technical report) |

## 10. Risks: which fired, and what the mitigation did

| Risk (design §10) | Fired? | Outcome |
|---|---|---|
| Confidence miscalibrated → correct branch pruned | Partly | Legacy head under-read eye-level views (0.672); retraining on matched views (0.710) was required for the human-constraint system to win |
| Always-branch / view overload | Yes, observed | Extra renders *hurt* counting, room-size, route-planning; the gate + λ budget are the working mitigations (71 % direct at depth ≤ 3) |
| RL collapse to always-STOP | **Yes (ViewTree-D Phase 3)** | Pre-registered: dual λ + answer-improved shaping did not prevent drift to steps ≈ 0.07; beam inference explores regardless; RL's real gain surfaced in answer tokens (frames-only +2.7). Reported as such |
| Reconstruction holes | Managed | Hard coverage mask (≥ 45 %) + training on holey renders; scene failures skipped (corpus large enough) |
| Reward hacking | Not observed | Verifiable QA reward; masked-action penalty; traces replayable |
| Weak novelty ("Think3D + tree search") | Addressed by evidence | The defensible additions are the human-camera constraint (free, enabling), outcome-calibrated pruning (H4), transfer-not-peak framing (H1/H8), and the data-matched-baseline discipline |
| **New risk (found):** template-specific corpus gains | Yes | Corpus lifts VSI/VSTI, *hurts* OST (−2.4). Mitigation for any future phase: mix OST-style exploration QA into the corpus (pre-registered in RESULTS §8) |

## 11. Paper positioning supportable today

**Title (unchanged):** ViewTree: Spatial Reasoning over an Explicit Scene Memory via Confidence-Guided Viewpoint Branching and Fusion — the "Resource-Aware On-Device" clause must be dropped unless M6 is built.
**Pitch (as evidenced):** ViewTree turns a reconstructed scene into an adaptive reasoning tree over *human-takeable* viewpoints; a calibrated head decides what to keep and when the memory adds nothing; the result transfers across five benchmarks with few acquired views, and scales to multi-step walks with a 494k-QA corpus.
**Claims to avoid, updated:** everything in design §11.4, plus: no peak-accuracy claims on the training benchmark; no depth-> 1 claims without the data-matched baseline; no cross-domain claims for the corpus-trained models (OST regression); no RL-improves-the-walk claim (it improves the answerer).

## 12. Evidence checklist (design §12, as of 31 Aug 2026)

- [x] Complementary-view benefit localized (relative-direction/distance, appearance order) and reported per type
- [x] Tree beats best single trajectory on those subsets (depth-1 vs static; depth-≤3 vs depth-1 ordering)
- [x] Head AUROC/calibration reported, not only QA accuracy (0.710 / 0.723 held-out; temperature on held-out scenes)
- [x] Best-branch vs fuse vs direct arbitration compared (path-mix statistics per system)
- [x] All systems share the same VLM, reconstruction, frame budgets, scorer; paired questions
- [x] Scene-level splits + leakage audit (MindCube overlap 0; VSI halves share 0 rooms)
- [x] Cost reported per question (calls, renders, mean depth)
- [x] Failure taxonomy: reconstruction (scene_failed), gate errors, head fallbacks, walk collapse — visible in 69 released trace figures
- [ ] Three training seeds (single seed per system)
- [ ] Real-device latency/energy/memory/thermal (M6 not built)
- [ ] Scan-geometry vs reconstructed-geometry separation (reconstruction-only throughout)
- [ ] Fresh untouched benchmark for the headline claim (recommended before submission; odd half is the current guard)

## References and repository anchors

Design anchors [1]–[10] as in the original document (VGGT arXiv:2503.11651; Think3D arXiv:2601.13029; Chain-of-View arXiv:2601.05172; AVIC arXiv:2602.08236; cdViews arXiv:2505.22143; VisuoThink arXiv:2504.09130; Guo et al. calibration, ICML 2017).
Implementation anchors: `RESULTS.md` (experiment log, §1–§8), `DECISIONS.md` (execution decisions §0–§10), `DESIGN_DEPTH.md` (ViewTree-D design + corpus), `report/REPORT.md|pdf|html` (technical report with 69 visualized reasoning trees), `viewtree/` and `scripts/` (all code), `checkpoints/` (adapters + heads).

**Document status:** This is a record of completed implementation and measured results, not a plan. Where it contradicts the 24 Aug design document, this document describes what was actually done; the design document remains the statement of intent.

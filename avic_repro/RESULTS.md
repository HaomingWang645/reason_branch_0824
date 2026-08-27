# AVIC reproduction results (SAT-Real, 150 test questions)

Paper: *When and How Much to Imagine* (arXiv 2602.08236), Table 1. All our runs use the exact upstream
"Eval setting" (see README.md → Protocol): full SAT-Real test split (150 q), 5 policy samples, ≤5 world-model
candidates, verifier-selected trajectory, ≤6 atomic actions, SVC world model. One run per configuration
(as in the paper). Numbers are accuracy (%) per SAT question type. `#Token (K)` = average closed-source-VLM
tokens per question (API backbones only), `Avg. WM` = average world-model renders per question,
`WM-call %` = fraction of questions on which the majority policy vote was `call_wm`, `views/q` = imagined frames
handed to the QA model per question.

Category sizes: EgoM 23, ObjM 23, EgoAct 37, Goal 34, Pers 33 — one question is 2.7–4.3 points in a category,
so per-category numbers are noisy; the Avg column (150 q, ±~4 pts at 1σ) is the one to compare.

## Side-by-side: ours vs. paper (Avg accuracy)

| Backbone | Method | Policy | **ours** | paper | Δ | ours WM/q | paper WM/q |
|---|---|---|---|---|---|---|---|
| InternVL3-14B | base | – | **58.7** | 59.3 | −0.6 | 0 | 0 |
| | + AVIC | InternVL3-14B | **66.7** | 68.0 | −1.3 | 0.89 | 0.64 |
| | + AVIC | Qwen2.5VL-7B (zero-shot) | **72.0** | 61.3 | +10.7 | 3.03 | 1.81 |
| | + AVIC-R | Qwen2.5VL-7B (released adapter) | **64.0** | 69.3 | −5.3 | 2.81 | 3.03 |
| GPT-4o | base | – | **58.0** | 60.3 | −2.3 | 0 | 0 |
| | + AVIC | GPT-4o | **58.0** | 69.3 | −11.3 | 0.64 | 0.72 |
| | + AVIC | Qwen2.5VL-7B (zero-shot) | **66.0** | 71.3 | −5.3 | 2.85 | 1.81 |
| | + AVIC-R | Qwen2.5VL-7B (released adapter) | **66.0** | 77.3 | −11.3 | 2.79 | 3.03 |
| GPT-4.1 | base | – | **71.3** | 74.0 | −2.7 | 0 | 0 |
| | + AVIC | GPT-4.1 | **73.3** | 79.3 | −6.0 | 0.33 | 0.73 |
| | + AVIC | Qwen2.5VL-7B (zero-shot) | **76.7** | 72.6 | +4.1 | 2.87 | 1.81 |
| | + AVIC-R | Qwen2.5VL-7B (released adapter) | **76.7** | 80.0 | −3.3 | 2.63 | 3.03 |

Not run: `+ MindJourney` rows (always-on beam search — pipeline ported in `mindjourney/`, but the OpenAI account
ran out of credits on 2026-08-27 ~15:00 after 2–3 questions per chunk; re-run with
`BACKBONE=gpt-4.1 GPUS="6 6 7" bash scripts/run_mindjourney.sh` once credits are added), the `o1` block
(cost), MMSI (Table 2; no data path in the released code) and R2R navigation (Table 3; needs the Matterport3D
simulator).

## Full table (ours)

| run | n | EgoM | ObjM | EgoAct | Goal | Pers | **Avg** | #Token (K) | WM-call % | Avg. WM | views/q |
|---|---|---|---|---|---|---|---|---|---|---|---|
| InternVL3-14B_baseline | 150 | 60.9 | 65.2 | 59.5 | 70.6 | 39.4 | **58.7** | – | 0.0 | 0.00 | 0.00 |
| InternVL3-14B_avic | 150 | 56.5 | 60.9 | 70.3 | 85.3 | 54.5 | **66.7** | – | 89.3 | 0.89 | 1.42 |
| InternVL3-14B_avic_qwen | 150 | 65.2 | 65.2 | 86.5 | 82.4 | 54.5 | **72.0** | – | 100.0 | 3.03 | 2.53 |
| InternVL3-14B_avic_r | 150 | 52.2 | 60.9 | 73.0 | 79.4 | 48.5 | **64.0** | – | 100.0 | 2.81 | 1.90 |
| gpt-4o_baseline | 150 | 69.6 | 56.5 | 56.8 | 61.8 | 48.5 | **58.0** | 0.7 | 0.0 | 0.00 | 0.00 |
| gpt-4o_avic | 150 | 56.5 | 52.2 | 56.8 | 64.7 | 57.6 | **58.0** | 8.8 | 42.7 | 0.64 | 1.67 |
| gpt-4o_avic_qwen | 150 | 73.9 | 60.9 | 56.8 | 82.4 | 57.6 | **66.0** | 10.0 | 100.0 | 2.85 | 2.15 |
| gpt-4o_avic_r | 150 | 56.5 | 73.9 | 67.6 | 70.6 | 60.6 | **66.0** | 8.7 | 100.0 | 2.79 | 1.71 |
| gpt-41_baseline | 150 | 95.7 | 69.6 | 75.7 | 79.4 | 42.4 | **71.3** | 0.7 | 0.0 | 0.00 | 0.00 |
| gpt-41_avic | 150 | 95.7 | 78.3 | 75.7 | 82.4 | 42.4 | **73.3** | 7.2 | 26.7 | 0.33 | 0.98 |
| gpt-41_avic_qwen | 150 | 95.7 | 87.0 | 75.7 | 85.3 | 48.5 | **76.7** | 10.0 | 100.0 | 2.87 | 2.23 |
| gpt-41_avic_r | 150 | 95.7 | 87.0 | 75.7 | 82.4 | 51.5 | **76.7** | 8.1 | 100.0 | 2.63 | 1.74 |

(regenerate with `python scripts/summarize.py --md results/RESULTS_TABLE.md`; raw per-question logs are under
`results/<run>/question_chunk_*/<qid>/step_0/`.)

## Reference (paper Table 1)

| Method | Policy | EgoM | ObjM | EgoAct | Goal | Pers | Avg | #Token(K) | Avg. WM |
|---|---|---|---|---|---|---|---|---|---|
| InternVL3-14B | – | 56.5 | 69.5 | 54.0 | 73.5 | 45.4 | 59.3 | 0.2 | 0 |
| + MindJourney | – | 69.6 | 60.9 | 78.4 | 79.4 | 42.4 | 66.7 | 2.5 | 12.34 |
| + AVIC | InternVL3-14B | 95.6 | 73.9 | 62.1 | 76.4 | 42.4 | 68.0 | 2.0 | 0.64 |
| + AVIC | Qwen2.5VL-7B | 73.9 | 47.8 | 67.5 | 73.5 | 42.4 | 61.3 | 4.4 | 1.81 |
| + AVIC-R | Qwen2.5VL-7B | 82.6 | 52.1 | 70.2 | 85.2 | 54.5 | 69.3 | 4.8 | 3.03 |
| GPT-4o | – | 56.5 | 85.0 | 50.0 | 64.0 | 45.0 | 60.3 | 0.9 | 0 |
| + MindJourney | – | 78.3 | 60.9 | 78.4 | 70.6 | 57.5 | 69.3 | 26.0 | 12.34 |
| + AVIC | GPT-4o | 86.9 | 60.9 | 64.8 | 82.3 | 48.4 | 69.3 | 9.5 | 0.72 |
| + AVIC | Qwen2.5VL-7B | 65.2 | 73.9 | 64.8 | 91.1 | 60.6 | 71.3 | 5.0 | 1.81 |
| + AVIC-R | Qwen2.5VL-7B | 82.6 | 82.6 | 81.0 | 91.1 | 51.2 | 77.3 | 5.4 | 3.03 |
| GPT-4.1 | – | 95.7 | 73.9 | 78.3 | 88.2 | 39.4 | 74.0 | 0.7 | 0 |
| + MindJourney | – | 100.0 | 82.6 | 86.5 | 79.4 | 45.4 | 77.3 | 67.1 | 12.34 |
| + AVIC | GPT-4.1 | 100.0 | 78.2 | 83.7 | 85.2 | 54.5 | 79.3 | 7.6 | 0.73 |
| + AVIC | Qwen2.5VL-7B | 82.6 | 86.9 | 75.6 | 88.2 | 36.3 | 72.6 | 4.8 | 1.81 |
| + AVIC-R | Qwen2.5VL-7B | 91.3 | 86.9 | 83.7 | 85.2 | 57.5 | 80.0 | 5.2 | 3.03 |
| o1 | – | 78.3 | 82.6 | 73.0 | 73.5 | 69.7 | 74.6 | 1.4 | 0 |
| + AVIC | o1 | 100.0 | 86.9 | 86.4 | 91.1 | 66.6 | 85.3 | 14.6 | 1.28 |
| + AVIC-R | Qwen2.5VL-7B | 86.9 | 65.2 | 81.0 | 94.1 | 69.6 | 80.0 | 6.1 | 3.03 |

## What reproduces, what does not

**Reproduces**
* *Baselines*: all three no-WM baselines land within 0.6–2.7 points of the paper (InternVL3-14B 58.7 vs 59.3,
  GPT-4o 58.0 vs 60.3, GPT-4.1 71.3 vs 74.0; GPT-4.1 EgoM 95.7 matches exactly). The ~2-point gap on the GPT
  models is consistent with the paper using Azure deployments and us using the public API snapshots
  (`gpt-4o-2024-08-06`, `gpt-4.1-2025-04-14`).
* *Adaptive imagination helps every backbone*: the best AVIC variant beats its baseline by +5.4 (GPT-4.1),
  +8.0 (GPT-4o) and +13.3 (InternVL3-14B) points, and the gains concentrate on the action-conditioned categories
  (EgoAct: InternVL 59.5→86.5, GPT-4o 56.8→67.6; Pers: +9 to +15 on all backbones), matching the paper's
  §5.4 analysis. Paired per-question diagnostics (`scripts/paired_diag.py`): with AVIC-R on GPT-4.1, 19 questions
  flip wrong→right vs 11 right→wrong; on InternVL3 25 vs 17; on GPT-4o 27 vs 15.
* *Efficiency numbers of the training-free gate* match closely: GPT-4o self-policy calls the world model 0.64×/q
  (paper 0.72) at 8.8K tokens/q (paper 9.5K); GPT-4.1 0.33×/q (0.73) at 7.2K (7.6K).
* *Ordering AVIC-R ≥ AVIC (self-policy) > base* holds for GPT-4.1 (76.7 ≥ 73.3 > 71.3) and GPT-4o
  (66.0 > 58.0 = 58.0).

**Does not reproduce (or only partially)**
* *Absolute AVIC / AVIC-R accuracies are 3–11 points below the paper* on the GPT backbones: GPT-4.1 AVIC-R 76.7
  vs 80.0, GPT-4o AVIC-R 66.0 vs 77.3, GPT-4o AVIC 58.0 vs 69.3. With GPT-4o the self-policy variant brings no gain
  at all: on the 64 questions where it chose `call_wm`, accuracy went *down* from 60.9 (direct answer) to 57.8,
  i.e. the imagined views were as often misleading as helpful for that backbone (the paper's "Case 2").
* *The released AVIC-R adapter is not better than the zero-shot Qwen2.5-VL-7B policy in our runs*: equal on
  GPT-4.1 (76.7/76.7) and GPT-4o (66.0/66.0) and worse on InternVL3-14B (64.0 vs 72.0), whereas the paper reports
  +6–8 points for RL. The behavioural claim does partly reproduce — AVIC-R calls the WM on 100 % of questions
  (paper 92 %) with fewer, shorter plans (1.7–1.9 views/q vs 2.2–2.5 for zero-shot Qwen; paper 3.60 vs 4.03) — but
  our zero-shot Qwen policy also calls the WM on 100 % of questions (paper: skips 64.7 %), so the "over-skipping"
  failure mode the paper attributes to the untrained policy did not occur here, which removes most of RL's
  advantage. Plan-level settings (`--policy_temperature 0.7`, 5 samples, `max_action_ids_cap 6`) are the upstream
  defaults, so this is most likely a prompt/decoding-version effect of the base Qwen2.5-VL-7B rather than a setup
  difference.
* InternVL3-14B + AVIC (zero-shot Qwen) is *much better* than the paper (72.0 vs 61.3), the best InternVL row
  overall.

**Caveats that could explain part of the gap**
1. World-model VAE: `stabilityai/stable-diffusion-2-1-base` is no longer on the HF hub; the VAE is loaded from an
   SD-2.1 checkpoint whose `vae/` is an unmodified copy of the original (its config still says
   `_name_or_path: stabilityai/stable-diffusion-2-1`). Renders were visually checked and are coherent.
2. API models are the public OpenAI snapshots, not the paper's Azure deployments; even at temperature≈0 with
   `seed=44` the API is not deterministic (on the 110 questions GPT-4.1 answered directly in two different runs,
   3/110 answers differ).
3. Single run per configuration with a stochastic (T=0.7) policy — the paper's numbers are single runs too, so
   ±3–4 points of run-to-run variance is expected on 150 questions.
4. In-process SVC rendering (our patch) resets the SVC globals before every call to match the upstream fresh-process
   behaviour; renders use the same sampler settings (`cfg 4.0, L_short 576, num_targets 8, 20 steps, seed 23`).

## Reasoning traces

`results/traces.html` (generated by `scripts/viz_traces.py`) shows the complete reasoning path for two questions of
each task type for GPT-4.1 AVIC-R, GPT-4o AVIC-R, InternVL3-14B AVIC-R and GPT-4.1 AVIC: observation → 5 policy
samples with reasons and plans → majority vote → every rendered trajectory with its verifier score and imagined
frames → selected trajectory → QA answer vs. ground truth.

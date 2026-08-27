# Think3D reproduction — results

Paper: *Think3D: Thinking with Space for Spatial Reasoning* (arXiv 2601.13029v3), code
https://github.com/zhangzaibin/spagent. Reproduced on GPUs 5/6 only (see README.md).

## Protocol (paper Sec. 4.1 + App. C, as implemented here)

| Item | Paper | This reproduction |
|---|---|---|
| Policy | Qwen3-VL-4B-Instruct (open-weight rows) | same (`Qwen/Qwen3-VL-4B-Instruct`, served by vLLM 0.11.2) |
| 3D tool | Pi3 / Pi3X point-cloud reconstruction + global / ego novel-view rendering | Pi3X server from the repo (`yyfz233/Pi3X` weights) |
| Rounds | ≤ 3 interaction rounds, ≤ 1 reconstruction per query | `--max-iterations 3` (baseline: 1, no tools) |
| Decoding | temperature 1.0, ≤ 1024 response tokens, mean of 3 runs | same (3 runs, seeds differ) |
| MindCube | 40 q / category (rotation, around, among) = 120 q | 40 q / category sampled (seed 42) from `MindCube_tinybench`, **excluding scenes that appear in the RL training set** |
| BLINK | all Multi-view questions | BLINK val `Multi-view_Reasoning`, 133 q |
| VSI-Bench | VSI-Bench-tiny, 4 MC tasks, 7 frames / video | 50 q / task sampled (seed 42) from the test split (rel-dir easy/medium/hard merged), 7 uniformly sampled frames |
| Think3D-RL | GRPO, 977 MindCube samples, 1 epoch, 8 rollouts, lr 1e-6 cosine 5 % warm-up, max completion 1024, 8×H200 | same recipe on 976 samples (24 have images missing from the public MindCube release), 1×H100 with ZeRO-2 optimizer offload, 16 prompts (128 completions) / optimizer step |
| Scoring | MC accuracy | option letter inside `<answer>` tags vs. GT (`scripts/score.py`) |

Rows for GPT-4.1 / Gemini-2.5-Pro are not reproduced (no API keys on this machine).

## Paper numbers (open-weight rows)

Table 2 (BLINK multi-view + MindCube):

| Model | BLINK (MV) | MC Rotation | MC Among | MC Around | Avg |
|---|---|---|---|---|---|
| Qwen3-VL-4B | 47.87 | 34.17 | 20.00 | 41.67 | 35.92 |
| Think3D (Qwen3-VL-4B) | 48.62 | 35.83 | 28.33 | 33.33 | 36.53 |
| Qwen3-VL-4B-T3RL (no tool) | 46.11 | 30.83 | 25.83 | 35.83 | 34.65 |
| Think3D (Qwen3-VL-4B-T3RL) | 53.39 | 42.50 | 37.47 | 42.50 | 43.97 |

Table 1 (VSI-Bench-tiny):

| Model | Route Plan | Rel. Dir. | Rel. Dist. | App. Order | Avg |
|---|---|---|---|---|---|
| Qwen3-VL-4B | 34.69 | 40.67 | 35.33 | 42.44 | 38.28 |
| Think3D (Qwen3-VL-4B) | 30.61 | 44.00 | 29.33 | 52.38 | 39.08 |
| Qwen3-VL-4B-T3RL (no tool) | 27.89 | 30.67 | 32.00 | 42.86 | 33.36 |
| Think3D (Qwen3-VL-4B-T3RL) | 36.73 | 39.00 | 44.67 | 61.22 | 45.41 |


## Implementation notes / deviations discovered while reproducing

* **Compute layout.** GPU 5 is in `Exclusive_Process` mode (one process only) → it hosts the single-process
  GRPO job; GPU 6 hosts the Pi3X server and the vLLM policy servers. `CUDA_DEVICE_ORDER=PCI_BUS_ID` is
  required on this box (CUDA's default order differs from nvidia-smi's).
* **RL data.** 24 of the 1000 Think3DQA samples reference images that are not in the public MindCube
  release (11 `among` files with different zero-padding, 13 `linear` scenes) → 976 samples (paper: 977).
  8 of the 598 unique scenes could not be reconstructed by Pi3X offline (oversized / mixed-size `around`
  images); their tool calls return an error during rollouts.
* **Test/train overlap.** 362 of the 1050 `MindCube_tinybench` questions share a scene with the RL training
  set; the 120-question eval subset is drawn only from the remaining 688.
* **VSI-Bench-tiny** ids are not public; the paper's per-task denominators (49–50 questions / task / run)
  match the 400-question "tiny" subset of the original VSI-Bench paper (50 / task), so 50 questions per MC
  task were sampled with a fixed seed.
* **Rendered-view resolution.** The Pi3X server returns 3900×3900 (global) / up to 6000×5000 px renders.
  vLLM's default Qwen3-VL processor keeps up to 16.7 M px, i.e. one render costs ~15 k visual tokens at
  eval time, whereas RL training used `MAX_PIXELS=262144` (256 tokens / image). The main tables use the
  repo/vLLM defaults; an extra run with `--mm-processor-kwargs '{"max_pixels": 262144}'` (tag `px256k`)
  tests the resolution-matched setting.
* **Tool use at eval** (all 3 runs, `outputs/run*/spagent_traces`): the untrained Qwen3-VL-4B calls the tool
  on ~100 % of MindCube/BLINK questions but 22 % / 95 % of those calls are the "wasted" (0°, 0°) view the
  paper describes; the released RL checkpoint never issues (0°, 0°), prefers ego views at az ±90°, el 30°,
  and calls the tool on 61 % (MindCube) / 76 % (BLINK) / 24 % (VSI-Bench) of the questions.

| model | dataset | questions with ≥1 tool call | (0°,0°) calls | ego-view calls | mean agent iterations |
|---|---|---|---|---|---|
| Qwen3-VL-4B | MindCube / BLINK / VSI | 100 % / 99 % / 59 % | 22 % / 95 % / 65 % | 100 % | 1.99 / 1.99 / 1.63 |
| released SPAgent-4B (RL) | MindCube / BLINK / VSI | 61 % / 76 % / 24 % | 0 % / 2 % / 0 % | 90 % / 87 % / 62 % | 1.68 / 1.79 / 1.32 |

* **Scoring.** No LLM judge is available; the option letter inside `<answer>` tags is matched exactly
  (fallback: first `(X)`/`X.` pattern, as in the repo). Long chains of thought that hit the 1024-token limit
  without an `<answer>` tag count as wrong (this also happens in the repo's own scorer).
* **Training speed.** With the repo's HF-generate rollouts one optimizer step took 45 min (1.5 M-point
  matplotlib renders + sequential generation). Rendering was capped at 150 k points (the live server renders
  ~130 k filtered points anyway), rollouts moved to vLLM colocate mode, and the likely viewpoints were
  pre-rendered in parallel on CPU; the optimization itself (data, rewards, rollouts / prompt, lr schedule,
  epochs) is unchanged.
* **Infra.** vLLM 0.11.2's multimodal processor cache crashed one engine core (`AssertionError: Expected a
  cached item for mm_hash=...`); servers are now started with `--mm-processor-cache-gb 0`.

## Reproduced numbers

All numbers are accuracy (%), mean ± std over 3 independent runs (temperature 1.0); the paper's
number for the same row is given in brackets. Rows marked "released SPAgent-4B" use the authors'
public RL checkpoint (`jialianjie/SPAgent-4B`); rows for our own GRPO run are added when the run
finishes (`Think3D-RL-4B`, see `outputs/results_tables.md` for the live version).

### Table 2 — BLINK Multi-view + MindCube (accuracy %, mean ± std over runs; paper value in brackets)

| Model | Multi-view | rotation | among | around | Avg | runs |
|---|---|---|---|---|---|---|
| Qwen3-VL-4B (no tool) | 45.36 ± 1.1 [47.87] | 37.50 ± 6.6 [34.17] | 31.67 ± 5.2 [20.00] | 40.83 ± 1.4 [41.67] | 38.84 [35.92] | 3 |
| Think3D (Qwen3-VL-4B) | 48.62 ± 4.5 [48.62] | 40.83 ± 5.2 [35.83] | 44.17 ± 8.0 [28.33] | 35.00 ± 0.0 [33.33] | 42.16 [36.53] | 3 |
| Qwen3-VL-4B-T3RL, released SPAgent-4B (no tool) | 46.12 ± 2.6 [46.11] | 26.67 ± 3.8 [30.83] | 35.83 ± 3.8 [25.83] | 39.17 ± 6.3 [35.83] | 36.95 [34.65] | 3 |
| Think3D (released SPAgent-4B) | 49.62 ± 2.6 [53.39] | 35.83 ± 5.2 [42.50] | 41.67 ± 2.9 [37.47] | 44.17 ± 2.9 [42.50] | 42.82 [43.97] | 3 |
| Think3D (released SPAgent-4B), eval images capped at 262144 px (= RL training MAX_PIXELS) | 45.11 ± 3.8 [53.39] | 33.33 ± 8.0 [42.50] | 40.00 ± 6.6 [37.47] | 35.00 ± 2.5 [42.50] | 38.36 [43.97] | 3 |

### Table 1 — VSI-Bench-tiny, 4 MC tasks (accuracy %, mean ± std over runs; paper value in brackets)

| Model | route planning | object rel direction | object rel distance | obj appearance order | Avg | runs |
|---|---|---|---|---|---|---|
| Qwen3-VL-4B (no tool) | 36.00 ± 0.0 [34.69] | 36.67 ± 4.6 [40.67] | 39.33 ± 4.2 [35.33] | 35.33 ± 6.4 [42.44] | 36.83 [38.28] | 3 |
| Think3D (Qwen3-VL-4B) | 33.33 ± 5.0 [30.61] | 33.33 ± 12.9 [44.00] | 32.67 ± 4.2 [29.33] | 33.33 ± 6.1 [52.38] | 33.17 [39.08] | 3 |
| Qwen3-VL-4B-T3RL, released SPAgent-4B (no tool) | 32.67 ± 4.6 [27.89] | 41.33 ± 5.0 [30.67] | 44.67 ± 4.6 [32.00] | 32.67 ± 3.1 [42.86] | 37.83 [33.36] | 3 |
| Think3D (released SPAgent-4B) | 36.00 ± 8.7 [36.73] | 32.67 ± 10.1 [39.00] | 34.00 ± 4.0 [44.67] | 34.00 ± 2.0 [61.22] | 34.17 [45.41] | 3 |
| Think3D (released SPAgent-4B), eval images capped at 262144 px (= RL training MAX_PIXELS) | 34.00 ± 3.5 [36.73] | 37.33 ± 9.9 [39.00] | 35.33 ± 4.6 [44.67] | 32.67 ± 1.2 [61.22] | 34.83 [45.41] | 3 |

### Reading of the results

* **BLINK multi-view + MindCube (Table 2).** The qualitative picture of the paper reproduces: tools help
  the untrained Qwen3-VL-4B only a little on MindCube-style multi-view questions (+3.3 avg here vs. +0.6
  in the paper), the RL checkpoint without the tool is no better than the base model (36.9 vs 38.8;
  paper 34.7 vs 35.9), and RL + Think3D is the best configuration (42.8 avg here vs. 44.0 in the paper;
  gain over its own no-tool row +5.9 here vs. +9.3 in the paper). BLINK numbers match to within ~2-4
  points; MindCube per-category numbers are noisier (40 questions per category → ±5-8 points between runs).
* **VSI-Bench-tiny (Table 1).** Baseline rows are close to the paper (no-tool Qwen3-VL-4B 36.8 vs 38.3;
  no-tool RL 37.8 vs 33.4), but the tool does not help on VSI in this reproduction (RL + Think3D 34.2 vs
  45.4 in the paper). In our traces the RL checkpoint only invokes the Pi3X tool on ~25 % of the VSI
  questions (vs ~85 % on MindCube) and 7-frame video reconstructions are used as-is; the exact VSI-tiny
  question set, frame sampling and rendering settings of the paper could not be recovered from the paper
  or the repo, so this row is the one clear gap.
* **Resolution ablation (px256k).** Capping eval images at the RL training resolution (256 visual tokens
  per image) lowers the RL checkpoint's Think3D scores (38.4 vs 42.8 on Table 2, unchanged on VSI), so the
  default full-resolution renders are kept for the main tables.

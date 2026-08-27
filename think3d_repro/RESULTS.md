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
* **Tool use at eval.** The released RL checkpoint requests diverse non-zero viewpoints (mostly ego views
  at az ±90°, el 30°); the untrained Qwen3-VL-4B mostly issues the "wasted" (0°, 0°) call, as described in
  the paper. On VSI-Bench the RL model calls the tool for only ~25 % of the questions (MindCube: ~85 %).
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

_(filled in as runs complete — see `outputs/summary.csv`)_

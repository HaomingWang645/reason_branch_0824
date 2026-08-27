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

## Reproduced numbers

_(filled in as runs complete — see `outputs/summary.csv`)_

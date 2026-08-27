# Think3D (arXiv 2601.13029) reproduction with SPAgent — GPUs 5/6 only

Everything for this reproduction lives in this folder (nothing is mixed into the rest
of `reason_branch_0824`).

```
think3d_repro/
├── spagent/            clone of https://github.com/zhangzaibin/spagent (+ local patches, see below)
│   ├── checkpoints/pi3x/model.safetensors   Pi3X 3D reconstruction weights (yyfz233/Pi3X)
│   ├── dataset/        MindCube (linked from ~/mindcube_data), VSI-Bench videos, BLINK.tsv,
│   │                   RL training set (Think3DQA), Pi3X offline cache, eval subsets
│   ├── logs/           server / install / training logs + launch helpers
│   └── output/         GRPO checkpoints (Think3D-RL)
├── scripts/
│   ├── env.sh                        common env (CUDA_DEVICE_ORDER=PCI_BUS_ID, LMUData, ...)
│   ├── make_mindcube_eval_subsets.py 40 q / category MindCube test subset (no RL-train scene overlap)
│   ├── make_vsi_tiny.py              VSI-Bench-tiny-style split (4 MC tasks × 50 q)
│   ├── run_eval.sh                   evaluation driver (baseline / Think3D settings)
│   ├── score.py                      aggregate per-task accuracies over runs
│   └── train_grpo_4b.sh              Think3D-RL (GRPO) training on one GPU
├── outputs/run{1,2,3}/               predictions, traces, per-run results
└── RESULTS.md                        reproduced numbers vs. paper
```

## Environments
* `conda env spagent` (py3.11, torch 2.8 cu128, transformers 4.57.6, ms-swift 4.5.2, VLMEvalKit editable
  from `spagent/third_party/VLMEvalKit`) — evaluation, Pi3X server, GRPO training.
* `conda env vllm` (py3.12, vLLM 0.11.2, torch 2.9 cu128) — serves the policy models (OpenAI API).

## Local patches to the upstream clone
* `spagent/vllm_models/qwen_vllm.py`: vLLM base URL read from `$VLLM_BASE_URL` instead of a hard-coded IP.

## GPU layout (see `scripts/env.sh`)
* GPU 5 is in `Exclusive_Process` compute mode → single process only → GRPO training.
* GPU 6 (Default mode) → Pi3X server (:20031) + vLLM servers (Qwen3-VL-4B :30058, SPAgent-4B :30059,
  own Think3D-RL checkpoint :30060).

## What is reproduced
The open-weight rows of the paper's Tables 1–2 (Qwen3-VL-4B → +Think3D → Think3D-RL). GPT-4.1 /
Gemini-2.5-Pro rows need API keys that are not available on this machine.

# AVIC reproduction — "When and How Much to Imagine" (arXiv 2602.08236)

Reproduction of the main results (Table 1, SAT-Real) of
*When and How Much to Imagine: Adaptive Test-Time Scaling with World Models for Visual Spatial Reasoning*
(Yu, Zhang et al., 2026) from the official code
[Yui010206/Adaptive-Visual-Imagination-Control](https://github.com/Yui010206/Adaptive-Visual-Imagination-Control).

Everything for this reproduction lives in this folder; nothing outside it was modified.
See [RESULTS.md](RESULTS.md) for the numbers.

## Layout

```
avic_repro/
├── avic/                      # upstream clone (visual_spatial_reasoning/ is what we run) + local patches
├── patches/local_patches.diff # git diff of every change made to the upstream clone
├── mindjourney/               # MindJourney (always-on beam search) pipeline, for the "+ MindJourney" rows
├── checkpoints/AVIC-Qwen2.5-VL-7B-policy/   # released AVIC-R LoRA adapter (HF: Shoubin/AVIC-Qwen2.5-VL-7B-policy)
├── scripts/
│   ├── env.sh          # conda env `avic`, PYTHONPATH, CUDA_DEVICE_ORDER, OpenAI key
│   ├── run_avic.sh     # one Table-1 configuration: MODE={baseline,avic,avic_qwen,avic_r} × BACKBONE
│   ├── run_matrix.sh   # sequential queue of configurations
│   ├── fix_chunks.sh   # re-run crashed chunks (pipeline resumes from results.json)
│   ├── status.sh       # per-chunk progress of every run
│   └── summarize.py    # Table-1-style markdown table incl. tokens / WM-call rate / views
├── results/<backbone>_<mode>[_spatial_beam_search]_qc<N>/question_chunk_i/   # per-question logs + results.json
├── results/summary.csv, results/RESULTS_TABLE.md
└── logs/
```

## Setup (what was actually done)

```bash
conda create -n avic python=3.11 -y && conda activate avic
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126
pip install -e avic/visual_spatial_reasoning/stable_virtual_camera/
pip install -r avic/visual_spatial_reasoning/requirements_train.txt
pip install openai datasets decord numpy-quaternion "scipy<1.14" "numpy==1.24.4"
huggingface-cli download Shoubin/AVIC-Qwen2.5-VL-7B-policy --local-dir checkpoints/AVIC-Qwen2.5-VL-7B-policy
# SAT test images: extracted from the HF parquet `array/SAT` (SAT_test.parquet, 150 questions)
```

### Local patches to the upstream code (all recorded in `patches/local_patches.diff`)

| file | why |
|---|---|
| `utils/api.py` | upstream is hard-wired to Azure OpenAI with empty credentials and has a dead stub `return "understanding: ..."` in `get_system_response`. Patched to use the plain OpenAI API (`OPENAI_API_KEY`) when no Azure endpoint is set; Azure path unchanged. |
| `utils/data_process.py` | image saving was commented out upstream; re-enabled. |
| `utils/vlm_wrapper.py` | InternVL3 device map hard-coded to `cuda:1` / multi-GPU split; use `cuda:0` when one GPU is visible. |
| `stable_virtual_camera/seva/modules/autoencoder.py` | VAE path was a hard-coded `/nas-ssd2/...` path; and `stabilityai/stable-diffusion-2-1-base` is no longer on the HF hub (404). We use the unmodified `vae/` sub-folder of the cached `friedrichor/stable-diffusion-2-1-realistic` checkpoint (its config says `_name_or_path: stabilityai/stable-diffusion-2-1`, i.e. the original SD-2.1 VAE). Override with `SVC_VAE_PATH`. |
| `tools/aggregate_chunks.py` | `pipeline_baseline.py` never fills `results["current"]`; aggregator now derives it from `progress`. |

## Protocol

Exactly the README "Eval setting" of the upstream repo (= paper Table 1):
SAT-Real test split, all 150 questions, `max_images 2`, `spatial_beam_search`, `num_policy_samples 5`,
`max_wm_candidates 5`, `max_action_ids_cap 6`, thresholds 8/8, SVC `img2trajvid_s-prob`, `cfg 4.0`, `L_short 576`,
`num_targets 8`, `frame_interval 3`, `num_frames 9`; policy sampling T=0.7, top-p 1.0; API VLMs at temperature≈0, seed 44.
Questions are split into 6 chunks (3 per GPU on two H100s) and merged with `tools/aggregate_chunks.py`.

* **baseline**: `pipelines/pipeline_baseline.py` — direct QA, no world model.
* **AVIC**: `pipelines/pipeline_avic.py --policy_model_type gpt` — the backbone is policy, verifier and QA model.
* **AVIC (Qwen2.5VL-7B policy)**: `--policy_model_type qwen2.5vl` with the base `Qwen/Qwen2.5-VL-7B-Instruct`.
* **AVIC-R**: same, plus the released GRPO LoRA adapter (`adapter_step140`).
* **MindJourney**: `mindjourney/pipelines/pipeline_svc_scaling_spatial_beam_search.py` with the SAT-test settings shipped in the MindJourney repo.

Closed-source backbones (`gpt-4o`, `gpt-4.1`) are called through the OpenAI API (the paper used Azure deployments of the same models); the model snapshots resolved by the API at run time were `gpt-4o` → `gpt-4o-2024-08-06`, `gpt-4.1` → `gpt-4.1-2025-04-14`.

## Reproduce

```bash
source scripts/env.sh
MODE=baseline BACKBONE=gpt-4o GPUS="6" CHUNKS_PER_GPU=2 bash scripts/run_avic.sh
CONFIGS="avic_r:gpt-4o avic:gpt-4o avic_qwen:gpt-4o" GPUS="6 7" CHUNKS_PER_GPU=3 bash scripts/run_matrix.sh
python scripts/summarize.py --md results/RESULTS_TABLE.md
```

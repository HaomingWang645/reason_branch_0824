# ViewTree — Execution Decisions Beyond the Design Document

Everything in this file is **added by the implementation effort on 2026-08-24** and is
not specified in `ViewTree_Research_Design_Document.pdf`. The design doc left five
blocker categories open; per instruction, the mobile stack (blocker 4) is **ignored**
— all experiments run on server GPUs.

## 0. Hardware actually used

8× NVIDIA H100 (6× NVL 96GB, 2× PCIe 80GB), 1.5 TB RAM, 256 CPU cores, single node.

## 1. Model choices (design doc §3.1 left these open)

| Role | Choice | Rationale |
|---|---|---|
| VLM controller ("lightweight VLM") | **Qwen2.5-VL-7B-Instruct** | Already cached locally; strong spatial baseline in VSI-Bench literature; Qwen2.5-VL-3B can be swapped in later for the true mobile-class model. |
| Teacher VLM (Stage I SFT, later) | **Qwen2.5-VL-32B-Instruct** (cached) | Same family → same prompt/token conventions for distillation. |
| Reconstruction backbone | **facebook/VGGT-1B, frozen** | As prescribed by the doc (§5.2); already cached. |
| Confidence head (later stages) | Linear head on last-layer hidden state of a designated token | Doc §5.5; simplest instantiation first. |

## 2. Renderer choice (design doc §3.1 "Renderer" was a named box)

**Custom pure-PyTorch GPU point-splat renderer** (`viewtree/render.py`):
z-buffered splatting (3×3 pixel splats, two-pass scatter-reduce depth test),
white background, ~2.4 M points per 16-frame scene, renders a 518-wide view in
milliseconds on H100. No third-party graphics dependency (no Open3D/PyTorch3D/
nvdiffrast), so the same code path can later be ported to mobile.

Geometry source: VGGT depth-map branch + estimated cameras, unprojected to world
points (the depth+camera route is more accurate than VGGT's direct point-map head,
per the VGGT paper). Points below the 30th percentile of VGGT depth-confidence are
dropped (doc's "reliability metadata" made concrete).

**Viability check protocol (added):** one VGGT pass on 16 uniform frames; build the
cloud from the 8 even-indexed frames only; render at the 8 odd-indexed held-out
camera poses; compare to the real frames (PSNR, global SSIM, pixel coverage).
This measures *novel-view* quality, not self-reprojection. Script:
`scripts/render_check.py`. Run on 60 scenes (20 per source dataset).

## 3. Benchmark + first experiment (design doc §8.2, §9.2 check #1)

- Primary benchmark: **VSI-Bench** (full test split: 5,130 questions, 288 scenes;
  videos from ScanNet / ScanNet++ / ARKitScenes, all local).
- Scoring: official protocol — exact-match accuracy for multiple-choice;
  Mean Relative Accuracy (mean over thresholds θ ∈ {0.50…0.95, step 0.05} of
  1[relative error < 1−θ]) for the 4 numerical question types. Headline number =
  mean over the 10 question types.
- Prompts: VSI-Bench standard suffixes ("Answer with the option's letter…",
  "Do not respond with anything other than a single number!"), greedy decoding,
  max 32 new tokens, per-image cap 448² pixels (min 224²).

### Conditions (RQ1 instantiation — added; the doc only names them abstractly)

| Condition | Input to VLM | Design-doc role |
|---|---|---|
| `current` | last video frame only | Current-view baseline (§8.4) |
| `frames16` | 16 uniformly sampled frames | History-frame baseline (§8.4) |
| `memory` | 12 uniform frames + **5 rendered overview views** (4 elevated oblique at 45°-spaced azimuths + 1 top-down, aimed at the confident-point centroid; world-up estimated as −mean camera down-axis) | Explicit-memory condition, M0-level: no learned controller yet — a *fixed heuristic view policy* stands in for the controller to test whether reconstructed novel views add information at all |

All conditions share the same VLM, decoding, and scoring (doc §8.5 requirement).
Evaluation is paired: every condition answers all 5,130 questions.

### GPU allocation for the first run (all 8 H100s)

- GPU 0: `current` (1 shard)
- GPU 1–2: `frames16` (2 shards, sharded by scene)
- GPU 3–6: `memory` (4 shards; VGGT reconstruction runs once per scene, renders cached to `data/renders/`)
- GPU 7: `render_check.py` (60 scenes), then free for reruns

## 4. Quantities the doc left blank

- Frames per scene fed to VGGT: **16** at width 518 (VGGT native).
- Rendered views per scene: **5**; render resolution = VGGT frame resolution.
- Render-check scenes: **60**; held-out views per scene: **8**.
- First-run compute estimate: ≤ ~3 GPU-hours per condition shard (measured, see results).

## 5. Deferred checks resolved or noted

- **Dataset license note:** VSI-Bench is distributed under Apache-2.0; underlying
  ScanNet/ScanNet++/ARKitScenes have their own research-use terms — reconstruction
  and rendering here are internal research use, consistent with those terms; no
  rendered data will be redistributed.
- **Scene-level splits:** VSI-Bench is evaluation-only here (no training yet), so no
  leakage concern in this phase. When SFT/RL data generation starts, splits will be
  made at scene level per doc §6.1.

## 6. Repository layout (added)

```
viewtree/           # library: data, reconstruct, render, vlm, score
scripts/run_eval.py     # condition × shard evaluation driver (resumable)
scripts/render_check.py # renderer viability protocol
data/videos/            # extracted VSI-Bench videos (not committed)
data/renders/           # per-scene cached overview renders (not committed)
results/                # jsonl predictions + metrics + example renders
```

## 7. Results

(to be filled in as runs complete)

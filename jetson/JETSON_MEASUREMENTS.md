# ViewTree on Jetson AGX Orin — W0 feasibility and system measurements

**Scope.** This document records the first on-device execution (workpackage W0 of
DESIGN_MOBILE_JETSON.md) of the ViewTree stack on the target hardware, and the
system measurements — latency, memory, energy — of our method and its in-repo
baselines. **Accuracy is deliberately out of scope**: runs use random input
frames and the un-finetuned base model, with call shapes (image counts, prompt
token counts, decode lengths, reasoning depth) exactly matching the server
implementation, so system cost is representative while answers are meaningless.
**Date:** 31 August 2026. **Author:** benchmark harness in `jetson/` (this
directory), raw data `jetson/results/bench_raw.json` (untracked),
rendered summary `jetson/results/REPORT.md`.

## 1. Device and software environment

| | |
|---|---|
| Device | Jetson AGX Orin 64 GB Developer Kit (the plan's primary target) |
| OS / BSP | JetPack 6, L4T r36.4.7, Ubuntu 22.04, kernel 5.15.148-tegra |
| CUDA | 12.6 (driver 540.4.0), GPU sm_87, ~8.5 TFLOPS fp16 measured (torch matmul) |
| Power mode | **MAXN** for all numbers in this document (`nvpmodel -q`) |
| Python env | conda `mosaic-thinker`: torch 2.5.0a0+nv24.08 (NVIDIA aarch64 build), torchvision 0.20, transformers **4.56.1** (upgraded from 4.53), qwen-vl-utils 0.0.11, accelerate, scipy, opencv |
| Models | `Qwen/Qwen2.5-VL-7B-Instruct` (bf16, sdpa) and `facebook/VGGT-1B`, downloaded to `HF_HOME=/mnt/data/hf_cache`; `vggt` installed editable from `/mnt/data/vggt_src` |
| Storage | 1.8 TB NVMe at `/mnt/data` (root eMMC is nearly full — keep everything on NVMe) |
| Telemetry | INA3221 at `/sys/class/hwmon/hwmon1`: ch1 `VDD_GPU_SOC`, ch2 `VDD_CPU_CV`, ch3 `VIN_SYS_5V0`, sampled at 20 Hz by `jetson/telemetry.py` |

### Compatibility shims (the only two changes needed)

1. **transformers ≥ 4.56** — the repo's `viewtree/vlm.py` passes `dtype=` to
   `from_pretrained`, which 4.53 rejects (`torch_dtype` era). Upgraded in-place.
2. **SDPA `enable_gqa`** — transformers 4.56's version check treats torch
   `2.5.0a0` as ≥ 2.5 and passes `enable_gqa=` to
   `scaled_dot_product_attention`, which the NV alpha build lacks.
   `jetson/bench_system.py` patches
   `transformers.integrations.sdpa_attention.use_gqa_in_sdpa` to return False,
   forcing the numerically identical repeat-kv path.

Everything else runs unmodified: no flash-attn anywhere in the repo (sdpa only),
no bitsandbytes/vllm/xformers, renderer is pure torch, pose machinery is
NumPy/SciPy.

## 2. Feasibility verdict

**Feasible.** The complete server pipeline — Qwen2.5-VL-7B bf16, VGGT-1B
32-frame reconstruction, torch point-splat renderer, 97-pose bank, and the call
patterns of both tree drivers (`viewtree/tree.py`, `scripts/depth/run_tree_d.py`)
— runs end to end on the device with both models resident simultaneously.

- Peak CUDA allocation with VLM + VGGT resident during 32-frame reconstruction:
  **25.9 GB** (plan §8 estimated ~15 GB transient VGGT on top of ~7 GB VLM at
  W4A16; in bf16 the VLM alone is ~17 GB, so 25.9 GB total is consistent and
  comfortable on 64 GB). Confirms the plan's judgment that Orin NX 16 GB
  requires phase-exclusive residency.
- One transient `NvMapMemAlloc error 12` was observed at VGGT's allocation peak
  with no functional consequence.
- Power draw: **10.6 W idle → ≈ 45 W sustained** during VLM work,
  ≈ 49 W during reconstruction (3-rail sum, see §3).

## 3. Measurement methodology

- **Random data.** 32 context frames of smoothed random noise at 1080p (Qwen's
  processor caps every image at 448², i.e. ~252 visual tokens per frame;
  VGGT preprocesses to 518×294). Token counts match real data by construction.
- **Forced decode lengths.** `min_new_tokens = max_new_tokens` (gate 4, ViewTree-D
  answers 12, depth-1/baseline answers 32) so random inputs cannot shorten
  reasoning; decode cost is therefore a slight upper bound.
- **Forced reasoning depth.** ViewTree-D is driven to target depths 0–3 with
  consensus early-stop disabled and every level fully expanded — worst-case call
  counts per depth. Expected per-question cost is then computed at the
  server-measured path mix (RESULTS.md:821: direct 1815 / consensus-d1 379 /
  d2 157 / d3+best-state+fallback 206 of n = 2557 ⇒ 71.0 / 14.8 / 6.1 / 8.1 %),
  which reproduces the reported 4.5 mean calls (we get 4.7 with worst-case
  ladders).
- **Value-head feature readout.** The server code computes the head feature with
  a *second* full forward (`viewtree/tree.py:state_feature`), costing an extra
  ~4.0 s per scored answer on Orin. The harness reads the same feature from the
  prefill hidden states inside `generate()` at zero extra cost — precisely the
  backend-interface fusion the mobile plan specifies (§2 "head(hidden)"). All
  ViewTree numbers here use the fused readout; the naive path is measured
  separately (`feature_extra_forward`, 4.04 s). The head itself (2-layer MLP,
  random weights — cost-identical to trained) is µs-scale.
- **Energy.** Per-phase integration of the three INA3221 rails at 20 Hz;
  reported energy is the **3-rail sum (GPU+SOC, CPU, 5 V system) — a lower
  bound on board power** (the devkit does not expose total module input).
  Idle draw on the same rails: 10.6 W.
- **Adapters.** Base model, no LoRA. A LoRA merged into the base (the plan's
  deployment form) is latency-identical; an *unmerged* PEFT adapter would add
  overhead and is not the deployment form.
- **Warm/cold.** Model load and first-call warmup are excluded from per-question
  numbers and reported separately (§5). Renders are cached per scene exactly as
  `run_tree_d.py`'s `rcache` does; the first question of a scene pays cold
  renders.

## 4. Per-question results

### 4.1 Expected cost at the server path mix

| method | VLM calls | latency s | energy J (3-rail) |
|---|---:|---:|---:|
| frames16 baseline (1 call, 16 frames) | 1 | 11.7 | 529 |
| memory32 baseline (1 call, 12 frames + 5 renders) | 1 | 11.4 | 518 |
| depth-1 tree (mix 23.4 % direct / 33.1 % consensus / 43.5 % fused) | 2–8 | 46.3 | 2110 |
| **ViewTree-D (ours, mix 71.0/14.8/6.1/8.1 % depth 0–3)** | **4.7** | **24.5** | **1104** |

ViewTree-D costs ≈ 2.1× the static baselines and **undercuts the depth-1 tree
by ~1.9× on both latency and energy** — the server's call-count advantage
(4.5 vs ≤ 8) carried into device units.

### 4.2 Measured routes (median across repeats; worst-case expansion)

| route | share | VLM calls | latency s | energy J | GPU-rail J |
|---|---:|---:|---:|---:|---:|
| ViewTree-D depth 0 (gate answers directly) | 71.0 % | 2 | 9.6 | 423 | 299 |
| ViewTree-D depth 1 | 14.8 % | 5 | 26.0 | 1188 | 839 |
| ViewTree-D depth 2 | 6.1 % | 13 | 70.9 | 3201 | 2252 |
| ViewTree-D depth 3 (full beam) | 8.1 % | 21 | 117.2 | 5346 | 3771 |
| depth-1 tree, direct route | 23.4 % | 2 | 11.8 | 526 | 367 |
| depth-1 tree, full (5 branches + fuse) | 43.5 % (fused) | 8 | 59.4 | 2708 | 1863 |
| frames16 | — | 1 | 11.7 | 529 | 371 |
| memory32 | — | 1 | 11.4 | 518 | 362 |

Note the headline: **ViewTree-D's 71 %-frequency gated route (9.6 s) is cheaper
than a single 16-frame baseline call (11.7 s)** — two short-decode 8-image calls
(~2.1k tokens each) beat one 4.1k-token prefill with 32-token decode.

## 5. Where the time goes

### 5.1 VLM call shapes (median of 3 reps)

| call shape | images | prompt tok | decode tok | latency s | preproc s | energy J | avg W |
|---|---:|---:|---:|---:|---:|---:|---:|
| gate (8 frames) | 8 | 2091 | 4 | 4.33 | 0.19 | 191 | 44.4 |
| answer, 8 imgs | 8 | 2091 | 12 | 5.19 | 0.18 | 234 | 44.3 |
| answer, 8 imgs, long decode | 8 | 2091 | 32 | 7.42 | 0.18 | 333 | 44.7 |
| answer, +1 render | 9 | 2273 | 12 | 5.39 | 0.19 | 242 | 44.8 |
| answer, +2 renders | 10 | 2455 | 12 | 5.76 | 0.19 | 263 | 45.6 |
| answer, +3 renders | 11 | 2637 | 12 | 6.13 | 0.19 | 282 | 45.7 |
| controller action scoring (prefill-only) | 9 | 2273 | 0 | 4.30 | 0.22 | 190 | 44.5 |
| value-head extra forward (server's naive path) | 8 | 2091 | 0 | 4.04 | 0.21 | 182 | 44.8 |
| baseline, 16 frames | 16 | 4123 | 32 | 11.71 | 0.36 | 528 | 45.3 |
| baseline, 12 frames + 5 renders | 17 | 4017 | 32 | 11.37 | 0.30 | 522 | 45.7 |

Derived rates: **prefill ≈ 1.9 s per 1k prompt tokens** (≈ 4 s for a 2.1k-token
call), **decode ≈ 0.11 s/token** (~9 tok/s, memory-bandwidth-bound as expected
for 7B bf16 on 204.8 GB/s LPDDR5), CPU-side preprocessing ≈ 0.2 s. Prefill
dominates, exactly the mobile plan's premise — its two main mechanisms (shared
KV prefixes, sibling batching) target the dominant term. The controller's
action scoring decodes nothing, so a control step costs the same as a gate.

### 5.2 One-time and per-scene phases

| phase | latency s | energy J | avg W | peak CUDA GB |
|---|---:|---:|---:|---:|
| Model load, cold (VLM 14 s + VGGT 15 s) | 28.2 | 405 | 14.4 | 21.7 |
| VGGT reconstruction, 32 frames → 3.3 M points | 11.3 (warm; ~29 cold first run) | 555 | 49.0 | 25.9 |
| Pose bank build (97 poses, lazy renders) | 0.7 | 17 | 22.4 | — |
| 5 overview renders (human_poses, splat 2) | 3.1 | 85 | 27.5 | — |
| Render, per view (torch splat, lazy, n = 20) | 0.119 mean / 0.234 p95 | — | — | — |

**Amortization** at VSI's ≈ 17.8 questions/scene: reconstruction + pose bank add
**0.7 s and 32 J per question** to ViewTree-D (12 s upfront cold);
reconstruction + 5 renders add 0.8 s per question to memory32/depth-1.
frames16 adds nothing.

### 5.3 Memory

- Peak CUDA allocation overall: **25.9 GB** (during reconstruction, VLM
  resident). Steady-state VLM-only inference: ≈ 21.7 GB allocated.
- Peak system RAM used (MemTotal − MemAvailable) during the suite: ≈ 38.5 GB —
  unified memory, includes both models, CUDA context, and page cache pressure.
- The per-frame maps VGGT returns (`world_maps`/`color_maps`/`mask_maps`) are
  dead weight for inference and are dropped after reconstruction in the harness
  (the mobile plan's easy win; keeps the scene store at points + colors).

## 6. Frames-scaling study (frames-only baseline)

One VLM call (`These are frames of a video.` + question, decode 32 forced) with
the frame count grown until failure. Frames are 1080p random images, each capped
by the processor to 448² ≈ 254 visual tokens. 16–128 frames: median of 2 reps
(`jetson/results/frames_scale_raw.json`); 192–512: 1 rep
(`frames_scale_ext_raw.json`). Counts above 128 exceed Qwen2.5-VL's 32,768
max_position_embeddings — RoPE extrapolates without error, so latency/memory
remain valid measurements there even though answer quality would not be.

### 6.1 Results

| frames | prompt tok | latency s | preproc s (CPU) | energy J (3-rail) | peak CUDA GB | peak sys RAM GB |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 4,130 | 12.3 | 0.36 | 536 | 22.6 | 36.7 |
| 24 | 6,162 | 16.6 | 0.53 | 754 | 23.1 | 36.6 |
| 32 | 8,194 | 21.0 | 0.73 | 962 | 23.5 | 36.6 |
| 48 | 12,258 | 31.2 | 1.12 | 1,461 | 24.5 | 37.2 |
| 64 | 16,322 | 40.8 | 1.43 | 1,942 | 25.4 | 39.2 |
| 96 | 24,450 | 61.6 | 2.33 | 3,010 | 27.2 | 41.5 |
| 128 | 32,578 | 83.4 | 3.09 | 4,150 | 29.1 | 43.4 |
| 192 | 48,834 | 129.9 | 4.42 | 6,554 | 32.8 | 51.2 |
| 256 | 65,090 | 179.2 | 5.93 | 9,442 | 36.5 | 57.8 |
| 384 | 97,602 | 299.7 | 9.25 | 16,182 | 43.8 | 59.4 |
| **512** | ~130,146 | **OOM** | — | — | 45.3 at failure | ~61 (exhausted) |

At 512 frames the allocation fails inside the CUDA caching allocator
(`RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED …
CUDACachingAllocator.cpp:838` — how the NV Jetson torch build surfaces an
exhausted unified-memory allocation, rather than a clean
`torch.cuda.OutOfMemoryError`). The harness caught it and the process recovered.

### 6.2 Reading the curve

- **Latency is mildly superlinear**: ~0.64 s/frame at 16–48 frames rising to
  ~0.78 s/frame by 384 (attention's quadratic term). Extrapolated crossover with
  ViewTree-D's expected 24.5 s/question sits near ~38 frames: the frames-only
  baseline can afford only ≈ 5 extra frames of context (32 → ~38) for the price
  of ViewTree-D's *entire* adaptive exploration.
- **Energy scales the same way** (~32 J/frame early, ~42 J/frame late);
  ViewTree-D's 1,104 J buys ≈ 34 baseline frames.
- **Memory**: peak CUDA grows ≈ 55 MB/frame (weights ~19 GB + ViT activations +
  KV ≈ 57 KB/token). The 64 GB unified memory admits ≈ 384 frames / ~98k tokens
  in bf16; the ceiling is CUDA + OS sharing one pool (sys RAM hit 59.4/61 GB at
  384). W4A16 quantization (W1) frees ~12 GB of weights and would push the
  ceiling well past 512 frames — but the position-embedding limit (32k tokens ≈
  128 frames) binds first for any *accurate* use.
- **CPU preprocessing** grows linearly to a nontrivial 9.3 s at 384 frames
  (image resizing in the HF processor) — a pipelining target if long contexts
  ever matter.

## 7. Plan risks — status after W0

| risk (plan §10) | status |
|---|---|
| R1 TRT-LLM multimodal immaturity | Not yet exercised (W1); torch bf16 path fully works as the bring-up baseline |
| R2 VGGT memory on NX | AGX peak measured 25.9 GB incl. VLM ⇒ NX phase-exclusive confirmed necessary |
| R3 unified-memory contention | Not yet profiled (W4, Nsight); harness runs phases sequentially |
| R4 W4A16 head miscalibration | Deferred to W1 (no quantization yet); fused feature readout implemented and validated |
| R5 thermal throttling | GPU stayed < 60 °C throughout at MAXN; sustained-load run deferred to W5 |
| R6 cache staleness | Not applicable to random-data benchmark |
| R7 Pareto gain too small | Early signal is favorable: ours 24.5 s / 1104 J vs depth-1 46.3 s / 2110 J at higher server accuracy (0.530 vs 0.517) |

## 8. Known gaps / next steps

1. **Renderer:** 119 ms/view (25 scatter launches in torch) vs the plan's
   < 5 ms target — the W2 CUDA-kernel port is the fix; at current cost, 5
   renders ≈ 0.6 s, noticeable but not dominant.
2. **Quantization (W1):** all numbers are unoptimized bf16. W4A16 TensorRT-LLM
   (~3× smaller weights) should cut both prefill and decode substantially;
   these tables are the "before" column for that comparison.
3. **Power modes:** MAXN only. The 50/30/15 W sweep needs
   `sudo nvpmodel -m <N>` (password required on this box) and a rerun of
   `jetson/bench_system.py` per mode.
4. **KV prefix sharing (§4.2 of the plan):** not yet implemented — every call
   re-prefills the 8-frame prefix. The measured per-call prefill (~4 s, of which
   ~3 s is the shared prefix) bounds the saving: at ViewTree-D's 4.7 mean calls,
   prefix reuse could remove roughly 10–12 s from the 24.5 s expected latency.
5. **External baselines (AVIC, Think3D/SPAgent):** not measured — their repro
   clones are not in the repo, they use different backbones (Stable Virtual
   Camera diffusion world model; Qwen3-VL-4B + Pi3X over vLLM), and the server
   study never ran them head-to-head either (DESIGN_IMPLEMENTATION.md §
   "external reimplementations were not run").

## 9. Model-size comparison: Qwen2.5-VL 3B / 7B / 32B

Same experiments repeated with the 3B and 32B backbones (harness args
`--model`, `--quant`; raw data `bench_raw_3b.json`, `frames_scale_3b*.json`,
`bench_raw_32b.json`, `frames_scale_32b_raw.json`).

**32B feasibility:** bf16 weights are ≈ 66 GB — **more than the 64 GB unified
memory**, so the full-precision model cannot run on this device at all
(§9.4). 32B numbers are therefore measured with bitsandbytes **NF4 4-bit**
quantization (~19 GB weights; bnb 0.50.2 has working aarch64 wheels). NF4
dequantizes on the fly, so its latency is *not* comparable to an optimized
W4A16 engine — treat 32B rows as "what 4-bit torch costs today," an upper
bound that TensorRT-LLM would improve substantially.

### 9.1 Per-question, expected at the server path mix

| method | 3B bf16 | 7B bf16 | 32B NF4 |
|---|---:|---:|---:|
| frames16 (1 call) | 8.4 s / 338 J | 11.7 s / 529 J | 32.2 s / 1690 J |
| memory32 (1 call) | 8.0 s / 319 J | 11.4 s / 518 J | 31.4 s / 1673 J |
| depth-1 tree | 35.3 s / 1321 J | 46.3 s / 2110 J | 126.5 s / 6717 J |
| **ViewTree-D (ours)** | **17.6 s / 681 J** | **24.5 s / 1104 J** | **69.6 s / 3690 J** |

The cost *ratios* between methods are model-size-invariant (ViewTree-D ≈ 2.1×
the static baselines and ≈ 1.8–2.0× cheaper than the depth-1 tree at every
size); only the absolute scale moves (3B ≈ 0.7×, 32B-NF4 ≈ 2.8× the 7B
numbers).

### 9.2 Per-route (median), 3B

| route | calls | latency s | energy J | GPU-rail J |
|---|---:|---:|---:|---:|
| vtd depth 0 | 2 | 6.9 | 266 | 181 |
| vtd depth 1 | 5 | 19.4 | 748 | 505 |
| vtd depth 2 | 13 | 52.6 | 2037 | 1387 |
| vtd depth 3 | 20 | 81.1 | 3176 | 2174 |
| tree1 direct | 2 | 8.7 | 330 | 219 |
| tree1 full | 8 | 45.2 | 1695 | 1112 |

3B is only ≈ 1.4× faster than 7B (not the 2.3× parameter ratio): the shared
vision tower, CPU preprocessing, and short decodes don't shrink with the LM.
Method ordering is unchanged — ViewTree-D ≈ 2.1× the static baselines and
≈ 2× cheaper than the depth-1 tree at every model size measured.

### 9.3 Frames-scaling, 3B (frames-only baseline until OOM)

| frames | prompt tok | latency s | energy J | peak CUDA GB | peak sys RAM GB |
|---:|---:|---:|---:|---:|---:|
| 16 | 4,130 | 8.7 | 340 | 13.5 | 27.2 |
| 32 | 8,194 | 14.7 | 629 | 14.4 | 28.6 |
| 64 | 16,322 | 28.8 | 1,276 | 16.1 | 31.1 |
| 128 | 32,578 | 58.4 | 2,744 | 19.5 | 36.5 |
| 256 | 65,090 | 132.0 | 6,464 | 26.3 | 47.0 |
| 384 | 97,602 | 216.4 | 11,183 | 33.1 | 56.4 |
| 512 | 130,114 | 318.9 | 16,794 | 39.9 | 60.6 |
| **768** | ~195,266 | **OOM** | — | 43.6 at failure | exhausted |

3B's ceiling is 512 frames (~130k tokens) vs 7B's 384 — the same allocator
assert at the 64 GB unified-memory wall. Memory slope ≈ 52 MB/frame
(vs 7B's ≈ 55 — KV shrinks, ViT activations don't). Latency ≈ 0.55 → 0.62
s/frame, ~70 % of 7B's at every count.

### 9.4 32B results (NF4 4-bit)

**bf16 infeasible, measured:** loading `Qwen2.5-VL-32B-Instruct` in bf16 fails
after 256 s at the unified-memory wall (CUDA allocator assert while
materializing ≈ 66 GB of weights on a 61 GB-usable device).

**bitsandbytes on Jetson:** the PyPI aarch64 wheel (0.50.2) targets SBSA
datacenter GPUs (sm_90) and dies with `named symbol not found` on Orin (sm_87).
Fix: build from source — `cmake -DCOMPUTE_BACKEND=cuda
-DCMAKE_CUDA_ARCHITECTURES=87` (source at `/mnt/data/bnb_src`, installed as
0.50.3.dev0). NF4 kernels then work.

Per-route (median, nq = 2; ~19 GB weights, quantize-on-load ≈ minutes):

| route | calls | latency s | energy J | GPU-rail J |
|---|---:|---:|---:|---:|
| vtd depth 0 | 2 | 27.2 | 1443 | 1028 |
| vtd depth 1 | 5 | 75.6 | 4020 | 2860 |
| vtd depth 2 | 13 | 200.7 | 10663 | 7609 |
| vtd depth 3 | 20 | 331.7 | 17573 | 12514 |
| tree1 direct | 2 | 32.7 | 1721 | 1224 |
| tree1 full | 8 | 162.3 | 8624 | 6129 |
| frames16 | 1 | 32.2 | 1690 | 1206 |
| memory32 | 1 | 31.4 | 1673 | 1192 |

Frames-scaling (1 rep):

| frames | prompt tok | latency s | energy J | peak CUDA GB | peak sys RAM GB |
|---:|---:|---:|---:|---:|---:|
| 16 | 4,130 | 32.4 | 1,661 | 26.8 | 35.1 |
| 32 | 8,194 | 58.9 | 3,049 | 28.8 | 37.4 |
| 64 | 16,322 | 116.5 | 6,176 | 32.8 | 41.8 |
| 128 | 32,578 | 241.7 | 13,233 | 40.9 | 51.4 |
| 192 | 48,834 | 383.7 | 21,306 | 49.1 | 60.5 |
| **256** | ~65,090 | **OOM** | — | 45.7 at failure | exhausted |

Observations:

- NF4-torch is ≈ 2.8× slower than 7B bf16 per token (dequant-on-the-fly
  dominates; an optimized W4A16 engine would narrow this a lot — treat these as
  an upper bound). Sustained draw is ≈ 51–55 W (memory-bound dequant keeps the
  GPU busier than bf16's 45 W).
- Memory slope is steep: ≈ 127 MB/frame (32B's KV is ≈ 262 KB/token bf16 —
  4.6× the 7B's 57 KB/token — GQA 8 kv-heads × 64 layers). OOM ceiling:
  **192 frames (~49k tokens)**, half the 7B ceiling and ~2.7× below 3B's.
- ViewTree-D's gated route still beats one frames16 call (27.2 vs 32.2 s) even
  at 32B.

**OOM ceilings across sizes (frames-only baseline, 64 GB unified memory):**
3B bf16 → 512 frames (~130k tok); 7B bf16 → 384 (~98k); 32B NF4 → 192 (~49k).

## 10. Reproduce

```bash
# full suite (micro + end-to-end, ~30 min at MAXN for 7B)
HF_HOME=/mnt/data/hf_cache VIEWTREE_POSES=human \
  python jetson/bench_system.py --nq 3 --micro-reps 3   # --model ... --quant 4bit
python jetson/summarize.py <raw.json> <report.md>

# frames-scaling study (base + OOM extension)
HF_HOME=/mnt/data/hf_cache python jetson/bench_frames_scale.py  # --counts ...
python jetson/summarize_frames_scale.py <raw.json>

# whole campaigns per model size (download-wait + suite + scaling chained)
zsh jetson/run_3b.sh ; zsh jetson/run_32b.sh
```

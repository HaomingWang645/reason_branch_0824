# Jetson AGX Orin system measurements — Qwen/Qwen2.5-VL-3B-Instruct (none)

Device: Jetson AGX Orin 64GB, power mode MAXN, idle draw 11.0 W (sum of VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0; energy figures below are the same 3-rail sum, a lower bound on board power). Run: 2026-08-31 22:34:32.

## Scene-setup and one-time phases

| phase | latency s | energy J (3-rail) | avg W | peak CUDA GB | notes |
|---|---|---|---|---|---|
| load_models | 30.0 | 425 | 14.2 | 12.7 | VLM 13s + VGGT 17s |
| recon_32f | 11.3 | 529 | 46.7 | 16.9 | 3.3M points |
| posebank_build | 0.3 | 7 | 27.1 | 0.0 | 97 poses |
| renders_5 | 2.3 | 75 | 33.1 | 0.0 |  |
| render_per_view_lazy | 0.126 (mean/view, n=15) | | | | p95 0.231s |

## VLM call shapes (microbenchmark, median of reps)

| shape | images | prompt tok | decode tok | latency s | preproc s | energy J | avg W |
|---|---|---|---|---|---|---|---|
| gate_8i_d4 | 8 | 2091 | 4 | 3.03 | 0.20 | 117 | 38.2 |
| ans_8i_d12 | 8 | 2091 | 12 | 3.83 | 0.20 | 146 | 36.9 |
| ans_8i_d32 | 8 | 2091 | 32 | 5.77 | 0.18 | 208 | 36.4 |
| ans_9i_d12 | 9 | 2273 | 12 | 3.91 | 0.18 | 152 | 38.8 |
| ans_10i_d12 | 10 | 2455 | 12 | 4.23 | 0.19 | 163 | 38.2 |
| ans_11i_d12 | 11 | 2637 | 12 | 4.46 | 0.22 | 173 | 38.7 |
| ctrl_9i_prefill | 9 | 2273 | 0 | 3.12 | 0.22 | 123 | 38.8 |
| ans_16i_d32 | 16 | 4123 | 32 | 8.54 | 0.35 | 339 | 39.4 |
| ans_17i_d32 | 17 | 4017 | 32 | 8.21 | 0.33 | 330 | 40.2 |
| feature_extra_forward | 8 | 2091 | 0 | 2.75 | 0.19 | 115 | 41.0 |

## Per-question end-to-end (median across questions)

| method / route | VLM calls | latency s | energy J (3-rail) | GPU rail J | peak mem MB |
|---|---|---|---|---|---|
| frames16 | 1 | 8.4 | 338 | 232 | 31163 |
| memory32 | 1 | 8.0 | 319 | 217 | 31419 |
| tree1_direct | 2 | 8.7 | 330 | 219 | 31341 |
| tree1_full | 8 | 45.2 | 1695 | 1112 | 31274 |
| vtd_d0 | 2 | 6.9 | 266 | 181 | 31147 |
| vtd_d1 | 5 | 19.4 | 748 | 505 | 31222 |
| vtd_d2 | 13 | 52.6 | 2037 | 1387 | 31276 |
| vtd_d3 | 20 | 81.1 | 3176 | 2174 | 31281 |

## Expected per-question cost at the server-measured path mix

- **ViewTree-D (ours)**: 17.6 s, 681 J, 4.6 calls (mix 71.0/14.8/6.1/8.1 % for depth 0/1/2/3, RESULTS.md:821)
- **depth-1 tree**: 35.3 s, 1321 J (mix 23.4 % direct / 33.1 % consensus / 43.5 % fused, RESULTS.md:644)
- **frames16**: 8.4 s, 338 J (deterministic 1 call)
- **memory32**: 8.0 s, 319 J (deterministic 1 call)

## Scene-setup amortization

VGGT(32f) = 11.3 s / 529 J; pose bank 0.3 s; 5 renders 2.3 s. VSI mean ≈ 17.8 questions/scene:
- ViewTree-D adds (recon+bank)/17.8 = **0.7 s, 30 J per question** amortized (upfront 12 s cold).
- memory32/depth-1 add (recon+5 renders)/17.8 = **0.8 s per question** (upfront 14 s).
- frames16 adds nothing (no reconstruction).

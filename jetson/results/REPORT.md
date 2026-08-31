# Jetson AGX Orin system measurements (W0 bring-up, torch bf16)

Device: Jetson AGX Orin 64GB, power mode MAXN, idle draw 10.6 W (sum of VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0; energy figures below are the same 3-rail sum, a lower bound on board power). Run: 2026-08-31 15:32:29.

## Scene-setup and one-time phases

| phase | latency s | energy J (3-rail) | avg W | peak CUDA GB | notes |
|---|---|---|---|---|---|
| load_models | 28.2 | 405 | 14.4 | 21.7 | VLM 14s + VGGT 15s |
| recon_32f | 11.3 | 555 | 49.0 | 25.9 | 3.3M points |
| posebank_build | 0.7 | 17 | 22.4 | 0.0 | 97 poses |
| renders_5 | 3.1 | 85 | 27.5 | 0.0 |  |
| render_per_view_lazy | 0.119 (mean/view, n=20) | | | | p95 0.234s |

## VLM call shapes (microbenchmark, median of reps)

| shape | images | prompt tok | decode tok | latency s | preproc s | energy J | avg W |
|---|---|---|---|---|---|---|---|
| gate_8i_d4 | 8 | 2091 | 4 | 4.33 | 0.19 | 191 | 44.4 |
| ans_8i_d12 | 8 | 2091 | 12 | 5.19 | 0.18 | 234 | 44.3 |
| ans_8i_d32 | 8 | 2091 | 32 | 7.42 | 0.18 | 333 | 44.7 |
| ans_9i_d12 | 9 | 2273 | 12 | 5.39 | 0.19 | 242 | 44.8 |
| ans_10i_d12 | 10 | 2455 | 12 | 5.76 | 0.19 | 263 | 45.6 |
| ans_11i_d12 | 11 | 2637 | 12 | 6.13 | 0.19 | 282 | 45.7 |
| ctrl_9i_prefill | 9 | 2273 | 0 | 4.30 | 0.22 | 190 | 44.5 |
| ans_16i_d32 | 16 | 4123 | 32 | 11.71 | 0.36 | 528 | 45.3 |
| ans_17i_d32 | 17 | 4017 | 32 | 11.37 | 0.30 | 522 | 45.7 |
| feature_extra_forward | 8 | 2091 | 0 | 4.04 | 0.21 | 182 | 44.8 |

## Per-question end-to-end (median across questions)

| method / route | VLM calls | latency s | energy J (3-rail) | GPU rail J | peak mem MB |
|---|---|---|---|---|---|
| frames16 | 1 | 11.7 | 529 | 371 | 38465 |
| memory32 | 1 | 11.4 | 518 | 362 | 38457 |
| tree1_direct | 2 | 11.8 | 526 | 367 | 38453 |
| tree1_full | 8 | 59.4 | 2708 | 1863 | 38782 |
| vtd_d0 | 2 | 9.6 | 423 | 299 | 38278 |
| vtd_d1 | 5 | 26.0 | 1188 | 839 | 38249 |
| vtd_d2 | 13 | 70.9 | 3201 | 2252 | 38263 |
| vtd_d3 | 21 | 117.2 | 5346 | 3771 | 38274 |

## Expected per-question cost at the server-measured path mix

- **ViewTree-D (ours)**: 24.5 s, 1104 J, 4.7 calls (mix 71.0/14.8/6.1/8.1 % for depth 0/1/2/3, RESULTS.md:821)
- **depth-1 tree**: 46.3 s, 2110 J (mix 23.4 % direct / 33.1 % consensus / 43.5 % fused, RESULTS.md:644)
- **frames16**: 11.7 s, 529 J (deterministic 1 call)
- **memory32**: 11.4 s, 518 J (deterministic 1 call)

## Scene-setup amortization

VGGT(32f) = 11.3 s / 555 J; pose bank 0.7 s; 5 renders 3.1 s. VSI mean ≈ 17.8 questions/scene:
- ViewTree-D adds (recon+bank)/17.8 = **0.7 s, 32 J per question** amortized (upfront 12 s cold).
- memory32/depth-1 add (recon+5 renders)/17.8 = **0.8 s per question** (upfront 14 s).
- frames16 adds nothing (no reconstruction).

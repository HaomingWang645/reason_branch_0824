# Jetson AGX Orin system measurements — Qwen/Qwen2.5-VL-32B-Instruct (4bit)

Device: Jetson AGX Orin 64GB, power mode MAXN, idle draw 11.1 W (sum of VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0; energy figures below are the same 3-rail sum, a lower bound on board power). Run: 2026-08-31 23:20:27.

## Scene-setup and one-time phases

| phase | latency s | energy J (3-rail) | avg W | peak CUDA GB | notes |
|---|---|---|---|---|---|
| load_models | 131.3 | 1836 | 14.0 | 24.7 | VLM 116s + VGGT 16s |
| recon_32f | 11.2 | 549 | 48.8 | 28.9 | 3.3M points |
| posebank_build | 0.8 | 16 | 21.2 | 0.0 | 97 poses |
| renders_5 | 2.6 | 78 | 30.0 | 0.0 |  |
| render_per_view_lazy | 0.136 (mean/view, n=19) | | | | p95 0.238s |

## VLM call shapes (microbenchmark, median of reps)

| shape | images | prompt tok | decode tok | latency s | preproc s | energy J | avg W |
|---|---|---|---|---|---|---|---|
| gate_8i_d4 | 8 | 2091 | 4 | 12.51 | 0.20 | 647 | 51.7 |
| ans_8i_d12 | 8 | 2091 | 12 | 14.61 | 0.19 | 759 | 51.9 |
| ans_8i_d32 | 8 | 2091 | 32 | 20.00 | 0.19 | 1042 | 52.1 |
| ans_9i_d12 | 9 | 2273 | 12 | 15.20 | 0.20 | 802 | 52.7 |
| ans_10i_d12 | 10 | 2455 | 12 | 16.47 | 0.20 | 876 | 53.1 |
| ans_11i_d12 | 11 | 2637 | 12 | 17.44 | 0.22 | 918 | 52.6 |
| ctrl_9i_prefill | 9 | 2273 | 0 | 12.28 | 0.21 | 656 | 53.3 |
| ans_16i_d32 | 16 | 4123 | 32 | 32.29 | 0.41 | 1692 | 52.3 |
| ans_17i_d32 | 17 | 4017 | 32 | 31.75 | 0.31 | 1690 | 53.2 |
| feature_extra_forward | 8 | 2091 | 0 | 11.80 | 0.20 | 629 | 53.2 |

## Per-question end-to-end (median across questions)

| method / route | VLM calls | latency s | energy J (3-rail) | GPU rail J | peak mem MB |
|---|---|---|---|---|---|
| frames16 | 1 | 32.2 | 1690 | 1206 | 39718 |
| memory32 | 1 | 31.4 | 1673 | 1192 | 39446 |
| tree1_direct | 2 | 32.7 | 1721 | 1224 | 39422 |
| tree1_full | 8 | 162.3 | 8624 | 6129 | 39486 |
| vtd_d0 | 2 | 27.2 | 1443 | 1028 | 39459 |
| vtd_d1 | 5 | 75.6 | 4020 | 2860 | 39456 |
| vtd_d2 | 13 | 200.7 | 10663 | 7609 | 39455 |
| vtd_d3 | 20 | 331.7 | 17573 | 12514 | 39489 |

## Expected per-question cost at the server-measured path mix

- **ViewTree-D (ours)**: 69.6 s, 3690 J, 4.6 calls (mix 71.0/14.8/6.1/8.1 % for depth 0/1/2/3, RESULTS.md:821)
- **depth-1 tree**: 126.5 s, 6717 J (mix 23.4 % direct / 33.1 % consensus / 43.5 % fused, RESULTS.md:644)
- **frames16**: 32.2 s, 1690 J (deterministic 1 call)
- **memory32**: 31.4 s, 1673 J (deterministic 1 call)

## Scene-setup amortization

VGGT(32f) = 11.2 s / 549 J; pose bank 0.8 s; 5 renders 2.6 s. VSI mean ≈ 17.8 questions/scene:
- ViewTree-D adds (recon+bank)/17.8 = **0.7 s, 32 J per question** amortized (upfront 12 s cold).
- memory32/depth-1 add (recon+5 renders)/17.8 = **0.8 s per question** (upfront 14 s).
- frames16 adds nothing (no reconstruction).

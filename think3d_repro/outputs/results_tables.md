
### Table 2 — BLINK Multi-view + MindCube (accuracy %, mean ± std over runs; paper value in brackets)

| Model | Multi-view | rotation | among | around | Avg | runs |
|---|---|---|---|---|---|---|
| Qwen3-VL-4B (no tool) | 45.36 ± 1.1 [47.87] | 37.50 ± 6.6 [34.17] | 31.67 ± 5.2 [20.00] | 40.83 ± 1.4 [41.67] | 38.84 [35.92] | 3 |
| Think3D (Qwen3-VL-4B) | 48.62 ± 4.5 [48.62] | 40.83 ± 5.2 [35.83] | 44.17 ± 8.0 [28.33] | 35.00 ± 0.0 [33.33] | 42.16 [36.53] | 3 |
| Qwen3-VL-4B-T3RL, released SPAgent-4B (no tool) | 46.12 ± 2.6 [46.11] | 26.67 ± 3.8 [30.83] | 35.83 ± 3.8 [25.83] | 39.17 ± 6.3 [35.83] | 36.95 [34.65] | 3 |
| Think3D (released SPAgent-4B) | 49.37 ± 3.0 [53.39] | 35.83 ± 5.2 [42.50] | 40.83 ± 3.8 [37.47] | 44.17 ± 2.9 [42.50] | 42.55 [43.97] | 3 |
| Think3D (released SPAgent-4B), eval images capped at 262144 px (= RL training MAX_PIXELS) | 45.36 ± 3.4 [53.39] | 33.33 ± 8.0 [42.50] | 40.00 ± 6.6 [37.47] | 35.00 ± 2.5 [42.50] | 38.42 [43.97] | 3 |

### Table 1 — VSI-Bench-tiny, 4 MC tasks (accuracy %, mean ± std over runs; paper value in brackets)

| Model | route planning | object rel direction | object rel distance | obj appearance order | Avg | runs |
|---|---|---|---|---|---|---|
| Qwen3-VL-4B (no tool) | 36.00 ± 0.0 [34.69] | 36.67 ± 4.6 [40.67] | 39.33 ± 4.2 [35.33] | 35.33 ± 6.4 [42.44] | 36.83 [38.28] | 3 |
| Think3D (Qwen3-VL-4B) | 33.33 ± 5.0 [30.61] | 33.33 ± 12.9 [44.00] | 32.67 ± 4.2 [29.33] | 33.33 ± 6.1 [52.38] | 33.17 [39.08] | 3 |
| Qwen3-VL-4B-T3RL, released SPAgent-4B (no tool) | 32.67 ± 4.6 [27.89] | 41.33 ± 5.0 [30.67] | 44.67 ± 4.6 [32.00] | 32.67 ± 3.1 [42.86] | 37.83 [33.36] | 3 |
| Think3D (released SPAgent-4B) | 36.00 ± 8.7 [36.73] | 32.67 ± 10.1 [39.00] | 34.00 ± 4.0 [44.67] | 34.00 ± 2.0 [61.22] | 34.17 [45.41] | 3 |
| Think3D (released SPAgent-4B), eval images capped at 262144 px (= RL training MAX_PIXELS) | 34.00 ± 3.5 [36.73] | 37.33 ± 9.9 [39.00] | 36.00 ± 6.0 [44.67] | 32.00 ± 2.0 [61.22] | 34.83 [45.41] | 3 |
| released SPAgent-4B (no tool), 32 video frames instead of 7 | 37.33 ± 3.1 [27.89] | 46.00 ± 5.3 [30.67] | 48.67 ± 1.2 [32.00] | 52.00 ± 2.0 [42.86] | 46.00 [33.36] | 3 |
| Think3D (released SPAgent-4B), 32 video frames instead of 7 | 34.00 ± 3.5 [36.73] | 43.33 ± 3.1 [39.00] | 44.00 ± 3.5 [44.67] | 46.00 ± 7.2 [61.22] | 41.83 [45.41] | 3 |
| Qwen3-VL-4B (no tool), 32 video frames instead of 7 | 42.00 ± 2.0 [34.69] | 44.67 ± 5.0 [40.67] | 45.33 ± 3.1 [35.33] | 49.33 ± 2.3 [42.44] | 45.33 [38.28] | 3 |
| Think3D (Qwen3-VL-4B), 32 video frames instead of 7 | 34.00 ± 2.0 [30.61] | 44.00 ± 2.0 [44.00] | 39.33 ± 3.1 [29.33] | 45.33 ± 5.0 [52.38] | 40.67 [39.08] | 3 |

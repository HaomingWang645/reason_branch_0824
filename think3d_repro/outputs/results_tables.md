
### Table 2 — BLINK Multi-view + MindCube (accuracy %, mean ± std over runs; paper value in brackets)

| Model | Multi-view | rotation | among | around | Avg | runs |
|---|---|---|---|---|---|---|
| Qwen3-VL-4B (no tool) | 45.36 ± 1.1 [47.87] | 37.50 ± 6.6 [34.17] | 31.67 ± 5.2 [20.00] | 40.83 ± 1.4 [41.67] | 38.84 [35.92] | 3 |
| Think3D (Qwen3-VL-4B) | 48.62 ± 4.5 [48.62] | 40.83 ± 5.2 [35.83] | 44.17 ± 8.0 [28.33] | 35.00 ± 0.0 [33.33] | 42.16 [36.53] | 3 |
| Qwen3-VL-4B-T3RL, released SPAgent-4B (no tool) | 46.12 ± 2.6 [46.11] | 26.67 ± 3.8 [30.83] | 35.83 ± 3.8 [25.83] | 39.17 ± 6.3 [35.83] | 36.95 [34.65] | 3 |
| Think3D (released SPAgent-4B) | 51.13 ± 0.0 [53.39] | 35.83 ± 5.2 [42.50] | 41.67 ± 2.9 [37.47] | 44.17 ± 2.9 [42.50] | 43.20 [43.97] | 3 |

### Table 1 — VSI-Bench-tiny, 4 MC tasks (accuracy %, mean ± std over runs; paper value in brackets)

| Model | route planning | object rel direction | object rel distance | obj appearance order | Avg | runs |
|---|---|---|---|---|---|---|
| Qwen3-VL-4B (no tool) | 36.00 ± 0.0 [34.69] | 36.67 ± 4.6 [40.67] | 39.33 ± 4.2 [35.33] | 35.33 ± 6.4 [42.44] | 36.83 [38.28] | 3 |
| Think3D (Qwen3-VL-4B) | 36.00 ± 2.8 [30.61] | 26.00 ± 2.8 [44.00] | 35.00 ± 1.4 [29.33] | 36.00 ± 5.7 [52.38] | 33.25 [39.08] | 2 |
| Qwen3-VL-4B-T3RL, released SPAgent-4B (no tool) | 30.00 ± 0.0 [27.89] | 41.00 ± 7.1 [30.67] | 46.00 ± 5.7 [32.00] | 34.00 ± 2.8 [42.86] | 37.75 [33.36] | 2 |
| Think3D (released SPAgent-4B) | 34.00 ± 11.3 [36.73] | 28.00 ± 8.5 [39.00] | 32.00 ± 2.8 [44.67] | 35.00 ± 1.4 [61.22] | 32.25 [45.41] | 2 |

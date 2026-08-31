"""Render §6.1 of JETSON_MEASUREMENTS.md from frames_scale_raw.json."""
import json, sys
import numpy as np

RAW = sys.argv[1] if len(sys.argv) > 1 else "jetson/results/frames_scale_raw.json"
recs = json.load(open(RAW))
RAILS = ("VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0")

def tot_e(r):
    return sum(r.get(f"{x}_energy_j", 0) for x in RAILS)

rows = {}
for r in recs:
    if r.get("kind") == "scale":
        rows.setdefault(r["n_frames"], []).append(r)

print("| frames | prompt tok | latency s | preproc s | energy J (3-rail) | peak CUDA GB | peak sys RAM GB |")
print("|---:|---:|---:|---:|---:|---:|---:|")
for n in sorted(rows):
    rs = rows[n]
    med = lambda k: float(np.median([x[k] for x in rs]))
    print(f"| {n} | {rs[0]['n_prompt_tokens']} | {med('total_s'):.1f} | "
          f"{med('preproc_s'):.2f} | {np.median([tot_e(x) for x in rs]):.0f} | "
          f"{med('cuda_peak_gb'):.1f} | "
          f"{med('mem_peak_mb')/1024:.1f} |")
for r in recs:
    if r.get("kind") == "scale_fail":
        print(f"\n**Failure at {r['n_frames']} frames** (rep {r['rep']}): "
              f"`{r['error']}` — {r['msg']}"
              f" (peak CUDA at failure {r.get('cuda_peak_gb', 0):.1f} GB)")

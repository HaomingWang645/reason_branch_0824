"""Frames-scaling benchmark: the frames-only baseline (1 VLM call, decode 32)
with growing frame counts until OOM / context overflow.

Measures per count: latency (preproc + model), prompt tokens, peak CUDA GB,
peak system RAM, 3-rail energy. Counts: 16, 24, 32, 48, 64, 96, 128 — stops at
the first failure (recorded with its error type).
"""
import argparse, gc, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_system import Bench, load_models, make_frames, last_win, QUESTION  # noqa
from telemetry import Telemetry

COUNTS = [16, 24, 32, 48, 64, 96, 128]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results",
        "frames_scale_raw.json"))
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--counts", default="")
    a = ap.parse_args()
    global COUNTS
    if a.counts:
        COUNTS = [int(x) for x in a.counts.split(",")]

    tel = Telemetry(hz=20)
    idle_w = tel.baseline_idle_w(5.0)
    tel.start()
    bench = Bench(tel, a.out)
    bench.log(kind="meta", idle_total_w=idle_w, powermode="MAXN",
              device="Jetson AGX Orin 64GB", ts=time.strftime("%F %T"))

    vlm, head, t_vlm, t_vggt = load_models()
    frames = make_frames(128)
    bench.vlm_call(vlm, frames[:2], "warmup " + QUESTION, 4, label="warmup")

    for n in COUNTS:
        idx = np.linspace(0, 127, n).round().astype(int)
        imgs = [frames[i] for i in idx]
        ok = True
        for rep in range(a.reps):
            torch.cuda.reset_peak_memory_stats()
            gc.collect(); torch.cuda.empty_cache()
            try:
                with tel.window(f"frames{n}_{rep}"):
                    r, _ = bench.vlm_call(
                        vlm, imgs, "These are frames of a video.\n" + QUESTION,
                        32, label=f"frames{n}")
                bench.log(kind="scale", n_frames=n, rep=rep,
                          cuda_peak_gb=torch.cuda.max_memory_allocated() / 1e9,
                          **r, **last_win(tel))
                print(f"frames{n} rep{rep}: {r['total_s']:.1f}s "
                      f"{r['n_prompt_tokens']} tok", flush=True)
            except Exception as ex:
                tel.windows.append((f"frames{n}_{rep}_fail", time.monotonic(),
                                    time.monotonic()))
                bench.log(kind="scale_fail", n_frames=n, rep=rep,
                          error=type(ex).__name__, msg=str(ex)[:300],
                          cuda_peak_gb=torch.cuda.max_memory_allocated() / 1e9)
                print(f"frames{n} rep{rep} FAILED: {type(ex).__name__}: "
                      f"{str(ex)[:200]}", flush=True)
                ok = False
                gc.collect(); torch.cuda.empty_cache()
                break
        if not ok:
            break

    tel.stop()
    bench.log(kind="meta_end", ts=time.strftime("%F %T"))
    print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()

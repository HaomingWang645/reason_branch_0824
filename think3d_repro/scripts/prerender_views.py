#!/usr/bin/env python3
"""Pre-render common Pi3X novel views for every RL-training scene, in parallel on CPU.

The GRPO rollout scheduler executes tool calls sequentially, so rendering was the
bottleneck (~2 s per global view x ~150 views per optimizer step). Pi3XOfflineTool
keeps a disk cache (spagent/outputs/pi3x_<scene>_azim*_elev*[_refcamN][_camview].png)
that a running training job picks up immediately, so this script simply calls the
tool's own `call()` for the most likely (azimuth, elevation, ref_cam, camera_view)
combinations of each scene using a process pool.  Output is byte-identical to what
the rollout would have produced (same code path, same PI3X_OFFLINE_MAX_POINTS).
"""
import os, sys, json, re, collections, itertools
from multiprocessing import Pool
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'spagent')
os.chdir(ROOT); sys.path.insert(0, ROOT)
os.environ.setdefault('PI3X_CACHE_DIR', os.path.join(ROOT, 'dataset/pi3x_cache'))
os.environ.setdefault('PI3X_OFFLINE_MAX_POINTS', '150000')

# scenes: first sample per unique image tuple
scenes = {}
for l in open('dataset/crossviewQA_train_rl_fixed.jsonl'):
    d = json.loads(l); k = tuple(d['images'])
    scenes.setdefault(k, d['images'])
scenes = list(scenes.values())
# most likely global-view combos: from the training log histogram + the prompt's angle guide
# Ordered by observed likelihood in rollouts (top-down el=60 views dominate, then el=30, then el=0),
# and by ref cam, so that the most useful cache entries exist first for ALL scenes.
AZ = [0, 180, 90, -90, 45, -45]; EL = [60, 30, 0]
combos = [(a, e) for e in EL for a in AZ if not (a == 0 and e == 0)]
jobs = []
for a, e in combos:
    for ref in range(1, 5):
        for imgs in scenes:
            if ref <= len(imgs):
                jobs.append((imgs, float(a), float(e), ref, False))
print(f'{len(scenes)} scenes, {len(jobs)} global-view renders to (re)use', flush=True)

def work(job):
    import logging; logging.disable(logging.CRITICAL)
    from spagent.tools.pi3x_offline_tool import Pi3XOfflineTool
    global _tool
    try: _tool
    except NameError: _tool = Pi3XOfflineTool()
    imgs, a, e, ref, cv = job
    try:
        r = _tool.call(image_path=imgs, azimuth_angle=a, elevation_angle=e, rotation_reference_camera=ref, camera_view=cv)
        return bool(r.get('success'))
    except Exception:
        return False

if __name__ == '__main__':
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    ok = 0
    with Pool(nproc) as pool:
        for i, res in enumerate(pool.imap_unordered(work, jobs, chunksize=4), 1):
            ok += bool(res)
            if i % 500 == 0: print(f'{i}/{len(jobs)} done, ok={ok}', flush=True)
    print(f'FINISHED {ok}/{len(jobs)} ok', flush=True)

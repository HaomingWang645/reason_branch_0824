"""Paper Figure 3 assets: real Stable-Virtual-Camera (the MindJourney/AVIC world model)
predictions on the ARKit bedroom scene, for the world-model-failure comparison.

hop1: from frame_09 (facing the wardrobe), turn left ~86 deg (+ one small step)
      -> compare against captured frame_25 (the bed) and the ViewTree render.
hop2: cascade — feed hop1's *generated* view back in, move forward ~0.7 m
      -> compare against captured frame_23 and the ViewTree render at that pose.

Run:  source scripts/env.sh; CUDA_VISIBLE_DEVICES=7 python $AVIC_ROOT/scripts/wm_failure_gen.py <out_dir>
"""
import os, sys, copy, cv2
sys.path.insert(0, '.')
from argparse import Namespace
from pipelines.pipeline_avic import _run_one_candidate, resize_to_short_side, ActionSpace, _SVC_VERSION_DICT_INIT
import stable_virtual_camera.demo as _svc_demo
from stable_virtual_camera.demo import Model

WORK = sys.argv[1]
FRAMES = "/home/haoming/reason_branch_0824/figures/motivation_assets/vsi_arkit_47334117/frames_64"
model_args = Namespace(task="img2trajvid_s-prob", replace_or_include_input=True, cfg=4.0,
                       guider=1, L_short=576, num_targets=8, use_traj_prior=True,
                       chunk_strategy="interp")
model = Model()

def gen(tag, img_path, actions, magnitude, turn_size, forward_size):
    wdir = os.path.join(WORK, tag)
    os.makedirs(os.path.join(wdir, "step_0"), exist_ok=True)
    img = resize_to_short_side(cv2.imread(img_path), 512)
    inp = os.path.join(wdir, "step_0", "img_0.png")
    cv2.imwrite(inp, img)
    step_dir = os.path.join(wdir, "step_1")
    os.makedirs(step_dir, exist_ok=True)
    _svc_demo.VERSION_DICT.clear()
    _svc_demo.VERSION_DICT.update(copy.deepcopy(_SVC_VERSION_DICT_INIT))
    _run_one_candidate(actions, magnitude, inp, step_dir, "plan",
                       copy.deepcopy(model_args), model, forward_size, turn_size)
    vp = os.path.join(step_dir, "plan", "pred.mp4")
    cap = cv2.VideoCapture(vp)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        frames.append(fr)
    cap.release()
    for k, fr in enumerate(frames):
        cv2.imwrite(os.path.join(wdir, f"gen_{k:02d}.png"), fr)
    out = os.path.join(wdir, "final.png")
    cv2.imwrite(out, frames[-1])
    print(tag, "->", out, f"({len(frames)} frames in pred.mp4, meta said {n})", flush=True)
    return out

L, F = ActionSpace.TURN_LEFT, ActionSpace.MOVE_FORWARD
g1 = gen("hop1", f"{FRAMES}/frame_09.jpg", [L] * 10 + [F], 86, turn_size=8.6, forward_size=0.25)
gen("hop2", g1, [F] * 3, 0.7, turn_size=9.0, forward_size=0.23)
print("DONE")

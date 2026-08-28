"""Evaluate video spatial benchmarks STI-Bench and VSTI-Bench with the existing
checkpoints (no retraining).
  --system direct : frames + question -> answer (zero-shot / SFT-plain / GRPO-plain / any adapter)
  --system tree   : ViewTree depth-1 tree (gate -> constrained branch renders -> prune/fuse/arbitrate)
Frames: VSTI = 32 uniform frames (questions reference "frame k of 32");
STI = 16 frames inside [time_start, time_end] (±1 s window if the range is empty).
Scoring: MC letter exact match; numeric = VSI-style mean relative accuracy.
Sharded by video; reconstruction cached per video within a shard."""
import argparse, ast, json, os, re, sys, time
import cv2, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viewtree.data import load_questions
from viewtree.score import parse_number
from viewtree.reconstruct import reconstruct
from viewtree.render import overview_poses, render
from viewtree.tree import BRANCH_PRE, FUSE_PRE, ROOT_GATE, VIEW_DESCS, answer_logprob, load_conf_head
from viewtree.vlm import QwenVL
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); EXT = os.path.join(REPO, "data", "external")
MC_SUFFIX = "Answer with the option's letter from the given choices directly."
NUM_SUFFIX = "Answer with a single number only."
FRAMES_PRE = "These are frames of a video.\n"

def head_train_scenes():
    rows = load_questions(); sc = sorted({(r["dataset"], r["scene_name"]) for r in rows}); return {s[1] for s in sc[0::2]}

def read_frames(path, n, t0=None, t1=None):
    cap = cv2.VideoCapture(path); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total <= 0: raise RuntimeError(f"cannot read {path}")
    lo, hi = 0, total - 1
    if t0 is not None:
        if t1 is None or t1 <= t0: t0, t1 = t0 - 1.0, t0 + 1.0
        lo, hi = max(0, int(t0 * fps)), min(total - 1, int(t1 * fps))
        if hi <= lo: lo, hi = 0, total - 1
    idxs = np.linspace(lo, hi, n).round().astype(int); want = sorted(set(idxs.tolist())); out = {}
    for i in range(hi + 1):
        ok = cap.grab()
        if not ok: break
        if i in want:
            ok, fr = cap.retrieve()
            if ok: out[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    cap.release()
    frames = [out[i] for i in idxs if i in out]
    if not frames: raise RuntimeError("no frames")
    while len(frames) < n: frames.append(frames[-1])
    return frames

def load_sti():
    df = pd.read_parquet(f"{EXT}/sti/qa.parquet"); items = []
    for _, r in df.iterrows():
        cands = ast.literal_eval(r["Candidates"]) if isinstance(r["Candidates"], str) else dict(r["Candidates"])
        opts = [f"{k}. {v}" for k, v in cands.items() if v not in (None, "", "None")]
        extra = (str(r["Prompt"]).strip() + "\n") if str(r["Prompt"]).strip() not in ("", "nan") else ""
        items.append(dict(id=f"sti_{r['ID']}", video=f"{EXT}/sti/video/{r['Video']}", t0=float(r["time_start"]), t1=float(r["time_end"]), n=16,
                          prompt=f"{extra}{r['Question']}\n" + "\n".join(opts) + f"\n{MC_SUFFIX}", gt=str(r["Answer"]).strip(), numeric=False,
                          qtype=r["Task"], source=r["Source"], scene=str(r["Video"])[:-4]))
    return items

def load_vsti():
    d = json.load(open(f"{EXT}/vsti/test.json")); items = []
    for r in d:
        numeric = r["mc_answer"] in (None, "None")
        if numeric: prompt = f"{r['question']}\n{NUM_SUFFIX}"; gt = str(r["ground_truth"])
        else:
            opts = ast.literal_eval(r["options"]) if isinstance(r["options"], str) else r["options"]
            prompt = f"{r['question']}\n" + "\n".join(opts) + f"\n{MC_SUFFIX}"; gt = r["mc_answer"].strip()
        items.append(dict(id=f"vsti_{r['id']}", video=f"{EXT}/vsti/{r['video_path']}", t0=None, t1=None, n=32, prompt=prompt, gt=gt, numeric=numeric,
                          qtype=r["question_type"], source="ScanNet", scene=os.path.basename(r["video_path"])[:-4]))
    return items

def score(item, pred):
    if item["numeric"]:
        p = parse_number(pred); g = float(item["gt"])
        if p is None: return 0.0
        rel = abs(p - g) / max(abs(g), 1e-8); return float(np.mean(rel < (1.0 - np.arange(0.5, 1.0, 0.05))))
    m = re.search(r"\b([A-E])\b", pred.strip()); return float(bool(m) and m.group(1) == item["gt"])

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["sti", "vsti"], required=True); ap.add_argument("--system", choices=["direct", "tree"], required=True)
    ap.add_argument("--adapter", default=None); ap.add_argument("--conf-head", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    items = load_sti() if a.bench == "sti" else load_vsti()
    vids = sorted({it["video"] for it in items})[a.shard::a.num_shards]; vs = set(vids)
    items = [it for it in items if it["video"] in vs]
    if a.limit: items = items[: a.limit]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try: done.add(json.loads(l)["id"])
            except Exception: pass
    htrain = head_train_scenes()
    vlm = QwenVL("Qwen/Qwen2.5-VL-7B-Instruct", adapter=a.adapter); conf = load_conf_head(a.conf_head) if a.system == "tree" else None
    def sc(images, prompt):
        pred, lp, ft = answer_logprob(vlm, images, prompt, want_feature=True, max_new_tokens=12); return pred, float(conf(ft))
    fout = open(a.out, "a"); cache = {}; n = 0; t0 = time.time()
    for it in items:
        if it["id"] in done: continue
        try:
            key = (it["video"], it["t0"], it["t1"], it["n"])
            if key not in cache:
                cache = {key: {"frames": read_frames(it["video"], it["n"], it["t0"], it["t1"])}}  # keep one entry (memory)
            frames = cache[key]["frames"]; prompt = it["prompt"]; rec_out = dict(id=it["id"], qtype=it["qtype"], source=it["source"], scene=it["scene"], gt=it["gt"], numeric=it["numeric"], clean=it["scene"] not in htrain)
            if a.system == "direct":
                pred = vlm.ask(frames, FRAMES_PRE + prompt, max_new_tokens=12); rec_out.update(pred=pred, mode="direct")
            else:
                qtext = prompt.split("\n")[0] if not prompt.startswith("Object description") else prompt.split("\n")[1]
                gate, _ = answer_logprob(vlm, frames, ROOT_GATE.format(q=qtext), max_new_tokens=4); gate = gate.strip()
                dpred, dconf = sc(frames, FRAMES_PRE + prompt); rec_out.update(gate=gate, direct=dpred, dconf=round(dconf, 3))
                if "YES" in gate.upper(): rec_out.update(mode="direct", pred=dpred)
                else:
                    if "rec" not in cache[key]:
                        rec = reconstruct(frames); H, W = rec["size"]; K = rec["intrinsics"][0]
                        views = [(render(rec["points"], rec["colors"], pose, K, H, W, splat=2).clamp(0, 1) * 255).byte().cpu().numpy() for pose in overview_poses(rec)[:5]]
                        del rec; torch.cuda.empty_cache(); cache[key]["rec"] = views
                    views = cache[key]["rec"]; br = []
                    for vi, v8 in enumerate(views):
                        p, c = sc(frames + [v8], BRANCH_PRE.format(k=len(frames), desc=VIEW_DESCS[vi]) + prompt); br.append((p, c))
                    order = sorted(range(5), key=lambda j: -br[j][1]); kept = order[:2]
                    rec_out["branches"] = [{"pred": p, "conf": round(c, 3)} for p, c in br]; rec_out["kept"] = kept
                    agree = len({(parse_number(br[j][0]) if it["numeric"] else br[j][0].strip()) for j in kept}) == 1
                    if agree and br[kept[0]][1] > dconf: rec_out.update(mode="branch_consensus", pred=br[kept[0]][0])
                    else:
                        fpred, fconf = sc(frames + [views[j] for j in kept], FUSE_PRE.format(k=len(frames), m=2, descs=", ".join(VIEW_DESCS[j] for j in kept)) + prompt)
                        rec_out["fuse"] = {"pred": fpred, "conf": round(fconf, 3)}
                        if dconf > fconf and dconf > max(br[j][1] for j in kept): rec_out.update(mode="fused_fallback_direct", pred=dpred)
                        else: rec_out.update(mode="fused", pred=fpred)
            rec_out["score"] = score(it, rec_out["pred"]); fout.write(json.dumps(rec_out) + "\n"); n += 1
            if n % 25 == 0: fout.flush(); print(f"[{a.bench} {a.system} s{a.shard}] {n} {(time.time()-t0)/60:.1f} min", flush=True)
        except Exception as e:
            print("skip", it["id"], repr(e)[:120], flush=True); torch.cuda.empty_cache()
    fout.close(); print("DONE", flush=True)

if __name__ == "__main__":
    main()

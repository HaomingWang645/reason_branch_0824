"""Batch depth-1 reasoning tree over OST-Bench (sharded, resumable): image
history -> VGGT reconstruction -> (constrained) branch views -> prune/fuse/
arbitrate.  Output: one json line per item (id, qtype, gate, mode, final, gt, correct)."""
import argparse, json, os, re, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_external import load_ost
from viewtree.reconstruct import reconstruct
from viewtree.render import overview_poses, render
from viewtree.tree import BRANCH_PRE, FUSE_PRE, ROOT_GATE, VIEW_DESCS, answer_logprob, load_conf_head
from viewtree.vlm import QwenVL

def letter(t):
    m = re.search(r"\b([A-H])\b", t.strip()); return m.group(1) if m else ""

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True); ap.add_argument("--conf-head", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try: done.add(json.loads(l)["id"])
            except Exception: pass
    vlm = QwenVL("Qwen/Qwen2.5-VL-7B-Instruct", adapter=a.adapter); conf = load_conf_head(a.conf_head)
    def sc(images, prompt):
        pred, lp, ft = answer_logprob(vlm, images, prompt, want_feature=True, max_new_tokens=8); return letter(pred), float(conf(ft))
    fout = open(a.out, "a"); n = 0; t0 = time.time()
    for i, (qid, imgs, prompt, gt, qtype) in enumerate(load_ost()):
        if i % a.num_shards != a.shard or qid in done: continue
        try:
            base = [imgs[j] for j in np.linspace(0, len(imgs) - 1, min(8, len(imgs))).round().astype(int)]
            qtext = prompt.split("\n")[1]
            gate, _ = answer_logprob(vlm, base, ROOT_GATE.format(q=qtext), max_new_tokens=4); gate = gate.strip()
            dpred, dconf = sc(base, prompt)
            rec_out = {"id": qid, "qtype": qtype, "gt": gt, "gate": gate, "direct": dpred, "dconf": round(dconf, 3)}
            if "YES" in gate.upper():
                rec_out.update(mode="direct", final=dpred)
            else:
                rec = reconstruct(imgs); H, W = rec["size"]; K = rec["intrinsics"][0]
                views, br = [], []
                for vi, pose in enumerate(overview_poses(rec)[:5]):
                    v8 = (render(rec["points"], rec["colors"], pose, K, H, W, splat=2).clamp(0, 1) * 255).byte().cpu().numpy()
                    views.append(v8); p, c = sc(base + [v8], BRANCH_PRE.format(k=len(base), desc=VIEW_DESCS[vi]) + prompt); br.append((p, c))
                del rec; torch.cuda.empty_cache()
                order = sorted(range(5), key=lambda j: -br[j][1]); kept = order[:2]
                rec_out["branches"] = [{"pred": p, "conf": round(c, 3)} for p, c in br]; rec_out["kept"] = kept
                if len({br[j][0] for j in kept}) == 1 and br[kept[0]][1] > dconf:
                    rec_out.update(mode="branch_consensus", final=br[kept[0]][0])
                else:
                    fpred, fconf = sc(base + [views[j] for j in kept], FUSE_PRE.format(k=len(base), m=2, descs=", ".join(VIEW_DESCS[j] for j in kept)) + prompt)
                    rec_out["fuse"] = {"pred": fpred, "conf": round(fconf, 3)}
                    if dconf > fconf and dconf > max(br[j][1] for j in kept): rec_out.update(mode="fused_fallback_direct", final=dpred)
                    else: rec_out.update(mode="fused", final=fpred)
            rec_out["correct"] = rec_out["final"] == gt
            fout.write(json.dumps(rec_out) + "\n"); n += 1
            if n % 25 == 0: fout.flush(); print(f"[s{a.shard}] {n} items {(time.time()-t0)/60:.1f} min", flush=True)
        except Exception as e:
            print("skip", qid, repr(e)[:100], flush=True); torch.cuda.empty_cache()
    fout.close(); print("DONE", flush=True)

if __name__ == "__main__":
    main()

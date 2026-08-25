"""Confidence head v2: domain-adapted on MindCube ladder states + VSI tree states.

VSI states come only from the 144 even-indexed (head-train) scenes; the odd
scenes remain untouched for evaluation. Groups for splitting: MindCube group id
or VSI scene id. Reports AUROC/Brier/ECE overall and on VSI states alone.
"""
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_conf import auroc, ece, group_of, load_data

from viewtree.data import load_questions

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_vsi():
    qmeta = {r["id"]: r for r in load_questions()}
    X, y, gid = [], [], []
    for f in sorted(glob.glob(os.path.join(REPO, "results", "vsi_states_s*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            fp = os.path.join(REPO, "data", "feats_vsi", f"{r['id']}.pt")
            if not os.path.exists(fp):
                continue
            feats = torch.load(fp)
            scene = f"vsi_{qmeta[r['id']]['dataset']}_{qmeta[r['id']]['scene_name']}"
            for k, st in r["states"].items():
                if k not in feats:
                    continue
                X.append(feats[k].float())
                y.append(1.0 if st["score"] >= 0.5 else 0.0)
                gid.append(scene)
    return torch.stack(X), torch.tensor(y), np.array(gid)


def main():
    Xm, ym, lpm, gm, _, _ = load_data()
    Xv, yv, gv = load_vsi()
    print(f"mindcube states {len(ym)} (pos {ym.mean():.3f}); "
          f"vsi states {len(yv)} (pos {yv.mean():.3f})")
    X = torch.cat([Xm, Xv])
    y = torch.cat([ym, yv])
    gid = np.concatenate([gm, gv])
    is_vsi = np.array([False] * len(ym) + [True] * len(yv))

    groups = sorted(set(gid))
    rng = np.random.default_rng(0)
    rng.shuffle(groups)
    n = len(groups)
    gtr = set(groups[: int(0.8 * n)])
    gcal = set(groups[int(0.8 * n): int(0.9 * n)])
    gte = set(groups[int(0.9 * n):])
    mtr = np.array([g in gtr for g in gid])
    mcal = np.array([g in gcal for g in gid])
    mte = np.array([g in gte for g in gid])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    head = nn.Sequential(nn.Linear(3584, 512), nn.GELU(), nn.Dropout(0.1),
                         nn.Linear(512, 1)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr, ytr = X[mtr].to(dev), y[mtr].to(dev)
    # upweight VSI states (smaller pool) 2x
    w = torch.ones(len(ytr), device=dev)
    w[torch.tensor(is_vsi[mtr], device=dev)] = 2.0
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    for ep in range(20):
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(Xtr), 512):
            b = perm[i: i + 512]
            opt.zero_grad()
            out = head(Xtr[b]).squeeze(-1)
            loss = (lossf(out, ytr[b]) * w[b]).mean()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        if ep % 5 == 4:
            print(f"ep{ep} loss {tot/len(Xtr):.4f}")

    head.eval()
    with torch.no_grad():
        lcal = head(X[mcal].to(dev)).squeeze(-1).cpu()
        lte = head(X[mte].to(dev)).squeeze(-1).cpu()
    best_T, best_nll = 1.0, 1e9
    ycal = y[mcal]
    for T in np.arange(0.3, 5.01, 0.05):
        p = torch.sigmoid(lcal / T).clamp(1e-6, 1 - 1e-6)
        nll = -(ycal * p.log() + (1 - ycal) * (1 - p).log()).mean().item()
        if nll < best_nll:
            best_nll, best_T = nll, T
    print(f"T={best_T:.2f}")
    yt = y[mte].numpy()
    p = torch.sigmoid(lte / best_T).numpy()
    vte = is_vsi[mte]
    print(f"ALL  test: AUROC {auroc(p, yt):.3f} Brier {np.mean((p-yt)**2):.3f} "
          f"ECE {ece(p, yt):.3f}")
    print(f"VSI  test: AUROC {auroc(p[vte], yt[vte]):.3f} "
          f"Brier {np.mean((p[vte]-yt[vte])**2):.3f} ECE {ece(p[vte], yt[vte]):.3f} "
          f"(n={vte.sum()})")
    print(f"MC   test: AUROC {auroc(p[~vte], yt[~vte]):.3f} (n={(~vte).sum()})")
    torch.save({"head": head.state_dict(), "T": best_T},
               os.path.join(REPO, "checkpoints", "conf_head_v2.pt"))
    print("SAVED checkpoints/conf_head_v2.pt")


if __name__ == "__main__":
    main()

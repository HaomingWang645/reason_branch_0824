"""Stage II: train + calibrate the confidence head (doc §6.3-6.4).

Input: per-state hidden features (3584-d) from the SFT policy's ladders.
Label: eventual correctness — 1 if the ladder from this state onward reaches a
correct answer under the current policy.
Split: by MindCube group id (scene-level, §6.1). Temperature-calibrated on a
held-out split. Metrics: AUROC, Brier, ECE, plus state-selection accuracy
(head vs token-logprob vs oracle) — the operational pruning comparison (H4).
"""
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def group_of(item_id):
    return item_id.split("_q")[0]


def load_data():
    rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(
        REPO, "results", "traj_sft_s*.jsonl"))) for l in open(f)]
    X, y, lp, gid, iid, state = [], [], [], [], [], []
    for r in rows:
        fp = os.path.join(REPO, "data", "feats_train", f"{r['id']}.pt")
        if not os.path.exists(fp):
            continue
        feats = torch.load(fp)
        order = [k for k in
                 [f"s{i}" for i in range(1, r["n_views"] + 1)] + [f"s{r['n_views']}r"]
                 if k in r["states"]]
        corr = [bool(r["states"][k]["correct"]) for k in order]
        for i, k in enumerate(order):
            if k not in feats:
                continue
            X.append(feats[k].float())
            y.append(1.0 if any(corr[i:]) else 0.0)
            lp.append(r["states"][k]["lp"])
            gid.append(group_of(r["id"]))
            iid.append(r["id"])
            state.append(k)
    return (torch.stack(X), torch.tensor(y), np.array(lp), np.array(gid),
            np.array(iid), np.array(state))


def auroc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ece(probs, labels, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (probs >= edges[i]) & (probs < edges[i + 1])
        if m.sum():
            e += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return e


def main():
    X, y, lp, gid, iid, state = load_data()
    print(f"states: {len(y)}, positive rate {y.mean():.3f}")
    groups = sorted(set(gid))
    rng = np.random.default_rng(0)
    rng.shuffle(groups)
    n = len(groups)
    gtr, gcal, gte = (set(groups[: int(0.8 * n)]),
                      set(groups[int(0.8 * n): int(0.9 * n)]),
                      set(groups[int(0.9 * n):]))
    mtr = np.array([g in gtr for g in gid])
    mcal = np.array([g in gcal for g in gid])
    mte = np.array([g in gte for g in gid])
    print(f"split states: train {mtr.sum()} cal {mcal.sum()} test {mte.sum()}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    head = nn.Sequential(nn.Linear(3584, 512), nn.GELU(), nn.Dropout(0.1),
                         nn.Linear(512, 1)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr, ytr = X[mtr].to(dev), y[mtr].to(dev)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(20):
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(Xtr), 512):
            b = perm[i: i + 512]
            opt.zero_grad()
            out = head(Xtr[b]).squeeze(-1)
            loss = lossf(out, ytr[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        if ep % 5 == 4:
            print(f"ep{ep} loss {tot/len(Xtr):.4f}")

    head.eval()
    with torch.no_grad():
        logit_cal = head(X[mcal].to(dev)).squeeze(-1).cpu()
        logit_te = head(X[mte].to(dev)).squeeze(-1).cpu()
    # temperature on cal split
    best_T, best_nll = 1.0, 1e9
    ycal = y[mcal]
    for T in np.arange(0.3, 5.01, 0.05):
        p = torch.sigmoid(logit_cal / T).clamp(1e-6, 1 - 1e-6)
        nll = -(ycal * p.log() + (1 - ycal) * (1 - p).log()).mean().item()
        if nll < best_nll:
            best_nll, best_T = nll, T
    print(f"temperature T={best_T:.2f}")

    yt = y[mte].numpy()
    p = torch.sigmoid(logit_te / best_T).numpy()
    lpte = lp[mte]
    print("\n== test metrics (eventual-correctness) ==")
    print(f"head AUROC {auroc(p, yt):.3f}  Brier {np.mean((p-yt)**2):.3f}  "
          f"ECE {ece(p, yt):.3f}")
    print(f"token-logprob AUROC {auroc(lpte, yt):.3f}")

    # operational: pick one state per item by confidence, take its correctness
    items = {}
    idte, stte = iid[mte], state[mte]
    with torch.no_grad():
        pass
    corr_lookup = {}
    rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(
        REPO, "results", "traj_sft_s*.jsonl"))) for l in open(f)]
    for r in rows:
        for k, v in r["states"].items():
            corr_lookup[(r["id"], k)] = bool(v["correct"])
    for i in range(mte.sum()):
        items.setdefault(idte[i], []).append(
            (p[i], lpte[i], corr_lookup[(idte[i], stte[i])]))
    accs = {"head": [], "token": [], "oracle": [], "last_state": []}
    for it, cands in items.items():
        accs["head"].append(max(cands, key=lambda c: c[0])[2])
        accs["token"].append(max(cands, key=lambda c: c[1])[2])
        accs["oracle"].append(any(c[2] for c in cands))
        accs["last_state"].append(cands[-1][2])
    print("\n== state-selection accuracy (test items:", len(items), ") ==")
    for k, v in accs.items():
        print(f"{k:10s} {np.mean(v):.3f}")

    torch.save({"head": head.state_dict(), "T": best_T},
               os.path.join(REPO, "checkpoints", "conf_head.pt"))
    print("SAVED checkpoints/conf_head.pt")


if __name__ == "__main__":
    main()

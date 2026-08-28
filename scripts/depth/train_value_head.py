"""Value head for ViewTree-D: P(state answer correct) from the answerer's last-token
feature, trained on oracle-walk states (Phase 2), initialised from conf_head_v2_human.
Scene-level split (10 % held-in validation). Reports AUROC and saves temperature-calibrated head."""
import argparse, glob, json, os, sys, random
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts"))
from train_conf import auroc, ece
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--oracle", default=os.path.join(REPO, "data/train3r/oracle_s*.jsonl")); ap.add_argument("--feats", default=os.path.join(REPO, "data/train3r/feats_oracle"))
    ap.add_argument("--init", default=os.path.join(REPO, "checkpoints/conf_head_v2_human.pt")); ap.add_argument("--out", default=os.path.join(REPO, "checkpoints/depth/value_head.pt"))
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--max-items", type=int, default=0)
    a = ap.parse_args(); X, y, g = [], [], []
    files = sorted(glob.glob(a.oracle)); n = 0
    for f in files:
        for l in open(f):
            r = json.loads(l); fp = os.path.join(a.feats, r["id"].replace("::", "_") + ".pt")
            if not os.path.exists(fp): continue
            fe = torch.load(fp)
            for s in r["states"]:
                k = json.dumps(s["path"])
                if k in fe: X.append(fe[k].float()); y.append(float(s["correct"] >= 0.5)); g.append(r["scene"])
            n += 1
            if a.max_items and n >= a.max_items: break
    X = torch.stack(X); y = torch.tensor(y); scenes = sorted(set(g)); random.Random(0).shuffle(scenes); val = set(scenes[: max(1, len(scenes) // 10)])
    vm = torch.tensor([s in val for s in g]); print("states", len(y), "pos rate %.3f" % y.mean(), "val states", int(vm.sum()))
    ck = torch.load(a.init, weights_only=False); head = nn.Sequential(nn.Linear(X.shape[1], 512), nn.ReLU(), nn.Linear(512, 1)); head.load_state_dict(ck["head"])
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4); Xtr, ytr = X[~vm], y[~vm]; Xva, yva = X[vm], y[vm]
    for ep in range(a.epochs):
        perm = torch.randperm(len(ytr))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]; loss = nn.functional.binary_cross_entropy_with_logits(head(Xtr[b]).squeeze(-1), ytr[b]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): pv = head(Xva).squeeze(-1)
        print(f"ep{ep} val AUROC {auroc(torch.sigmoid(pv).numpy(), yva.numpy()):.3f}")
    with torch.no_grad():
        lv = head(Xva).squeeze(-1); best_T, best = 1.0, 1e9
        for T in np.linspace(0.5, 3.0, 26):
            e = ece(torch.sigmoid(lv / T).numpy(), yva.numpy())
            if e < best: best, best_T = e, float(T)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); torch.save({"head": head.state_dict(), "T": best_T}, a.out); print("SAVED", a.out, "T", best_T)

if __name__ == "__main__":
    main()

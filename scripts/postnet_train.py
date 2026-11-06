"""Train the learned grid posterior B0 (history only) or B (history + physics log-posterior) on n_train training
episodes and evaluate it, and the physics posterior M1, on the held-out test episodes (spec section 5.1).

  python scripts/postnet_train.py --set strong --model b --n_train 1000 --rep 0 \
      --data $RLS_DATA/postnet/data --out $RLS_DATA/postnet/runs

Training subsets are drawn from the first 9500 training episodes; episodes 9500+ are the validation set (early
stopping). Loss: -log q(true cell) averaged over n = 10, 20, ..., 80. Metrics at n = 20, 40, 80 on the test episodes:
log score, P(e < 0.1) and median e of the posterior mean, coverage of the 50% / 90% highest-posterior regions, area
of the 90% region; per test policy as well.
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlscalar.search.postnet import PostNet

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="strong")
ap.add_argument("--model", default="b", choices=["b0", "b"])
ap.add_argument("--n_train", type=int, default=1000)
ap.add_argument("--rep", type=int, default=0)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--max_epochs", type=int, default=300)
ap.add_argument("--patience", type=int, default=8)
ap.add_argument("--max_steps", type=int, default=60000, help="cap on gradient steps")
ap.add_argument("--clip", type=float, default=0.0, help="gradient-norm clip (0 = off)")
ap.add_argument("--vel", action="store_true", help="append the local velocity (u - U, v) to the tokens (E48)")
ap.add_argument("--cnn", type=int, default=0, help="channels of the residual CNN on the logit map (E51); 0 = none")
ap.add_argument("--attn", action="store_true", help="per-cell attention weights over the readings in place of the plain sum (E53)")
ap.add_argument("--bottleneck", type=int, default=0, help="rank of a learned linear map of the token before the GRU (E54); 0 = none")
ap.add_argument("--proj", default="", help="(F, r) .npy fixed projection of the token before the GRU (E54)")
ap.add_argument("--nomem", action="store_true", help="GRU state fixed at zero: independent per-reading terms (E54)")
ap.add_argument("--exclude_policy", type=int, default=-1, help="drop this path type (policy code) from training and validation (E57)")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(a.rep)
NS = list(range(10, 81, 10))
EVAL = [20, 40, 80]


VSCALE = {"strong": 0.18, "weak": 0.09}               # rms of (u - U, v) at the sensor (E48 data log)


def load(split):
    fs = sorted(glob.glob(f"{a.data}/{a.set}_{split}_[0-9][0-9][0-9].npz"))
    zs = [np.load(f) for f in fs]
    d = {k: np.concatenate([z[k] for z in zs]) for k in ("tok", "pos", "src", "policy", "lpost")}
    if a.vel:                                                  # velocity / VSCALE, VSCALE ~ rms of (u - U, v)
        uv = np.concatenate([np.load(f.replace(".npz", "_vel.npz"))["uv"] for f in fs])
        d["tok"] = np.concatenate([d["tok"], uv / VSCALE[a.set]], -1).astype(np.float32)
    d["gx"], d["gy"] = zs[0]["grid_x"], zs[0]["grid_y"]
    return d


tr, te = load("train"), load("test")
GX, GY = np.meshgrid(tr["gx"], tr["gy"], indexing="ij")
cx, cy = GX.ravel(), GY.ravel()                          # same cell order as Belief


def true_cell(src):
    return np.argmin((cx[None] - src[:, :1]) ** 2 + (cy[None] - src[:, 1:]) ** 2, 1)


pool = np.arange(min(9500, len(tr["src"])))
val = np.arange(9500, len(tr["src"]))
if a.exclude_policy >= 0:                                # E57: leave one path type out of training
    pool, val = pool[tr["policy"][pool] != a.exclude_policy], val[tr["policy"][val] != a.exclude_policy]
sub = np.sort(np.random.default_rng(1000 + a.rep).choice(pool, a.n_train, replace=False))


def tensors(d, idx):
    return (torch.tensor(d["tok"][idx], device=dev), torch.tensor(d["pos"][idx], device=dev),
            torch.tensor(d["lpost"][idx].astype(np.float32), device=dev), torch.tensor(true_cell(d["src"][idx]), device=dev))


net = PostNet(cx, cy, physics=(a.model == "b"), vel=a.vel, cnn=a.cnn, attn=a.attn, bottleneck=a.bottleneck,
             proj=np.load(a.proj) if a.proj else None, nomem=a.nomem).to(dev)
opt_params = [p for p in net.parameters() if p.requires_grad]
opt = torch.optim.Adam(opt_params, lr=a.lr)
Xtr, Xva = tensors(tr, sub), tensors(tr, val)


def batch_loss(X, idx):
    tok, pos, lp, j = (x[idx] for x in X)
    lq = net(tok, pos, NS, lp)                                            # (B, N, J)
    return -lq.gather(2, j[:, None, None].expand(-1, len(NS), 1)).mean()


def full_loss(X):
    with torch.no_grad():
        return float(np.mean([batch_loss(X, torch.arange(s, min(s + a.bs, len(X[0])), device=dev)).item()
                              for s in range(0, len(X[0]), a.bs)]))


t0, best, bad, state, steps = time.time(), np.inf, 0, None, 0
hist = dict(epoch=[], steps=[], train=[], val=[])          # per-epoch loss history (E43)
for ep in range(a.max_epochs):
    perm = torch.randperm(len(Xtr[0]), device=dev)
    tl = []
    for s in range(0, len(perm), a.bs):
        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            l = batch_loss(Xtr, perm[s:s + a.bs])
        l.backward()
        if a.clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip)
        opt.step(); steps += 1
        tl.append(l.item())
    v = full_loss(Xva)
    hist["epoch"].append(ep + 1); hist["steps"].append(steps); hist["train"].append(float(np.mean(tl))); hist["val"].append(v)
    if v < best - 1e-4:
        best, bad, state = v, 0, {k: x.detach().clone() for k, x in net.state_dict().items()}
    else:
        bad += 1
    if bad >= a.patience or steps >= a.max_steps:
        break
net.load_state_dict(state)
train_time = time.time() - t0


def metrics(lq, src, pol):
    """lq: (E, N, J) log posteriors at EVAL steps; returns dict of metrics (overall and per policy)."""
    q = np.exp(lq)
    j = true_cell(src)
    out = {}
    for i, n in enumerate(EVAL):
        qi = q[:, i]
        mean = np.stack([qi @ cx, qi @ cy], 1)
        err = np.hypot(*(mean - src).T)
        ls = np.log(np.maximum(qi[np.arange(len(j)), j], 1e-300))
        o = np.sort(qi, 1)[:, ::-1]; cs = np.cumsum(o, 1)
        qt = qi[np.arange(len(j)), j]
        level = (qi * (qi > qt[:, None])).sum(1)                          # mass strictly above the truth
        a90 = (np.argmax(cs >= 0.9, 1) + 1) * (cx[1] - cx[0] if cx[1] != cx[0] else cy[1] - cy[0]) ** 2
        for tag, m in [("all", np.ones(len(j), bool))] + [(f"pol{p}", pol == p) for p in np.unique(pol)]:
            out[f"{tag}_n{n}"] = dict(logscore=float(ls[m].mean()), P01=float(np.mean(err[m] < 0.1)),
                                      med=float(np.median(err[m])), cov50=float(np.mean(level[m] < 0.5)),
                                      cov90=float(np.mean(level[m] < 0.9)), area90=float(np.mean(a90[m])))
    return out


Xte = tensors(te, np.arange(len(te["src"])))
idx_eval = [NS.index(n) for n in EVAL]
with torch.no_grad():
    lq = np.concatenate([net(Xte[0][s:s + a.bs], Xte[1][s:s + a.bs], NS, Xte[2][s:s + a.bs])[:, idx_eval].float().cpu().numpy()
                         for s in range(0, len(Xte[0]), a.bs)])
lphys = te["lpost"][:, idx_eval].astype(np.float64)
lphys -= np.log(np.exp(lphys).sum(2, keepdims=True))
res = dict(history=hist, vel=a.vel, cnn=a.cnn, attn=a.attn, bottleneck=a.bottleneck, proj=a.proj, nomem=a.nomem, exclude_policy=a.exclude_policy, n_params=sum(x.numel() for x in net.parameters()), set=a.set, model=a.model, n_train=a.n_train, rep=a.rep, readings=int(a.n_train * 81), epochs=ep + 1,
           steps=steps, val_loss=best, train_time=train_time, alpha=float(net.alpha.detach()) if a.model == "b" else None,
           net=metrics(lq, te["src"], te["policy"]), physics=metrics(lphys, te["src"], te["policy"]))
os.makedirs(a.out, exist_ok=True)
tag = f"{a.set}_{a.model}_n{a.n_train}_r{a.rep}"
json.dump(res, open(f"{a.out}/{tag}.json", "w"), indent=1)
torch.save(state, f"{a.out}/{tag}.pt")
r = res["net"]["all_n80"]; p = res["physics"]["all_n80"]
print(f"{tag}: epochs {ep + 1} val {best:.3f} | n80 net logscore {r['logscore']:.2f} P01 {r['P01']:.2f} med {r['med']:.3f} "
      f"cov90 {r['cov90']:.2f} | physics {p['logscore']:.2f} {p['P01']:.2f} {p['med']:.3f} {p['cov90']:.2f} | {train_time:.0f}s", flush=True)

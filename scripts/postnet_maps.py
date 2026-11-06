"""Posterior maps and calibration data for the learned grid posteriors (E39; figure scripts/figures/fig_posterior_maps.py).

Runs B and B0 (E37, n_train = 9500, rep 0) on all held-out test episodes of one forcing and stores
  levels, qtrue (E, M, 8) HPD level of the true cell (posterior mass strictly above it) for physics M1, B, B0 at n = 10..80;
                          the source lies inside the alpha-HPD region iff level < alpha, so the CDF of level is the
                          coverage curve
  q (E0, M, 3, J) float16 full posteriors of M1, B, B0 at n = 20, 40, 80 for the EIG-path episodes (policy 0; all with --map_all)
  err (E, M, 8)           error of the posterior mean
  plus pos, tok, src, policy of every episode and the grid.
  python scripts/postnet_maps.py --set strong --out $RLS_OUT
E48: --runs runs_fp32 --runs_vel runs_vel adds the velocity-sensing models: methods M1, B, B0, Bvel, B0vel
(the velocity token is appended from <shard>_vel.npz, scaled as in postnet_train.py).
"""
import argparse, glob, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlscalar.search.postnet import PostNet

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="strong")
ap.add_argument("--pn", default=os.environ.get("RLS_DATA", "data") + "/postnet")
ap.add_argument("--n_train", type=int, default=9500)
ap.add_argument("--runs", default="runs")
ap.add_argument("--runs_vel", default="", help="also run the velocity-sensing models from this run folder (E48)")
ap.add_argument("--map_all", action="store_true", help="store the maps of all test episodes, not only the EIG paths")
ap.add_argument("--out", required=True)
a = ap.parse_args()
NS = list(range(10, 81, 10))
MAPN = [20, 40, 80]

VSCALE = {"strong": 0.18, "weak": 0.09}                   # as in postnet_train.py
fs = sorted(glob.glob(f"{a.pn}/data/{a.set}_test_[0-9][0-9][0-9].npz"))
zs = [np.load(f) for f in fs]
d = {k: np.concatenate([z[k] for z in zs]) for k in ("tok", "pos", "src", "policy", "lpost")}
MODELS = [("B", "b", a.runs, False), ("B0", "b0", a.runs, False)]
if a.runs_vel:
    uv = np.concatenate([np.load(f.replace(".npz", "_vel.npz"))["uv"] for f in fs])
    d["tokv"] = np.concatenate([d["tok"], uv / VSCALE[a.set]], -1).astype(np.float32)
    d["uv"] = uv
    MODELS += [("Bvel", "b", a.runs_vel, True), ("B0vel", "b0", a.runs_vel, True)]
gx, gy = zs[0]["grid_x"], zs[0]["grid_y"]
GX, GY = np.meshgrid(gx, gy, indexing="ij")
cx, cy = GX.ravel(), GY.ravel()
E = len(d["src"])
j = np.argmin((cx[None] - d["src"][:, :1]) ** 2 + (cy[None] - d["src"][:, 1:]) ** 2, 1)

lq = np.zeros((E, 1 + len(MODELS), len(NS), cx.size))
lp = d["lpost"].astype(np.float64)
lq[:, 0] = lp - np.logaddexp.reduce(lp, axis=2, keepdims=True)
for mi, (_, m, runs, vel) in enumerate(MODELS, start=1):
    tok = d["tokv"] if vel else d["tok"]
    net = PostNet(cx, cy, physics=(m == "b"), vel=vel)
    net.load_state_dict(torch.load(f"{a.pn}/{runs}/{a.set}_{m}_n{a.n_train}_r0.pt", map_location="cpu"))
    net.eval()
    with torch.no_grad():
        for s in range(0, E, 16):
            sl = slice(s, min(s + 16, E))
            lq[sl, mi] = net(torch.tensor(tok[sl]), torch.tensor(d["pos"][sl]), NS,
                             torch.tensor(d["lpost"][sl].astype(np.float32))).double().numpy()
q = np.exp(lq)
qt = q[np.arange(E), :, :, j]                                            # (E, 3, N)
levels = (q * (q > qt[..., None])).sum(-1)
mean = np.stack([q @ cx, q @ cy], -1)
err = np.hypot(*(mean - d["src"][:, None, None, :]).transpose(3, 0, 1, 2))
e0 = np.arange(E) if a.map_all else np.where(d["policy"] == 0)[0]
qmap = q[e0][:, :, [NS.index(n) for n in MAPN]].astype(np.float16)
os.makedirs(a.out, exist_ok=True)
names = ["M1"] + [m[0] for m in MODELS]
np.savez_compressed(f"{a.out}/{a.set}_maps.npz", uv=d.get("uv", np.zeros(0)), levels=levels, qtrue=qt, err=err, q=qmap, map_episodes=e0, map_ns=MAPN, ns=NS,
                    pos=d["pos"], tok=d["tok"], src=d["src"], policy=d["policy"], grid_x=gx, grid_y=gy,
                    methods=np.array(names))
for mi, m in enumerate(names):
    print(f"{a.set} {m}: n80 cov50 {np.mean(levels[:, mi, -1] < 0.5):.2f} cov90 {np.mean(levels[:, mi, -1] < 0.9):.2f} "
          f"P01 {np.mean(err[:, mi, -1] < 0.1):.2f}", flush=True)

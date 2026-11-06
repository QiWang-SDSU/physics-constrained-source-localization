"""E54: which information in the measurement sequence the learned posterior uses (sensitivity of the posterior to the
tokens that enter the GRU).

For each held-out episode, the posterior q = q(. | H_80) is differentiated with respect to the tokens X (81 x F) that
enter the GRU encoder, with the sensor positions in the decoder offsets and the decoder's direct inputs (d_k, z_k[, u_k])
held fixed ("GRU path"); separately with respect to the direct inputs of the decoder ("direct path"). Inputs are
standardised by their std over the test tokens. The Fisher-weighted sensitivity
    G = E_episodes[ J^T F J ] = E_episodes E_{j ~ q}[ grad log q(j) grad log q(j)^T ]
is estimated with M samples j ~ q per episode; a small input change dX changes the posterior by KL ~ dX^T G dX / 2.
G_star uses the true cell instead (grad log q(j*)). Stored: the F x F feature blocks summed over steps, the per-step
traces (sensitivity versus step k), and the 81 x 81 step-step block of z (detections only).
z_k at non-detections and the binary flag d_k are excluded from the gradients (z = 0 there by construction); the flag is
handled by flipping it, one step at a time (detection -> non-detection with z = 0; non-detection -> detection at the
median detected z), both in all inputs and in the GRU input only, recording the change of log q(j*) and KL(q || q').
Output: {out}/{set}_{tag}_sens.npz.
  python scripts/checks/gru_sensitivity.py --set strong --model b0 --out $RLS_OUT/E54
"""
import argparse, glob, os, sys, time
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from rlscalar.search.postnet import PostNet

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="strong")
ap.add_argument("--model", default="b0", choices=["b0", "b"])
ap.add_argument("--vel", action="store_true")
ap.add_argument("--pn", default=os.environ.get("RLS_DATA", "data") + "/postnet")
ap.add_argument("--M", type=int, default=16, help="samples j ~ q per episode")
ap.add_argument("--n", type=int, default=0, help="number of test episodes (0 = all)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
VSCALE = {"strong": 0.18, "weak": 0.09}
NAMES = ["d", "z", "x", "y", "t", "dx", "dy"] + (["u", "v"] if a.vel else [])
RD = [0, 1, 7, 8] if a.vel else [0, 1]                       # token columns that also enter the decoder directly

fs = sorted(glob.glob(f"{a.pn}/data/{a.set}_test_[0-9][0-9][0-9].npz"))
T = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("tok", "pos", "src", "lpost")}
if a.vel:
    uv = np.concatenate([np.load(f.replace(".npz", "_vel.npz"))["uv"] for f in fs])
    T["tok"] = np.concatenate([T["tok"], uv / VSCALE[a.set]], -1).astype(np.float32)
z0 = np.load(fs[0])
GX, GY = np.meshgrid(z0["grid_x"], z0["grid_y"], indexing="ij")
cx, cy = GX.ravel(), GY.ravel()
E = len(T["src"]) if a.n == 0 else a.n
jstar = np.argmin((cx[None] - T["src"][:, :1]) ** 2 + (cy[None] - T["src"][:, 1:]) ** 2, 1)

net = PostNet(cx, cy, physics=(a.model == "b"), vel=a.vel).to(dev)
run = "runs_vel" if a.vel else "runs_fp32"
net.load_state_dict(torch.load(f"{a.pn}/{run}/{a.set}_{a.model}_n9500_r0.pt", map_location=dev))
net.train()                                                  # no dropout / batch norm; cuDNN GRU backward needs it
for p in net.parameters():
    p.requires_grad_(False)

tok_all = T["tok"].reshape(-1, T["tok"].shape[-1])
det_all = tok_all[:, 0] > 0.5
std = tok_all.std(0)
std[1] = tok_all[det_all, 1].std()                           # z: spread of detected values only
std[0] = 1.0
zmed = float(np.median(tok_all[det_all, 1]))
F, K = len(NAMES), T["tok"].shape[1]
S = torch.tensor(std, device=dev)


def logq80(tg, td, pos, lp):
    """log q(. | H_80) with GRU input tg and decoder direct input td (both (B, K, F)): (B, J)."""
    g, _ = net.gru(net.enc(tg))
    g_prev = torch.cat([torch.zeros_like(g[:, :1]), g[:, :-1]], 1)
    rd = net.f_rd(torch.cat([td[..., RD], g_prev], -1))
    off = (pos[:, :, None, :] - net.cells[None, None]) / net.scale
    S_ = net.f_rest(net.f_off(off) + rd[:, :, None, :])[..., 0].sum(1)
    logits = S_ if not net.physics else lp + net.alpha * S_
    return torch.log_softmax(logits, -1)


Gg, Gd = np.zeros((2, F, F)), np.zeros((2, len(RD), len(RD)))    # [Fisher, true cell]
lag_g, lag_d = np.zeros((2, K)), np.zeros((2, K))
Gz = np.zeros((2, K, K))
flip = np.full((E, K, 2, 2), np.nan, np.float32)                 # [all inputs, GRU only] x [d log q(j*), KL]
rng = np.random.default_rng(0)
t0 = time.time()
for e in range(E):
    tok = torch.tensor(T["tok"][e:e + 1], device=dev)
    pos = torch.tensor(T["pos"][e:e + 1], device=dev)
    lp = torch.tensor(T["lpost"][e:e + 1, -1].astype(np.float32), device=dev)
    det = torch.tensor(T["tok"][e, :, 0] > 0.5, device=dev)
    tg, td = tok.clone().requires_grad_(True), tok.clone().requires_grad_(True)
    lq = logq80(tg, td, pos, lp)[0]
    q = lq.detach().exp().double().cpu().numpy()
    js = list(rng.choice(len(q), a.M, p=q / q.sum())) + [int(jstar[e])]
    for i, j in enumerate(js):
        gg, gd = torch.autograd.grad(lq[j], (tg, td), retain_graph=(i < len(js) - 1))
        gg, gd = gg[0] * S, gd[0][:, RD] * S[RD]                    # standardised inputs
        gg[:, 0] = 0; gd[:, 0] = 0                                  # binary flag: handled by flipping
        gg[:, 1] *= det; gd[:, 1] *= det                            # z only where detected
        gg, gd = gg.double().cpu().numpy(), gd.double().cpu().numpy()
        w, s = (1.0 / a.M, 0) if i < a.M else (1.0, 1)
        Gg[s] += w * np.einsum("kf,kg->fg", gg, gg); Gd[s] += w * np.einsum("kf,kg->fg", gd, gd)
        lag_g[s] += w * (gg ** 2).sum(1); lag_d[s] += w * (gd ** 2).sum(1)
        zt = gg[:, 1] + gd[:, 1]                                    # z through both paths
        Gz[s] += w * np.outer(zt, zt)
    if e == 0:
        torch.cuda.synchronize() if dev == "cuda" else None
        print("dev", dev, "gradients of episode 0: %.1f s" % (time.time() - t0), flush=True)
    with torch.no_grad():                                           # flip the detection flag, one step at a time
        base = logq80(tok, tok, pos, lp)[0]
        pb = base.exp()
        for mode in (0, 1):
            V = tok.repeat(K, 1, 1)
            ks = torch.arange(K, device=dev)
            was = V[ks, ks, 0] > 0.5
            V[ks, ks, 0] = (~was).float()
            V[ks, ks, 1] = torch.where(was, torch.zeros_like(V[ks, ks, 1]), torch.full_like(V[ks, ks, 1], zmed))
            for c in range(0, K, 9):
                sl = slice(c, min(c + 9, K))
                n = sl.stop - sl.start
                tgv = V[sl]; tdv = V[sl] if mode == 0 else tok.repeat(n, 1, 1)
                l2 = logq80(tgv, tdv, pos.repeat(n, 1, 1), lp.repeat(n, 1))
                flip[e, sl, mode, 0] = (l2[:, jstar[e]] - base[jstar[e]]).cpu().numpy()
                flip[e, sl, mode, 1] = (pb * (base - l2)).sum(-1).cpu().numpy()
    if e % 50 == 0:
        print(e, "%.1f s" % (time.time() - t0), flush=True)

Gg /= E; Gd /= E; lag_g /= E; lag_d /= E; Gz /= E
off = T["pos"][:E] - T["src"][:E, None, :]                         # sensor offset from the true source
tag = a.model + ("_vel" if a.vel else "")
os.makedirs(a.out, exist_ok=True)
np.savez(f"{a.out}/{a.set}_{tag}_sens.npz", names=np.array(NAMES), rd=np.array(RD), std=std, zmed=zmed, Gg=Gg, Gd=Gd,
         lag_g=lag_g, lag_d=lag_d, Gz=Gz, flip=flip, det=T["tok"][:E, :, 0] > 0.5, off=off.astype(np.float32), E=E)
w, V = np.linalg.eigh(Gg[0])
print(a.set, tag, "E", E, "GRU-path Fisher eigvals", np.round(w[::-1] / w.sum(), 3))
for i in range(3):
    print("  dir", i, dict(zip(NAMES, np.round(V[:, -1 - i], 2))))
print("  diag share GRU path", dict(zip(NAMES, np.round(np.diag(Gg[0]) / np.trace(Gg[0]), 3))))
print("  direct path diag", dict(zip([NAMES[r] for r in RD], np.round(np.diag(Gd[0]), 3))), "GRU trace", round(np.trace(Gg[0]), 3))
print("  flip |dlogq*| mean all/GRU-only, det->non / non->det:",
      [round(float(np.nanmean(np.abs(flip[:, :, m, 0][(T['tok'][:E, :, 0] > 0.5) == d]))), 3) for m in (0, 1) for d in (True, False)])

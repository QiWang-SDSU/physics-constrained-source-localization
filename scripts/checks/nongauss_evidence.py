"""E52: distributional evidence for the non-Gaussian measurements and for how the learned posterior copes with them.

(a) DNS log-concentration at fixed offsets (xi, eta) from the source, pooled over all records and plumes of the training
    runs (seeds 1-5), with the measurement noise of DNSWorld, against the physics model (intermittency factor gamma and
    in-plume Gaussian N(m, s^2)) and the corrected model at the same offsets.
(b) Randomised PIT of the held-out test measurements (postnet data, strong/weak_test_*.npz) under the physics and the
    corrected model at the true source: u = F(z) for a detection, u ~ U(0, F(log m_min)) for a non-detection; uniform if
    the model's distribution of the measurement is right.
(c) PIT of the posterior: mass of the cells more probable than the true cell at n = 80 on the test episodes, for the
    physics posterior and the learned posteriors B and B0 (runs_fp32, 9500 episodes); uniform if calibrated.
(d) Evidence of one detection for candidate A over candidate B, log p(z | A) - log p(z | B), as a function of z, where
    the sensor is at offset A or B from the candidate: DNS (histograms of (a)), physics and corrected models (kappa = 1),
    and the learned B0 term f(offset, 1, z, g) averaged over GRU states g from test histories.
Output: {out}/{set}_nongauss.npz.  python scripts/checks/nongauss_evidence.py --set strong --out $RLS_OUT/E52
"""
import argparse, glob, json, os, sys
from dataclasses import replace
import numpy as np
from scipy.special import ndtr
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from rlscalar.search.core import Params, Belief

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="strong")
ap.add_argument("--root", default=os.environ.get("RLS_DATA", "data") + "/dns")
ap.add_argument("--fitdir", default=os.environ.get("RLS_DATA", "data") + "/dns/fits")
ap.add_argument("--pn", default=os.environ.get("RLS_DATA", "data") + "/postnet")
ap.add_argument("--out", required=True)
a = ap.parse_args()
ZI = ("zi_a0", "zi_a1", "zi_c0", "zi_c1", "leak_g", "leak_mu", "leak_s")
OFFS = np.array([[0.5, 0.0], [1.5, 0.0], [1.5, 0.3]])     # (xi, eta) offsets of the sensor from the source; A, B, C


def m1_params():
    """As postnet_closedloop.py."""
    fit = json.load(open(f"{a.fitdir}/{a.set}_fit_taylor_src.json"))
    sur = fit["surrogate"]
    z2 = json.load(open(f"{a.fitdir}/{a.set}_lik_taylor_src.json"))["zi2"]
    return Params(U=sur["U"], K=sur["K"], sigma=z2["sigma"], tau=sur["tau"], L=sur["L"],
                  Q=float(np.median([q["mass_flux"] for q in fit["plumes"]])), T_L=sur["T_L"],
                  s0sq=sur.get("s0sq_taylor", 0.0), src_r=sur.get("src_r", 0.0), n_steps=80, snap_src=True,
                  temper=True, obs="zi", **{k: z2[k] for k in ZI})


def m3_params():
    """As postnet_closedloop.py."""
    base = f"{a.fitdir}/nn/{a.set}_N1000000_map_fix_traj"
    fit, z2 = json.load(open(f"{base}_fit.json")), json.load(open(f"{base}_lik.json"))["zi2"]
    sur = fit["surrogate"]
    pd = dict(U=sur["U"], K=sur["K"], tau=sur["tau"], L=sur["L"], Q=float(np.median([q["mass_flux"] for q in fit["plumes"]])),
              s0sq=max(0.0, float(np.median([q["s0sq"] for q in fit["plumes"]]))), n_steps=80, snap_src=True, temper=True)
    if sur.get("T_L"):
        pd.update(T_L=sur["T_L"], s0sq=sur.get("s0sq_taylor", 0.0))
    if sur.get("src_r"):
        pd.update(src_r=sur["src_r"])
    pd.update(obs="zi", sigma=z2["sigma"], **{k: z2[k] for k in ZI})
    return replace(Params(**pd, nn_file=f"{base}_hyb.npz"), mean_model="nn", obs="zi")


P1, P3 = m1_params(), m3_params()
B1, B3 = Belief(P1, correlated=False), Belief(P3, correlated=False)
THR = np.log(P1.m_min)
J0 = int(np.argmin(B1.cx ** 2 + B1.cy ** 2))               # a candidate cell; offsets are taken from it
X0, Y0 = B1.cx[J0], B1.cy[J0]
rng = np.random.default_rng(0)


def comp(b, x, y, j):
    g, m, s = b.components(x, y, 0.0, np.array([j]))
    return float(g[0]), float(m[0]), float(s[0])


# ---------------------------------------------------------------- (a) DNS log c at fixed offsets
zs = []
for o in OFFS:
    vals = []
    for f in sorted(glob.glob(f"{a.root}/{a.set}/seed0[1-5].npz")):
        z = np.load(f)
        P = json.loads(str(z["params"]))
        c, sx, sy = z["c_sub"], z["sub_x"], z["sub_y"]
        for p, (xs, ys) in enumerate(P["sources"]):
            X, Y = xs + o[0], (ys + o[1]) % P["Ly"]
            fi, fj = (X - sx[0]) / (sx[1] - sx[0]), (Y - sy[0]) / (sy[1] - sy[0])
            i0, j0 = int(fi), int(fj)
            u, v = fi - i0, fj - j0
            j1 = (j0 + 1) % len(sy)
            cc = c[:, p].astype(np.float32)
            val = ((1 - u) * ((1 - v) * cc[:, i0, j0] + v * cc[:, i0, j1]) + u * ((1 - v) * cc[:, i0 + 1, j0] + v * cc[:, i0 + 1, j1]))
            vals.append(val)
    v = np.concatenate(vals)
    zz = np.full(v.shape, -np.inf)
    zz[v > 0] = np.log(v[v > 0]) + np.sqrt(P1.nug) * rng.standard_normal(int((v > 0).sum()))
    zs.append(zz)
NS = min(len(z) for z in zs)
zs = np.stack([z[:NS] for z in zs])                         # (3, NS) log c with noise; -inf for c = 0
mod = np.array([[comp(b, X0 + o[0], Y0 + o[1], J0) for o in OFFS] for b in (B1, B3)])   # (2 models, 3 offsets, 3)

# ---------------------------------------------------------------- (b) PIT of test measurements, (c) posterior PIT
fs = sorted(glob.glob(f"{a.pn}/data/{a.set}_test_[0-9][0-9][0-9].npz"))
T = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("tok", "pos", "zraw", "src", "lpost")}
E = len(T["src"])
jstar = np.array([int(np.argmin((B1.cx - s[0]) ** 2 + (B1.cy - s[1]) ** 2)) for s in T["src"]])
pit = np.zeros((2, E, 81))
for e in range(E):
    for k in range(81):
        x, y = T["pos"][e, k]
        det = T["tok"][e, k, 0] > 0.5
        z = THR + 5 * T["tok"][e, k, 1]
        for i, b in enumerate((B1, B3)):
            g, m, s = comp(b, x, y, jstar[e])
            F0 = 1 - g + g * ndtr((THR - m) / s)            # P(non-detection)
            pit[i, e, k] = F0 + g * (ndtr((z - m) / s) - ndtr((THR - m) / s)) if det else rng.uniform(0, F0)

import torch
from rlscalar.search.postnet import PostNet
nets = {}
for m in ("b", "b0"):
    n = PostNet(B1.cx, B1.cy, physics=(m == "b"))
    n.load_state_dict(torch.load(f"{a.pn}/runs_fp32/{a.set}_{m}_n9500_r0.pt", map_location="cpu"))
    nets[m] = n.eval()


def level(lq, j):
    q = np.exp(lq - np.logaddexp.reduce(lq))
    return float(q[q > q[j]].sum())


lev = np.zeros((3, E))                                      # physics, B, B0 at n = 80
with torch.no_grad():
    for e in range(E):
        tok, pos = torch.tensor(T["tok"][e:e + 1]), torch.tensor(T["pos"][e:e + 1])
        lp = torch.tensor(T["lpost"][e:e + 1].astype(np.float32))
        lev[0, e] = level(T["lpost"][e, -1].astype(np.float64), jstar[e])
        for i, m in enumerate(("b", "b0")):
            lev[1 + i, e] = level(nets[m](tok, pos, [80], lp[:, -1:])[0, 0].double().numpy(), jstar[e])

# ---------------------------------------------------------------- (d) evidence of one detection, A over B
zg = np.linspace(THR, THR + 12, 121)
net = nets["b0"]
with torch.no_grad():
    tok = torch.tensor(T["tok"])
    g, _ = net.gru(net.enc(tok))
    gp = torch.cat([torch.zeros_like(g[:, :1]), g[:, :-1]], 1).reshape(-1, g.shape[-1])
    gp = gp[torch.as_tensor(rng.choice(len(gp), 2000, replace=False))]          # GRU states from test histories
    lrn = np.zeros((2, len(zg)))
    for i, o in enumerate(OFFS[:2]):
        off = net.f_off(torch.tensor(o, dtype=torch.float32) / net.scale)       # sensor - candidate = (xi, eta)
        for iz, z in enumerate(zg):
            rd = net.f_rd(torch.cat([torch.tensor([[1.0, float(z - THR) / 5]], dtype=torch.float32).expand(len(gp), 2), gp], -1))
            lrn[i, iz] = float(net.f_rest(off + rd)[:, 0].mean())
os.makedirs(a.out, exist_ok=True)
np.savez(f"{a.out}/{a.set}_nongauss.npz", offs=OFFS, z_dns=zs.astype(np.float32), model=mod, thr=THR, nug=P1.nug,
         pit=pit.astype(np.float32), level=lev, zg=zg, learned=lrn)
print(a.set, "samples per offset", NS, "P(det) DNS", [(np.isfinite(z) & (z >= THR)).mean().round(3) for z in zs],
      "model (g, m, s)", mod.round(2).tolist(), flush=True)
for i, nm in enumerate(("physics", "corrected")):
    h = np.histogram(pit[i].ravel(), 10, (0, 1))[0] / pit[i].size * 10
    print(nm, "PIT density per decile", h.round(2), flush=True)
for i, nm in enumerate(("physics", "B", "B0")):
    print(nm, "posterior level deciles", (np.histogram(lev[i], 10, (0, 1))[0] / E * 10).round(2), flush=True)

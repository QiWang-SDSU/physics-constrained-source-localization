"""E31: data efficiency of the physics vs the learned observation model, measured in NUMBER OF READINGS.

E30 counted data in flow time, but every recorded instant carries the whole field, so even 0.2% of the record was
0.5 M readings. Here both models see the same N point readings (random (t, x, y) of the training seeds 1-5, region
-2.5 < dx < 7, |dy| < 2), N = 10^2 ... 10^6, and both are fitted by maximum likelihood on those readings only:
  physics: 11 parameters (K, T_L, s0^2 of the adjoint mean field; zi2 a0, a1, c0, c1, s; leak g, mu, s), U and the
           source radius fixed (measured mean wind, known source geometry), generic initial values;
  nn:      the E30 MLP (2-64-64-3), 10% validation split, early stopping, generic initialisation.
  hyb:     the same MLP with two more inputs, the adjoint log-mean of the physics fit at the same N and its support
           flag (physics prior + learned correction).
Test: per-point NLL on held-out seeds 6-8 of both forcings. Three random subsets per N. Replicate 0 of every N
writes {set}_N{N}_{nn,hyb}.npz (core.nn_components format) and {set}_N{N}_{fit,lik}.json (search_sweep formats).
Usage (Fermi): python scripts/checks/dns2_data_efficiency.py --set strong --out $RLS_DATA/dns/fits/nn/strong_eff.json
"""
import argparse, json, os, sys, time, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import ndtr
from rlscalar.search.core import Params, log_cbar, nn_features

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="strong")
ap.add_argument("--root", default=os.environ.get("RLS_DATA", "data") + "/dns")
ap.add_argument("--fitdir", default=os.environ.get("RLS_DATA", "data") + "/dns/fits")
ap.add_argument("--Ns", type=int, nargs="+", default=[100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000])
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--m_min", type=float, default=1e-2)
ap.add_argument("--ntest", type=int, default=200_000)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--map", action="store_true", help="E31b: MAP physics fit with generic order-of-magnitude priors")
ap.add_argument("--fixmean", action="store_true",
                help="E35: mean field (K, T_L, s0^2) fixed at the plume-width (Taylor) fit; only the 8 zi2/leak coefficients from the N readings")
ap.add_argument("--sampling", default="random", choices=["random", "traj"],
                help="E36: 'traj' = readings along sensor paths (half zigzag survey, half persistent random walk, 80 moves, "
                     "search-task geometry, training seeds); subsets are whole paths, so the readings stay correlated")
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(a.threads)
thr, L2P = np.log(a.m_min), np.log(2 * np.pi)
base = json.load(open(f"{a.fitdir}/{a.set}_fit_taylor_src.json"))["surrogate"]
U, SRC_R = base["U"], base.get("src_r", 0.1)


def load(set_, seeds, nmax, rng):
    DX, DY, LC = [], [], []
    per = max(1, nmax // (len(seeds) * 4))
    for s in seeds:
        z = np.load(f"{a.root}/{set_}/seed{s:02d}.npz")
        P = json.loads(str(z["params"]))
        X, Y = np.meshgrid(z["sub_x"], z["sub_y"], indexing="ij")
        c = z["c_sub"]
        for i, (xs, ys) in enumerate(P["sources"]):
            xi, dy = X - xs, (Y - ys + P["Ly"] / 2) % P["Ly"] - P["Ly"] / 2
            kk = np.flatnonzero(((xi > -2.5) & (xi < 7.0) & (np.abs(dy) < 2.0)).ravel())
            sel = rng.choice(c.shape[0] * kk.size, per, replace=False)
            tt, jj = sel // kk.size, kk[sel % kk.size]
            DX.append(xi.ravel()[jj]); DY.append(dy.ravel()[jj])
            LC.append(np.log(np.maximum(c[:, i].reshape(c.shape[0], -1)[tt, jj].astype(np.float32), 1e-30)))
    return np.concatenate(DX), np.concatenate(DY), np.concatenate(LC)


def load_traj(set_, seeds, nmax, rng):
    """Readings along sensor paths in the training seeds. Returns (dx, dy, log c) and the path index of each
    reading. Censored readings get log c = -30 (below threshold)."""
    from rlscalar.search.seek import SeekTasks, Episode
    from rlscalar.search.policies import Zigzag
    p = Params(n_steps=80, snap_src=True)
    tasks = SeekTasks(p, [f"{a.root}/{set_}/seed{s:02d}.npz" for s in seeds])
    DX, DY, LC, PID = [], [], [], []
    i = 0
    while sum(map(len, LC)) < nmax:
        ep = Episode(tasks, 5_000_000 + i, r_succ=-1.0)
        zz = Zigzag(); zz.reset(); heading = int(rng.integers(8))
        rec = [(ep.pos.copy(), ep.z)]
        for k in range(p.n_steps):
            if i % 2 == 0:
                act = zz.act(dict(pos=ep.pos))
            else:
                if rng.uniform() < 0.2:
                    heading = int(rng.integers(8))
                act = heading
            z, _, _ = ep.step(act)
            rec.append((ep.pos.copy(), z))
        pos = np.array([r[0] for r in rec])
        DX.append(pos[:, 0] - ep.src[0]); DY.append(pos[:, 1] - ep.src[1])
        LC.append(np.array([-30.0 if r[1] is None else r[1] for r in rec])); PID.append(np.full(len(rec), i))
        i += 1
    return np.concatenate(DX), np.concatenate(DY), np.concatenate(LC), np.concatenate(PID)


def ppnll(g, m, s, lc):
    det = lc >= thr
    with np.errstate(divide="ignore", over="ignore"):
        lcen = np.log(np.clip((1 - g) + g * ndtr((thr - m) / s), 1e-300, None))
        ld = np.log(np.clip(g, 1e-300, None)) - 0.5 * ((lc - m) / s) ** 2 - np.log(s) - 0.5 * L2P
    return -np.where(det, ld, lcen)


sig = lambda x: 1 / (1 + np.exp(-np.clip(x, -50, 50)))


def phys_eval(th, dx, dy, lc):
    lK, lT, lS, a0, a1, c0, c1, ls, lg, lmu, lsd = th
    PM = Params(U=U, K=np.exp(lK), Q=1.0, T_L=np.exp(lT), s0sq=np.exp(lS), src_r=SRC_R)
    lb = log_cbar(0.0, 0.0, np.maximum(dx, -5.0), dy, PM)
    fin = np.isfinite(lb); lbf = np.maximum(np.where(fin, lb, 0.0), -30.0)
    g = np.where(fin, sig(a0 + a1 * lbf), sig(lg))
    m = np.where(fin, c0 + c1 * lbf, lmu)
    s = np.where(fin, np.exp(np.clip(ls, -5, 3)), np.exp(np.clip(lsd, -5, 3)))
    return ppnll(g, m, s, lc)


# E31b priors (independent Gaussians on the fitted coordinates): mean field log K ~ N(log 0.05, 1),
# log T_L ~ N(log 2, 1), log s0^2 ~ N(log 0.02, 1); zi2 and leak coefficients N(generic start, 2^2), log s ~ N(log 1.5, 1).
PRIOR_MU = np.array([np.log(0.05), np.log(2.0), np.log(0.02), 1.0, 1.0, -0.5, 1.0, np.log(1.5), -1.5, -4.0, np.log(0.5)])
PRIOR_SD = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0])


FIX = np.log([base["K"], base["T_L"], base["s0sq_taylor"]])


def fit_phys(dx, dy, lc):
    if a.fixmean:                                    # E35: optimise the 8 observation coefficients only
        pen8 = (lambda q: 0.5 * np.sum(((q - PRIOR_MU[3:]) / PRIOR_SD[3:]) ** 2) / lc.size) if a.map else (lambda q: 0.0)
        g = lambda q: float(phys_eval(np.concatenate([FIX, q]), dx, dy, lc).mean()) + pen8(np.asarray(q))
        best = None
        for x0 in ([1.0, 1.0, -0.5, 1.0, np.log(1.5), -1.5, -4.0, np.log(0.5)], [0.0, 0.5, -1.0, 0.5, 0.0, -3.0, -3.0, 0.0]):
            r = minimize(g, x0, method="Nelder-Mead", options=dict(maxiter=6000, maxfev=6000, xatol=1e-4, fatol=1e-7))
            r = minimize(g, r.x, method="Powell", options=dict(maxiter=6000, xtol=1e-4, ftol=1e-8))
            if best is None or r.fun < best.fun:
                best = r
        return np.concatenate([FIX, best.x])
    pen = (lambda th: 0.5 * np.sum(((th - PRIOR_MU) / PRIOR_SD) ** 2) / lc.size) if a.map else (lambda th: 0.0)
    f = lambda th: float(phys_eval(th, dx, dy, lc).mean()) + pen(np.asarray(th))
    starts = [[np.log(0.05), 0.0, np.log(0.02), 1.0, 1.0, -0.5, 1.0, np.log(1.5), -1.5, -4.0, np.log(0.5)],
              [np.log(0.02), np.log(3.0), np.log(0.01), 0.0, 0.5, -1.0, 0.5, 0.0, -3.0, -3.0, 0.0]]
    best = None
    for x0 in starts:
        r = minimize(f, x0, method="Nelder-Mead", options=dict(maxiter=8000, maxfev=8000, xatol=1e-4, fatol=1e-7))
        r = minimize(f, r.x, method="Powell", options=dict(maxiter=8000, xtol=1e-4, ftol=1e-8))
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def fit_nn(F, lc, seed):
    torch.manual_seed(seed)
    X = torch.tensor(F, dtype=torch.float32)
    Z = torch.tensor(lc, dtype=torch.float32); D = Z >= thr
    net = torch.nn.Sequential(torch.nn.Linear(F.shape[1], 64), torch.nn.Tanh(), torch.nn.Linear(64, 64), torch.nn.Tanh(),
                              torch.nn.Linear(64, 3))
    with torch.no_grad():
        net[-1].bias.copy_(torch.tensor([0.0, -2.0, 0.5]))
    n = X.shape[0]; perm = torch.randperm(n); nv = max(10, n // 10)
    iv, it = perm[:nv], perm[nv:]
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    bs = int(min(16384, max(32, it.numel() // 8)))

    def nll(idx):
        o = net(X[idx]); al, m, ls = o[:, 0], o[:, 1], o[:, 2].clamp(-5, 3)
        z, d = Z[idx], D[idx]
        lcen = torch.logaddexp(torch.nn.functional.logsigmoid(-al),
                               torch.nn.functional.logsigmoid(al) + torch.special.log_ndtr((thr - m) / ls.exp()))
        ldet = torch.nn.functional.logsigmoid(al) - 0.5 * ((z - m) / ls.exp()) ** 2 - ls - 0.5 * L2P
        return -torch.where(d, ldet, lcen).mean()
    best, bad, state = np.inf, 0, copy.deepcopy(net.state_dict())
    for ep in range(2000):
        for b in it[torch.randperm(it.numel())].split(bs):
            opt.zero_grad(); nll(b).backward(); opt.step()
        with torch.no_grad():
            v = float(nll(iv))
        if v < best - 1e-4:
            best, bad, state = v, 0, copy.deepcopy(net.state_dict())
        elif (bad := bad + 1) >= 15:
            break
    net.load_state_dict(state)
    return net


def nn_eval(net, F, lc):
    with torch.no_grad():
        o = net(torch.tensor(F, dtype=torch.float32)).numpy().astype(np.float64)
    return ppnll(sig(o[:, 0]), o[:, 1], np.exp(np.clip(o[:, 2], -5, 3)), lc)


def weights(net, meta):
    lin = [m for m in net if isinstance(m, torch.nn.Linear)]
    W = {f"W{i}": l.weight.detach().numpy().T.astype(np.float64) for i, l in enumerate(lin)}
    W.update({f"b{i}": l.bias.detach().numpy().astype(np.float64) for i, l in enumerate(lin)})
    W.update(n_layers=len(lin), **meta)
    return W


def phys_files(th, tag):
    """Physics fit at this N in the search_sweep formats (correlation parameters from the full fit)."""
    lK, lT, lS, a0, a1, c0, c1, ls, lg, lmu, lsd = th
    fit = json.load(open(f"{a.fitdir}/{a.set}_fit_taylor_src.json"))
    fit["surrogate"].update(K=float(np.exp(lK)), T_L=float(np.exp(lT)), s0sq_taylor=float(np.exp(lS)), src_r=SRC_R)
    fit["plumes"] = [dict(mass_flux=1.0, s0sq=0.0)]
    zi2 = dict(zi_a0=a0, zi_a1=a1, zi_c0=c0, zi_c1=c1, sigma=np.exp(ls), leak_g=sig(lg), leak_mu=lmu, leak_s=np.exp(lsd))
    json.dump(fit, open(f"{OUT}/{tag}_fit.json", "w"), indent=1)
    json.dump(dict(zi2={k: float(v) for k, v in zi2.items()}), open(f"{OUT}/{tag}_lik.json", "w"), indent=1)


OUT = os.path.dirname(os.path.abspath(a.out))
BASE = dict(scale=np.array([3.0, 1.0]))
if a.sampling == "traj":
    *pool, PATH = load_traj(a.set, [1, 2, 3, 4, 5], max(a.Ns) * 2, np.random.default_rng(0))
    pool = tuple(pool)
else:
    pool = load(a.set, [1, 2, 3, 4, 5], max(a.Ns) * 2, np.random.default_rng(0))
tests = {st: load(st, [6, 7, 8], a.ntest, np.random.default_rng(1)) for st in ("strong", "weak")}
rows = []
for N in a.Ns:
    for r in range(a.reps if N < max(a.Ns) else 1):
        t0 = time.time()
        if a.sampling == "traj":                       # whole paths, in random order, truncated to N readings
            order = np.random.default_rng(100 + r).permutation(PATH.max() + 1)
            idx = np.concatenate([np.flatnonzero(PATH == q) for q in order[: N // 81 + 1]])[:N]
        else:
            idx = np.random.default_rng(100 + r).choice(pool[0].size, N, replace=False)
        dx, dy, lc = (v[idx] for v in pool)
        th = fit_phys(dx, dy, lc)
        HYB = dict(BASE, phys=np.array([U, np.exp(th[0]), np.exp(th[1]), np.exp(th[2]), SRC_R, 1.0]))
        net = fit_nn(nn_features(0.0, 0.0, dx, dy, BASE), lc, seed=r)
        hyb = fit_nn(nn_features(0.0, 0.0, dx, dy, HYB), lc, seed=r)
        row = dict(N=N, rep=r, n_det=int((lc >= thr).sum()), phys_params=[float(v) for v in th])
        for st, (tx, ty, tl) in tests.items():
            row[f"{st}_phys"] = float(phys_eval(th, tx, ty, tl).mean())
            row[f"{st}_nn"] = float(nn_eval(net, nn_features(0.0, 0.0, tx, ty, BASE), tl).mean())
            row[f"{st}_hyb"] = float(nn_eval(hyb, nn_features(0.0, 0.0, tx, ty, HYB), tl).mean())
        if r == 0:
            tag = f"{a.set}_N{N}" + ("_map" if a.map else "") + ("_fix" if a.fixmean else "") + ("_traj" if a.sampling == "traj" else "")
            np.savez(f"{OUT}/{tag}_nn.npz", **weights(net, BASE))
            np.savez(f"{OUT}/{tag}_hyb.npz", **weights(hyb, HYB))
            phys_files(th, tag)
        row["wall"] = time.time() - t0
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "phys_params"}), flush=True)
        rows.append(row)
        json.dump(dict(set=a.set, U=U, src_r=SRC_R, rows=rows), open(a.out, "w"), indent=1)

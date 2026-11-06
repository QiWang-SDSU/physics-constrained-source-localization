"""DNS2: (1) fit the planner's surrogate parameters on TRAINING seeds only; (2) out-of-sample test of
the moving-sensor decorrelation law with longer windows than E1b allowed (recurrence 12.6).

(1) Mean plume: for each plume i, fit the time-mean crosswind variance sigma_y^2(xi) = 2 K xi / U + s0^2
    over xi in [0.5, 6]. Also report the mass flux U * int cbar dy (should be about 1, the source rate).
    Fluctuation: log-anomaly phi' = log c - <log c> at in-plume points (cbar > 20 m_min), with
    sigma^2 = var(phi'). Report skewness, excess kurtosis, and the intermittency fraction
    P(c < m_min | cbar > 20 m_min).
    Correlation (tau, L) of the separable Gaussian closure: fitted to along-path correlations of the
    drift (v = U) and stationary cases at 3 stations, then pooled.
(2) Law test: T_int predicted vs measured for held-out velocities on TEST seeds, window T.

Usage (Fermi): python scripts/checks/dns2_fit_and_law.py --set strong --train 1 2 3 4 5 --test 6 7 8 \
                  --T 6.0 --out $RLS_DATA/dns/strong_fit.json
"""
import argparse, glob, json, os
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.stats import skew, kurtosis

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="strong")
ap.add_argument("--root", default=os.environ.get("RLS_DATA", "data") + "/dns")
ap.add_argument("--train", type=int, nargs="+", default=[1, 2, 3, 4, 5])
ap.add_argument("--test", type=int, nargs="+", default=[6, 7, 8])
ap.add_argument("--T", type=float, default=5.0)
ap.add_argument("--m_min", type=float, default=1e-2)
ap.add_argument("--floor", type=float, default=1e-3)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def load(seeds):
    out = []
    for s in seeds:
        z = np.load(f"{a.root}/{a.set}/seed{s:02d}.npz")
        out.append(z)
    return out


tr, te = load(a.train), load(a.test)
P = json.loads(str(tr[0]["params"]))
U = P["U"]
sources = P["sources"]
sx, sy = tr[0]["sub_x"], tr[0]["sub_y"]
dx, dy = sx[1] - sx[0], sy[1] - sy[0]
dt_rec = float(tr[0]["times"][1] - tr[0]["times"][0])
res = {"set": a.set, "train": a.train, "test": a.test, "plumes": []}

# ------------------------------------------------------------------ (1) mean plume + fluctuation
cm = np.mean([z["c_mean"] for z in tr], axis=0)            # (n_src, Nx, Ny) on the full grid
Nx, Ny = cm.shape[1:]
xf = np.arange(Nx) * P["Lx"] / Nx
yf = np.arange(Ny) * P["Ly"] / Ny
for i, (xs, ys) in enumerate(sources):
    c = cm[i]
    xi = xf - xs
    sel = (xi > 0.5) & (xi < 6.0)
    yy = (yf - ys + P["Ly"] / 2) % P["Ly"] - P["Ly"] / 2         # periodic distance to the source line
    m0 = c[sel].sum(1) * (yf[1] - yf[0])
    m1 = (c[sel] * yy).sum(1) * (yf[1] - yf[0]) / m0
    m2 = (c[sel] * yy ** 2).sum(1) * (yf[1] - yf[0]) / m0 - m1 ** 2
    A = np.vstack([xi[sel], np.ones(sel.sum())]).T
    (slope, s0sq), *_ = np.linalg.lstsq(A, m2, rcond=None)
    K = slope * U / 2
    flux = float(np.median(U * m0))
    # fluctuation statistics on the recorded sub-grid (training seeds)
    ix = [np.argmin(np.abs(xf - x)) for x in sx]
    iy = [np.argmin(np.abs(yf - y)) for y in sy]
    cbar_sub = c[np.ix_(ix, iy)]
    inpl = cbar_sub > 20 * a.m_min
    logs, inter = [], []
    for z in tr:
        cs = z["c_sub"][:, i].astype(np.float32)
        inter.append(np.mean(cs[:, inpl] < a.m_min))
        lc = np.log(np.maximum(cs[:, inpl], a.floor))
        logs.append((lc - lc.mean(0)).ravel())
    lg = np.concatenate(logs)
    res["plumes"].append(dict(i=i, xs=xs, ys=ys, K=float(K), s0sq=float(s0sq), mass_flux=flux,
                              sigma=float(lg.std()), skew=float(skew(lg)), exkurt=float(kurtosis(lg)),
                              intermittency=float(np.mean(inter)), n_inplume_points=int(inpl.sum())))

# ------------------------------------------------------------------ (2) correlation law
n = int(round(a.T / dt_rec)) + 1
s = dt_rec * np.arange(n)


def logsub(z, i):
    return np.log(np.maximum(z["c_sub"][:, i].astype(np.float32), a.floor))


def rho_paths(datasets, i, v, xc, yc):
    srel = dt_rec * (np.arange(n) - (n - 1) / 2)
    xs_, ys_ = xc + v[0] * srel, yc + v[1] * srel
    fi, fj = (xs_ - sx[0]) / dx, (ys_ - sy[0]) / dy
    Pth = []
    for z in datasets:
        L = logsub(z, i)
        base = map_coordinates(L.mean(0), [fi, fj], order=1)
        for t0 in range(0, L.shape[0] - n, 4):
            vals = np.array([map_coordinates(L[t0 + k], [[fi[k]], [fj[k]]], order=1)[0] for k in range(n)])
            Pth.append(vals - base)
    Pth = np.array(Pth); Pth -= Pth.mean(0)
    C = Pth.T @ Pth / len(Pth); sd = np.sqrt(np.diag(C))
    with np.errstate(invalid="ignore", divide="ignore"):
        R = C / np.outer(sd, sd)                  # NaN where a path point never leaves the floor
    return np.array([np.nanmean(np.diag(R, k)) for k in range(n)])


def tint(rho):
    k = np.argmax(rho <= 0) if np.any(rho <= 0) else len(rho)
    r = rho[:k]
    return dt_rec * (r.sum() - 0.5 * r[0] - (0.5 * r[-1] if k == len(rho) else 0.0))


def rho_model(w, tau, L):
    return np.exp(-s / tau - 0.5 * (w * s / L) ** 2)


cases = {"drift": (1.0, 0.0), "stationary": (0.0, 0.0), "half drift": (0.5, 0.0),
         "upwind": (-1.0, 0.0), "slow upwind": (-0.5, 0.0), "crosswind 0.5": (0.0, 0.5)}
fit_c = ["drift", "stationary"]
law = []
i0 = 0
xs0, ys0 = sources[i0]
# stations chosen so every path (half-length |v| T/2 <= 2.5 for T = 5) stays downstream of the
# source and inside the recorded range (x < 0.75 Lx = 9.42); the first attempt (job 95342) put
# paths upstream of the source -> zero-variance points -> NaN correlations
for xoff in (3.0, 4.5, 6.0):
    xc, yc = xs0 + xoff + (n - 1) / 2 * dt_rec * 0, ys0
    rf = {c: rho_paths(tr[:3], i0, v, xc, yc) for c, v in cases.items()}
    rt = {c: rho_paths(te, i0, v, xc, yc) for c, v in cases.items()}
    wv = {c: float(np.hypot(v[0] - U, v[1])) for c, v in cases.items()}
    def resid(p):
        r = np.concatenate([rf[c] - rho_model(wv[c], *np.exp(p)) for c in fit_c])
        return np.where(np.isfinite(r), r, 0.0)
    sol = least_squares(resid, np.log([2.0, 0.4]))
    tau, L = np.exp(sol.x)
    for c in cases:
        law.append(dict(xoff=xoff, case=c, w=wv[c], tau=float(tau), L=float(L), fit=c in fit_c,
                        tint_test=float(tint(rt[c])), tint_pred=float(tint(rho_model(wv[c], tau, L))),
                        rho_test=rt[c].tolist()))
    print(f"xoff={xoff}: tau={tau:.2f} L={L:.3f}", flush=True)
    for r in law[-len(cases):]:
        print(f"   {r['case']:14s} w={r['w']:.2f} Tint test={r['tint_test']:.3f} pred={r['tint_pred']:.3f} "
              f"err={100*(r['tint_pred']-r['tint_test'])/r['tint_test']:+.1f}% {'(fit)' if r['fit'] else ''}",
              flush=True)
res["law"] = law
res["surrogate"] = dict(U=U, K=float(np.median([p["K"] for p in res["plumes"]])),
                        sigma=float(np.median([p["sigma"] for p in res["plumes"]])),
                        tau=float(np.median([r["tau"] for r in law])),
                        L=float(np.median([r["L"] for r in law])))
print(json.dumps(res["surrogate"]), flush=True)
for p_ in res["plumes"]:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in p_.items()}, flush=True)
json.dump(res, open(a.out, "w"), indent=1)

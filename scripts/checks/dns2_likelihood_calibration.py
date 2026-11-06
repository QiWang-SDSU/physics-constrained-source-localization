"""E8b: is the surrogate's single-point observation model consistent with DNS2? (TRAINING seeds only)

For every recorded (t, x, y) of plume i, the surrogate predicts log cbar = log(Q c_dag(x_s; x)) from
the fitted K (s0^2 clamped at 0, as in E8). Points are binned by predicted log cbar and compared with:
  model (lognormal + censoring):  P(cens) = Phi((log m_min - (log cbar - s^2/2)) / s)
                                  log c | det ~ N(log cbar - s^2/2, s^2) truncated at log m_min
  DNS:                            empirical P(c < m_min) and moments of log c | c >= m_min.
Also reported: the detection rate where the model predicts cbar = 0 (upstream "leak").
Then a zero-inflated model is fitted by maximum likelihood (bins pooled):
  P(cens) = pi0 + (1 - pi0) Phi((thr - m)/s),  log c | det ~ (1 - pi0) N(m, s^2) on [thr, inf)
  with m = log cbar + b.  Free parameters: pi0, b, s.
Output: JSON with the calibration table and the fitted (pi0, b, s, leak).

Usage (Fermi): python scripts/checks/dns2_likelihood_calibration.py --set strong \
                  --fit $RLS_DATA/dns/strong_fit.json --out .../strong_lik.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rlscalar.search.core import Params, log_cbar
import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr, log_ndtr

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="strong")
ap.add_argument("--root", default=os.environ.get("RLS_DATA", "data") + "/dns")
ap.add_argument("--train", type=int, nargs="+", default=[1, 2, 3, 4, 5])
ap.add_argument("--fit", required=True)
ap.add_argument("--m_min", type=float, default=1e-2)
ap.add_argument("--tstride", type=int, default=10)
ap.add_argument("--mean", default="adjoint", choices=["adjoint", "iso"], help="E27: isotropic no-physics mean field")
ap.add_argument("--iso_l", type=float, default=0.5)
ap.add_argument("--out", required=True)
a = ap.parse_args()
fit = json.load(open(a.fit))
sur = fit["surrogate"]
U, K, sig = sur["U"], sur["K"], sur["sigma"]
T_L = sur.get("T_L", 0.0)
S0 = sur.get("s0sq_taylor", 0.0) if T_L > 0 else 0.0
Q = float(np.median([q["mass_flux"] for q in fit["plumes"]]))
PM = Params(U=U, K=K, Q=Q, T_L=T_L, s0sq=S0, src_r=sur.get("src_r", 0.0), mean_model=a.mean, iso_l=a.iso_l)
thr = np.log(a.m_min)

LC, LB = [], []                       # log c (or -inf) and predicted log cbar (or -inf)
for s in a.train:
    z = np.load(f"{a.root}/{a.set}/seed{s:02d}.npz")
    P = json.loads(str(z["params"]))
    sx, sy, Ly = z["sub_x"], z["sub_y"], P["Ly"]
    X, Y = np.meshgrid(sx, sy, indexing="ij")
    for i, (xs, ys) in enumerate(P["sources"]):
        xi = X - xs
        dy = (Y - ys + Ly / 2) % Ly - Ly / 2
        lb_all = log_cbar(0.0, 0.0, np.maximum(xi, -5.0), dy, PM)  # the belief's own mean field (E26; iso for E27)
        keep = (np.abs(dy) < 1.2) & (xi > -0.5) & (xi < 7.0)       # neighbourhood of the plume
        cs = z["c_sub"][::a.tstride, i].astype(np.float32)[:, keep]
        lb = np.where(np.isfinite(lb_all[keep]), lb_all[keep], np.nan)
        if a.mean == "adjoint":
            lb = np.maximum(lb, -30.0)                             # E24: clip numerically zero means (log cbar < -30)
        LC.append(np.log(np.maximum(cs, 1e-30)).ravel())
        LB.append(np.broadcast_to(lb, cs.shape).ravel())
LC, LB = np.concatenate(LC), np.concatenate(LB)
det = LC >= thr

# ---------------------------------------------------------------- leak where the model predicts cbar = 0
zero = ~np.isfinite(LB)
leak = float(np.mean(det[zero]))
# ---------------------------------------------------------------- calibration table
bins = np.arange(-8, 3.01, 0.5)
rows = []
fin = np.isfinite(LB)
for lo, hi in zip(bins[:-1], bins[1:]):
    m = fin & (LB >= lo) & (LB < hi)
    if m.sum() < 200:
        continue
    mb = 0.5 * (lo + hi)
    pc_dns = 1 - det[m].mean()
    pc_mod = ndtr((thr - (mb - sig ** 2 / 2)) / sig)
    d = LC[m & det]
    rows.append(dict(logcbar=mb, n=int(m.sum()), Pcens_dns=float(pc_dns), Pcens_model=float(pc_mod),
                     mean_logc_det_dns=float(d.mean()) if d.size else np.nan,
                     std_logc_det_dns=float(d.std()) if d.size else np.nan))

# ---------------------------------------------------------------- zero-inflated fit
X_lb, X_det, X_lc = LB[fin], det[fin], LC[fin]
rng = np.random.default_rng(0)
if X_lb.size > 400_000:
    sel = rng.choice(X_lb.size, 400_000, replace=False)
    X_lb, X_det, X_lc = X_lb[sel], X_det[sel], X_lc[sel]


def nll(p):
    logit_pi, b, log_s = p
    pi0 = 1 / (1 + np.exp(-logit_pi)); s = np.exp(log_s)
    m = X_lb + b
    lc = np.log(pi0 + (1 - pi0) * ndtr((thr - m) / s))
    ld = np.log(1 - pi0) - 0.5 * ((X_lc - m) / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi)
    return -np.mean(np.where(X_det, ld, lc))


def nll_lognormal(p):                         # the E8 model: pi0 = 0, b = -s^2/2 with s = sigma
    s = sig
    m = X_lb - s ** 2 / 2
    lc = log_ndtr((thr - m) / s)
    ld = -0.5 * ((X_lc - m) / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi)
    return -np.mean(np.where(X_det, ld, lc))


r = minimize(nll, [0.0, -1.0, np.log(1.5)], method="Nelder-Mead", options=dict(maxiter=2000, xatol=1e-4))
pi0 = float(1 / (1 + np.exp(-r.x[0]))); b = float(r.x[1]); s_fit = float(np.exp(r.x[2]))
# ---------------------------------------------------------------- intermittent-plume (zi2) fit, E8c
# gamma = sigmoid(a0 + a1 lc), in-puff log c ~ N(c0 + c1 lc, s^2), censored below thr
def nll_zi2(p):
    a0, a1, c0, c1, log_s = p
    g = 1 / (1 + np.exp(-(a0 + a1 * X_lb))); s = np.exp(log_s); m = c0 + c1 * X_lb
    lc_ = np.log(np.clip((1 - g) + g * ndtr((thr - m) / s), 1e-300, None))
    ld = np.log(g) - 0.5 * ((X_lc - m) / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi)
    return -np.mean(np.where(X_det, ld, lc_))


r2 = minimize(nll_zi2, [1.0, 1.0, -0.5, 1.0, np.log(1.5)], method="Nelder-Mead",
              options=dict(maxiter=8000, maxfev=8000, xatol=1e-4, fatol=1e-7))
a0, a1, c0, c1, ls = r2.x
# leak branch (model zero-support): detection prob and in-puff moments
Lz_det = LC[zero & det]


def nll_leak(p):
    lg, mu_, lsd = p
    g = 1 / (1 + np.exp(-lg)); sdv = np.exp(lsd)
    n_c = np.sum(zero & ~det)
    lcens = n_c * np.log(np.clip((1 - g) + g * ndtr((thr - mu_) / sdv), 1e-300, None))
    ldet = np.sum(np.log(g) - 0.5 * ((Lz_det - mu_) / sdv) ** 2 - np.log(sdv) - 0.5 * np.log(2 * np.pi))
    return -(lcens + ldet) / zero.sum()


r3 = minimize(nll_leak, [-1.5, -4.0, np.log(0.5)], method="Nelder-Mead", options=dict(maxiter=4000)) if zero.any() else None
if r3 is None:                                # iso mean field: full support, no leak branch
    from types import SimpleNamespace
    r3 = SimpleNamespace(x=[-30.0, -4.0, np.log(0.5)], fun=0.0)
zi2 = dict(zi_a0=float(a0), zi_a1=float(a1), zi_c0=float(c0), zi_c1=float(c1), sigma=float(np.exp(ls)),
           leak_g=float(1 / (1 + np.exp(-r3.x[0]))), leak_mu=float(r3.x[1]), leak_s=float(np.exp(r3.x[2])),
           nll=float(r2.fun), nll_leak=float(r3.fun), converged=bool(r2.success))
print(f"  zi2 fit: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in zi2.items()})}")
print(f"  mean NLL per point (finite-support points): zi2 {r2.fun:.4f} vs constant-ZI {r.fun:.4f} vs E8 lognormal {nll_lognormal(None):.4f}")
# calibration of the zi2 model on the same table
for q in rows:
    m_ = c0 + c1 * q["logcbar"]; g_ = 1 / (1 + np.exp(-(a0 + a1 * q["logcbar"]))); s_ = np.exp(ls)
    q["Pcens_zi2"] = float((1 - g_) + g_ * ndtr((thr - m_) / s_))
    q["mean_det_zi2_untrunc"] = float(m_)

out = dict(set=a.set, mean=a.mean, iso_l=a.iso_l, zi2=zi2, train=a.train, m_min=a.m_min, sigma_E8=sig, leak_det_rate_where_model_zero=leak,
           n_points=int(LB.size), frac_model_zero=float(zero.mean()),
           zi_fit=dict(pi0=pi0, b=b, s=s_fit, nll=float(r.fun)),
           nll_E8_lognormal=float(nll_lognormal(None)), table=rows)
print(f"{a.set}: leak (det where model cbar=0) = {leak:.4f}  (model-zero fraction {zero.mean():.3f})")
print(f"  zero-inflated fit: pi0={pi0:.3f}  b={b:+.3f} (lognormal would be {-sig**2/2:+.3f})  s={s_fit:.3f} (E8 sigma {sig:.3f})")
print(f"  mean NLL per point: zero-inflated {r.fun:.4f}  vs E8 lognormal {out['nll_E8_lognormal']:.4f}")
print(f"  {'log cbar':>8s} {'n':>8s} {'Pcens DNS':>9s} {'Pcens E8':>9s} {'Pcens zi2':>9s} {'E[log c|det] DNS':>16s} {'zi2 m':>7s} {'sd DNS':>6s}")
for q in rows:
    print(f"  {q['logcbar']:8.2f} {q['n']:8d} {q['Pcens_dns']:9.3f} {q['Pcens_model']:9.3f} {q['Pcens_zi2']:9.3f} "
          f"{q['mean_logc_det_dns']:16.3f} {q['mean_det_zi2_untrunc']:7.3f} {q['std_logc_det_dns']:6.3f}")
json.dump(out, open(a.out, "w"), indent=1, default=float)

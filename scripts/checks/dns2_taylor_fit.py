"""E24: fit Taylor's dispersion law sigma_y^2(xi) = s0^2 + 2 K T_L [r - 1 + exp(-r)], r = xi / (U T_L), to the time-mean
cross-wind variance of the DNS plumes (TRAINING seeds only, full grid, as in dns2_fit_and_law.py).
The fitted law is the slender adjoint solution with a developing eddy diffusivity K (1 - exp(-xi / U T_L)); it
replaces the Fickian law 2 K xi / U with the negative virtual-origin offset s0^2 clamped to 0 (E8-E22).
s0^2 >= 0 is the initial variance set by the finite source radius.
Writes a copy of the existing fit JSON with surrogate K and T_L replaced.
Usage (Fermi): python scripts/checks/dns2_taylor_fit.py --set weak --out $RLS_DATA/dns/fits/weak_fit_taylor.json
"""
import argparse, json, os
import numpy as np
from scipy.optimize import least_squares

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="weak")
ap.add_argument("--root", default=os.environ.get("RLS_DATA", "data") + "/dns")
ap.add_argument("--train", type=int, nargs="+", default=[1, 2, 3, 4, 5])
ap.add_argument("--xi", type=float, nargs=2, default=[0.2, 6.0])
ap.add_argument("--out", required=True)
a = ap.parse_args()

fit = json.load(open(f"{a.root}/{a.set}_fit.json"))
U = fit["surrogate"]["U"]
zs = [np.load(f"{a.root}/{a.set}/seed{s:02d}.npz") for s in a.train]
P = json.loads(str(zs[0]["params"]))
cm = np.mean([z["c_mean"] for z in zs], axis=0)
Nx, Ny = cm.shape[1:]
xf, yf = np.arange(Nx) * P["Lx"] / Nx, np.arange(Ny) * P["Ly"] / Ny
dyf = yf[1] - yf[0]
XI, M2 = [], []
for i, (xs, ys) in enumerate(P["sources"]):
    xi = xf - xs
    sel = (xi > a.xi[0]) & (xi < a.xi[1]) & (xf < P["sponge_start"] * P["Lx"])
    yy = (yf - ys + P["Ly"] / 2) % P["Ly"] - P["Ly"] / 2
    c = cm[i][sel]
    m0 = c.sum(1) * dyf
    m1 = (c * yy).sum(1) * dyf / m0
    XI.append(xi[sel]); M2.append((c * yy ** 2).sum(1) * dyf / m0 - m1 ** 2)
XI, M2 = np.concatenate(XI), np.concatenate(M2)


def taylor(q, xi):
    K, TL, s0sq = np.exp(q)
    r = xi / (U * TL)
    return s0sq + 2 * K * TL * (r - 1 + np.exp(-r))


sol = least_squares(lambda q: taylor(q, XI) - M2, np.log([fit["surrogate"]["K"], 0.5, 0.01]))
K, TL, S0 = map(float, np.exp(sol.x))
rms_taylor = float(np.sqrt(np.mean((taylor(sol.x, XI) - M2) ** 2)))
rms_fick = float(np.sqrt(np.mean((2 * fit["surrogate"]["K"] * XI / U - M2) ** 2)))   # the E8-E22 law (s0^2 = 0)
out = json.loads(json.dumps(fit))
out["surrogate"].update(K=K, T_L=TL, s0sq_taylor=S0)
out["taylor_fit"] = dict(K=K, T_L=TL, s0sq=S0, K_fick=fit["surrogate"]["K"], xi_range=a.xi, train=a.train, n=int(XI.size),
                         rms_var_taylor=rms_taylor, rms_var_fick_s0clamped=rms_fick)
json.dump(out, open(a.out, "w"), indent=1)
print(f"{a.set}: K={K:.4f} T_L={TL:.3f} s0^2={S0:.4f} (Fickian K {fit['surrogate']['K']:.4f}); "
      f"rms(var) Taylor {rms_taylor:.4f} vs Fickian s0=0 {rms_fick:.4f}")
for x in (0.5, 1.0, 2.0, 3.0, 4.0):
    m = np.abs(XI - x) < 0.05
    print(f"  xi={x}: DNS sd {np.sqrt(M2[m].mean()):.3f}  Taylor {np.sqrt(taylor(sol.x, x)):.3f}  "
          f"Fickian {np.sqrt(2 * fit['surrogate']['K'] * x / U):.3f}")

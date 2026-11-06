"""Generate one DNS2 dataset (one seed): spin-up, then record sub-sampled scalar fields.

Output <out>/seedXX.npz:
  times (Nt,), sub_x (nx,), sub_y (ny,), c_sub (Nt, n_src, nx, ny) float16,
  energy (Nt,), neg_frac = fraction of cells with c < -1e-3 (Gibbs monitor), c_mean (n_src, Nx, Ny) float32 (time mean over the record),
  neg_frac (Nt,), params (json)
Checkpoints every --ckpt t.u. to <out>/seedXX.ckpt.npz; a restart resumes from it.

Usage: python scripts/dns2_run.py --seed 1 --out data/dns2_strong --eps 0.005 \
          [--t_spin 100 --t_rec 500 --dt_rec 0.1 --stride_x 3 --stride_y 4 --xmax_frac 0.75]
"""
import argparse, json, os, sys, time
from dataclasses import asdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlscalar.dns2 import DNS2, DNS2Params

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--eps", type=float, default=0.005)
ap.add_argument("--mu", type=float, default=0.05)
ap.add_argument("--t_spin", type=float, default=100.0)
ap.add_argument("--t_rec", type=float, default=500.0)
ap.add_argument("--dt_rec", type=float, default=0.1)
ap.add_argument("--stride_x", type=int, default=3)
ap.add_argument("--stride_y", type=int, default=4)
ap.add_argument("--xmax_frac", type=float, default=0.75)
ap.add_argument("--ckpt", type=float, default=25.0)
ap.add_argument("--Nx", type=int, default=384)
ap.add_argument("--Ny", type=int, default=192)
ap.add_argument("--workers", type=int, default=1)
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
p = DNS2Params(eps=a.eps, mu=a.mu, Nx=a.Nx, Ny=a.Ny)
d = DNS2(p, seed=a.seed, workers=a.workers)
ix = np.arange(0, int(a.xmax_frac * p.Nx), a.stride_x)
iy = np.arange(0, p.Ny, a.stride_y)
nsteps_rec = int(round(a.dt_rec / p.dt))
Nt = int(round(a.t_rec / a.dt_rec))
ck = os.path.join(a.out, f"seed{a.seed:02d}.ckpt.npz")
c_sub = np.zeros((Nt, len(p.sources), len(ix), len(iy)), np.float16)
energy, negf, times = np.zeros(Nt), np.zeros(Nt), np.zeros(Nt)
c_sum = np.zeros((len(p.sources), p.Nx, p.Ny))
k0 = 0
if os.path.exists(ck):
    z = np.load(ck)
    d.w, d.c, d.t, k0 = z["w"], z["c"], float(z["t"]), int(z["k"])
    d.rng = np.random.default_rng([a.seed, k0])
    c_sub[:k0], energy[:k0], negf[:k0], times[:k0] = z["c_sub"][:k0], z["energy"][:k0], z["negf"][:k0], z["times"][:k0]
    c_sum = z["c_sum"]
    print(f"resumed at t={d.t:.1f}, record index {k0}", flush=True)
else:
    t0 = time.time()
    while d.t < a.t_spin - 1e-9:
        d.step()
    print(f"spin-up {a.t_spin} t.u. in {time.time()-t0:.0f}s, E={d.energy():.4f}", flush=True)
t0 = time.time()
for k in range(k0, Nt):
    for _ in range(nsteps_rec):
        d.step()
    c = d.scalars()
    c_sum += c
    c_sub[k] = c[:, ix][:, :, iy].astype(np.float16)
    energy[k], negf[k], times[k] = d.energy(), np.mean(c < -1e-3), d.t
    if (k + 1) % int(round(a.ckpt / a.dt_rec)) == 0 or k == Nt - 1:
        np.savez(ck, w=d.w, c=d.c, t=d.t, k=k + 1, c_sub=c_sub, energy=energy, negf=negf,
                 times=times, c_sum=c_sum)
        print(f"t={d.t:.1f} k={k+1}/{Nt} E={energy[k]:.4f} neg={negf[k]:.4f} "
              f"{(time.time()-t0)/(k+1-k0):.2f}s/rec", flush=True)
pd = asdict(p)
pd.update(vars(a))
np.savez(os.path.join(a.out, f"seed{a.seed:02d}.npz"), times=times, sub_x=d.x[ix], sub_y=d.y[iy],
         c_sub=c_sub, energy=energy, neg_frac=negf, c_mean=(c_sum / Nt).astype(np.float32),
         params=json.dumps(pd, default=float))
os.remove(ck)
print("done", flush=True)

"""Replay one DNS2 dataset run in full and record the velocity on the sensor sub-grid (E44: velocity sensing).

The dataset (scripts/dns2_run.py) stores only the scalar c_sub. This replays the same run (same code, seed,
parameters and worker count, as scripts/dns2_replay.py, which matched the stored c_sub to within one float16 ulp
on seeds 6 and 7) and saves the total velocity (U + u, v) at every record on the same sub-grid (stride 3 x 4),
so a sensor can read (c, u, v) at the same point and time. Two conditions for an exact replay (E44): restarts of
the original run must be repeated (--resume), and the CPU must give the same floating-point results as the original
node (the original strong runs were on Intel Gold6240 / Platinum8268; on an AMD EPYC-7452 node the replay departs by
1e-7 after ~200 records and turbulence amplifies this to O(1) by record ~2500). At every record the replayed c on the sub-grid is
compared with the stored c_sub; the worst difference is saved and printed (it must stay at float16 rounding).
Output <out>/<set>/seed0X_vel.npz: u_sub, v_sub (Nt, nx, ny) float16, sub_x, sub_y, times, sub_maxdiff (Nt,).
Usage (Fermi): python scripts/dns2_velocity.py --seed 1 --set strong --out $RLS_DATA/dns/vel [--kmax 50]
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlscalar.dns2 import DNS2, DNS2Params

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--set", default="strong")
ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--kmax", type=int, default=None, help="stop after this many records (smoke test)")
ap.add_argument("--resume", type=int, nargs="*", default=[],
                help="record indices where the original run was restarted from a checkpoint (dns2_run.py reseeds the "
                     "forcing RNG there with default_rng([seed, k0])); see the 'resumed at' lines of its logs")
a = ap.parse_args()

ref = np.load(f"{os.environ.get('RLS_DATA', 'data')}/dns/{a.set}/seed{a.seed:02d}.npz")
R = json.loads(str(ref["params"]))
c_ref = ref["c_sub"]
assert int(R["workers"]) == a.workers, R["workers"]
p = DNS2Params(eps=R["eps"], mu=R["mu"], Nx=R["Nx"], Ny=R["Ny"])
d = DNS2(p, seed=a.seed, workers=a.workers)
ix = np.arange(0, int(R["xmax_frac"] * p.Nx), R["stride_x"])
iy = np.arange(0, p.Ny, R["stride_y"])
nsteps_rec = int(round(R["dt_rec"] / p.dt))
Nt = c_ref.shape[0] if a.kmax is None else a.kmax
u_sub = np.zeros((Nt, len(ix), len(iy)), np.float16)
v_sub = np.zeros_like(u_sub)
md, times = np.zeros(Nt), np.zeros(Nt)

t0 = time.time()
while d.t < R["t_spin"] - 1e-9:
    d.step()
print(f"spin-up in {time.time()-t0:.0f}s, E={d.energy():.4f}", flush=True)
for k in range(Nt):
    if k in a.resume:
        d.rng = np.random.default_rng([a.seed, k])
    for _ in range(nsteps_rec):
        d.step()
    c = d.scalars()
    md[k] = np.max(np.abs(c[:, ix][:, :, iy].astype(np.float16).astype(np.float32) - c_ref[k].astype(np.float32)))
    u, v = d.velocity()
    u_sub[k], v_sub[k], times[k] = u[ix][:, iy], v[ix][:, iy], d.t
    if (k + 1) % 250 == 0 or k == Nt - 1:
        print(f"k={k+1}/{Nt} t={d.t:.1f} worst c_sub mismatch {md[:k+1].max():.3g} {(time.time()-t0)/(k+1):.2f}s/rec",
              flush=True)
os.makedirs(f"{a.out}/{a.set}", exist_ok=True)
f = f"{a.out}/{a.set}/seed{a.seed:02d}_vel{'' if a.kmax is None else f'_k{Nt}'}.npz"
np.savez_compressed(f, u_sub=u_sub, v_sub=v_sub, sub_x=d.x[ix], sub_y=d.y[iy], times=times, sub_maxdiff=md,
                    times_ref=ref["times"][:Nt])
print(f"saved {f}; worst c_sub mismatch {md.max():.3g}; time match {np.abs(times - ref['times'][:Nt]).max():.2e}",
      flush=True)

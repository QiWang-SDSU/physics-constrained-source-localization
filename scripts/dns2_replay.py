"""Replay one DNS2 dataset run bit-for-bit and save FULL-resolution fields over selected record windows.

Why: the dataset (scripts/dns2_run.py) stores the scalar only on a sub-grid (c_sub), which is also what the
search sensor reads. For figures we want the full-resolution field over the time window of a given episode.
The replay uses the same code, seed, parameters and worker count, and checks at every record that its
sub-grid equals the stored c_sub (any mismatch is logged; it would mean the replay diverged).
The episodes themselves are NOT re-run: the sensor still reads the stored sub-grid.

Usage (Fermi): python scripts/dns2_replay.py --seed 6 --set strong --eps 0.005 --windows 2029:2112 287:370 \
                   --out $RLS_DATA/dns/replay
Output <out>/<set>_seed0X_k<k0>.npz per window: k (records), times, x, y, c (n_rec, n_src, Nx, Ny) float16,
omega (n_rec, Nx, Ny) float16, u, v (n_rec, Nx, Ny) float16, sub_maxdiff (n_rec,), params.
"""
import argparse, json, os, sys, time
from dataclasses import asdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlscalar.dns2 import DNS2, DNS2Params

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--set", default="strong")
ap.add_argument("--eps", type=float, default=0.005)
ap.add_argument("--windows", nargs="+", required=True, help="k0:k1 record ranges (k1 exclusive)")
ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=2)
a = ap.parse_args()

root = os.environ.get("RLS_DATA", "data") + "/dns"
ref = np.load(f"{root}/{a.set}/seed{a.seed:02d}.npz")
R = json.loads(str(ref["params"]))
c_ref = ref["c_sub"]
assert abs(R["eps"] - a.eps) < 1e-12 and int(R["workers"]) == a.workers, (R["eps"], R["workers"])
p = DNS2Params(eps=R["eps"], mu=R["mu"], Nx=R["Nx"], Ny=R["Ny"])
d = DNS2(p, seed=a.seed, workers=a.workers)
ix = np.arange(0, int(R["xmax_frac"] * p.Nx), R["stride_x"])
iy = np.arange(0, p.Ny, R["stride_y"])
nsteps_rec = int(round(R["dt_rec"] / p.dt))
wins = [tuple(map(int, w.split(":"))) for w in a.windows]
kmax = max(k1 for _, k1 in wins)
buf = {w: dict(k=[], t=[], c=[], om=[], u=[], v=[], md=[]) for w in wins}
os.makedirs(a.out, exist_ok=True)

t0 = time.time()
while d.t < R["t_spin"] - 1e-9:
    d.step()
print(f"spin-up in {time.time()-t0:.0f}s, E={d.energy():.4f}", flush=True)
worst = 0.0
for k in range(kmax):
    for _ in range(nsteps_rec):
        d.step()
    c = d.scalars()
    md = float(np.max(np.abs(c[:, ix][:, :, iy].astype(np.float16).astype(np.float32) - c_ref[k].astype(np.float32))))
    worst = max(worst, md)
    for (k0, k1), b in buf.items():
        if k0 <= k < k1:
            u, v = d.velocity()
            b["k"].append(k); b["t"].append(d.t); b["md"].append(md)
            b["c"].append(c.astype(np.float16)); b["om"].append(d._ifft(d.w).astype(np.float16))
            b["u"].append(u.astype(np.float16)); b["v"].append(v.astype(np.float16))
    if (k + 1) % 250 == 0:
        print(f"k={k+1}/{kmax} t={d.t:.1f} max|c_sub replay - stored| so far {worst:.3g} "
              f"{(time.time()-t0)/(k+1):.2f}s/rec", flush=True)
for (k0, k1), b in buf.items():
    f = os.path.join(a.out, f"{a.set}_seed{a.seed:02d}_k{k0}.npz")
    np.savez_compressed(f, k=np.array(b["k"]), times=np.array(b["t"]), x=d.x, y=d.y, c=np.array(b["c"]),
                        omega=np.array(b["om"]), u=np.array(b["u"]), v=np.array(b["v"]),
                        sub_maxdiff=np.array(b["md"]), params=json.dumps(asdict(p), default=float))
    print(f"saved {f}: records {k0}-{k1-1}, max sub-grid mismatch {max(b['md']):.3g}", flush=True)
print(f"done; worst sub-grid mismatch over all {kmax} records: {worst:.3g}", flush=True)

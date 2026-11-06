"""DNS plume snapshots and sensor readings for the E40 posterior-map tasks (figure scripts/figures/fig_posterior_planners.py).

For each task: rebuild the DNS world exactly as postnet_closedloop.py does (same file, plume, start record, noise seed),
sample the plume concentration c in task coordinates on a 0.01 grid over [0, 3.7] x [-0.85, 0.85] at n = --ns (default 20, 40, 80),
and replay the stored sensor paths of the m1 and m3 planners through world.measure (same call order -> identical noise
draws -> identical readings to the closed-loop run).
Output {out}/{set}_plume_task{task}.npz: c (3, nx, ny) float16, xg, yg, ns, z (2, 81) log readings (nan = no detection).
  python scripts/plume_snapshots.py --set strong --task 14 --out $RLS_OUT/E40
"""
import argparse, os, sys
import numpy as np
from scipy.ndimage import map_coordinates

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="strong")
ap.add_argument("--task", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ns", type=int, nargs="+", default=[20, 40, 80])
a = ap.parse_args()
sys.argv = [sys.argv[0], "--set", a.set, "--out", a.out]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import postnet_closedloop as cl                                   # M1 parameters and DNS files, as in E38 / E40
from search_sweep import dns_factory
from rlscalar.search.episode import sample_task

NS = a.ns
P1 = cl.P1
cl_q = np.load(f"{a.out}/{a.set}_closedloop_q.npz")
pols = list(cl_q["policies"])
make = dns_factory(cl.FILES)
src, _ = sample_task(P1, a.task)
assert np.allclose(src, cl_q["src"][a.task]), (src, cl_q["src"][a.task])

z = np.full((2, 81), np.nan)
for r, pol in enumerate(("m1", "m3")):
    world = make(P1, src, np.random.default_rng(10_000 + a.task), a.task)
    path = cl_q["path"][pols.index(pol), a.task]
    for k in range(81):
        v = world.measure(*path[k], k * P1.dt)
        z[r, k] = np.nan if v is None else v

d, i, off = world.d, world.i, world.off
xg, yg = np.arange(0.0, 3.701, 0.01), np.arange(-0.85, 0.851, 0.01)
X, Y = np.meshgrid(xg + off[0], yg + off[1], indexing="ij")
fi = (X - d.sx[0]) / (d.sx[1] - d.sx[0])
fj = ((Y % d.Ly) - d.sy[0]) / (d.sy[1] - d.sy[0])
c = np.zeros((len(NS), len(xg), len(yg)), np.float16)
for m, n in enumerate(NS):
    f = d.c[world.t0 + int(round(n * P1.dt / d.dt)), i].astype(np.float32)
    f = np.concatenate([f, f[:, :1]], axis=1)
    v = map_coordinates(f, [fi.ravel(), fj.ravel()], order=1, mode="constant", cval=0.0).reshape(X.shape)
    v[(fi < 0) | (fi > len(d.sx) - 1)] = 0.0
    c[m] = v
np.savez_compressed(f"{a.out}/{a.set}_plume_task{a.task}.npz", c=c, xg=xg, yg=yg, ns=NS, z=z, src=src,
                    m_min=P1.m_min, policies=np.array(["m1", "m3"]))
print(a.set, a.task, "detections m1/m3:", int(np.isfinite(z[0]).sum()), int(np.isfinite(z[1]).sum()),
      "c max", float(c.max()), flush=True)

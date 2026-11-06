"""Posterior distributions of the source location with and without velocity sensing (E48 data, scripts/postnet_maps.py
--runs runs_fp32 --runs_vel runs_vel).

One held-out test episode (any of the three path types), chosen by rule: among the episodes where velocity turns a
failure (e80 >= 0.1) into a success (e80 < 0.1), the one at the median (--rank) of the error without velocity.
--flip 0 instead takes the median gain over all episodes. The printed counts give how often each flip occurs. Colour = q(x_j | H_n) / max_j q,
linear, over the prior support. Columns: learned posterior with the physics input, without and with velocity sensing;
rows: n = 20, 40, 80 moves. Green ring and cross: true source; line: sensor path; dots: detections; solid contour: 90%
highest-posterior region.
Colour is used because a 2-D probability field over six panels is unreadable in grey at this size.
Run (from repo/): python scripts/figures/fig_posterior_velocity.py [--set weak] [--rank 0.5]
"""
import argparse, os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from plotstyle import use_style, panel_label, save

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="strong")
ap.add_argument("--rank", type=float, default=0.5)
ap.add_argument("--flip", type=int, default=1)
a = ap.parse_args()
z = np.load(f"docs/research/data/E59/{a.set}_maps.npz")
gx, gy, ns, mapn = z["grid_x"], z["grid_y"], list(z["ns"]), list(z["map_ns"])
meth = list(z["methods"])
COLS = [(meth.index("B"), r"no velocity"), (meth.index("Bvel"), r"with velocity")]
h = gx[1] - gx[0]
e0 = z["map_episodes"]
e_no, e_v = z["err"][e0, COLS[0][0], ns.index(80)], z["err"][e0, COLS[1][0], ns.index(80)]
up, down = (e_no >= 0.1) & (e_v < 0.1), (e_no < 0.1) & (e_v >= 0.1)
print(f"{a.set}: {len(e0)} episodes, velocity turns {up.sum()} failures into successes and {down.sum()} successes into failures")
if a.flip:
    cand = np.where(up)[0]
    k = int(cand[np.argsort(e_no[cand])[int(round(a.rank * (len(cand) - 1)))]])
else:
    k = int(np.argsort(e_no - e_v)[int(round(a.rank * (len(e0) - 1)))])
ep = int(e0[k])
src, pos, det = z["src"][ep], z["pos"][ep], z["tok"][ep][:, 0] > 0.5
X0, X1, Y0, Y1 = gx[0] - h / 2, gx[-1] + h / 2, gy[0] - h / 2, gy[-1] + h / 2


def hpd(q, level=0.9):
    o = np.argsort(q)[::-1]
    m = np.zeros(q.size, bool)
    m[o[:np.searchsorted(np.cumsum(q[o]), level) + 1]] = True
    return m


use_style(base=12)
fig, axes = plt.subplots(3, 2, figsize=(8.0, 6.6), constrained_layout=True, sharex=True, sharey=True)
for row, n in enumerate(mapn):
    for col, (mi, name) in enumerate(COLS):
        ax = axes[row, col]
        q = z["q"][k, mi, row].astype(np.float64)
        q /= q.sum()
        Q = (q / q.max()).reshape(len(gx), len(gy))
        im = ax.pcolormesh(gx, gy, Q.T, cmap="magma_r", vmin=0, vmax=1, shading="nearest", rasterized=True)
        ax.contour(gx, gy, hpd(q).reshape(len(gx), len(gy)).T.astype(float), levels=[0.5], colors="k", linewidths=0.9)
        p = pos[:n + 1]
        ax.plot(p[:, 0], p[:, 1], color="#4a7ab5", lw=1.0)
        ax.plot(p[det[:n + 1], 0], p[det[:n + 1], 1], "o", ms=3, color="#4a7ab5")
        ax.plot(*src, "o", ms=13, mfc="none", mec="#2ca02c", mew=1.6, zorder=6)    # ring around the source
        ax.plot(*src, "+", ms=5, mew=1.0, color="#2ca02c", zorder=6)
        ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.set_aspect("equal")
        err = z["err"][ep, mi, ns.index(n)]
        ax.text(0.0, 1.03, rf"{name}, $n = {n}$", transform=ax.transAxes, ha="left", va="bottom", fontsize=11)
        ax.text(1.0, 1.03, rf"$e = {err:.3f}$", transform=ax.transAxes, ha="right", va="bottom", fontsize=11)
        if row == 2:
            ax.set_xlabel(r"$x$")
        if col == 0:
            ax.set_ylabel(r"$y$")
        panel_label(ax, "abcdef"[2 * row + col], dx=-0.14, dy=1.03)
cb = fig.colorbar(im, ax=axes, shrink=0.55, pad=0.02)
cb.set_label(r"$q(\boldsymbol{x}_j\,|\,H_n)\,/\,\max_j q$")
out = f"figures/fig_posterior_velocity_{a.set}" + ("" if a.flip else "_typical") + ("" if a.rank == 0.5 else f"_r{a.rank:g}")
save(fig, out)
print(f"episode {ep} (policy {z['policy'][ep]}); rank {a.rank}; wrote {out}.{{png,eps}}")

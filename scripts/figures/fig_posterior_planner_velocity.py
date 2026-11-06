"""Source posterior with and without velocity sensing along a planner-chosen path (E49 data,
scripts/postnet_closedloop.py --runs_vel runs_vel --save_q --q_ns 10 20 30 40 80).

The path is chosen by one planner (--plan, default the corrected planner m3, EIG on M3), which does not use the learned
posterior, so both columns see the same path and the same concentration readings; only the velocity input differs.
Columns: learned posterior with the physics input, without (B) and with (Bvel) velocity; rows: n = --ns moves.
The task is chosen by rule: among the tasks where velocity turns a failure (e >= 0.1) into a success (e < 0.1) at
n = --pick, the one at the median (--rank) of the error without velocity. The printed counts give how often each flip
occurs over the 100 tasks.
Colour = q(x_j | H_n) / max_j q, linear, over the prior support. Green ring and cross: true source; line: sensor path
(the closed-loop output stores no readings, so detections are not marked); open square: current position; solid contour: 90% highest-posterior region.
Colour is used because a 2-D probability field over six panels is unreadable in grey at this size.
Run (from repo/): python scripts/figures/fig_posterior_planner_velocity.py [--set weak] [--plan m1]
"""
import argparse, os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from plotstyle import use_style, panel_label, save

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="strong")
ap.add_argument("--plan", default="m3")
ap.add_argument("--pick", type=int, default=30)
ap.add_argument("--rank", type=float, default=0.5)
ap.add_argument("--ns", type=int, nargs=3, default=[10, 20, 30])
a = ap.parse_args()
z = np.load(f"docs/research/data/E59/{a.set}_closedloop_q.npz")
gx, gy, qn = z["grid_x"], z["grid_y"], list(z["q_ns"])
pols, ests = list(z["policies"]), list(z["estimators"])
pi = pols.index(a.plan)
COLS = [(ests.index("B"), r"no velocity"), (ests.index("Bvel"), r"with velocity")]
PLAN = {"m1": r"physics infotaxis", "m3": r"corrected infotaxis", "chase": r"plume tracing", "surgecast": r"cast-and-surge"}
h = gx[1] - gx[0]
e_no, e_v = z["err"][pi, :, a.pick, COLS[0][0]], z["err"][pi, :, a.pick, COLS[1][0]]
up, down = (e_no >= 0.1) & (e_v < 0.1), (e_no < 0.1) & (e_v >= 0.1)
print(f"{a.set} {a.plan} n={a.pick}: velocity turns {up.sum()} failures into successes and {down.sum()} successes "
      f"into failures (of {len(e_no)} tasks)")
cand = np.where(up)[0]
task = int(cand[np.argsort(e_no[cand])[int(round(a.rank * (len(cand) - 1)))]])
src, path = z["src"][task], z["path"][pi, task]
X0, X1, Y0, Y1 = gx[0] - h / 2, gx[-1] + h / 2, gy[0] - h / 2, gy[-1] + h / 2


def hpd(q, level=0.9):
    o = np.argsort(q)[::-1]
    m = np.zeros(q.size, bool)
    m[o[:np.searchsorted(np.cumsum(q[o]), level) + 1]] = True
    return m


use_style(base=12)
fig, axes = plt.subplots(3, 2, figsize=(8.0, 6.6), constrained_layout=True, sharex=True, sharey=True)
for row, n in enumerate(a.ns):
    for col, (ei, name) in enumerate(COLS):
        ax = axes[row, col]
        q = z["q"][pi, task, qn.index(n), ei].astype(np.float64)
        q /= q.sum()
        Q = (q / q.max()).reshape(len(gx), len(gy))
        im = ax.pcolormesh(gx, gy, Q.T, cmap="magma_r", vmin=0, vmax=1, shading="nearest", rasterized=True)
        ax.contour(gx, gy, hpd(q).reshape(len(gx), len(gy)).T.astype(float), levels=[0.5], colors="k", linewidths=0.9)
        p = path[:n + 1]
        ax.plot(p[:, 0], p[:, 1], color="#4a7ab5", lw=1.0)
        ax.plot(*path[n], "s", ms=5.5, mfc="white", mec="#4a7ab5", mew=1.2, zorder=5)
        ax.plot(*src, "o", ms=13, mfc="none", mec="#2ca02c", mew=1.6, zorder=6)
        ax.plot(*src, "+", ms=5, mew=1.0, color="#2ca02c", zorder=6)
        ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.set_aspect("equal")
        ax.text(0.0, 1.03, rf"{name}, $n = {n}$", transform=ax.transAxes, ha="left", va="bottom", fontsize=11)
        ax.text(1.0, 1.03, rf"$e = {z['err'][pi, task, n, ei]:.3f}$", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=11)
        if row == 2:
            ax.set_xlabel(r"$x$")
        if col == 0:
            ax.set_ylabel(r"$y$")
        panel_label(ax, "abcdef"[2 * row + col], dx=-0.14, dy=1.03)
cb = fig.colorbar(im, ax=axes, shrink=0.55, pad=0.02)
cb.set_label(r"$q(\boldsymbol{x}_j\,|\,H_n)\,/\,\max_j q$")
out = f"figures/fig_posterior_planner_velocity_{a.set}_{a.plan}"
save(fig, out)
print(f"task {task}; wrote {out}.{{png,eps}}")

"""Source posteriors on one held-out DNS episode, v6 redesign of fig_posterior_compare (E48 data,
scripts/postnet_maps.py --runs runs_real --runs_vel runs_vel --map_all; docs/research/data/E59/{set}_maps.npz).

Selection rule (fixed before plotting, identical to fig_posterior_compare): among the held-out test episodes whose path
was planned by physics infotaxis (policy 0; 100 of 300), take the episode with the MEDIAN error of the learned
posterior (with physics input) at n = 40 (rank 0.5 of the sorted errors). Strong forcing: episode 105 (E58).
Rows: (a,b) physics posterior, (c,d) learned posterior (with physics input); the velocity variant is in the appendix;
columns: n = 40 and n = 80 steps. Colour: log10 q(x_j | H_n), probability per grid cell (h = 0.02), on ONE shared
scale for all panels, so that both the spread and the peak height are comparable across posteriors (cells below
1e-4 share the lightest colour). Black contour: 90% highest-posterior region, with its area A_90; e: error of the
posterior mean. Sensor path (grey line, from its start at x = 3.5 outside the frame), detections (white dots, size
growing with the reading), true source (white ring with black edge, centred on the source cell).
Colour is necessary: a 2-D probability field; magma is perceptually uniform and readable in greyscale.
Run (from repo/): py -3.11 scripts/figures/fig_posterior_compare_v6.py [--set weak] [--cols 40] (Fig. 4) or [--cols 20 40] (appendix)
  --cols 40: one row, (a) physics and (b) learned posterior at n = 40 (paper; n = 80 shows the late back-and-forth of the sensor upwind of the source) -> figures/fig_posterior_compare_{set}_row_v6
"""
import argparse, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
sys.path.insert(0, os.path.dirname(__file__))
from plotstyle import panel_label, save
from style_v6 import use_style, TEXTW

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="strong")
ap.add_argument("--cols", type=int, nargs="*", default=[40, 80],
                help="one value: one row, (a) physics and (b) learned posterior at that n")
a = ap.parse_args()
z = np.load(f"docs/research/data/E59/{a.set}_maps.npz")
gx, gy, ns, mapn = z["grid_x"], z["grid_y"], list(z["ns"]), list(z["map_ns"])
meth = list(z["methods"])
ROWS = [("M1", r"physics posterior"), ("B", r"learned posterior")]
COLN = list(a.cols)
ROW1 = len(COLN) == 1                                       # one row: physics | learned at a single n
PICK, RANK = 40, 0.5
h = gx[1] - gx[0]
e0 = z["map_episodes"]
cand = np.where(z["policy"][e0] == 0)[0]
eb = z["err"][e0[cand], meth.index("B"), ns.index(PICK)]
k = int(cand[np.argsort(eb)[int(round(RANK * (len(eb) - 1)))]])
ep = int(e0[k])
src, pos, tok = z["src"][ep], z["pos"][ep], z["tok"][ep]
det = tok[:, 0] > 0.5
X0, X1, Y0, Y1 = gx[0] - h / 2, gx[-1] + h / 2, gy[0] - h / 2, gy[-1] + h / 2
VMIN, VMAX = -4.0, 0.0


def hpd(q, level=0.9):
    o = np.argsort(q)[::-1]
    m = np.zeros(q.size, bool)
    m[o[:np.searchsorted(np.cumsum(q[o]), level) + 1]] = True
    return m


use_style()
if ROW1:
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.05), sharex=True, sharey=True, squeeze=False,
                             gridspec_kw=dict(wspace=0.12, left=0.07, right=0.85, bottom=0.2, top=0.97))
    axes = axes.T                                            # axes[row = posterior, col = 0]
else:
    fig, axes = plt.subplots(2, 2, figsize=(TEXTW, 3.85), sharex=True, sharey=True,
                             gridspec_kw=dict(hspace=0.12, wspace=0.12, left=0.07, right=0.85, bottom=0.1, top=0.98))
for row, (mk, name) in enumerate(ROWS):
    mi = meth.index(mk)
    for col, n in enumerate(COLN):
        ax = axes[row, col]
        q = z["q"][k, mi, mapn.index(n)].astype(np.float64)
        q /= q.sum()
        L = np.log10(np.maximum(q, 1e-30)).reshape(len(gx), len(gy))
        im = ax.pcolormesh(gx, gy, L.T, cmap="magma_r", vmin=VMIN, vmax=VMAX, shading="nearest", rasterized=True,
                           zorder=-10)
        m90 = hpd(q)
        ax.contour(gx, gy, m90.reshape(len(gx), len(gy)).T.astype(float), levels=[0.5], colors="k", linewidths=0.8)
        p = pos[:n + 1]
        seg = {}                                             # draw each segment once, shaded by its last visit
        for i in range(n):
            seg[tuple(sorted((tuple(p[i].round(4)), tuple(p[i + 1].round(4)))))] = i
        ax.add_collection(LineCollection([np.array(s) for s in seg], colors=[str(0.75 - 0.6 * i / n) for i in seg.values()],
                                         linewidths=0.9, zorder=2))
        ax.plot(*p[-1], "s", ms=4.5, mfc="0.15", mec="white", mew=0.6, zorder=5)
        dd = det[:n + 1]
        ax.scatter(p[dd, 0], p[dd, 1], s=4 + 14 * tok[:n + 1][dd, 1], facecolor="white", edgecolor="0.15",
                   linewidths=0.6, zorder=3)
        ax.plot(*src, "o", ms=14, mfc="none", mec="k", mew=2.6, zorder=6)
        ax.plot(*src, "o", ms=14, mfc="none", mec="white", mew=1.2, zorder=7)
        ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.set_aspect("equal")
        ax.set_xticks([0.5, 1.0, 1.5, 2.0]); ax.set_yticks([-0.4, 0, 0.4])
        e = z["err"][ep, mi, ns.index(n)]
        ax.text(0.975, 0.04, rf"{name}, $n={n}$", transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
        ax.text(0.03, 0.04, rf"$e={e:.3f}$" "\n" rf"$A_{{90}}={m90.sum() * h * h:.3g}$", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9, linespacing=1.2)
        if ROW1:                                             # one row: (a) physics, (b) learned
            ax.set_xlabel(r"$x$", labelpad=1)
            if row == 0:
                ax.set_ylabel(r"$y$", labelpad=1)
            panel_label(ax, "ab"[row], dx=-0.02 if row else -0.1, dy=1.0, va="top", fontsize=10)
            continue
        if row == 1:
            ax.set_xlabel(r"$x$", labelpad=1)
        if col == 0:
            ax.set_ylabel(r"$y$", labelpad=1)
        panel_label(ax, "abcd"[2 * row + col], dx=-0.02 if col else -0.1, dy=1.0, va="top", fontsize=10)
cax = fig.add_axes([0.87, 0.27, 0.02, 0.63] if ROW1 else [0.87, 0.2, 0.02, 0.7])
cb = fig.colorbar(im, cax=cax, extend="min")
cb.set_label(r"$\log_{10} q(\boldsymbol{x}_j\,|\,H_n)$")
cb.ax.tick_params(direction="in")
out = (f"figures/fig_posterior_compare_{a.set}_row_v6" if ROW1 else f"figures/fig_posterior_compare_{a.set}_v6")
save(fig, out)
print(f"episode {ep}; wrote {out}.{{png,eps}}")

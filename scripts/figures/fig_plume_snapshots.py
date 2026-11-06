"""Instantaneous DNS plume behind the E40 tasks (data: scripts/plume_snapshots.py), shown in colour at high levels.

Columns: (left) eps = 5e-3, task 14; (right) eps = 1.2e-3, task 44. Rows: t at reading n = 20, 40, 80.
Colour: c / max c of that snapshot, linear, blank below 0.05 (log readings are a factor 20 above the detection
threshold there). The stored DNS field is sampled every 0.098 in x and 0.131 in y (stride 3 x 4 of the 384 x 192 run);
the 0.01 grid here is bilinear between those samples. Green ring: source.
Colour is used because a thresholded concentration field with filaments needs more levels than grey resolves.
Run (from repo/): python scripts/figures/fig_plume_snapshots.py
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from plotstyle import use_style, panel_label, save

D = "docs/research/data/E40"
CASES = [("strong", 14, r"$\varepsilon = 5\times10^{-3}$"), ("weak", 44, r"$\varepsilon = 1.2\times10^{-3}$")]
use_style(base=12)
fig, axes = plt.subplots(3, 2, figsize=(8.4, 5.0), constrained_layout=True, sharex=True, sharey=True)
for col, (st, task, lab) in enumerate(CASES):
    pl = np.load(f"{D}/{st}_plume_task{task}.npz")
    for row, n in enumerate(pl["ns"]):
        ax = axes[row, col]
        c = pl["c"][row].astype(np.float64)
        C = np.ma.masked_less(c / c.max(), 0.05)
        im = ax.pcolormesh(pl["xg"], pl["yg"], C.T, cmap="inferno_r", vmin=0.05, vmax=1, shading="nearest", rasterized=True)
        ax.plot(*pl["src"], "o", ms=12, mfc="none", mec="#2ca02c", mew=1.6)
        ax.set_aspect("equal")
        ax.set_xlim(0, 3.7); ax.set_ylim(-0.85, 0.85)
        ax.text(0.0, 1.03, rf"{lab}, $n = {n}$", transform=ax.transAxes, ha="left", va="bottom", fontsize=11)
        if row == 2:
            ax.set_xlabel(r"$x$")
        if col == 0:
            ax.set_ylabel(r"$y$")
        panel_label(ax, "abcdef"[2 * row + col], dx=-0.14, dy=1.03)
cb = fig.colorbar(im, ax=axes, shrink=0.55, pad=0.02)
cb.set_label(r"$c\,/\,\max c$")
save(fig, "figures/fig_plume_snapshots")
print("wrote figures/fig_plume_snapshots.{png,eps}")

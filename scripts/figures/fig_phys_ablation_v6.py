"""E50: replay ablation of the physics posterior along the E49 paths (docs/research/data/E50_{strong,weak}_ablation.npz,
scripts/checks/phys_ablation_replay.py). One row per variant of the physics posterior, one marker per motion policy that
chose the path. (a, b) success P(e_80 < 0.1); (c, d) coverage of the 90% region at n = 80; left strong, right weak.
Colour = motion policy, as in every v6 result figure (Okabe-Ito, colour-blind safe, each policy has its own marker).
Run (from repo/): py -3.11 scripts/figures/fig_phys_ablation_v6.py
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(__file__))
from plotstyle import panel_label, save
from style_v6 import use_style, POL, SETS, TEXTW

ROWS = [("base", r"physics posterior"), ("k1", r"$\kappa_k=1$"), ("k05", r"$\kappa_k/2$"), ("k025", r"$\kappa_k/4$"),
        ("bgM3", r"corrected background"), ("plumeM3", r"corrected in-plume")]
PATHS = ["m3", "m1", "chase"]
DY = {"m3": 0.2, "m1": 0.0, "chase": -0.2}               # small offsets so equal values stay visible
use_style()
fig, axes = plt.subplots(2, 2, figsize=(TEXTW, 4.1), sharey=True,
                         gridspec_kw=dict(wspace=0.06, hspace=0.42, left=0.21, right=0.99, bottom=0.09, top=0.92))
for j, st in enumerate(["strong", "weak"]):
    z = np.load(f"docs/research/data/E50_{st}_ablation.npz")
    P, V = list(z["policies"]), list(z["variants"])
    for i, (metric, lab) in enumerate([("err", r"$P(e_{80}<0.1)$"), ("in90", r"$P(\bm x_s\in R_{0.9})$ at $n=80$")]):
        ax = axes[i, j]
        for g, (v, _) in enumerate(ROWS):
            ax.axhspan(-g - 0.5, -g + 0.5, color="0.95" if g % 2 == 0 else "white", lw=0, zorder=0)
            for pk in PATHS:
                a = z[metric][P.index(pk), :, 80, V.index(v)]
                val = (a < 0.1).mean() if metric == "err" else a.mean()
                ax.plot(val, -g + DY[pk], ls="none", marker=POL[pk]["marker"], ms=5.5, mfc=POL[pk]["color"],
                        mec=POL[pk]["color"], alpha=0.9, zorder=3)
        ax.set_xlim(-0.03, 1.03); ax.set_ylim(-len(ROWS) + 0.5, 0.5)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
        ax.tick_params(axis="y", which="both", left=False, right=False)
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        ax.tick_params(axis="x", which="major", direction="out", top=False)
        ax.set_xlabel(lab, labelpad=1)
        if i == 0:
            ax.text(0.5, 1.03, SETS[st], transform=ax.transAxes, ha="center", va="bottom", fontsize=9)
        panel_label(ax, "abcd"[2 * i + j], dx=-0.01 if j else -0.3, dy=1.02)
    axes[0, 0].set_yticks([-g for g in range(len(ROWS))])
    axes[0, 0].set_yticklabels([r for _, r in ROWS])
h = [Line2D([], [], ls="none", marker=POL[k]["marker"], ms=5.5, mfc=POL[k]["color"], mec=POL[k]["color"],
            label=POL[k]["label"] + r" paths") for k in PATHS]
fig.legend(handles=h, loc="upper center", ncol=3, fontsize=8.5, handletextpad=0.3, columnspacing=1.5,
           bbox_to_anchor=(0.6, 1.005))
save(fig, "figures/fig_phys_ablation_v6")
print("wrote figures/fig_phys_ablation_v6.{png,eps}")

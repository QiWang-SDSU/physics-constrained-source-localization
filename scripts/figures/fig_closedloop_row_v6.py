"""Closed-loop localization success vs steps, one-row version of fig_closedloop_v6 for the page budget (E58 data,
docs/research/data/E59/{strong,weak}_closedloop.npz, scripts/postnet_closedloop.py).

(a) strong eps = 5e-3, (b) weak eps = 1.2e-3. y: fraction of the 100 held-out episodes with posterior-mean error
e_n < 0.1 after n steps (raw per-step fractions, no smoothing). Colour + marker = policy (markers every 10 steps).
Solid, filled markers: learned posterior (with physics input, estimator B); dashed, open markers: learned posterior
with velocity (Bvel), same colour. Grey dotted: physics posterior along physics-infotaxis paths (M1 on m1).
Colour: four policy curves cross and cannot be separated in grey (Okabe-Ito, colour-blind safe; vermillion =
physics infotaxis, blue = corrected infotaxis, as in all v6 figures).
Run (from repo/): py -3.11 scripts/figures/fig_closedloop_row_v6.py
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(__file__))
from plotstyle import panel_label, save
from style_v6 import use_style, POLICIES, POST, SETS, TEXTW

N = np.arange(81)
MK = np.arange(0, 81, 10)
DASH = (0, (3.5, 1.5))
DOT = (0, (1, 1.5))
use_style()
fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.7), sharey=True,
                         gridspec_kw=dict(wspace=0.07, left=0.085, right=0.99, bottom=0.165, top=0.765))
for j, st in enumerate(["strong", "weak"]):
    z = np.load(f"docs/research/data/E59/{st}_closedloop.npz")
    P, E, err = list(z["policies"]), list(z["estimators"]), z["err"]
    ax = axes[j]
    ref = (err[P.index("m1"), :, :, E.index("M1")] < 0.1).mean(0)
    ax.plot(N, ref, color=POST["M1"]["color"], ls=DOT, lw=1.3, zorder=1)
    for key, name, col, mk in POLICIES:
        for est, ls, mfc in (("Bvel", DASH, "white"), ("B", "-", col)):
            y = (err[P.index(key), :, :, E.index(est)] < 0.1).mean(0)
            ax.plot(N, y, color=col, ls=ls, lw=1.3, zorder=3)
            ax.plot(MK, y[MK], ls="none", marker=mk, mfc=mfc, mec=col, mew=1.0, ms=4.3, zorder=4)
    ax.set_xlim(0, 80); ax.set_ylim(0, 1.03)
    ax.set_xticks([0, 20, 40, 60, 80]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"$n$", labelpad=1)
    ax.text(0.97, 0.05, SETS[st], transform=ax.transAxes, ha="right", va="bottom")
    panel_label(ax, "ab"[j], dx=-0.02 if j else -0.12, dy=1.0)
axes[0].set_ylabel(r"$P(e_n<0.1)$")
h = [Line2D([], [], color=c, marker=m, mfc=c, mec=c, ms=4.5, lw=1.3, label=l) for _, l, c, m in POLICIES]
h += [Line2D([], [], color="0.2", ls="-", lw=1.3, marker="o", mfc="0.2", mec="0.2", ms=4, label=r"learned posterior"),
      Line2D([], [], color="0.2", ls=DASH, lw=1.3, marker="o", mfc="white", mec="0.2", ms=4,
             label=r"learned, with velocity"),
      Line2D([], [], color=POST["M1"]["color"], ls=DOT, lw=1.3, label=r"physics posterior, physics infotaxis")]
kw = dict(loc="upper center", fontsize=8, handlelength=2.0, columnspacing=1.5, handletextpad=0.5)
fig.legend(handles=h[:4], ncol=4, bbox_to_anchor=(0.54, 1.0), **kw)       # row 1: policies (colour)
fig.legend(handles=h[4:], ncol=3, bbox_to_anchor=(0.54, 0.925), **kw)     # row 2: posteriors (line style)
save(fig, "figures/fig_closedloop_row_v6")
print("wrote figures/fig_closedloop_row_v6.{png,eps}")

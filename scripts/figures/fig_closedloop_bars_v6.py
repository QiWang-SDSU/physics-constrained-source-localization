"""Closed-loop success at a short (n = 30) and the full (n = 80) budget, per policy and posterior: grouped bars that
replace the dumbbell plot fig_closedloop_dots_v6 (same data: docs/research/data/E59/{strong,weak}_closedloop.npz).

(a) strong eps = 5e-3, (b) weak eps = 1.2e-3. One group per motion policy (label in the policy colour, as in all
closed-loop figures); within it one bar per posterior: physics posterior (light grey, M1), learned posterior (dark grey,
B), learned with velocity (black, Bvel). Solid bar: P(e_30 < 0.1); open bar with the same edge: P(e_80 < 0.1).
Values are fractions of the same 100 held-out episodes; binomial standard error <= 0.05.
Greyscale for the posteriors; colour only on the policy labels (Okabe-Ito, colour-blind safe).
Run (from repo/): py -3.11 scripts/figures/fig_closedloop_bars_v6.py
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, os.path.dirname(__file__))
from plotstyle import panel_label, save
from style_v6 import use_style, POL, SETS, TEXTW

ORDER = ["m1", "m3", "surgecast", "chase"]
LAB = {"m1": "physics\ninfotaxis", "m3": "corrected\ninfotaxis", "surgecast": "cast-and-\nsurge",
       "chase": "plume\ntracing"}
EST = [("M1", r"physics posterior", "0.72"), ("B", r"learned posterior", "0.38"), ("Bvel", r"learned, with velocity", "0.0")]
BW = 0.26
use_style()
fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.55), sharey=True,
                         gridspec_kw=dict(wspace=0.06, left=0.075, right=0.995, bottom=0.2, top=0.86))
for j, st in enumerate(["strong", "weak"]):
    z = np.load(f"docs/research/data/E59/{st}_closedloop.npz")
    P, E, err = list(z["policies"]), list(z["estimators"]), z["err"]
    ax = axes[j]
    for g, key in enumerate(ORDER):
        for r, (est, _, col) in enumerate(EST):
            x = g + (r - 1) * BW
            s = err[P.index(key), :, :, E.index(est)] < 0.1
            p30, p80 = s[:, 30].mean(), s[:, 80].mean()
            ax.bar(x, p80, width=BW * 0.92, fc="white", ec=col, lw=1.1, zorder=2)
            ax.bar(x, p30, width=BW * 0.92, fc=col, ec=col, lw=1.1, zorder=3)
            print(st, key, est, f"{p30:.2f} {p80:.2f}")
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([LAB[k] for k in ORDER], linespacing=1.0)
    for t, k in zip(ax.get_xticklabels(), ORDER):
        t.set_color(POL[k]["color"])
    ax.tick_params(axis="x", which="both", length=0, pad=3)
    ax.tick_params(axis="y", which="minor", left=False, right=False)
    ax.tick_params(axis="y", which="major", right=False)
    ax.set_xlim(-0.55, len(ORDER) - 0.45); ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.0, 1.02, SETS[st], transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5)
    panel_label(ax, "ab"[j], dx=-0.02 if j else -0.1, dy=1.02)
axes[0].set_ylabel(r"$P(e_n<0.1)$", labelpad=2)
h = [Patch(fc=c, ec=c, label=l) for _, l, c in EST]
h += [Patch(fc="none", ec="none", label=r"solid: $n=30$; open: $n=80$")]
fig.legend(handles=h, loc="upper center", ncol=4, fontsize=8.5, handlelength=1.1, columnspacing=1.1,
           bbox_to_anchor=(0.54, 1.01), handletextpad=0.4, frameon=False)
save(fig, "figures/fig_closedloop_bars_v6")
print("wrote figures/fig_closedloop_bars_v6.{png,eps}")

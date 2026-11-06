"""New result panels from existing closed-loop data (E58, scripts/postnet_closedloop.py; docs/research/data/E58).

(a) Calibration at n = 80 on the 100 held-out closed-loop episodes: empirical coverage, the fraction of episodes whose
    true source cell lies inside the nominal alpha highest-posterior region, versus alpha (alpha = 0..0.995, step 0.005).
    Computed from the stored posterior maps {set}_closedloop_q.npz (q at n = 80); the HPD level of the true cell is the
    posterior mass strictly above it, as in scripts/postnet_maps.py, and the true cell is the grid cell nearest the
    source. At alpha = 0.9 this reproduces in90 of {set}_closedloop.npz (e.g. strong, physics posterior on corrected-
    infotaxis paths: 0.13). Line style = posterior (dotted: physics posterior M1; solid: learned posterior B);
    colour = the policy that drove the path (physics infotaxis m1, corrected infotaxis m3; same colours as
    fig_closedloop_v6). Below the diagonal: overconfident; above: conservative.
(b) Reaching is not localizing: x = median over the 100 episodes of the closest approach to the source so far,
    d_n = min_{k<=n} |p_k - x_s| (log scale, reversed so that moving right = closer); y = P(e_n < 0.1) of the learned
    posterior (B); one curve per policy parametrized by n = 0..80; open marker n = 30, filled marker n = 80 (all eight drawn on top of the curves); arrowheads at n = 22 and 45 show increasing n.
Both are raw fractions/medians from the data (no smoothing).
Colour: policy identity, shared with fig_closedloop_v6 (Okabe-Ito, colour-blind safe).
Run (from repo/): py -3.11 scripts/figures/fig_calib_reach_v6.py [--set weak]
"""
import argparse, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter
sys.path.insert(0, os.path.dirname(__file__))
from plotstyle import panel_label, save
from style_v6 import use_style, POLICIES, POL, POST, SETS, TEXTW

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="strong")
a = ap.parse_args()
D = "docs/research/data/E59"
c = np.load(f"{D}/{a.set}_closedloop.npz")
cq = np.load(f"{D}/{a.set}_closedloop_q.npz")
P, E = list(c["policies"]), list(c["estimators"])
gx, gy = cq["grid_x"], cq["grid_y"]
src = cq["src"]
jt = np.abs(src[:, 0, None] - gx[None]).argmin(1) * len(gy) + np.abs(src[:, 1, None] - gy[None]).argmin(1)
iq = list(cq["q_ns"]).index(80)
AL = np.linspace(0, 0.995, 200)


def coverage(pol, est):
    q = cq["q"][P.index(pol), :, iq, E.index(est)].astype(np.float64)
    q /= q.sum(1, keepdims=True)
    qt = q[np.arange(len(q)), jt]
    lev = (q * (q > qt[:, None])).sum(1)
    chk = c["in90"][P.index(pol), :, 80, E.index(est)].mean()
    assert abs(np.mean(lev < 0.9) - chk) < 0.015, (pol, est, np.mean(lev < 0.9), chk)
    return np.array([np.mean(lev < x) for x in AL])


use_style()
fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.85),
                         gridspec_kw=dict(wspace=0.3, left=0.085, right=0.985, bottom=0.16, top=0.8))
# (a) calibration
ax = axes[0]
ax.plot([0, 1], [0, 1], color="0.75", lw=0.9, zorder=0)
for pol in ["m1", "m3"]:
    for est, ls in (("M1", (0, (1, 1.3))), ("B", "-")):
        ax.plot(AL, coverage(pol, est), color=POL[pol]["color"], ls=ls, lw=1.5 if est == "B" else 1.6,
                drawstyle="steps-post")
ax.set_xlim(0, 1); ax.set_ylim(-0.12, 1)        # strip below 0 holds the direct label of the curve lying on 0
ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
ax.set_xlabel(r"$\alpha$", labelpad=1)
ax.set_ylabel(r"$P(\boldsymbol{x}_s\in R_\alpha)$")
ax.text(0.04, 0.96, SETS[a.set] + r", $n=80$" + "\n" + r"infotaxis paths", transform=ax.transAxes, ha="left",
        va="top", multialignment="left", fontsize=8.5)
ax.text(0.985, -0.065, r"physics posterior, corrected infotaxis", ha="right", va="center", fontsize=8,
        color=POL["m3"]["color"])
panel_label(ax, "a", dx=-0.13, dy=1.0)
# (b) reaching vs localizing
ax = axes[1]
d = np.linalg.norm(c["path"] - c["src"][None, :, None, :], axis=-1)
dmin = np.minimum.accumulate(d, axis=2)
for key, name, col, mk in POLICIES:
    pi = P.index(key)
    x = np.median(dmin[pi], axis=0)
    y = (c["err"][pi, :, :, E.index("B")] < 0.1).mean(0)
    ax.plot(x, y, color=col, ls="-", lw=1.4, zorder=2)
    for n0 in (22, 45):                            # arrowheads along the curve: direction of increasing n
        n1 = n0 + 1
        while n1 < 80 and abs(np.log(x[n1] / x[n0])) < 0.04 and abs(y[n1] - y[n0]) < 0.02:
            n1 += 1
        if n1 > n0 + 6:                            # curve is (near) stationary here: no arrow
            continue
        ax.annotate("", xy=(x[n1], y[n1]), xytext=(x[n0], y[n0]), zorder=3,
                    arrowprops=dict(arrowstyle="-|>,head_length=0.75,head_width=0.38", color=col, lw=0, mutation_scale=10,
                                    shrinkA=0, shrinkB=0))
    ax.plot(x[30], y[30], ls="none", marker=mk, ms=7, mfc="white", mec=col, mew=1.6, zorder=6)
    ax.plot(x[80], y[80], ls="none", marker=mk, ms=7, mfc=col, mec=col, mew=1.6, zorder=6)
ax.set_xscale("log")
ax.set_xlim(2.4, 0.07); ax.set_ylim(0, 1.05)
ax.xaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1, 2]))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_yticks([0, 0.5, 1])
ax.set_xlabel(r"$\tilde d_n$", labelpad=1)
ax.set_ylabel(r"$P(e_n<0.1)$")
ax.text(0.97, 0.04, r"open: $n=30$" "\n" r"filled: $n=80$", transform=ax.transAxes, ha="right", va="bottom",
        multialignment="right", fontsize=8.5)
panel_label(ax, "b", dx=-0.13, dy=1.0)
h = [Line2D([], [], color=col, marker=mk, mfc=col, mec=col, ms=5, lw=1.4, label=name) for _, name, col, mk in POLICIES]
h.append(Line2D([], [], color="0.2", ls="-", lw=1.5, label=r"learned posterior"))
h.append(Line2D([], [], color="0.2", ls=(0, (1, 1.3)), lw=1.6, label=r"physics posterior"))
fig.legend(handles=h, loc="upper center", ncol=3, fontsize=8.5, handlelength=2.0, columnspacing=1.6,
           bbox_to_anchor=(0.53, 1.0), labelspacing=0.3)
out = "figures/fig_calib_reach" + ("" if a.set == "strong" else f"_{a.set}") + "_v6"
save(fig, out)
print(f"wrote {out}.{{png,eps}}")

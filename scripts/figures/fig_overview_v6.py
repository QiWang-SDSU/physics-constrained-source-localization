"""Overview (teaser) figure: the closed loop of mobile source localization and the two places where the paper asks
"physics or data?".

Left column: instantaneous DNS plume (E40 strong forcing, task 14, snapshot at n = 20; scripts/plume_snapshots.py)
with the sensor path chosen by infotaxis on the corrected observation model (policy m3, E40 closed loop), and below it
the measurement history H_n of that path (log readings z_k; bars at the floor: no detection; dotted: log m_min).
Middle: the posterior q(x_j | H_n) over the candidate cells at n = 20 on the same path, physics posterior (M1) and
learned posterior (B), both from docs/research/data/E40/strong_closedloop_q.npz. Right: the motion policy (infotaxis
with the physics or the corrected observation model; plume tracing and cast-and-surge). The arrow over the top closes
the loop (next position p_{n+1}).
Colour: the posterior maps use magma_r (a 2-D probability field, as in the result figures); one accent hue marks
physics (Okabe-Ito vermillion #D55E00) and one marks learned / data-corrected components (Okabe-Ito blue #0072B2),
a colour-blind-safe pair that also separates in greyscale by lightness. The plume itself is greyscale.
Run (from repo/): py -3.11 scripts/figures/fig_overview_v6.py
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, os.path.dirname(__file__))
from plotstyle import use_style, save

PHYS, LEARN, GREY = "#D55E00", "#0072B2", "0.45"
D = "docs/research/data/E40"
TASK, POL, N = 14, "m3", 20
FS = 10.0                                                    # prints at ~9 pt at the 6.0 in PMLR text width

q = np.load(f"{D}/strong_closedloop_q.npz")
pl = np.load(f"{D}/strong_plume_task{TASK}.npz")
pols, ests, qn = list(q["policies"]), list(q["estimators"]), list(q["q_ns"])
gx, gy = q["grid_x"], q["grid_y"]
h = gx[1] - gx[0]
path = q["path"][pols.index(POL), TASK][:N + 1]
src = q["src"][TASK]
zr = pl["z"][list(pl["policies"]).index(POL)][:N + 1]     # log readings along this path (nan = no detection)
thr = np.log(float(pl["m_min"]))
det = ~np.isnan(zr)

use_style(base=FS)
W, H = 6.5, 2.55
fig = plt.figure(figsize=(W, H))
bg = fig.add_axes([0, 0, 1, 1], zorder=-10)
bg.set_xlim(0, W); bg.set_ylim(0, H); bg.axis("off")


def axes_in(x, y, w, hh, **kw):
    return fig.add_axes([x / W, y / H, w / W, hh / H], **kw)


def arrow(p0, p1, color="k", lw=1.1, ls="-", rad=0.0, ms=10):
    bg.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def rbox(x, y, w, hh, ec="k", fc="white", lw=1.0, ls="-"):
    bg.add_patch(FancyBboxPatch((x, y), w, hh, boxstyle="round,pad=0.0,rounding_size=0.06", ec=ec, fc=fc, lw=lw,
                                ls=ls))


# ---------------------------------------------------------------- column 1: plume + sensor path, history H_n
X0p, Wp = 0.10, 2.30
Hp = Wp * 1.7 / 3.7
Y0p = 1.03
axp = axes_in(X0p, Y0p, Wp, Hp)
c = pl["c"][qn.index(N)].astype(np.float64)
axp.pcolormesh(pl["xg"], pl["yg"], np.ma.masked_less(c / c.max(), 0.04).T, cmap="Greys", vmin=0.0, vmax=0.9,
               shading="nearest", rasterized=True)
axp.plot(path[:, 0], path[:, 1], color="k", lw=1.0)
axp.plot(path[det, 0], path[det, 1], "o", ms=3.2, mfc="white", mec="k", mew=0.7)
axp.plot(*path[-1], "s", ms=5.5, mfc="k", mec="k")
axp.plot(*src, "o", ms=8, mfc="none", mec="k", mew=1.2)
axp.plot(*src, "+", ms=5, mew=1.0, color="k")
axp.set_xlim(0, 3.7); axp.set_ylim(-0.85, 0.85); axp.set_aspect("equal")
axp.set_xticks([]); axp.set_yticks([])
for s in axp.spines.values():
    s.set_linewidth(0.8)
axp.annotate("", xy=(2.95, -0.70), xytext=(2.35, -0.70), arrowprops=dict(arrowstyle="-|>", lw=1.1, color="k"))
axp.text(3.05, -0.70, r"$U$", ha="left", va="center")
halo = [pe.withStroke(linewidth=1.6, foreground="white")]
axp.text(src[0] - 0.16, src[1] + 0.13, r"source", ha="left", va="bottom", fontsize=FS, path_effects=halo)
axp.text(path[-1, 0] - 0.12, path[-1, 1], r"sensor $\boldsymbol{p}_n$", ha="right", va="center", fontsize=FS)
x0s, x1s, y0s, y1s = gx[0] - h / 2, gx[-1] + h / 2, gy[0] - h / 2, gy[-1] + h / 2   # prior support = candidate cells
axp.plot([x0s, x1s, x1s, x0s, x0s], [y0s, y0s, y1s, y1s, y0s], color="0.35", ls=":", lw=0.9)
axp.text(x0s + 0.06, y0s + 0.05, r"candidate cells", ha="left", va="bottom", fontsize=FS, color="0.3")
bg.text(X0p, Y0p + Hp + 0.05, r"DNS plume", ha="left", va="bottom")

# measurement history: z_k against step k
X0h, Y0h, Wh, Hh = 0.42, 0.30, 1.98, 0.42
axh = axes_in(X0h, Y0h, Wh, Hh)
k = np.arange(N + 1)
floor = thr - 1.4
axh.vlines(k[det], floor, zr[det], color="k", lw=1.0)
axh.plot(k[det], zr[det], "o", ms=2.6, color="k")
axh.plot(k[~det], np.full((~det).sum(), floor), "|", ms=4, color=GREY, mew=0.9)
axh.axhline(thr, color="k", ls=":", lw=0.8)
axh.set_xlim(-1, N + 1); axh.set_ylim(floor - 0.3, 1.6)
axh.set_xticks([]); axh.set_yticks([])
for side in ("top", "right"):
    axh.spines[side].set_visible(False)
for s in ("left", "bottom"):
    axh.spines[s].set_linewidth(0.8)
axh.set_ylabel(r"$z_k$", labelpad=2, fontsize=FS)
axh.text(N + 1, floor - 0.45, r"$k$", ha="right", va="top", fontsize=FS)
bg.text(X0h, Y0h + Hh + 0.05, r"measurement history $H_n$", ha="left", va="bottom")
arrow((X0p + 0.17, Y0p - 0.02), (X0p + 0.17, Y0h + Hh + 0.02))
bg.text(X0p + 0.12, 0.5 * (Y0p + Y0h + Hh) - 0.0, r"$z_n$", ha="right", va="center")

# ---------------------------------------------------------------- column 2: posterior, physics vs learned
X0q = 2.78
Wm = 0.90
Hm = Wm * (gy[-1] - gy[0] + h) / (gx[-1] - gx[0] + h)
rows = [(ests.index("M1"), r"physics", r"adjoint $+$" "\n" r"likelihood", PHYS, 1.33),
        (ests.index("B"), r"learned", r"GRU $+$" "\n" r"decoder", LEARN, 0.36)]
for ei, name, how, col, y0 in rows:
    axm = axes_in(X0q, y0, Wm, Hm)
    qq = q["q"][pols.index(POL), TASK, qn.index(N), ei].astype(np.float64)
    Q = (qq / qq.max()).reshape(len(gx), len(gy))               # every cell drawn, same colour range for both maps
    axm.pcolormesh(gx, gy, Q.T, cmap="magma_r", vmin=0, vmax=1, shading="nearest", rasterized=True)
    axm.plot(*src, "o", ms=6.5, mfc="none", mec="k", mew=1.0)
    axm.plot(*path[-1], "s", ms=3.5, mfc="k", mec="k")
    axm.set_xlim(gx[0] - h / 2, gx[-1] + h / 2); axm.set_ylim(gy[0] - h / 2, gy[-1] + h / 2)
    axm.set_aspect("equal"); axm.set_xticks([]); axm.set_yticks([])
    for s in axm.spines.values():
        s.set_edgecolor(col); s.set_linewidth(1.6)
    bg.text(X0q + Wm + 0.08, y0 + Hm - 0.02, name, ha="left", va="top", color=col)
    bg.text(X0q + Wm + 0.08, y0 + Hm - 0.20, how, ha="left", va="top", linespacing=1.15)
    arrow((X0h + Wh + 0.05, Y0h + Hh / 2), (X0q - 0.05, y0 + Hm / 2), rad=0.0)
bg.text(X0q, 1.98, r"posterior $q(\boldsymbol{x}_j\,|\,H_n)$", ha="left", va="bottom")

# ---------------------------------------------------------------- column 3: motion policy
X0c, Wc = 4.86, 1.48
arrow((X0q + Wm + 0.78, 1.33 + Hm / 2), (X0c - 0.04, 1.42), rad=0.0)
arrow((X0q + Wm + 0.78, 0.36 + Hm / 2), (X0c - 0.04, 1.30), rad=0.0)
# infotaxis box
yi, hi = 1.04, 0.82
rbox(X0c, yi, Wc, hi)
bg.text(X0c + Wc / 2, yi + hi - 0.06, r"infotaxis with the", ha="center", va="top")
tw = Wc - 0.16
for i, (lab, col, ls) in enumerate([(r"physics model", PHYS, "-"), (r"corrected model", LEARN, (0, (3, 1.5)))]):
    yt = yi + 0.34 - i * 0.27
    rbox(X0c + 0.08, yt, tw, 0.22, ec=col, lw=1.4, ls=ls)         # corrected model dashed: not the learned posterior
    bg.text(X0c + 0.08 + tw / 2, yt + 0.11, lab, ha="center", va="center", color=col)
# heuristics box
yh, hh2 = 0.30, 0.55
rbox(X0c, yh, Wc, hh2, ec=GREY)
bg.text(X0c + Wc / 2, yh + hh2 / 2, r"plume tracing," "\n" r"cast-and-surge", ha="center", va="center",
        linespacing=1.2)
bg.text(X0c, 1.98, r"motion policy", ha="left", va="bottom")
bg.text(0.5 * (X0q + X0c + Wc), 2.20, r"\textit{physics or data?}", ha="center", va="bottom", fontsize=FS + 1)

# ---------------------------------------------------------------- loop closure: policy -> next sensor position
ytop = 2.42
xr = X0c + Wc + 0.08
xl = X0p + path[-1, 0] / 3.7 * Wp
bg.plot([X0c + Wc, xr], [yi + hi / 2] * 2, color="k", ls="-", lw=1.1, solid_capstyle="butt")
bg.plot([X0c + Wc, xr], [yh + hh2 / 2] * 2, color="k", ls="-", lw=1.1, solid_capstyle="butt")
bg.plot([xr, xr, xl], [yh + hh2 / 2, ytop, ytop], color="k", ls="-", lw=1.1, solid_capstyle="butt", solid_joinstyle="miter")
xsn = X0p + path[-1, 0] / 3.7 * Wp
bg.plot([xsn, xsn], [ytop, Y0p + Hp], color="k", ls="-", lw=1.1, solid_capstyle="butt")
axp.annotate("", xy=(path[-1, 0], path[-1, 1] + 0.09), xytext=(path[-1, 0], 0.86), annotation_clip=False,
             arrowprops=dict(arrowstyle="-|>", lw=1.1, color="k", shrinkA=0, shrinkB=0, mutation_scale=10))
bg.text(0.5 * (xl + xr), ytop + 0.03, r"move to $\boldsymbol{p}_{n+1}$", ha="center", va="bottom")

save(fig, "figures/fig_overview_v6")
print("wrote figures/fig_overview_v6.{png,eps}")

"""Schematic of the learned posterior (v7: visual redesign of schematic_posterior_net_v6.py), drawn at the printed size
(PMLR text width 6.0 in, included at \\linewidth).

Data flow:
  (top lane)    measurement history H_n (the real sensor path) -> tokens (d_k, z~_k, p_k, t_k, Delta p_k [, u_k]), one
                column per step k -> embedding MLP -> GRU unrolled over k; its state g_{k-1} before measurement k goes
                down to the decoder;
  (bottom lane) every pair (j, k) of a candidate cell x_j and a sensor position p_k, offset (p_k - x_j)/Lambda ->
                pairwise MLP f (drawn as layers of nodes; 68 (70) -> 64 -> 64 -> 1), shared over all (j, k) -> l_jk ->
                sum over k -> map of sum_k l_jk;
  (right)       physics posterior log b^phys_n (frozen, dashed vermillion) + alpha sum_k l_jk -> softmax over the J cells
                -> q(x_j | H_n); training loss -log q at the true cell j* (ring).
Maps are REAL data: the held-out strong-forcing episode of Fig. 4 (episode 105, physics infotaxis, n = 40) from
docs/research/data/E59/strong_maps.npz, zoomed to x in [0.2, 1.4], y in [-0.3, 0.6].  Physics posterior = M1; output =
B (with physics input); the summed-score map is that of the network without physics input (B0), whose log q^0 equals
sum_k l_jk up to a constant (eq. q0).  All three: log10 q per cell, cells below 1e-6 share the lightest colour.
Colour (paper-wide rule): vermillion = physics, blue = data (trainable blocks); magma_r for the 2-D probability maps.
Run (from repo/): py -3.11 scripts/figures/schematic_posterior_net_v7.py
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
sys.path.insert(0, os.path.dirname(__file__))
from plotstyle import use_style, save

PHYS, LEARN = "#D55E00", "#0072B2"
LEARN_FILL, PHYS_FILL, LOSS_FILL = "#DCEBF6", "#FBE1CF", "0.88"
DASH = (0, (4, 2))
FS = 9.0
FT = FS - 1
use_style(base=FS)
W, H = 6.0, 3.30
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
G = 0.035

# ------------------------------------------------------------------ data (episode of Fig. 4)
z = np.load("docs/research/data/E59/strong_maps.npz")
gx, gy, meth = z["grid_x"], z["grid_y"], list(z["methods"])
EP, N = 105, 40
kk = list(z["map_episodes"]).index(EP)
src, pos, tok = z["src"][EP], z["pos"][EP][:N + 1], z["tok"][EP][:N + 1]
det = tok[:, 0] > 0.5
h = gx[1] - gx[0]
X0, X1, Y0, Y1 = gx[0] - h / 2, gx[-1] + h / 2, gy[0] - h / 2, gy[-1] + h / 2
ZX, ZY = (0.2, 1.4), (-0.3, 0.6)                             # zoom of the maps
MW = 0.92                                                    # map width (in)
MH = MW * (ZY[1] - ZY[0]) / (ZX[1] - ZX[0])


def logq(mk):
    q = z["q"][kk, meth.index(mk), list(z["map_ns"]).index(N)].astype(np.float64)
    q /= q.sum()
    return np.log10(np.maximum(q, 1e-30)).reshape(len(gx), len(gy)).T


def inset(x, y, w, hh):
    a = fig.add_axes([x / W, y / H, w / W, hh / H])
    a.set_xticks([]); a.set_yticks([])
    return a


def mapax(x, y, L, ec="k", ls="-"):
    a = inset(x, y, MW, MH)
    im = a.pcolormesh(gx, gy, L, cmap="magma_r", vmin=-6, vmax=0, shading="nearest", rasterized=True)
    a.plot(*src, "o", ms=7, mfc="none", mec="k", mew=1.8)
    a.plot(*src, "o", ms=7, mfc="none", mec="white", mew=0.7)
    a.set_xlim(*ZX); a.set_ylim(*ZY)
    for s in a.spines.values():
        s.set_edgecolor(ec); s.set_linestyle(ls); s.set_linewidth(0.8 if ec == "k" else 1.3)
    return im, dict(l=x, r=x + MW, b=y, t=y + MH, cx=x + MW / 2, cy=y + MH / 2, ax=a)


def box(x, y, w, hh, lines, fc="white", ec="k", ls="-", lw=1.0, fs=FS):
    ax.add_patch(FancyBboxPatch((x, y), w, hh, boxstyle="round,pad=0,rounding_size=0.05", fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + hh / 2, "\n".join(lines), ha="center", va="center", linespacing=1.2, fontsize=fs)
    return dict(l=x, r=x + w, b=y, t=y + hh, cx=x + w / 2, cy=y + hh / 2)


def arrow(p0, p1, color="k", lw=1.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0))


def line(xs, ys, color="k", ls="-", lw=1.0):
    ax.plot(xs, ys, color=color, ls=ls, lw=lw, solid_joinstyle="miter", solid_capstyle="butt")


def tag(x, y, s, **kw):
    ax.text(x, y, s, ha=kw.pop("ha", "left"), va=kw.pop("va", "bottom"), fontsize=kw.pop("fontsize", FS), **kw)


# ================================================================== top lane: history -> tokens -> MLP -> GRU
HX, HY = (0.1, 3.7), (-0.75, 0.9)                            # full domain view of the path
hw = 1.28
hh = hw * (HY[1] - HY[0]) / (HX[1] - HX[0])
hist = dict(l=0.06, r=0.06 + hw, b=2.08, t=2.08 + hh, cx=0.06 + hw / 2, cy=2.08 + hh / 2)
a = inset(hist["l"], hist["b"], hw, hh)
a.add_patch(Rectangle((X0, Y0), X1 - X0, Y1 - Y0, fill=False, ls=(0, (1.5, 1.5)), lw=0.7, ec="0.35"))
a.plot(pos[:, 0], pos[:, 1], "-", color="0.45", lw=0.7)
a.scatter(pos[det, 0], pos[det, 1], s=5 + 10 * tok[det, 1], facecolor="white", edgecolor="0.1", lw=0.5, zorder=3)
a.plot(*pos[-1], "s", ms=3.5, mfc="0.15", mec="white", mew=0.5, zorder=4)
a.plot(*src, "o", ms=5, mfc="none", mec="k", mew=1.4)
a.set_xlim(*HX); a.set_ylim(*HY)
tag(hist["l"], hist["t"] + 0.04, r"history $H_n$")

# tokens: one column per step k (first entry = detection flag, second = scaled log-measurement)
tx0, cw = 1.56, 0.085
ty0 = hist["cy"] - 3.5 * cw
for c, k in enumerate([0, 1, 2, None, N]):
    x = tx0 + c * 0.13
    if k is None:
        ax.text(x + cw / 2, ty0 + 3.5 * cw, r"$\cdots$", ha="center", va="center"); continue
    for r in range(7):
        fc = "0.1" if (r == 0 and det[k]) else ("0.6" if r == 1 and det[k] else "white")
        ax.add_patch(Rectangle((x, ty0 + (6 - r) * cw), cw, cw, fc=fc, ec="0.3", lw=0.5))
    ax.text(x + cw / 2, ty0 - 0.04, r"$n$" if k == N else rf"${k}$", ha="center", va="top", fontsize=FT)
tok_r = tx0 + 4 * 0.13 + cw
tag(tx0 - 0.02, ty0 + 7 * cw + 0.04, r"tokens")
tag(tok_r + 0.05, ty0 + 3.5 * cw,
    r"$\begin{array}{l}d_k,\,\tilde z_k,\\ \boldsymbol p_k,\,t_k,\\ \Delta\boldsymbol p_k\,[,\boldsymbol u_k]\end{array}$",
    va="center", fontsize=FT)
arrow((hist["r"] + G, ty0 + 3.5 * cw), (tx0 - G, ty0 + 3.5 * cw))

# embedding MLP (top) -> GRU (below), unrolled over k
cx0, bw, dx = 2.98, 0.38, 0.46
ym, yg, bh = 2.62, 2.10, 0.28
gru = []
for c, lab in enumerate([r"0", r"1", None, r"n"]):
    x = cx0 + c * dx
    if lab is None:
        ax.text(x + bw / 2, yg + bh / 2, r"$\cdots$", ha="center", va="center"); gru.append(None); continue
    m = box(x, ym, bw, bh, [r"MLP"], fc=LEARN_FILL, fs=FT)
    g = box(x, yg, bw, bh, [r"GRU"], fc=LEARN_FILL, fs=FT)
    arrow((m["cx"], m["b"] - G / 2), (g["cx"], g["t"] + G / 2))
    ax.text(m["cx"], m["t"] + 0.03, rf"$k={lab}$", ha="center", va="bottom", fontsize=FT)
    gru.append(g)
for c in range(3):
    x0 = cx0 + c * dx + bw
    arrow((x0 + 0.015, yg + bh / 2), (x0 + dx - bw - 0.015, yg + bh / 2))
# token bus into the embeddings
line([tok_r + 0.62, cx0 - 0.08], [ym + bh / 2, ym + bh / 2])
arrow((cx0 - 0.09, ym + bh / 2), (cx0 - G / 2, ym + bh / 2))

# ================================================================== bottom lane: pair geometry -> f -> sum
pw = 1.28
ph = pw * (Y1 - Y0) / (X1 - X0)
YM = 0.96                                                    # centre line of the bottom lane
pair = dict(l=0.06, r=0.06 + pw, b=YM - ph / 2, t=YM + ph / 2, cy=YM)
a2 = inset(pair["l"], pair["b"], pw, ph)
xs, ys = np.meshgrid(np.linspace(0.3, 1.9, 9), np.linspace(-0.5, 0.5, 6))
a2.plot(xs, ys, ".", ms=1.6, color="0.55")
xj, pk = np.array([0.7, 0.3]), np.array([1.5, -0.15])
a2.add_patch(Rectangle(xj - 0.07, 0.14, 0.14, fc=LEARN, ec="none"))
a2.plot(*pk, "s", ms=5, mfc="0.15", mec="white", mew=0.6)
a2.annotate("", xy=pk, xytext=xj, arrowprops=dict(arrowstyle="-|>", lw=1.0, color="k", shrinkA=3, shrinkB=4,
                                                    mutation_scale=8))
a2.text(*(xj + [0, 0.10]), r"$\boldsymbol x_j$", ha="center", va="bottom")
a2.text(*(pk + [0, -0.10]), r"$\boldsymbol p_k$", ha="center", va="top")
a2.text(1.10, 0.20, r"$\boldsymbol p_k-\boldsymbol x_j$", ha="left", va="bottom", fontsize=FT)
a2.set_xlim(X0, X1); a2.set_ylim(Y0, Y1)
tag(pair["l"], pair["t"] + 0.04, r"cell $j$, step $k$")

# pairwise MLP f as layers of nodes
fb = box(1.62, YM - 0.50, 1.48, 1.06, [], ec=LEARN)
lx, nn, sp = [1.86, 2.26, 2.64, 2.98], [6, 5, 5, 1], 0.14
ymid = YM + 0.06
nodes = [[(x, ymid + (i - (m - 1) / 2) * sp) for i in range(m)] for x, m in zip(lx, nn)]
for L0, L1 in zip(nodes[:-1], nodes[1:]):
    for p0 in L0:
        for p1 in L1:
            line([p0[0], p1[0]], [p0[1], p1[1]], color="0.72", lw=0.35)
for L_ in nodes:
    for p in L_:
        ax.add_patch(Circle(p, 0.042, fc=LEARN_FILL, ec=LEARN, lw=0.8, zorder=3))
ax.text(fb["cx"], fb["b"] + 0.04, r"$68\,(70){\to}64{\to}64{\to}1$", ha="center", va="bottom", fontsize=FT)
ax.text(fb["cx"], fb["b"] - 0.04, r"pairwise MLP $f$, shared over all $(j,k)$", ha="center", va="top", fontsize=FT)
arrow((pair["r"] + G, ymid), (fb["l"] - G, ymid))
tag(pair["r"] + 0.03, ymid + 0.04, r"$\frac{\boldsymbol p_k-\boldsymbol x_j}{\boldsymbol\Lambda}$", fontsize=FT)
# GRU states down to f (bus)
yb = 1.86
for g in gru:
    if g is not None:
        line([g["cx"], g["cx"]], [g["b"] - G / 2, yb])
line([fb["cx"], gru[-1]["cx"]], [yb, yb])
arrow((fb["cx"], yb), (fb["cx"], fb["t"] + G / 2))
ax.text(fb["cx"] + 0.05, 0.5 * (yb + fb["t"]), r"$\boldsymbol g_{k-1},\ d_k,\ \tilde z_k\,[,\boldsymbol u_k]$",
        ha="left", va="center", fontsize=FT)

# sum over k -> summed-score map (network without physics input: log q^0 = sum_k l_jk + const)
sx, sr = 3.47, 0.14
ax.add_patch(Circle((sx, ymid), sr, fc="white", ec="k", lw=1.0))
ax.text(sx, ymid, r"$\sum_k$", ha="center", va="center")
arrow((fb["r"] + G, ymid), (sx - sr - G, ymid))
ax.text(0.5 * (fb["r"] + sx - sr), ymid + 0.04, r"$l_{jk}$", ha="center", va="bottom", fontsize=FT)
_, mS = mapax(3.78, ymid - MH / 2, logq("B0"), ec=LEARN)
arrow((sx + sr + G, ymid), (mS["l"] - G, ymid))
mS["ax"].text(0.04, 0.95, r"$\sum_k l_{jk}$", transform=mS["ax"].transAxes, ha="left", va="top", color=LEARN)

# ================================================================== right: physics posterior, combination, output
im, mP = mapax(5.00, 2.08, logq("M1"), ec=PHYS, ls=DASH)
tag(mP["l"] - 0.06, mP["t"] + 0.04, r"physics $b^{\mathrm{phys}}_n$ (frozen)", color=PHYS, fontsize=FT)
yl = 3.14
line([hist["cx"], hist["cx"]], [hist["t"] + 0.24, yl], color=PHYS, ls=DASH, lw=1.1)
line([hist["cx"], 4.88], [yl, yl], color=PHYS, ls=DASH, lw=1.1)
line([4.88, 4.88], [yl, mP["cy"]], color=PHYS, ls=DASH, lw=1.1)
arrow((4.88, mP["cy"]), (mP["l"] - G, mP["cy"]), color=PHYS, lw=1.1)
tag(2.55, yl + 0.03, r"physics likelihood at all cells $\boldsymbol x_j$, Bayes' rule", color=PHYS, fontsize=FT,
    ha="center")

cpx, cpy, cr = mP["cx"], 1.72, 0.13
ax.add_patch(Circle((cpx, cpy), cr, fc="white", ec="k", lw=1.0))
ax.text(cpx, cpy, r"$+$", ha="center", va="center")
arrow((cpx, mP["b"] - G), (cpx, cpy + cr + G), color=PHYS, lw=1.1)
ax.text(cpx + 0.05, 0.5 * (mP["b"] + cpy + cr), r"$\log$", ha="left", va="center", fontsize=FT, color=PHYS)
line([mS["cx"], mS["cx"]], [mS["t"] + G, cpy])
arrow((mS["cx"], cpy), (cpx - cr - G, cpy))
ax.text(mS["cx"] + 0.05, cpy + 0.03, r"$\times\alpha$", ha="left", va="bottom", fontsize=FT)
_, mQ = mapax(5.00, mS["b"], logq("B"))
arrow((cpx, cpy - cr - G), (cpx, mQ["t"] + G))
ax.text(cpx + 0.05, 0.5 * (cpy - cr + mQ["t"]), r"softmax$_j$", ha="left", va="center", fontsize=FT)
mQ["ax"].text(0.04, 0.95, r"$q(\boldsymbol x_j\,|\,H_n)$", transform=mQ["ax"].transAxes, ha="left", va="top")

loss = box(3.78, 0.05, 2.12, 0.30, [r"loss $-\log q(\boldsymbol x_{j^*}\,|\,H_n)$, $j^*$: ring"], fc=LOSS_FILL,
           fs=FT)
arrow((mQ["cx"], mQ["b"] - G), (mQ["cx"], loss["t"] + G))

cax = fig.add_axes([0.20 / W, 0.20 / H, 1.00 / W, 0.06 / H])
cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="min")
cb.set_ticks([-6, -3, 0])
cb.ax.tick_params(direction="in", labelsize=FT, pad=1)
cb.ax.set_title(r"maps: $\log_{10}$ per cell", fontsize=FT, pad=4)

save(fig, "figures/schematic_posterior_net_v7")
print("wrote figures/schematic_posterior_net_v7.{png,eps}")

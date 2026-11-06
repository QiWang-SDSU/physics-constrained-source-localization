"""E52: non-Gaussian measurements and the evidence of one detection (docs/research/data/E52_{strong,weak}_nongauss.npz,
scripts/checks/nongauss_evidence.py).

(a, b) Strong forcing: probability density of the measured log-concentration z at a fixed offset (xi, eta) of the sensor
from the source, DNS (all records and plumes of the training runs) against the physics model gamma N(m, s^2) and the
corrected model at the same offset. (c, d) Evidence of one detection for a candidate at offset A = (0.5, 0) over one at
B = (1.5, 0), log p(z|A) - log p(z|B), versus z: DNS histograms, physics and corrected models, and the learned term
(B0, averaged over GRU states of test histories), shown over the range of z where the DNS has samples at both offsets;
(c) strong, (d) weak forcing.
Styles as in style_v6: physics grey dotted, learned black solid; corrected model blue dashed (the colour of corrected
infotaxis); DNS black line / dots. Colour only for the corrected model, to keep it apart from the learned term.
Run (from repo/): py -3.11 scripts/figures/fig_nongauss_v6.py
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(__file__))
from plotstyle import panel_label, save
from style_v6 import use_style, POL, GREY_PHYS, SETS, TEXTW

BLUE = POL["m3"]["color"]
use_style()
fig, axes = plt.subplots(2, 2, figsize=(TEXTW, 4.6),
                         gridspec_kw=dict(wspace=0.28, hspace=0.42, left=0.09, right=0.99, bottom=0.1, top=0.9))
bins = np.linspace(-10, 3, 131)
zc = 0.5 * (bins[1:] + bins[:-1])


def dens(v):
    return np.histogram(v[np.isfinite(v)], bins)[0] / len(v) / (bins[1] - bins[0])


Z = {s: np.load(f"docs/research/data/E52_{s}_nongauss.npz") for s in ("strong", "weak")}
z = Z["strong"]
thr = float(z["thr"])
for c, i in enumerate((0, 2)):
    ax = axes[0, c]
    d = dens(z["z_dns"][i])
    ax.semilogy(zc[d > 0], d[d > 0], "k-", lw=1.4)
    for k, (col, ls) in enumerate(((GREY_PHYS, ":"), (BLUE, "--"))):
        g, m, sd = z["model"][k, i]
        ax.semilogy(zc, g * norm.pdf(zc, m, sd), color=col, ls=ls, lw=1.6)
    ax.axvline(thr, color="0.6", lw=0.7)
    ax.text(thr + 0.15, 2.0, r"$\log m_{\min}$", fontsize=8.5, va="top", color="0.4")
    xi, eta = z["offs"][i]
    ax.text(0.03, 0.97, r"$(\xi,\eta)=(%g,%g)$" % (xi, eta), transform=ax.transAxes, ha="left", va="top", fontsize=9)
    ax.set_xlim(-10, 3); ax.set_ylim(1e-4, 4)
    ax.set_xlabel(r"$z=\log c$", labelpad=1); ax.set_ylabel(r"$p(z)$")
    panel_label(ax, "ab"[c])
for c, s in enumerate(("strong", "weak")):
    z = Z[s]
    ax = axes[1, c]
    dA, dB = dens(z["z_dns"][0]), dens(z["z_dns"][1])
    ok = (dA > 0) & (dB > 0)
    lo, hi = zc[ok].min(), zc[ok].max()
    ax.plot(zc[ok], np.log(dA[ok] / dB[ok]), "o", ms=2.8, mfc="k", mec="k")
    zz = np.linspace(lo, hi, 200)
    for k, (col, ls) in enumerate(((GREY_PHYS, ":"), (BLUE, "--"))):
        (gA, mA, sA), (gB, mB, sB) = z["model"][k, 0], z["model"][k, 1]
        ax.plot(zz, norm.logpdf(zz, mA, sA) - norm.logpdf(zz, mB, sB) + np.log(gA / gB), color=col, ls=ls, lw=1.6)
    m = (z["zg"] >= lo) & (z["zg"] <= hi)
    ax.plot(z["zg"][m], z["learned"][0, m] - z["learned"][1, m], "k-", lw=1.6)
    ax.axhline(0, color="0.8", lw=0.6, zorder=0)
    ax.set_xlim(lo - 0.2, hi + 0.2); ax.set_ylim(-12, 6)
    ax.set_xlabel(r"$z$", labelpad=1); ax.set_ylabel(r"$\log p(z\mid A)-\log p(z\mid B)$")
    ax.text(0.97, 0.05, SETS[s], transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5)
    panel_label(ax, "cd"[c])
axes[0, 1].text(0.97, 0.97, SETS["strong"], transform=axes[0, 1].transAxes, ha="right", va="top", fontsize=8.5)
from matplotlib.lines import Line2D
h = [Line2D([], [], color="k", lw=1.4, marker="o", ms=2.8, label=r"DNS"),
     Line2D([], [], color=GREY_PHYS, ls=":", lw=1.6, label=r"physics model"),
     Line2D([], [], color=BLUE, ls="--", lw=1.6, label=r"corrected model"),
     Line2D([], [], color="k", lw=1.6, label=r"learned term (c,\,d)")]
fig.legend(handles=h, loc="upper center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.54, 1.0), handlelength=2.2)
save(fig, "figures/fig_nongauss_v6")
print("wrote figures/fig_nongauss_v6.{png,eps}")

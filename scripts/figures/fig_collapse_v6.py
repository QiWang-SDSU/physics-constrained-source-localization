"""Why the physics posterior collapses along corrected-infotaxis paths (one row, two panels).

(a) E50 replay ablation (docs/research/data/E50_{strong,weak}_ablation.npz, scripts/checks/phys_ablation_replay.py),
    paths chosen by corrected infotaxis only (policy key "m3"): one row per variant of the physics posterior, as in
    fig_phys_ablation_v6. Large circle: success P(e_80 < 0.1); small square: coverage of the 90% region at n = 80,
    P(x_s in R_0.9). Filled: strong forcing, open: weak forcing. Colour/marker: corrected infotaxis (style_v6.POL).
(b) E52 (docs/research/data/E52_strong_nongauss.npz, scripts/checks/nongauss_evidence.py), strong forcing: evidence of
    one detection for a candidate at offset A = (0.5, 0) over one at B = (1.5, 0), log p(z|A) - log p(z|B), versus z;
    computed exactly as fig_nongauss_v6 panel (c): DNS histograms (black dots), physics model (grey dotted), corrected
    model (blue dashed), learned term (black solid), over the range of z where the DNS has samples at both offsets.
Colour only for corrected infotaxis / the corrected model (blue #0072B2), as in the other v6 figures.
Run (from repo/): py -3.11 scripts/figures/fig_collapse_v6.py
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(__file__))
from plotstyle import panel_label, save
from style_v6 import use_style, POL, SETS, TEXTW, GREY_PHYS

ROWS = [("base", r"physics posterior"), ("k1", r"$\kappa_k=1$"), ("k05", r"$\kappa_k/2$"), ("k025", r"$\kappa_k/4$"),
        ("bgM3", r"corrected background"), ("plumeM3", r"corrected in-plume")]
PK = "m3"
BLUE = POL[PK]["color"]
MK = POL[PK]["marker"]
DY = {"strong": 0.17, "weak": -0.17}
MS_S, MS_C = 5.5, 3.6                                     # success / coverage marker sizes

use_style()
fig, (a, b) = plt.subplots(1, 2, figsize=(TEXTW, 2.2),
                           gridspec_kw=dict(wspace=0.42, left=0.215, right=0.99, bottom=0.2, top=0.9))

# (a) ablation along corrected-infotaxis paths
for st in ("strong", "weak"):
    z = np.load(f"docs/research/data/E50_{st}_ablation.npz")
    P, V = list(z["policies"]), list(z["variants"])
    for g, (v, _) in enumerate(ROWS):
        succ = (z["err"][P.index(PK), :, 80, V.index(v)] < 0.1).mean()
        cov = z["in90"][P.index(PK), :, 80, V.index(v)].mean()
        fc = BLUE if st == "strong" else "white"
        a.plot(succ, -g + DY[st], ls="none", marker=MK, ms=MS_S, mfc=fc, mec=BLUE, mew=1.0, zorder=4)
        a.plot(cov, -g + DY[st], ls="none", marker="s", ms=MS_C, mfc=fc, mec=BLUE, mew=0.8, zorder=3)
for g in range(len(ROWS)):
    a.axhspan(-g - 0.5, -g + 0.5, color="0.95" if g % 2 == 0 else "white", lw=0, zorder=0)
a.set_xlim(-0.03, 1.03); a.set_ylim(-len(ROWS) + 0.5, 0.5)
a.set_xticks([0, 0.25, 0.5, 0.75, 1]); a.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
a.set_yticks([-g for g in range(len(ROWS))]); a.set_yticklabels([r for _, r in ROWS])
a.tick_params(axis="y", which="both", left=False, right=False)
a.tick_params(axis="x", which="minor", bottom=False, top=False)
a.tick_params(axis="x", which="major", direction="out", top=False)
a.set_xlabel(r"fraction of episodes at $n=80$", labelpad=1)
h = [Line2D([], [], ls="none", marker=MK, ms=MS_S, mfc=BLUE, mec=BLUE, label=r"success, strong"),
     Line2D([], [], ls="none", marker="s", ms=MS_C, mfc=BLUE, mec=BLUE, label=r"$90\%$ coverage, strong"),
     Line2D([], [], ls="none", marker=MK, ms=MS_S, mfc="white", mec=BLUE, label=r"success, weak"),
     Line2D([], [], ls="none", marker="s", ms=MS_C, mfc="white", mec=BLUE, label=r"$90\%$ coverage, weak")]
a.legend(handles=h, frameon=False, fontsize=7, loc="center right", bbox_to_anchor=(1.0, 0.6), ncol=1,
         handletextpad=0.2, labelspacing=0.25, borderaxespad=0.2)
panel_label(a, "a", dx=0.0, dy=1.02)

# (b) evidence of one detection, strong forcing (as fig_nongauss_v6 panel c)
bins = np.linspace(-10, 3, 131)
zc = 0.5 * (bins[1:] + bins[:-1])


def dens(v):
    return np.histogram(v[np.isfinite(v)], bins)[0] / len(v) / (bins[1] - bins[0])


z = np.load("docs/research/data/E52_strong_nongauss.npz")
dA, dB = dens(z["z_dns"][0]), dens(z["z_dns"][1])
ok = (dA > 0) & (dB > 0)
lo, hi = zc[ok].min(), zc[ok].max()
b.plot(zc[ok], np.log(dA[ok] / dB[ok]), "o", ms=2.8, mfc="k", mec="k", zorder=3)
zz = np.linspace(lo, hi, 200)
for k, (col, ls) in enumerate(((GREY_PHYS, ":"), (BLUE, "--"))):
    (gA, mA, sA), (gB, mB, sB) = z["model"][k, 0], z["model"][k, 1]
    b.plot(zz, norm.logpdf(zz, mA, sA) - norm.logpdf(zz, mB, sB) + np.log(gA / gB), color=col, ls=ls, lw=1.6)
m = (z["zg"] >= lo) & (z["zg"] <= hi)
b.plot(z["zg"][m], z["learned"][0, m] - z["learned"][1, m], color="k", ls="-", lw=1.6)
b.axhline(0, color="0.8", lw=0.6, zorder=0)
b.set_xlim(lo - 0.2, hi + 0.2); b.set_ylim(-12, 7.5)
b.set_xlabel(r"$z$", labelpad=1); b.set_ylabel(r"$\log p(z\mid A)-\log p(z\mid B)$", fontsize=9)
b.text(0.97, 0.04, SETS["strong"], transform=b.transAxes, ha="right", va="bottom", fontsize=8)
h = [Line2D([], [], ls="none", marker="o", ms=2.8, mfc="k", mec="k", label=r"DNS"),
     Line2D([], [], color=GREY_PHYS, ls=":", lw=1.6, label=r"physics model"),
     Line2D([], [], color=BLUE, ls="--", lw=1.6, label=r"corrected model"),
     Line2D([], [], color="k", ls="-", lw=1.6, label=r"learned term")]
b.legend(handles=h, frameon=False, fontsize=7, loc="upper left", handlelength=1.8, labelspacing=0.25,
         borderaxespad=0.2)
panel_label(b, "b", dy=1.02)
save(fig, "figures/fig_collapse_v6")
print("wrote figures/fig_collapse_v6.{png,eps}")

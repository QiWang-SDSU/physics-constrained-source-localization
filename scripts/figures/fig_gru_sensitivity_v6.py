"""E54: what the learned posterior takes from the measurement sequence (docs/research/data/E54,
scripts/checks/gru_sensitivity.py, scripts/checks/gru_sens_proj.py, scripts/hpc/fermi_postnet_bottleneck.sbatch).

(a) Share of the Fisher-weighted sensitivity of q(. | H_80) to each standardised input of the GRU (summed over steps),
    learned posterior without physics input, without and with velocity sensing, both forcings. Features ordered by
    their mean share; x_k, y_k, t_k summed into one group "position, time" (light-grey background).
(b) Sensitivity versus step k (5-step running mean): through the GRU memory (solid) and through the decoder's direct
    inputs (dashed), without physics input, strong (black) and weak (grey) forcing; each curve divided by its total.
(c) Log score at n = 80 when the GRU input is restricted to rank r: learned map (filled) or fixed to the r leading
    sensitivity directions (open; r = 0: no measurement enters the GRU; missing runs are skipped); horizontal dotted
    lines: full network.
(d) Change of log q at the true cell when one detection flag is flipped: a detection removed or a false detection
    added; filled: flip in all inputs, open: in the GRU (memory) input only; hatched: with velocity sensing.
Greyscale only.
Run (from repo/): py -3.11 scripts/figures/fig_gru_sensitivity_v6.py [--main]
  --main: one row, (a) feature shares and (b) detection-flag flips -> figures/fig_gru_sensitivity_main_v6
"""
import argparse, json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
sys.path.insert(0, os.path.dirname(__file__))
from plotstyle import panel_label, save
from style_v6 import use_style, SETS, TEXTW

ap = argparse.ArgumentParser()
ap.add_argument("--main", action="store_true")
args = ap.parse_args()

D = "docs/research/data/E54"
LAB = {"z": r"$\tilde z_k$", "dx": r"$\Delta x_k$", "dy": r"$\Delta y_k$", "pt": r"position," "\n" r"time",
       "u": r"$u_k$", "v": r"$v_k$"}
SHADE = {"strong": "k", "weak": "0.6"}
LEGFS = 7.5


def shares():
    """Share of tr G per feature group and case; x, y, t summed into 'pt'."""
    cases = [("strong", "b0", "k", ""), ("weak", "b0", "0.6", ""), ("strong", "b0_vel", "white", "////"),
             ("weak", "b0_vel", "white", "...")]
    out = []
    for s, t, fc, h in cases:
        z = np.load(f"{D}/{s}_{t}_sens.npz")
        names, G = list(z["names"]), z["Gg"][0]
        sh = dict(zip(names, np.diag(G) / np.trace(G)))
        v = {f: sh.get(f, np.nan) for f in ("z", "dx", "dy", "u", "v")}
        v["pt"] = sh["x"] + sh["y"] + sh["t"]
        out.append((s, t, fc, h, v))
    return out


def panel_shares(a):
    cs = shares()
    feats = ["z", "dx", "dy", "pt", "u", "v"]
    feats.sort(key=lambda f: -np.nanmean([v[f] for *_, v in cs]))
    wb = 0.2
    ipt = feats.index("pt")
    a.axvspan(ipt - 0.48, ipt + 0.48, ymax=0.12 / 1.12, color="0.88", lw=0, zorder=0)
    for i, (s, t, fc, h, v) in enumerate(cs):
        vals = [v[f] for f in feats]
        x = np.arange(len(feats)) + (i - 1.5) * wb
        a.bar(x, vals, wb, color=fc, ec="k", lw=0.6, hatch=h, zorder=2,
              label=(SETS[s] + (r", velocity" if "vel" in t else "")))
        iz = feats.index("z")
        a.text(x[iz], 0.9, f"{vals[iz]:.2f}", rotation=90, ha="center", va="bottom", fontsize=7)
    a.set_xticks(np.arange(len(feats)), [LAB[f] for f in feats])
    a.tick_params(axis="x", which="both", length=0)
    a.set_xlim(-0.6, len(feats) - 0.4)
    a.set_ylabel(r"share of $\mathrm{tr}\,G$")
    a.set_ylim(0, 1.12)
    a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    a.legend(frameon=False, fontsize=7, loc="center right", bbox_to_anchor=(0.97, 0.55), handlelength=1.2, borderaxespad=0.0)


def smooth(y, w=5):
    k = np.ones(w)
    return np.convolve(y, k, "same") / np.convolve(np.ones_like(y), k, "same")


def panel_lag(b):
    k = np.arange(81)
    for s in ("strong", "weak"):
        z = np.load(f"{D}/{s}_b0_sens.npz")
        lg, ld = z["lag_g"][0], z["lag_d"][0]
        b.plot(k, smooth(lg / lg.sum()), ls="-", color=SHADE[s], lw=1.4)
        b.plot(k, smooth(ld / ld.sum()), ls="--", color=SHADE[s], lw=1.2)
    b.set_xlabel(r"$k$", labelpad=1); b.set_ylabel(r"sensitivity share")
    b.set_xlim(0, 80); b.set_ylim(0, 0.05)
    h = [Line2D([], [], color="k", ls="-", lw=1.4, label=r"memory"),
         Line2D([], [], color="k", ls="--", lw=1.2, label=r"direct"),
         Line2D([], [], color="k", lw=3, label=r"strong"),
         Line2D([], [], color="0.6", lw=3, label=r"weak")]
    b.legend(handles=h, frameon=False, fontsize=LEGFS, loc="upper left", ncol=4, handlelength=1.6,
             columnspacing=0.9, handletextpad=0.5, borderaxespad=0.3)


def panel_bottleneck(c):
    for s in ("strong", "weak"):
        full = json.load(open(f"{D}/{s}_b0_full.json"))["net"]["all_n80"]["logscore"]
        c.axhline(full, color=SHADE[s], lw=0.8, ls=":")
        c.text(-0.15, full + 0.015, r"full network", color=SHADE[s] if s == "strong" else "0.45", fontsize=7.5,
               ha="left", va="bottom")
        for mode, mk, ls in (("learned", "o", "-"), ("fixed", "s", "--")):
            rs, lsc = [], []
            for r in (0, 1, 2, 3):
                f = f"{D}/{s}_b0_bn_{mode}_r{r}.json"
                if os.path.exists(f):
                    rs.append(r); lsc.append(json.load(open(f))["net"]["all_n80"]["logscore"])
            if not rs:
                continue
            c.plot(rs, lsc, marker=mk, mfc=SHADE[s] if mode == "learned" else "white", mec=SHADE[s], color=SHADE[s],
                   lw=1.0, ls=ls, label=SETS[s] + (r", learned" if mode == "learned" else r", fixed to $G$"))
    c.set_xticks([0, 1, 2, 3]); c.set_xlim(-0.2, 3.2)
    c.set_xlabel(r"rank $r$ of the GRU input", labelpad=1)
    c.set_ylabel(r"$\log q(\boldsymbol{x}_{j^*}\,|\,H_{80})$")
    c.legend(frameon=False, fontsize=LEGFS, loc="center right", handlelength=1.8)


def panel_flips(d, legloc="lower left"):
    for i, t in enumerate(("b0", "b0_vel")):
        for j, s in enumerate(("strong", "weak")):
            z = np.load(f"{D}/{s}_{t}_sens.npz")
            fl, det = z["flip"], z["det"]
            for m in (0, 1):                               # 0: all inputs, 1: GRU input only
                for sel, x0 in ((det, 0), (~det, 1)):
                    x = x0 * 9 + i * 2.2 + j * 1.0 + m * 0.45
                    v = np.nanmean(fl[:, :, m, 0][sel])
                    hatch = "////" if i else ""
                    if m == 0:
                        d.bar(x, v, 0.42, color=SHADE[s], ec="white", hatch=hatch, lw=0)
                        d.bar(x, v, 0.42, fill=False, ec="k", lw=0.6)
                    else:
                        d.bar(x, v, 0.42, color="white", ec=SHADE[s], hatch=hatch, lw=0.6)
    d.axhline(0, color="k", lw=0.6)
    d.set_xticks([1.6, 10.6], [r"true detection removed", r"false detection added"])
    d.tick_params(axis="x", which="both", length=0)
    d.set_xlim(-0.5, 12.7)
    d.set_ylabel(r"$\Delta\log q(\boldsymbol{x}_{j^*}\,|\,H_{80})$")
    h = [Patch(fc="k", ec="k", lw=0.6, label=r"all inputs, concentration"),
         Patch(fc="k", ec="white", hatch="////", lw=0, label=r"all inputs, with velocity"),
         Patch(fc="white", ec="k", lw=0.6, label=r"memory only, concentration"),
         Patch(fc="white", ec="k", hatch="////", lw=0.6, label=r"memory only, with velocity")]
    d.legend(handles=h, frameon=False, fontsize=LEGFS, loc=legloc, handlelength=1.4, handleheight=0.9,
             borderaxespad=0.3, labelspacing=0.3)


use_style()
if args.main:
    fig, ax = plt.subplots(1, 2, figsize=(TEXTW, 2.2),
                           gridspec_kw=dict(wspace=0.32, left=0.09, right=0.99, bottom=0.14, top=0.9))
    panel_shares(ax[0]); panel_label(ax[0], "a")
    panel_flips(ax[1]); panel_label(ax[1], "b")
    out = "figures/fig_gru_sensitivity_main_v6"
else:
    fig, ax = plt.subplots(2, 2, figsize=(TEXTW, 4.9),
                           gridspec_kw=dict(wspace=0.32, hspace=0.5, left=0.09, right=0.99, bottom=0.08, top=0.95))
    panel_shares(ax[0, 0]); panel_label(ax[0, 0], "a")
    panel_lag(ax[0, 1]); panel_label(ax[0, 1], "b")
    panel_bottleneck(ax[1, 0]); panel_label(ax[1, 0], "c")
    panel_flips(ax[1, 1]); panel_label(ax[1, 1], "d")
    out = "figures/fig_gru_sensitivity_v6"
save(fig, out)
print(f"wrote {out}.{{png,eps}}")

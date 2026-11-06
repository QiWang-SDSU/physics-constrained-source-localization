"""Learning curves of the learned source posteriors with and without the physics input (E47: E37 sweep rerun with
fp32 accumulation, rlscalar/search/postnet.py).

Columns: (a, c) eps = 5e-3, (b, d) eps = 1.2e-3. x: training readings N = 81 n_train. Top: P(e < 0.1) after 80 moves on
the 300 held-out test histories (policy mix EIG / zigzag / random walk). Bottom: best validation loss
-mean log q(x_s | H_n). Black: learned posterior with the physics input; grey: without it; markers: mean of 3 random
training subsets (n_train <= 1000), bars min-max. Dotted: physics posterior alone.
Solid: concentration readings only (E47, docs/research/data/E47); dashed: the sensor also reads the local velocity
(E48, docs/research/data/E48).
--top writes fig_learning_curve_top (success row only, for the main text).
Run (from repo/): python scripts/figures/fig_learning_curve.py [--top]
"""
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from plotstyle import use_style, panel_label, save

DS = [("E47", "-", r""), ("E48", "--", r", velocity")]
NT = [10, 30, 100, 300, 1000, 3000, 9500]
SETS = [("strong", r"$\varepsilon = 5\times10^{-3}$"), ("weak", r"$\varepsilon = 1.2\times10^{-3}$")]
MOD = [("b", dict(color="k", marker="o", mfc="k"), r"learned, with physics input"),
       ("b0", dict(color="#888888", marker="s", mfc="white"), r"learned, without physics input")]


def curve(D, st, m, get):
    D = f"docs/research/data/{D}"
    out = []
    for n in NT:
        v = [get(json.load(open(f"{D}/{st}_{m}_n{n}_r{r}.json"))) for r in range(3)
             if os.path.exists(f"{D}/{st}_{m}_n{n}_r{r}.json")]
        out.append((np.mean(v), np.min(v), np.max(v)))
    return np.array(out).T


TOP = "--top" in sys.argv                     # success row only, (a) strong and (b) weak (main text)
use_style(base=13)
if TOP:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0), layout="constrained", sharex=True, squeeze=False)
else:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.2), layout="constrained", sharex=True)
N = 81 * np.array(NT)
for j, (st, lab) in enumerate(SETS):
    for i, get in enumerate([lambda r: r["net"]["all_n80"]["P01"], lambda r: r["val_loss"]][:axes.shape[0]]):
        ax = axes[i, j]
        for D, ls, tag in DS:
            for m, sty, name in MOD:
                y, lo, hi = curve(D, st, m, get)
                ax.errorbar(N, y, yerr=[y - lo, hi - y], lw=1.6, ls=ls, ms=6, mew=1.2, capsize=0, label=name + tag, **sty)
        if i == 0:
            ph = json.load(open(f"docs/research/data/E47/{st}_b_n9500_r0.json"))["physics"]["all_n80"]["P01"]
            ax.axhline(ph, color="k", ls=":", lw=1.2, label=r"physics posterior")
            ax.set_ylim(0, 0.8)
            ax.text(0.97, 0.05, lab, transform=ax.transAxes, ha="right", va="bottom")
        ax.set_xscale("log"); ax.set_xlim(5e2, 1.3e6)
        panel_label(ax, "abcd"[2 * i + j])
    axes[-1, j].set_xlabel(r"$N$")
axes[0, 0].set_ylabel(r"$P(e_{80} < 0.1)$")
if not TOP:
    axes[1, 0].set_ylabel(r"$-\overline{\log q(\boldsymbol{x}_s\,|\,H_n)}$")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=11, ncol=2, loc="outside lower center")
out = "figures/fig_learning_curve" + ("_top" if TOP else "")
save(fig, out)
print(f"wrote {out}.{{png,eps}}")

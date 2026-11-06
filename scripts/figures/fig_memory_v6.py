"""What the memory of the learned posterior adds, and on which paths (E57; L4DC v6, Fig. memory).

Learned posterior without physics input, along the paths of the four motion policies on 300 fresh tasks (task seeds
1000-1299, DNS runs 6-8), at n = 80, for three training sets:
  generic paths       physics infotaxis, zigzag survey, random walk (E37/E54; 3 training seeds)
  planner held out    the four policies' own paths, with the scored policy removed (E57 leave-one-out; 1 seed)
  planners' paths     the four policies' own paths (E56; 3 training seeds)
(a) gain from memory: log score of the full network minus that of the network with the GRU state fixed at zero;
(b) coverage of the 90% region of the full network. Colour/marker: motion policy (style_v6); filled: strong, open: weak.
Data: docs/research/data/E57/{set}_fresh.json (O/ON, R/RN, seeds 0-2) and {set}_lopo.json (X{c}/XN{c}).
Run (from repo/): py -3.11 scripts/figures/fig_memory_v6.py
"""
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(__file__))
from plotstyle import panel_label, save
from style_v6 import use_style, POLICIES, TEXTW

CODE = {"m1": 0, "m3": 5, "chase": 3, "surgecast": 4}
COND = ["generic\npaths", "policy\nheld out", "policies'\nown paths"]


def values(st):
    """{policy: [(gain, cov90 of full) per condition]}"""
    fr = json.load(open(f"docs/research/data/E57/{st}_fresh.json"))
    lo = json.load(open(f"docs/research/data/E57/{st}_lopo.json"))
    g = lambda d, p, k: d[f"{p}_{k}_n80"]
    out = {}
    for p in CODE:
        rows = []
        for full, nom, d in [(["O0", "O1", "O2"], ["ON0", "ON1", "ON2"], fr), ([f"X{CODE[p]}"], [f"XN{CODE[p]}"], lo),
                             (["R0", "R1", "R2"], ["RN0", "RN1", "RN2"], fr)]:
            ls_f = np.mean([g(d, p, k)["logscore"] for k in full])
            ls_n = np.mean([g(d, p, k)["logscore"] for k in nom])
            rows.append((ls_f - ls_n, np.mean([g(d, p, k)["cov90"] for k in full])))
        out[p] = rows
    return out


use_style()
fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.2), gridspec_kw=dict(wspace=0.3, left=0.085, right=0.99,
                                                                          bottom=0.235, top=0.85))
V = {st: values(st) for st in ("strong", "weak")}
DX = {"m3": -0.15, "m1": -0.05, "surgecast": 0.05, "chase": 0.15}
for i, ax in enumerate(axes):
    for key, lab, col, mk in POLICIES:
        for st, filled in (("strong", True), ("weak", False)):
            y = np.array([r[i] for r in V[st][key]])
            x = np.arange(3) + DX[key] + (0.02 if not filled else -0.02)
            if i == 0:
                y = np.clip(y, -2.6, None)                  # one weak held-out value is off scale (marked below)
            ax.plot(x, y, ls="-" if filled else ":", lw=0.8, color=col, marker=mk, ms=4.5, mfc=col if filled else "white",
                    mec=col, mew=0.9)
    ax.set_xticks(range(3)); ax.set_xticklabels(COND, linespacing=1.0)
    ax.set_xlim(-0.4, 2.4)
    ax.tick_params(axis="x", which="both", top=False, bottom=False)
    panel_label(ax, "ab"[i], dx=-0.13, dy=1.0)
axes[0].axhline(0, color="0.6", lw=0.7, zorder=0)
axes[0].set_ylabel(r"$\Delta\log q(\bm x^*\mid H_{80})$")
axes[0].set_ylim(-2.7, 0.45)
off = V["weak"]["m1"][1][0]
axes[0].annotate(rf"${off:.1f}$", xy=(1 - 0.05 + 0.02, -2.6), xytext=(1.25, -2.35), fontsize=8, color="#D55E00",
                 arrowprops=dict(arrowstyle="-", lw=0.6, color="#D55E00"))
axes[1].axhline(0.9, color="0.6", lw=0.7, zorder=0)
axes[1].set_ylabel(r"$C_{90}$")
axes[1].set_ylim(0.55, 1.0)
h = [Line2D([], [], ls="none", marker=m, ms=4.5, color=c, label=l) for _, l, c, m in POLICIES]
h += [Line2D([], [], ls="none", marker="o", ms=4.5, color="0.3", label="strong"),
      Line2D([], [], ls="none", marker="o", ms=4.5, mfc="white", mec="0.3", label="weak")]
fig.legend(handles=h, loc="upper center", ncol=6, fontsize=8.5, handletextpad=0.3, columnspacing=1.0,
           bbox_to_anchor=(0.54, 1.02))
save(fig, "figures/fig_memory_v6")
for st in V:
    print(st, {p: [tuple(round(v, 2) for v in r) for r in V[st][p]] for p in V[st]})

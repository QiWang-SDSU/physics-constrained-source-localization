"""Shared encodings for the v6 result figures (L4DC 2027), so that every method keeps one colour/marker everywhere.

Policies (closed loop) are the only categorical colour: four crossing success curves cannot be told apart in grey.
Colours are Okabe-Ito (colour-blind safe); each policy also has its own marker, so the figures survive greyscale print.
Posteriors are encoded in greyscale by line style / marker fill:
  physics posterior      : grey dotted line, open circle
  learned posterior      : black solid line, filled circle (with physics input)
  without physics input  : grey solid line, open square
  with velocity          : dashed line of the same posterior; filled diamond in dot plots
Printed width: text width 6.5 in; fonts sized for that width (base 9 pt).
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from plotstyle import use_style as _use_style

TEXTW = 6.5                     # in, final printed text width

POLICIES = [  # key in E49, label, colour, marker
    ("m3", r"corrected infotaxis", "#0072B2", "o"),
    ("m1", r"physics infotaxis", "#D55E00", "s"),
    ("surgecast", r"cast-and-surge", "#CC79A7", "^"),
    ("chase", r"plume tracing", "#009E73", "D"),
]
POL = {k: dict(label=l, color=c, marker=m) for k, l, c, m in POLICIES}

GREY_PHYS = "0.55"
POST = {
    "M1": dict(label=r"physics posterior", color=GREY_PHYS, ls=":", marker="o", mfc="white"),
    "B": dict(label=r"learned posterior", color="k", ls="-", marker="o", mfc="k"),
    "B0": dict(label=r"learned, without physics input", color="0.5", ls="-", marker="s", mfc="white"),
    "Bvel": dict(label=r"learned, with velocity", color="k", ls="--", marker="D", mfc="k"),
}
SETS = {"strong": r"$\varepsilon = 5\times10^{-3}$", "weak": r"$\varepsilon = 1.2\times10^{-3}$"}


def use_style():
    import matplotlib as mpl
    _use_style(base=9)
    mpl.rcParams.update({"axes.labelsize": 10.5, "lines.linewidth": 1.4, "lines.markersize": 4.5,
                         "lines.markeredgewidth": 0.9, "axes.linewidth": 0.8, "xtick.major.size": 3.5,
                         "ytick.major.size": 3.5, "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
                         "xtick.major.width": 0.8, "ytick.major.width": 0.8, "legend.fontsize": 8.5,
                         "legend.handlelength": 2.2})

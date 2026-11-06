"""Shared matplotlib style for research figures.

Encodes the project figure rules so they are applied by construction rather
than remembered:

  * no grid lines
  * LaTeX as the text interpreter (with a graceful fallback)
  * large fonts, sized for a one-column figure
  * greyscale + linestyle + linewidth cycle; color only when asked for
  * legends never boxed
  * panel labels (a), (b), ... rendered by LaTeX, placed outside the axes
  * saves .png + .eps; PDF output is refused

Usage
-----
    from plotstyle import use_style, panel_label, save, add_colorbar

    use_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(x, u1, label=r'$Re_\\tau=180$')
    ax.plot(x, u2, label=r'$Re_\\tau=550$')
    ax.set_xlabel(r'$y^{+}$')
    ax.set_ylabel(r'$\\overline{u}^{+}$')
    ax.legend()                      # frameless by default
    panel_label(ax, 'a')
    save(fig, 'figures/mean_profile')
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

__all__ = ["use_style", "panel_label", "save", "add_colorbar", "GREY_CYCLE", "COLOR_CYCLE"]


# Greyscale-first: every curve is distinguishable in black and white by
# grey level AND linestyle AND width, so color carries no information.
GREY_CYCLE = (
    cycler(color=["0.0", "0.45", "0.0", "0.45", "0.65"])
    + cycler(linestyle=["-", "--", "-.", ":", (0, (3, 1, 1, 1, 1, 1))])
    + cycler(linewidth=[2.0, 2.0, 1.8, 1.8, 1.6])
)

# Only for the cases where color is genuinely necessary (many curves, or a
# physical quantity mapped to hue). Colorblind-safe, still varies linestyle.
COLOR_CYCLE = (
    cycler(color=["#000000", "#0072B2", "#D55E00", "#009E73", "#CC79A7"])
    + cycler(linestyle=["-", "--", "-.", ":", (0, (3, 1, 1, 1))])
)


def _latex_available() -> bool:
    """True if a LaTeX toolchain matplotlib can drive is on PATH.

    ``latex`` + ``dvipng`` covers raster output; EPS additionally wants
    ``dvips`` (and ghostscript), which is checked lazily at save time.
    """
    return all(shutil.which(exe) for exe in ("latex", "dvipng"))


def use_style(usetex: bool | None = None, base: float = 15.0, color: bool = False) -> None:
    """Apply the project figure style to matplotlib's global rcParams.

    Parameters
    ----------
    usetex : bool or None
        Force LaTeX text rendering on/off. ``None`` (default) auto-detects and
        falls back to matplotlib's Computer Modern mathtext if LaTeX is absent,
        so figures still *look* the same when the toolchain is missing.
    base : float
        Base font size in points. Axis labels sit one step above this.
    color : bool
        ``False`` (default) installs the greyscale cycle. Pass ``True`` only
        when color is necessary, and say why in the report.
    """
    if usetex is None:
        usetex = _latex_available()
        if not usetex:
            warnings.warn(
                "LaTeX toolchain not found; falling back to mathtext (cm). "
                "Install a TeX distribution for manuscript-quality text.",
                stacklevel=2,
            )

    mpl.rcParams.update(
        {
            # --- text -------------------------------------------------------
            "text.usetex": usetex,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
            # --- sizes: large, for one-column reproduction ------------------
            "font.size": base,
            "axes.labelsize": base + 2,
            "axes.titlesize": base + 2,
            "xtick.labelsize": base,
            "ytick.labelsize": base,
            "legend.fontsize": base,
            "figure.titlesize": base + 3,
            # --- no grid, ever ----------------------------------------------
            "axes.grid": False,
            "axes.axisbelow": True,
            # --- frame and ticks --------------------------------------------
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 5.5,
            "ytick.major.size": 5.5,
            "xtick.minor.size": 3.0,
            "ytick.minor.size": 3.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            # --- lines and markers ------------------------------------------
            "lines.linewidth": 2.0,
            "lines.markersize": 6.0,
            "lines.markeredgewidth": 1.2,
            "axes.prop_cycle": COLOR_CYCLE if color else GREY_CYCLE,
            # --- legend: NEVER a box ----------------------------------------
            "legend.frameon": False,
            "legend.fancybox": False,
            "legend.shadow": False,
            "legend.borderpad": 0.2,
            "legend.handlelength": 2.8,   # long enough to read a linestyle
            "legend.labelspacing": 0.35,
            "legend.handletextpad": 0.6,
            # --- output ------------------------------------------------------
            "figure.figsize": (4.4, 3.4),
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "savefig.transparent": False,
            "ps.papersize": "figure",  # EPS bbox follows the figure, not a paper size
            "ps.usedistiller": None,
        }
    )


def panel_label(ax, letter: str, dx: float = -0.20, dy: float = 1.06, **kwargs):
    """Place a LaTeX-rendered panel label ``(a)`` outside the axes box.

    ``dx``/``dy`` are in axes-fraction coordinates; the default sits above and
    to the left of the axes. Nudge ``dx`` toward 0 for panels with short tick
    labels, or use ``dy=1.02`` for tight vertical stacks.
    """
    opts = dict(ha="right", va="bottom", clip_on=False)
    opts.update(kwargs)
    return ax.text(dx, dy, rf"$({letter})$" if len(letter) == 1 else letter,
                   transform=ax.transAxes, **opts)


def add_colorbar(mappable, ax, label: str, pad: float = 0.03, fraction: float = 0.046, **kwargs):
    """Attach a labeled colorbar. Required for every contourf/pcolormesh plot.

    ``label`` must be a mathematical symbol, e.g. ``r'$\\omega_z \\delta / U_\\infty$'``.
    """
    cbar = ax.figure.colorbar(mappable, ax=ax, pad=pad, fraction=fraction, **kwargs)
    cbar.set_label(label)
    cbar.outline.set_linewidth(1.0)
    cbar.ax.tick_params(direction="in")
    return cbar


def save(fig, stem, formats=("png", "eps"), rasterize_below: int | None = 0, close: bool = False):
    """Save ``fig`` to ``stem.<ext>`` for each format. PDF is refused.

    Parameters
    ----------
    stem : str or Path
        Path without extension, e.g. ``'figures/mean_profile'``. Parent
        directories are created.
    formats : iterable of str
        Any of ``png``, ``eps``, ``jpeg``/``jpg``, ``svg``. ``pdf`` raises.
    rasterize_below : int or None
        zorder below which artists are rasterized, keeping filled-contour EPS
        files small while text and axes stay vector. Draw heavy artists with
        ``zorder=-5`` (e.g. ``ax.contourf(..., zorder=-5)``) to opt them in.
        Pass ``None`` to disable.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    bad = [f for f in formats if f.lower().lstrip(".") == "pdf"]
    if bad:
        raise ValueError("PDF figures are not allowed; use eps, png, or jpeg.")

    if rasterize_below is not None:
        for ax in fig.axes:
            ax.set_rasterization_zorder(rasterize_below)

    written = []
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        out = stem.parent / f"{stem.name}.{fmt}"
        fig.savefig(out, format=fmt)
        written.append(str(out))

    if close:
        plt.close(fig)
    return written

"""
ieee_style.py

Shared matplotlib style settings for IEEE DASC 2026 paper figures.

IEEE requirements enforced:
  - Single-column: 3.5 in  |  Double-column: 7.16 in
  - ≥600 DPI (vector PDF primary, PNG backup)
  - Minimum 8 pt font, Helvetica/DejaVu Sans
  - Grayscale-safe: every series uses BOTH distinct color AND distinct
    linestyle/marker/hatch so the figure reads in black-and-white print
  - No transparency

Usage:
    from spkdiar.analysis.ieee_style import apply_ieee_style, save_fig, SYSTEM_STYLES
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 3.0))
    ...
    save_fig(fig, Path("results/paper_figures/fig1_der_comparison"))
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Dimension and resolution constants
# ---------------------------------------------------------------------------

IEEE_SINGLE_COL  = 3.5    # inches (88.9 mm)
IEEE_DOUBLE_COL  = 7.16   # inches (181.9 mm)
IEEE_DPI         = 600
IEEE_LINE_WIDTH  = 1.0    # minimum 0.5 pt; 1.0 is safe

# ---------------------------------------------------------------------------
# Grayscale-safe system style palette
# Colors are distinguishable in colour; linestyle + marker make them
# distinguishable in black-and-white print.
# ---------------------------------------------------------------------------

SYSTEM_STYLES = {
    "pretrained": {"color": "#1f77b4", "marker": "o",  "linestyle": "-",   "hatch": ""},
    "finetuned":  {"color": "#d62728", "marker": "s",  "linestyle": "--",  "hatch": "//"},
    "streaming":  {"color": "#2ca02c", "marker": "^",  "linestyle": "-.",  "hatch": ".."},
    "pyannote":   {"color": "#9467bd", "marker": "D",  "linestyle": ":",   "hatch": "xx"},
    "lseend":     {"color": "#8c564b", "marker": "v",  "linestyle": "--",  "hatch": "\\\\"},
}

# Speaker probability curves — distinct linestyle per slot (colour is secondary)
SPK_STYLES = {
    0: {"color": "#d62728", "marker": None, "linestyle": "-"},
    1: {"color": "#1f77b4", "marker": None, "linestyle": "--"},
    2: {"color": "#2ca02c", "marker": None, "linestyle": "-."},
    3: {"color": "#9467bd", "marker": None, "linestyle": ":"},
}

# ATC role bar colours / hatch patterns
ATC_STYLE   = {"color": "#E8702A", "hatch": "//"}   # controller — orange + hatch
PILOT_STYLE = {"color": "#4C72B0", "hatch": ""}     # pilot — blue, no hatch


# ---------------------------------------------------------------------------
# DER stacked bar component colours (all three readable in greyscale via
# progressively darker grey equivalents)
# ---------------------------------------------------------------------------

BAR_COLORS = {
    "FA":   "#aec7e8",   # lightest blue
    "MISS": "#6baed6",   # medium blue
    "CER":  "#08519c",   # dark blue (most important component)
}


# ---------------------------------------------------------------------------
# Font selection
# ---------------------------------------------------------------------------

def _select_font() -> str:
    """Return the best available IEEE-compatible font name."""
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in ("Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"):
        if candidate in available:
            return candidate
    return "sans-serif"


IEEE_FONT = _select_font()


# ---------------------------------------------------------------------------
# Apply global rcParams
# ---------------------------------------------------------------------------

def apply_ieee_style() -> None:
    """Set matplotlib rcParams to IEEE DASC submission standards."""
    matplotlib.rcParams.update({
        # Font
        "font.family":         "sans-serif",
        "font.sans-serif":     [IEEE_FONT, "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":           9,
        "axes.labelsize":      9,
        "axes.titlesize":      9,
        "xtick.labelsize":     8,
        "ytick.labelsize":     8,
        "legend.fontsize":     8,
        "legend.title_fontsize": 8,
        # Lines
        "lines.linewidth":     IEEE_LINE_WIDTH,
        "axes.linewidth":      0.6,
        "xtick.major.width":   0.6,
        "ytick.major.width":   0.6,
        "patch.linewidth":     0.6,
        # Resolution
        "figure.dpi":          150,      # screen preview
        "savefig.dpi":         IEEE_DPI,
        "savefig.bbox":        "tight",
        "savefig.transparent": False,
        "savefig.format":      "pdf",
        # Padding
        "figure.constrained_layout.use": False,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
    })


# ---------------------------------------------------------------------------
# Save helper — writes both PDF (primary) and PNG (backup)
# ---------------------------------------------------------------------------

def save_fig(fig: "plt.Figure", base_path: Path, close: bool = True) -> None:
    """Save figure to <base_path>.pdf and <base_path>.png.

    Args:
        fig: The matplotlib Figure to save.
        base_path: Path without extension (e.g. Path("results/paper_figures/fig1")).
        close: Whether to close the figure after saving (default True).
    """
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)

    pdf_path = base.with_suffix(".pdf")
    png_path = base.with_suffix(".png")

    fig.savefig(
        str(pdf_path),
        dpi=IEEE_DPI,
        bbox_inches="tight",
        transparent=False,
        format="pdf",
    )
    fig.savefig(
        str(png_path),
        dpi=IEEE_DPI,
        bbox_inches="tight",
        transparent=False,
        format="png",
    )
    if close:
        plt.close(fig)
    print(f"Saved: {pdf_path}  |  {png_path}")

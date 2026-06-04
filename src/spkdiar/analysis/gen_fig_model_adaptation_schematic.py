"""Create a compact conceptual schematic for Sortformer adaptation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from spkdiar.analysis.ieee_style import IEEE_SINGLE_COL, apply_ieee_style, save_fig


def box(ax, x, y, w, h, text, *, facecolor="#f7f7f7", hatch="", fontsize=8.2, weight="normal"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=0.8,
        edgecolor="#303030",
        facecolor=facecolor,
        hatch=hatch,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
    )
    return patch


def arrow(ax, start, end, *, linestyle="-"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.0,
            linestyle=linestyle,
            color="#303030",
            shrinkA=2,
            shrinkB=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Sortformer adaptation schematic.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/paper_figures"))
    args = parser.parse_args()

    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 4.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.03, 0.72, 0.18, 0.14, "Audio\nwindow", facecolor="#ffffff", fontsize=7.8, weight="bold")
    box(ax, 0.26, 0.72, 0.18, 0.14, "NEST\nencoder", facecolor="#f0f0f0", fontsize=7.8, weight="bold")
    box(ax, 0.49, 0.72, 0.24, 0.14, "Decoder", facecolor="#ffffff", hatch="//", fontsize=7.8, weight="bold")
    box(ax, 0.79, 0.72, 0.16, 0.14, "4 slots", facecolor="#ffffff", hatch="..", fontsize=7.8, weight="bold")

    arrow(ax, (0.21, 0.79), (0.26, 0.79))
    arrow(ax, (0.44, 0.79), (0.49, 0.79))
    arrow(ax, (0.73, 0.79), (0.79, 0.79))

    ax.text(0.35, 0.64, "Frozen", ha="center", va="center", fontsize=6.7, fontweight="bold")
    ax.text(0.61, 0.64, "Updated", ha="center", va="center", fontsize=6.7, fontweight="bold")

    box(ax, 0.05, 0.26, 0.30, 0.16, "Fine-tuning\n90 s, AdamW\n1000", facecolor="#ffffff", fontsize=7.2)
    arrow(ax, (0.37, 0.38), (0.54, 0.71), linestyle="--")

    box(ax, 0.41, 0.28, 0.22, 0.14, "Streaming\nAOSC", facecolor="#f7f7f7", hatch="..", fontsize=7.2)
    arrow(ax, (0.54, 0.42), (0.61, 0.72), linestyle=":")

    box(ax, 0.68, 0.24, 0.27, 0.18, "Calibration\nselect threshold,\nthen transfer", facecolor="#f7f7f7", fontsize=7.0)
    arrow(ax, (0.87, 0.72), (0.84, 0.42), linestyle=":")

    ax.text(
        0.50,
        0.10,
        "Same Sortformer backbone; only adaptation, streaming state,\n"
        "and threshold choice are varied.",
        ha="center",
        va="center",
        fontsize=6.9,
    )

    fig.tight_layout(pad=0.2)
    save_fig(fig, args.out_dir / "fig_model_adaptation_schematic")


if __name__ == "__main__":
    main()

"""Create a compact schematic for ATC data partitioning."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from spkdiar.analysis.ieee_style import IEEE_SINGLE_COL, apply_ieee_style, save_fig


def add_box(
    ax,
    xy,
    wh,
    text,
    *,
    facecolor="#f7f7f7",
    edgecolor="#303030",
    hatch="",
    fontsize=8.2,
    weight="normal",
):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=0.8,
        edgecolor=edgecolor,
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


def add_arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.0,
            color="#303030",
            shrinkA=2,
            shrinkB=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ATC partition schematic.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/paper_figures"))
    args = parser.parse_args()

    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 4.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(
        ax,
        (0.11, 0.84),
        (0.78, 0.11),
        "ATC0 corpus\n16 VHF recordings, about 22 h",
        facecolor="#f0f0f0",
        fontsize=8.7,
        weight="bold",
    )

    add_box(
        ax,
        (0.06, 0.58),
        (0.30, 0.18),
        "Evaluation\n4 recs\n10 s / 5 s\n3989",
        facecolor="#ffffff",
        hatch="..",
        fontsize=7.0,
    )
    add_box(
        ax,
        (0.60, 0.60),
        (0.27, 0.15),
        "Shared source\n12 recs",
        facecolor="#ffffff",
        hatch="//",
        fontsize=7.1,
    )
    add_arrow(ax, (0.50, 0.84), (0.23, 0.76))
    add_arrow(ax, (0.50, 0.84), (0.72, 0.74))

    ax.text(
        0.72,
        0.53,
        "Train, val., calib.\nall use same 12 recs",
        ha="center",
        va="center",
        fontsize=6.9,
        fontweight="bold",
    )

    boxes = [
        ((0.05, 0.19), (0.24, 0.18), "FT train\n90 s\n1453", "#f7f7f7", ""),
        ((0.38, 0.19), (0.24, 0.18), "FT val.\n90 s\n456", "#f7f7f7", "//"),
        ((0.71, 0.19), (0.24, 0.18), "Calibration\n10 s no overlap\n2172", "#f7f7f7", ".."),
    ]
    for xy, wh, text, face, hatch in boxes:
        add_box(ax, xy, wh, text, facecolor=face, hatch=hatch, fontsize=6.7)

    for x in (0.17, 0.50, 0.83):
        add_arrow(ax, (0.72, 0.59), (x, 0.38))

    ax.text(
        0.50,
        0.09,
        "Calibration uses the first 30 min of each shared-source recording.\n"
        "Evaluation recordings are separate and never used for training or threshold selection.",
        ha="center",
        va="center",
        fontsize=6.8,
    )

    fig.tight_layout(pad=0.2)
    save_fig(fig, args.out_dir / "fig_data_partitions_schematic")


if __name__ == "__main__":
    main()

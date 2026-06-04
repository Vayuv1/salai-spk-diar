"""Generate the segment-level evaluation protocol figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from spkdiar.analysis.ieee_style import IEEE_SINGLE_COL, apply_ieee_style, save_fig


EDGE = "#333333"
GRID = "#cfcfcf"
COLLAR = "#ebebeb"
COLORS = {
    "w1": "#dbe7f3",
    "w2": "#f5e2c9",
    "w3": "#e1ecd6",
}


def double_arrow(
    ax,
    x0: float,
    x1: float,
    y: float,
    label: str,
    label_x: float,
    label_y: float,
    *,
    label_va: str = "bottom",
) -> None:
    ax.annotate(
        "",
        xy=(x1, y),
        xytext=(x0, y),
        arrowprops=dict(arrowstyle="<->", linewidth=0.9, color=EDGE, shrinkA=0, shrinkB=0),
        annotation_clip=False,
    )
    ax.text(label_x, label_y, label, ha="center", va=label_va, fontsize=6.8)


def draw_top(ax) -> None:
    ax.set_xlim(0, 20)
    ax.set_ylim(0.20, 3.05)
    ax.set_yticks([1.90, 1.30, 0.70])
    ax.set_yticklabels([r"$W_{k-1}$", r"$W_k$", r"$W_{k+1}$"])
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.tick_params(axis="x", length=3, width=0.8, pad=1.5)
    ax.tick_params(axis="y", length=0, pad=4)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)

    row_h = 0.24
    rows = [
        (1.82, row_h, 0.0, 10.0, COLORS["w1"]),
        (1.20, row_h, 5.0, 10.0, COLORS["w2"]),
        (0.58, row_h, 10.0, 10.0, COLORS["w3"]),
    ]
    for y, h, x0, width, face in rows:
        ax.add_patch(Rectangle((x0, y), width, h, facecolor=face, edgecolor=EDGE, linewidth=1.0))

    double_arrow(ax, 0.0, 10.0, 2.82, "10 s window", 5.9, 2.91)
    double_arrow(ax, 0.0, 5.0, 2.52, "5 s shift", 2.2, 2.43, label_va="top")


def draw_bottom(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.42, 2.86)
    ax.set_xticks([0, 5, 10])
    ax.set_xlabel("Time (s)", labelpad=1.5)
    ax.set_yticks([2.0, 1.0, 0.0])
    ax.set_yticklabels(["Ref. 1", "Ref. 2", "Hyp."])
    ax.tick_params(axis="x", length=3, width=0.8, pad=1.5)
    ax.tick_params(axis="y", length=0)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)

    for y in [0.0, 1.0, 2.0]:
        ax.hlines(y, 0, 10, colors=GRID, linewidth=0.8, zorder=0)

    collar_center = 1.90
    ax.axvspan(collar_center - 0.25, collar_center + 0.25, facecolor=COLLAR, edgecolor="none", zorder=0)

    ref_h = 0.30
    hyp_h = 0.26
    ax.broken_barh(
        [(0.90, 1.00), (6.00, 2.00)],
        (2.0 - ref_h / 2, ref_h),
        facecolors=COLORS["w1"],
        edgecolors=EDGE,
        linewidth=0.9,
    )
    ax.broken_barh(
        [(7.00, 1.00)],
        (1.0 - ref_h / 2, ref_h),
        facecolors=COLORS["w3"],
        edgecolors=EDGE,
        linewidth=0.9,
    )
    ax.broken_barh(
        [(0.95, 1.10), (6.05, 1.95)],
        (0.0 - hyp_h / 2, hyp_h),
        facecolors=COLORS["w2"],
        edgecolors=EDGE,
        linewidth=0.9,
    )

    overlap_box = Rectangle((7.00, 0.82), 1.00, 1.34, facecolor="none", edgecolor="#666666", linewidth=1.0, linestyle="--")
    ax.add_patch(overlap_box)

    ax.annotate(
        "0.25 s collar",
        xy=(collar_center, 2.12),
        xytext=(1.25, 2.58),
        fontsize=7.0,
        ha="center",
        va="bottom",
        arrowprops=dict(arrowstyle="-|>", linewidth=0.8, color=EDGE),
    )
    ax.annotate(
        "Overlap retained",
        xy=(7.50, 1.48),
        xytext=(8.10, 2.56),
        fontsize=7.0,
        ha="center",
        va="bottom",
        arrowprops=dict(arrowstyle="-|>", linewidth=0.8, color=EDGE),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the protocol overview figure.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/paper_figures"))
    args = parser.parse_args()

    apply_ieee_style()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(IEEE_SINGLE_COL, 3.02),
        gridspec_kw={"height_ratios": [1.0, 1.25]},
    )
    draw_top(axes[0])
    draw_bottom(axes[1])
    fig.subplots_adjust(left=0.20, right=0.98, top=0.98, bottom=0.14, hspace=0.40)
    save_fig(fig, args.out_dir / "fig_protocol_overview")


if __name__ == "__main__":
    main()

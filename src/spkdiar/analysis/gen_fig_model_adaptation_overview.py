"""Generate the Sortformer inference/adaptation overview figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from spkdiar.analysis.ieee_style import IEEE_SINGLE_COL, apply_ieee_style, save_fig


EDGE = "#333333"
FACES = {
    "input": "#f7f7f7",
    "encoder": "#dbe7f3",
    "heads": "#f5e2c9",
    "slots": "#e1ecd6",
}


def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    kind: str,
    *,
    fontsize: float = 6.8,
    linestyle: str = "-",
    linewidth: float = 1.0,
) -> None:
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor=FACES[kind],
            edgecolor=EDGE,
            linewidth=linewidth,
            linestyle=linestyle,
        )
    )
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize, fontweight="bold")


def add_arrow(ax, x0: float, y0: float, x1: float, y1: float, dashed: bool = False) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.9,
            linestyle="--" if dashed else "-",
            color=EDGE,
            shrinkA=1.5,
            shrinkB=1.5,
        )
    )


def draw_top(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.94, "Inference", fontsize=7.2, fontweight="bold", ha="left", va="top")

    y = 0.44
    h = 0.26
    x = [0.02, 0.23, 0.51, 0.80]
    w = [0.16, 0.22, 0.20, 0.16]
    labels = ["Audio\nwindow", "NEST/Fast\nConformer", "Transformer\nheads", "4 local\nslots"]
    kinds = ["input", "encoder", "heads", "slots"]

    for xi, wi, label, kind in zip(x, w, labels, kinds, strict=True):
        add_box(ax, xi, y, wi, h, label, kind, fontsize=6.7)

    add_arrow(ax, x[0] + w[0], y + h / 2, x[1], y + h / 2)
    add_arrow(ax, x[1] + w[1], y + h / 2, x[2], y + h / 2)
    add_arrow(ax, x[2] + w[2], y + h / 2, x[3], y + h / 2)

    aosc_x, aosc_y, aosc_w, aosc_h = 0.54, 0.12, 0.22, 0.11
    ax.add_patch(
        Rectangle(
            (aosc_x, aosc_y),
            aosc_w,
            aosc_h,
            facecolor="white",
            edgecolor=EDGE,
            linewidth=1.0,
            linestyle="--",
        )
    )
    ax.text(aosc_x + aosc_w / 2, aosc_y + aosc_h / 2, "AOSC", ha="center", va="center", fontsize=6.8, fontweight="bold")
    add_arrow(ax, aosc_x + aosc_w / 2, aosc_y + aosc_h, x[2] + w[2] / 2, y, dashed=True)
    ax.text(aosc_x + aosc_w / 2, aosc_y - 0.040, "streaming only", ha="center", va="top", fontsize=6.6)


def draw_bottom(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.95, "Frozen-encoder adaptation", fontsize=7.2, fontweight="bold", ha="left", va="top")

    y = 0.48
    h = 0.26
    x = [0.02, 0.27, 0.53, 0.81]
    w = [0.18, 0.18, 0.18, 0.15]
    labels = ["Vanilla\noffline", "Encoder", "Transformer\nheads", "Fine-tuned\noffline"]
    kinds = ["input", "encoder", "heads", "slots"]

    for xi, wi, label, kind in zip(x, w, labels, kinds, strict=True):
        style = "--" if label.startswith("Fine-tuned") else "-"
        add_box(ax, xi, y, wi, h, label, kind, fontsize=6.6, linestyle=style)

    add_arrow(ax, x[0] + w[0], y + h / 2, x[1], y + h / 2)
    add_arrow(ax, x[1] + w[1], y + h / 2, x[2], y + h / 2)
    add_arrow(ax, x[2] + w[2], y + h / 2, x[3], y + h / 2)

    ax.text(x[1] + w[1] / 2, y - 0.060, "frozen", ha="center", va="top", fontsize=6.6)
    ax.text(x[2] + w[2] / 2 + 0.03, y - 0.070, "trainable", ha="center", va="top", fontsize=6.6)

    data_x, data_y, data_w, data_h = 0.35, 0.07, 0.30, 0.10
    ax.add_patch(
        Rectangle(
            (data_x, data_y),
            data_w,
            data_h,
            facecolor="white",
            edgecolor=EDGE,
            linewidth=1.0,
            linestyle="--",
        )
    )
    ax.text(data_x + data_w / 2, data_y + data_h / 2, "ATC 90 s windows", ha="center", va="center", fontsize=6.3, fontweight="bold")
    add_arrow(ax, data_x + data_w / 2, data_y + data_h, x[2] + 0.06, y, dashed=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Sortformer overview figure.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/paper_figures"))
    args = parser.parse_args()

    apply_ieee_style()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(IEEE_SINGLE_COL, 3.05),
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )
    draw_top(axes[0])
    draw_bottom(axes[1])
    fig.subplots_adjust(left=0.04, right=0.98, top=0.97, bottom=0.08, hspace=0.22)
    save_fig(fig, args.out_dir / "fig_model_adaptation_overview")


if __name__ == "__main__":
    main()

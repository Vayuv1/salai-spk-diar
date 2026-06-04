from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle


OUT_DIR = Path(__file__).resolve().parent


def add_box(ax, xy, w, h, text, facecolor="#f3f3f3", fontsize=9.2):
    x, y = xy
    patch = Rectangle((x, y), w, h, linewidth=1.1, edgecolor="black", facecolor=facecolor)
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.08,
    )
    return patch


def add_arrow(ax, start, end, style="-|>", lw=1.2):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=11,
        linewidth=lw,
        color="black",
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arrow)
    return arrow


def main():
    fig, ax = plt.subplots(figsize=(7.45, 3.55))
    ax.set_xlim(0, 11.45)
    ax.set_ylim(0, 4.25)
    ax.axis("off")

    proc_fill = "#f1f1f1"
    white_fill = "#ffffff"

    add_box(ax, (0.35, 1.62), 1.30, 0.76, "ATC audio", facecolor=white_fill, fontsize=9.4)

    add_box(ax, (2.10, 2.55), 1.95, 0.78, "Streaming\ndiarization\n(Sortformer)", facecolor=proc_fill, fontsize=8.9)
    add_box(ax, (2.10, 0.92), 1.95, 0.78, "Domain-tuned ASR", facecolor=proc_fill, fontsize=9.0)

    add_box(ax, (4.60, 2.55), 1.55, 0.78, "Role attribution", facecolor=white_fill, fontsize=9.0)
    add_box(ax, (4.60, 0.92), 1.55, 0.78, "Callsign\nextraction", facecolor=white_fill, fontsize=8.9)

    add_box(ax, (6.70, 1.64), 1.90, 0.86, "Joint attribution", facecolor=proc_fill, fontsize=9.2)

    ax.text(10.00, 3.58, "Avionics support outputs", ha="center", va="center", fontsize=9.1)
    add_box(ax, (9.15, 2.78), 1.70, 0.42, "Instr./readback\ncontext", facecolor=white_fill, fontsize=8.1)
    add_box(ax, (9.15, 2.16), 1.70, 0.42, "Communication\nsafety events", facecolor=white_fill, fontsize=8.1)
    add_box(ax, (9.15, 1.54), 1.70, 0.42, "Safety / ATM\nsupport", facecolor=white_fill, fontsize=8.1)
    add_box(ax, (9.15, 0.92), 1.70, 0.42, "Review log", facecolor=white_fill, fontsize=8.3)

    add_arrow(ax, (1.65, 2.00), (2.10, 2.94))
    add_arrow(ax, (1.65, 2.00), (2.10, 1.31))
    add_arrow(ax, (4.05, 2.94), (4.60, 2.94))
    add_arrow(ax, (4.05, 1.31), (4.60, 1.31))
    add_arrow(ax, (6.15, 2.94), (6.70, 2.15))
    add_arrow(ax, (6.15, 1.31), (6.70, 1.99))

    add_arrow(ax, (8.60, 2.07), (8.95, 2.07))
    ax.add_line(Line2D([8.95, 8.95], [1.13, 2.99], linewidth=1.2, color="black"))
    add_arrow(ax, (8.95, 2.99), (9.15, 2.99))
    add_arrow(ax, (8.95, 2.37), (9.15, 2.37))
    add_arrow(ax, (8.95, 1.75), (9.15, 1.75))
    add_arrow(ax, (8.95, 1.13), (9.15, 1.13))

    fig.tight_layout(pad=0.15)
    fig.savefig(OUT_DIR / "fig_role_callsign_framework.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_role_callsign_framework.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()

"""
gen_fig1_der_comparison.py

Fig 1: DER comparison bar chart — pretrained vs fine-tuned Sortformer on 4 eval
recordings. Stacked bars: FA (lightest) / MISS / CER (darkest). Two bars per
recording with hatching to distinguish conditions. Single column (3.5 × 3.0 in).

Output: results/paper_figures/fig1_der_comparison.{pdf,png}

Usage:
    uv run python -m spkdiar.analysis.gen_fig1_der_comparison
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spkdiar.analysis.ieee_style import (
    IEEE_SINGLE_COL, apply_ieee_style, save_fig, BAR_COLORS
)

# ---------------------------------------------------------------------------
# Data (full-recording evaluation, collar=0.25s)
# ---------------------------------------------------------------------------

RECORDINGS = ["dca_d1_1", "dca_d2_2", "dfw_a1_1", "log_id_1"]
REC_LABELS  = ["DCA D1-1", "DCA D2-2", "DFW A1-1", "LOG ID-1"]

DATA = {
    "pretrained": {
        "dca_d1_1": {"FA": 2.66,  "MISS": 1.42, "CER": 16.80},
        "dca_d2_2": {"FA": 2.41,  "MISS": 0.67, "CER": 17.14},
        "dfw_a1_1": {"FA": 30.96, "MISS": 0.50, "CER": 15.92},
        "log_id_1": {"FA": 33.14, "MISS": 1.53, "CER": 10.06},
    },
    "finetuned": {
        "dca_d1_1": {"FA": 1.60,  "MISS": 1.90, "CER": 9.08},
        "dca_d2_2": {"FA": 1.56,  "MISS": 1.61, "CER": 3.75},
        "dfw_a1_1": {"FA": 13.13, "MISS": 2.01, "CER": 6.10},
        "log_id_1": {"FA": 4.32,  "MISS": 2.09, "CER": 5.37},
    },
}

COMPONENTS = ["FA", "MISS", "CER"]
COMP_LABELS = {"FA": "False Alarm", "MISS": "Missed Speech", "CER": "Confusion (CER)"}


def main(out_dir: Path = Path("results/paper_figures")) -> None:
    apply_ieee_style()

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 3.0))

    n_recs = len(RECORDINGS)
    bar_w  = 0.32
    x      = np.arange(n_recs)

    # Color cycle: progressively darker for FA→MISS→CER
    component_colors = {
        "FA":   "#aec7e8",
        "MISS": "#4292c6",
        "CER":  "#084594",
    }

    conditions = [
        ("pretrained", "Pretrained",   -bar_w / 2 - 0.005, ""),
        ("finetuned",  "Fine-tuned",    bar_w / 2 + 0.005, "///"),
    ]

    # Draw stacked bars
    for cond_key, cond_label, x_offset, hatch in conditions:
        bottom = np.zeros(n_recs)
        for comp_idx, comp in enumerate(COMPONENTS):
            vals = np.array([DATA[cond_key][r][comp] for r in RECORDINGS])
            label = f"{COMP_LABELS[comp]} ({cond_label})" if comp_idx == 2 else None
            ax.bar(
                x + x_offset,
                vals,
                bar_w,
                bottom=bottom,
                color=component_colors[comp],
                hatch=hatch,
                edgecolor="white" if hatch == "" else "#555555",
                linewidth=0.4,
                label=label,
                zorder=3,
            )
            bottom += vals

    # Add DER value labels on top of each bar
    for cond_key, _, x_offset, _ in conditions:
        for i, rec in enumerate(RECORDINGS):
            der = sum(DATA[cond_key][rec][c] for c in COMPONENTS)
            ax.text(
                x[i] + x_offset, der + 0.5,
                f"{der:.1f}",
                ha="center", va="bottom",
                fontsize=6.5, color="#222222",
            )

    # Legend — two entries for condition + three component color patches
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#084594", hatch="",    edgecolor="#555", label="Pretrained"),
        Patch(facecolor="#084594", hatch="///", edgecolor="#555", label="Fine-tuned"),
        Patch(facecolor=component_colors["FA"],   edgecolor="none", label="FA"),
        Patch(facecolor=component_colors["MISS"], edgecolor="none", label="MISS"),
        Patch(facecolor=component_colors["CER"],  edgecolor="none", label="CER"),
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=7,
        ncol=2,
        loc="upper left",
        framealpha=0.85,
        handlelength=1.0,
        handleheight=0.8,
        borderpad=0.5,
        columnspacing=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(REC_LABELS, fontsize=8)
    ax.set_ylabel("Diarization Error Rate (%)", fontsize=9)
    ax.set_ylim(0, 57)
    ax.yaxis.grid(True, linewidth=0.4, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)

    fig.tight_layout(pad=0.4)
    save_fig(fig, out_dir / "fig1_der_comparison")


if __name__ == "__main__":
    main()

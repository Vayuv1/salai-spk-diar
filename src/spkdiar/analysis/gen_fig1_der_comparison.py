"""Generate the DER component comparison figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from spkdiar.analysis.ieee_style import IEEE_SINGLE_COL, apply_ieee_style, save_fig

RECORDINGS = ["dca_d1_1", "dca_d2_2", "dfw_a1_1", "log_id_1"]
REC_LABELS = {
    "dca_d1_1": "R1\n(DCA)",
    "dca_d2_2": "R2\n(DCA)",
    "dfw_a1_1": "R3\n(DFW)",
    "log_id_1": "R4\n(LOG)",
}
COMPONENTS = ["FA", "MISS", "CER"]
COMPONENT_COLORS = {
    "FA": "#9ecae1",
    "MISS": "#fdae6b",
    "CER": "#807dba",
}
CONDITIONS = [
    ("pretrained", "Vanilla", -0.165, ""),
    ("finetuned", "Fine-tuned", 0.165, "////"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Fig. 1 from rerun metrics.")
    parser.add_argument(
        "--pretrained-json",
        type=Path,
        default=Path("results/repro/sortformer_pretrained_eval4_rerun/eval_metrics.json"),
    )
    parser.add_argument(
        "--finetuned-json",
        type=Path,
        default=Path("results/repro/sortformer_finetuned_eval4_rerun/eval_metrics.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/paper_figures"),
    )
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, dict[str, float]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    rows: dict[str, dict[str, float]] = {}
    for row in payload["recordings"]:
        rows[row["recording"]] = {comp: float(row[comp]) for comp in COMPONENTS}
    missing = [rec for rec in RECORDINGS if rec not in rows]
    if missing:
        raise ValueError(f"{path} is missing recordings: {', '.join(missing)}")
    return rows


def main() -> None:
    args = parse_args()
    apply_ieee_style()

    data = {
        "pretrained": load_metrics(args.pretrained_json),
        "finetuned": load_metrics(args.finetuned_json),
    }

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 3.2))
    bar_width = 0.30
    x = np.arange(len(RECORDINGS))
    max_der = 0.0

    for cond_key, _cond_label, x_offset, hatch in CONDITIONS:
        bottom = np.zeros(len(RECORDINGS))
        for comp in COMPONENTS:
            values = np.array([data[cond_key][rec][comp] for rec in RECORDINGS])
            ax.bar(
                x + x_offset,
                values,
                bar_width,
                bottom=bottom,
                color=COMPONENT_COLORS[comp],
                hatch=hatch,
                edgecolor="#303030",
                linewidth=0.5,
                zorder=3,
            )
            bottom += values

        max_der = max(max_der, float(bottom.max()))
        for idx, rec in enumerate(RECORDINGS):
            der = sum(data[cond_key][rec][comp] for comp in COMPONENTS)
            ax.text(
                x[idx] + x_offset,
                der + 0.55,
                f"{der:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#222222",
            )

    condition_handles = [
        Patch(facecolor="#ffffff", hatch="", edgecolor="#303030", label="Vanilla"),
        Patch(facecolor="#d9d9d9", hatch="////", edgecolor="#303030", label="Fine-tuned"),
    ]
    component_handles = [
        Patch(facecolor=COMPONENT_COLORS["FA"], edgecolor="#303030", label="FA"),
        Patch(facecolor=COMPONENT_COLORS["MISS"], edgecolor="#303030", label="MISS"),
        Patch(facecolor=COMPONENT_COLORS["CER"], edgecolor="#303030", label="CER"),
    ]
    legend_system = ax.legend(
        handles=condition_handles,
        fontsize=7,
        ncol=1,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        framealpha=0.95,
        handlelength=1.2,
        borderpad=0.45,
        columnspacing=0.8,
        title="System",
    )
    ax.add_artist(legend_system)
    ax.legend(
        handles=component_handles,
        fontsize=7,
        ncol=1,
        loc="lower left",
        bbox_to_anchor=(0.48, 1.02),
        framealpha=0.95,
        handlelength=1.2,
        borderpad=0.45,
        columnspacing=0.8,
        title="DER component",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([REC_LABELS[rec] for rec in RECORDINGS], fontsize=8)
    ax.set_ylabel("Diarization Error Rate (%)", fontsize=9)
    ax.set_ylim(0, min(60, max_der + 6.0))
    ax.yaxis.grid(True, linewidth=0.4, color="#d0d0d0", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), pad=0.4)
    save_fig(fig, args.out_dir / "fig1_der_comparison")


if __name__ == "__main__":
    main()

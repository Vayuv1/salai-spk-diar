"""
gen_fig_checkpoint_sweep.py

Plot held-out DER/CER/FA/MISS as a function of fine-tuning step.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from spkdiar.analysis.ieee_style import IEEE_DOUBLE_COL, apply_ieee_style, save_fig


def load_rows(summary_csv: Path) -> list[dict]:
    with open(summary_csv, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["step"] = int(row["step"])
        for key in ("DER", "FA", "MISS", "CER"):
            row[key] = float(row[key])
    rows.sort(key=lambda item: item["step"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot held-out metrics over fine-tuning checkpoints.")
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--pretrained-json", type=Path,
                        default=Path("results/repro/sortformer_pretrained_eval4_rerun/eval_metrics.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/paper_figures"))
    args = parser.parse_args()

    rows = load_rows(args.summary_csv)
    pretrained = json.loads(args.pretrained_json.read_text())["overall"]
    steps = [row["step"] for row in rows]

    apply_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_DOUBLE_COL, 2.6))

    axes[0].plot(steps, [row["DER"] for row in rows], marker="o", color="#1f77b4", label="DER")
    axes[0].plot(steps, [row["CER"] for row in rows], marker="s", color="#d62728", linestyle="--", label="CER")
    axes[0].axhline(pretrained["DER"], color="#1f77b4", linestyle=":", linewidth=0.9)
    axes[0].axhline(pretrained["CER"], color="#d62728", linestyle=":", linewidth=0.9)
    axes[0].set_xlabel("Fine-Tuning Step")
    axes[0].set_ylabel("Error Rate (%)")
    axes[0].set_title("Evaluation DER and CER")
    axes[0].grid(axis="y", linestyle=":", linewidth=0.5)
    axes[0].legend(
        handles=[
            Line2D([], [], marker="o", color="#1f77b4", linestyle="-", label="DER"),
            Line2D([], [], marker="s", color="#d62728", linestyle="--", label="CER"),
            Line2D([], [], color="#1f77b4", linestyle=":", label="Vanilla DER"),
            Line2D([], [], color="#d62728", linestyle=":", label="Vanilla CER"),
        ],
        loc="upper right",
        framealpha=1.0,
    )

    axes[1].plot(steps, [row["FA"] for row in rows], marker="^", color="#2ca02c", label="FA")
    axes[1].plot(steps, [row["MISS"] for row in rows], marker="D", color="#9467bd", linestyle="--", label="MISS")
    axes[1].axhline(pretrained["FA"], color="#2ca02c", linestyle=":", linewidth=0.9)
    axes[1].axhline(pretrained["MISS"], color="#9467bd", linestyle=":", linewidth=0.9)
    axes[1].set_xlabel("Fine-Tuning Step")
    axes[1].set_ylabel("Error Rate (%)")
    axes[1].set_title("Evaluation FA and MISS")
    axes[1].grid(axis="y", linestyle=":", linewidth=0.5)
    axes[1].legend(
        handles=[
            Line2D([], [], marker="^", color="#2ca02c", linestyle="-", label="FA"),
            Line2D([], [], marker="D", color="#9467bd", linestyle="--", label="MISS"),
            Line2D([], [], color="#2ca02c", linestyle=":", label="Vanilla FA"),
            Line2D([], [], color="#9467bd", linestyle=":", label="Vanilla MISS"),
        ],
        loc="upper right",
        framealpha=1.0,
    )

    fig.tight_layout(pad=0.4, w_pad=1.0)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, args.out_dir / "fig_checkpoint_sweep")


if __name__ == "__main__":
    main()

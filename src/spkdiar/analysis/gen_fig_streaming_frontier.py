"""
gen_fig_streaming_frontier.py

Plot the streaming latency frontier against the offline baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spkdiar.analysis.ieee_style import IEEE_DOUBLE_COL, apply_ieee_style, save_fig

LATENCY_SECONDS = {
    "ultra-low": 0.32,
    "low": 1.04,
    "medium": 10.0,
}


def load_rows(summary_csv: Path) -> list[dict]:
    with open(summary_csv, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["latency_sec"] = LATENCY_SECONDS[row["latency"]]
        for key in ("DER", "FA", "MISS", "CER"):
            row[key] = float(row[key])
    rows.sort(key=lambda item: item["latency_sec"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the streaming latency frontier.")
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--offline-json", type=Path,
                        default=Path("results/repro/sortformer_pretrained_eval4_rerun/eval_metrics.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/paper_figures"))
    args = parser.parse_args()

    rows = load_rows(args.summary_csv)
    offline = json.loads(args.offline_json.read_text())["overall"]
    x = [row["latency_sec"] for row in rows]

    apply_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_DOUBLE_COL, 2.6))

    axes[0].plot(x, [row["DER"] for row in rows], marker="o", color="#1f77b4", label="Streaming DER")
    axes[0].axhline(offline["DER"], color="#1f77b4", linestyle="--", linewidth=0.9, label="Offline DER")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Latency (s)")
    axes[0].set_ylabel("DER (%)")
    axes[0].set_title("Latency vs DER")
    axes[0].grid(axis="y", linestyle=":", linewidth=0.5)
    axes[0].legend(loc="upper right", framealpha=1.0)

    axes[1].plot(x, [row["CER"] for row in rows], marker="s", color="#d62728", linestyle="--", label="Streaming CER")
    axes[1].plot(x, [row["FA"] for row in rows], marker="^", color="#2ca02c", linestyle="-.", label="Streaming FA")
    axes[1].axhline(offline["CER"], color="#d62728", linestyle="--", linewidth=0.9, label="Offline CER")
    axes[1].axhline(offline["FA"], color="#2ca02c", linestyle="--", linewidth=0.9, label="Offline FA")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Latency (s)")
    axes[1].set_ylabel("Error Rate (%)")
    axes[1].set_title("Latency vs CER / FA")
    axes[1].grid(axis="y", linestyle=":", linewidth=0.5)
    axes[1].legend(loc="upper right", framealpha=1.0)

    fig.tight_layout(pad=0.4, w_pad=1.0)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, args.out_dir / "fig_streaming_frontier")


if __name__ == "__main__":
    main()

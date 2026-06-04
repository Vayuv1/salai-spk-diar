"""
gen_fig_streaming_threshold_transfer.py

Plot calibration threshold-sweep behavior and the corresponding evaluation
transfer result for the selected streaming operating point.
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


def load_sweep_rows(summary_csv: Path) -> list[dict]:
    with open(summary_csv, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["threshold"] = float(row["threshold"])
        for key in ("DER", "FA", "MISS", "CER"):
            row[key] = float(row[key])
    rows.sort(key=lambda item: item["threshold"])
    return rows


def choose_threshold(rows: list[dict], selected_threshold: float | None) -> dict:
    if selected_threshold is None:
        return min(rows, key=lambda item: (item["DER"], item["FA"], item["CER"]))

    for row in rows:
        if abs(row["threshold"] - selected_threshold) < 1e-9:
            return row
    raise ValueError(f"Selected threshold {selected_threshold:.4f} not present in sweep.")


def load_overall_metrics(metrics_json: Path) -> dict:
    return json.loads(metrics_json.read_text())["overall"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot calibration threshold sweep and evaluation transfer metrics."
    )
    parser.add_argument("--calibration-csv", type=Path, required=True)
    parser.add_argument("--default-test-json", type=Path, required=True)
    parser.add_argument("--calibrated-test-json", type=Path, default=None)
    parser.add_argument("--selected-threshold", type=float, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("results/paper_figures"))
    args = parser.parse_args()

    rows = load_sweep_rows(args.calibration_csv)
    selected = choose_threshold(rows, args.selected_threshold)
    default_test = load_overall_metrics(args.default_test_json)
    calibrated_test = (
        load_overall_metrics(args.calibrated_test_json)
        if args.calibrated_test_json is not None
        else None
    )

    apply_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_DOUBLE_COL, 2.6))

    thresholds = [row["threshold"] for row in rows]
    axes[0].plot(thresholds, [row["DER"] for row in rows], marker="o", color="#1f77b4", label="DER")
    axes[0].plot(thresholds, [row["FA"] for row in rows], marker="^", color="#2ca02c", linestyle="-.", label="FA")
    axes[0].plot(thresholds, [row["CER"] for row in rows], marker="s", color="#d62728", linestyle="--", label="CER")
    axes[0].axvline(selected["threshold"], color="black", linestyle=":", linewidth=0.9)
    axes[0].set_xlabel("Calibration Onset / Offset Threshold")
    axes[0].set_ylabel("Calibration Error Rate (%)")
    axes[0].set_title("Calibration Sweep")
    axes[0].grid(axis="y", linestyle=":", linewidth=0.5)
    axes[0].legend(loc="upper right", framealpha=1.0)

    metric_names = ["DER", "FA", "CER"]
    x = range(len(metric_names))
    width = 0.34 if calibrated_test is not None else 0.5
    axes[1].bar(
        [idx - width / 2 for idx in x] if calibrated_test is not None else list(x),
        [default_test[name] for name in metric_names],
        width=width,
        color="#7f7f7f",
        label="Vanilla",
    )
    if calibrated_test is not None:
        axes[1].bar(
            [idx + width / 2 for idx in x],
            [calibrated_test[name] for name in metric_names],
            width=width,
            color="#1f77b4",
            label=f"Calibrated @ {selected['threshold']:.2f}",
        )
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(metric_names)
    axes[1].set_ylabel("Evaluation Error Rate (%)")
    axes[1].set_title("Evaluation Transfer")
    axes[1].grid(axis="y", linestyle=":", linewidth=0.5)
    axes[1].legend(loc="upper right", framealpha=1.0)

    fig.tight_layout(pad=0.4, w_pad=1.0)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, args.out_dir / "fig_streaming_threshold_transfer")

    summary = {
        "selected_threshold": selected["threshold"],
        "validation_selected_metrics": {
            "DER": selected["DER"],
            "FA": selected["FA"],
            "MISS": selected["MISS"],
            "CER": selected["CER"],
        },
        "default_test_metrics": {name: default_test[name] for name in metric_names + ["MISS"]},
        "calibrated_test_metrics": (
            {name: calibrated_test[name] for name in metric_names + ["MISS"]}
            if calibrated_test is not None
            else None
        ),
    }
    (args.out_dir / "streaming_threshold_transfer_summary.json").write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main()

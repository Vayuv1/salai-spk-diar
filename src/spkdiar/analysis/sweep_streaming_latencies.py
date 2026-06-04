"""
sweep_streaming_latencies.py

Evaluate all configured streaming latency presets on the same held-out manifest
and export a single summary table.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from spkdiar.inference.run_streaming import (
    LATENCY_PRESETS,
    filter_manifest,
    run_streaming_inference,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all requested streaming latency presets on a held-out manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rec-ids", type=str, required=True)
    parser.add_argument("--latencies", type=str, default="ultra-low,low,medium")
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--collar", type=float, default=0.25)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rec_ids = [item.strip() for item in args.rec_ids.split(",")]
    latencies = [item.strip() for item in args.latencies.split(",") if item.strip()]
    for latency in latencies:
        if latency not in LATENCY_PRESETS:
            raise ValueError(f"Unknown latency preset: {latency}")

    tmp_manifest = filter_manifest(args.manifest, rec_ids=rec_ids)
    rows: list[dict] = []

    try:
        for latency in latencies:
            run_dir = args.out_dir / latency.replace("-", "_")
            metrics_path = run_dir / "eval_metrics.json"
            if args.force or not metrics_path.exists():
                run_streaming_inference(
                    model_path=args.model_path,
                    manifest_path=tmp_manifest,
                    out_dir=run_dir,
                    latency=latency,
                    precision=args.precision,
                    collar=args.collar,
                    save_probs=False,
                )
            summary = json.loads(metrics_path.read_text())
            rows.append({
                "latency": latency,
                "latency_label": LATENCY_PRESETS[latency]["label"],
                "DER": summary["overall"]["DER"],
                "FA": summary["overall"]["FA"],
                "MISS": summary["overall"]["MISS"],
                "CER": summary["overall"]["CER"],
                "metrics_json": str(metrics_path.resolve()),
            })
    finally:
        if tmp_manifest.exists():
            tmp_manifest.unlink()

    rows.sort(key=lambda item: item["DER"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = args.out_dir / "latency_sweep_summary.json"
    summary_csv = args.out_dir / "latency_sweep_summary.csv"
    summary_json.write_text(json.dumps(rows, indent=2))

    with open(summary_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["latency", "latency_label", "DER", "FA", "MISS", "CER", "metrics_json"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

"""
sweep_streaming_thresholds.py

Sweep onset/offset thresholds over saved streaming probability tensors and
export a summary table.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from spkdiar.analysis.reevaluate_prob_tensors import (
    filter_manifest,
    reevaluate_prob_tensors,
)


def parse_thresholds(spec: str) -> list[float]:
    values = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("No thresholds parsed from spec.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep thresholds over saved streaming probability tensors."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prob-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--thresholds", type=str, required=True,
                        help="Comma-separated onset/offset values, e.g. 0.50,0.55,0.60")
    parser.add_argument("--collar", type=float, default=0.25)
    parser.add_argument("--rec-ids", type=str, default=None)
    parser.add_argument("--exclude-rec-ids", type=str, default=None)
    parser.add_argument("--max-offset", type=float, default=None)
    args = parser.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    rec_ids = [item.strip() for item in args.rec_ids.split(",")] if args.rec_ids else None
    exclude_rec_ids = (
        [item.strip() for item in args.exclude_rec_ids.split(",")]
        if args.exclude_rec_ids else None
    )

    manifest = args.manifest
    tmp_manifest = None
    if rec_ids or exclude_rec_ids or args.max_offset is not None:
        tmp_manifest = filter_manifest(
            manifest,
            rec_ids=rec_ids,
            exclude_rec_ids=exclude_rec_ids,
            max_offset=args.max_offset,
        )
        manifest = tmp_manifest

    rows: list[dict] = []
    try:
        for threshold in thresholds:
            tag = f"thr_{threshold:.2f}".replace(".", "p")
            subdir = args.out_dir / tag
            reevaluate_prob_tensors(
                manifest_path=manifest,
                prob_dir=args.prob_dir,
                out_dir=subdir,
                collar=args.collar,
                onset=threshold,
                offset=threshold,
                bypass_postprocessing=False,
            )
            summary = json.loads((subdir / "eval_metrics.json").read_text())
            rows.append({
                "threshold": threshold,
                "DER": summary["overall"]["DER"],
                "FA": summary["overall"]["FA"],
                "MISS": summary["overall"]["MISS"],
                "CER": summary["overall"]["CER"],
                "summary_json": str((subdir / "eval_metrics.json").resolve()),
            })
    finally:
        if tmp_manifest and tmp_manifest.exists():
            tmp_manifest.unlink()

    rows.sort(key=lambda item: item["threshold"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = args.out_dir / "threshold_sweep_summary.json"
    summary_csv = args.out_dir / "threshold_sweep_summary.csv"
    summary_json.write_text(json.dumps(rows, indent=2))

    with open(summary_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["threshold", "DER", "FA", "MISS", "CER", "summary_json"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

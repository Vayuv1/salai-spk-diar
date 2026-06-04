"""Helpers for exporting grouped diarization metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def base_recording_id(uniq_id: str) -> str:
    parts = uniq_id.rsplit("-", 2)
    return parts[0] if len(parts) == 3 else uniq_id


def build_metric_summary(metric, collar: float, ignore_overlap: bool) -> dict:
    grouped: dict[str, dict[str, float | int]] = {}

    for uniq_id, detail in metric.results_:
        rec_id = base_recording_id(uniq_id)
        bucket = grouped.setdefault(
            rec_id,
            {
                "recording": rec_id,
                "units_evaluated": 0,
                "false_alarm": 0.0,
                "missed_detection": 0.0,
                "confusion": 0.0,
                "total": 0.0,
            },
        )
        bucket["units_evaluated"] += 1
        bucket["false_alarm"] += float(detail["false alarm"])
        bucket["missed_detection"] += float(detail["missed detection"])
        bucket["confusion"] += float(detail["confusion"])
        bucket["total"] += float(detail["total"])

    recordings = []
    for rec_id in sorted(grouped):
        item = grouped[rec_id]
        total = float(item["total"])
        fa = float(item["false_alarm"]) / total if total else 0.0
        miss = float(item["missed_detection"]) / total if total else 0.0
        cer = float(item["confusion"]) / total if total else 0.0
        der = fa + miss + cer
        recordings.append(
            {
                "recording": rec_id,
                "units_evaluated": int(item["units_evaluated"]),
                "total_ref_speech_sec": total,
                "DER": 100.0 * der,
                "FA": 100.0 * fa,
                "MISS": 100.0 * miss,
                "CER": 100.0 * cer,
                "der_fraction": der,
                "fa_fraction": fa,
                "miss_fraction": miss,
                "cer_fraction": cer,
            }
        )

    overall_total = float(metric["total"])
    overall = {
        "DER": 100.0 * abs(metric),
        "FA": 100.0 * (float(metric["false alarm"]) / overall_total if overall_total else 0.0),
        "MISS": 100.0 * (float(metric["missed detection"]) / overall_total if overall_total else 0.0),
        "CER": 100.0 * (float(metric["confusion"]) / overall_total if overall_total else 0.0),
        "total_ref_speech_sec": overall_total,
    }

    return {
        "collar_sec": collar,
        "ignore_overlap": ignore_overlap,
        "recordings": recordings,
        "overall": overall,
    }


def save_metric_summary(summary: dict, out_dir: Path, stem: str = "eval_metrics") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    fieldnames = [
        "recording",
        "units_evaluated",
        "total_ref_speech_sec",
        "DER",
        "FA",
        "MISS",
        "CER",
        "der_fraction",
        "fa_fraction",
        "miss_fraction",
        "cer_fraction",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary["recordings"])

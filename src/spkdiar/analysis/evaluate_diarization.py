"""
evaluate_diarization.py

Export reproducible DER/FA/MISS/CER metrics from predicted RTTMs.

Supports two evaluation modes:

1. `windowed`
   Intended for per-window RTTMs named `{rec_id}-{start_ms}-{dur_ms}.rttm`.
   Each window is scored independently against the clipped ground-truth RTTM
   over the same UEM, then errors are aggregated across all windows for a
   recording. This is useful for ad hoc analysis, but the paper-facing
   Sortformer tables should prefer the machine-generated `eval_metrics.json`
   files exported directly by the inference scripts.

2. `full`
   Intended for full-recording RTTMs named `{rec_id}.rttm`. Each recording is
   scored once against its reference RTTM, optionally using a duration from a
   manifest as the UEM.

Outputs:
  - JSON summary with per-recording metrics and macro averages
  - CSV table for direct paper/plot use

Usage examples:

    # Ad hoc window-wise evaluation on 4 held-out recordings
    uv run python -m spkdiar.analysis.evaluate_diarization \
        --mode windowed \
        --pred-dir results/sortformer_offline/pred_rttm \
        --ref-dir data/processed/rttm \
        --rec-ids dca_d1_1,dca_d2_2,dfw_a1_1,log_id_1 \
        --out-json results/repro/pretrained_windowed_eval.json \
        --out-csv results/repro/pretrained_windowed_eval.csv

    # Full-recording evaluation for a baseline that outputs one RTTM per file
    uv run python -m spkdiar.analysis.evaluate_diarization \
        --mode full \
        --pred-dir results/lseend/pred_rttm \
        --ref-dir data/processed/rttm \
        --rec-ids dca_d1_1,dca_d2_2 \
        --manifest data/processed/manifests/full_manifest.jsonl \
        --out-json results/repro/lseend_full_eval.json
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate


@dataclass(frozen=True)
class WindowSpec:
    rec_id: str
    start_sec: float
    duration_sec: float
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate diarization RTTM outputs.")
    parser.add_argument("--mode", choices=["windowed", "full"], required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--ref-dir", type=Path, required=True)
    parser.add_argument("--rec-ids", type=str, required=True)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Optional manifest with per-recording durations for UEM clipping.")
    parser.add_argument("--collar", type=float, default=0.25)
    parser.add_argument("--crop-left-sec", type=float, default=0.0,
                        help="Optional left crop applied to each window UEM in windowed mode.")
    parser.add_argument("--crop-right-sec", type=float, default=0.0,
                        help="Optional right crop applied to each window UEM in windowed mode.")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, default=None)
    return parser.parse_args()


def load_manifest_durations(manifest_path: Path | None) -> dict[str, float]:
    if manifest_path is None:
        return {}
    durations: dict[str, float] = {}
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            durations[entry["uniq_id"]] = float(entry["duration"])
    return durations


def load_rttm_annotation(
    rttm_path: Path,
    clip_start: float | None = None,
    clip_end: float | None = None,
) -> Annotation:
    annotation = Annotation(uri=rttm_path.stem)
    if not rttm_path.exists():
        return annotation

    with open(rttm_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 9 or parts[0] != "SPEAKER":
                continue

            start = float(parts[3])
            end = start + float(parts[4])
            speaker = parts[7]

            if clip_start is not None:
                start = max(start, clip_start)
            if clip_end is not None:
                end = min(end, clip_end)

            if end > start:
                annotation[Segment(start, end)] = speaker

    return annotation


def parse_window_filename(path: Path) -> WindowSpec:
    rec_id, start_ms, dur_ms = path.stem.rsplit("-", 2)
    return WindowSpec(
        rec_id=rec_id,
        start_sec=int(start_ms) / 1000.0,
        duration_sec=int(dur_ms) / 1000.0,
        path=path,
    )


def percent(value: float) -> float:
    return 100.0 * float(value)


def summarise_recording(
    rec_id: str,
    totals: dict[str, float],
    n_units: int,
    mode: str,
) -> dict:
    total_ref = totals["total"]
    fa = totals["false_alarm"] / total_ref if total_ref else 0.0
    miss = totals["missed_detection"] / total_ref if total_ref else 0.0
    conf = totals["confusion"] / total_ref if total_ref else 0.0
    der = fa + miss + conf

    return {
        "recording": rec_id,
        "mode": mode,
        "units_evaluated": n_units,
        "total_ref_speech_sec": total_ref,
        "der_fraction": der,
        "fa_fraction": fa,
        "miss_fraction": miss,
        "cer_fraction": conf,
        "DER": percent(der),
        "FA": percent(fa),
        "MISS": percent(miss),
        "CER": percent(conf),
    }


def evaluate_windowed(
    pred_dir: Path,
    ref_dir: Path,
    rec_ids: list[str],
    collar: float,
    crop_left_sec: float,
    crop_right_sec: float,
) -> list[dict]:
    results: list[dict] = []

    for rec_id in rec_ids:
        ref_rttm = ref_dir / f"{rec_id}.rttm"
        ref_full = load_rttm_annotation(ref_rttm)
        metric = DiarizationErrorRate(collar=collar, skip_overlap=False)

        totals = {
            "false_alarm": 0.0,
            "missed_detection": 0.0,
            "confusion": 0.0,
            "total": 0.0,
        }
        n_windows = 0

        for path in sorted(pred_dir.glob(f"{rec_id}-*.rttm")):
            window = parse_window_filename(path)
            clip_start = window.start_sec + crop_left_sec
            clip_end = window.start_sec + window.duration_sec - crop_right_sec
            if clip_end <= clip_start:
                continue

            ref = load_rttm_annotation(ref_rttm, clip_start=clip_start, clip_end=clip_end)
            hyp = load_rttm_annotation(path, clip_start=clip_start, clip_end=clip_end)
            uem = Timeline([Segment(clip_start, clip_end)])
            detail = metric(ref, hyp, uem=uem, detailed=True)

            totals["false_alarm"] += float(detail["false alarm"])
            totals["missed_detection"] += float(detail["missed detection"])
            totals["confusion"] += float(detail["confusion"])
            totals["total"] += float(detail["total"])
            n_windows += 1

        results.append(summarise_recording(rec_id, totals, n_windows, mode="windowed"))

    return results


def evaluate_full(
    pred_dir: Path,
    ref_dir: Path,
    rec_ids: list[str],
    manifest_durations: dict[str, float],
    collar: float,
) -> list[dict]:
    results: list[dict] = []

    for rec_id in rec_ids:
        pred_rttm = pred_dir / f"{rec_id}.rttm"
        ref_rttm = ref_dir / f"{rec_id}.rttm"
        ref = load_rttm_annotation(ref_rttm)
        hyp = load_rttm_annotation(pred_rttm)
        metric = DiarizationErrorRate(collar=collar, skip_overlap=False)

        duration = manifest_durations.get(rec_id)
        if duration is not None:
            detail = metric(ref, hyp, uem=Timeline([Segment(0.0, duration)]), detailed=True)
        else:
            detail = metric(ref, hyp, detailed=True)

        totals = {
            "false_alarm": float(detail["false alarm"]),
            "missed_detection": float(detail["missed detection"]),
            "confusion": float(detail["confusion"]),
            "total": float(detail["total"]),
        }
        results.append(summarise_recording(rec_id, totals, 1, mode="full"))

    return results


def build_summary(
    results: list[dict],
    args: argparse.Namespace,
) -> dict:
    macro = {}
    if results:
        for key in ("DER", "FA", "MISS", "CER"):
            macro[key] = sum(row[key] for row in results) / len(results)

    return {
        "mode": args.mode,
        "pred_dir": str(args.pred_dir),
        "ref_dir": str(args.ref_dir),
        "recordings": results,
        "macro_average_percent": macro,
        "collar_sec": args.collar,
        "crop_left_sec": args.crop_left_sec,
        "crop_right_sec": args.crop_right_sec,
    }


def write_csv(path: Path, results: list[dict]) -> None:
    fieldnames = [
        "recording",
        "mode",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    args = parse_args()
    rec_ids = [item.strip() for item in args.rec_ids.split(",") if item.strip()]
    manifest_durations = load_manifest_durations(args.manifest)

    if args.mode == "windowed":
        results = evaluate_windowed(
            pred_dir=args.pred_dir,
            ref_dir=args.ref_dir,
            rec_ids=rec_ids,
            collar=args.collar,
            crop_left_sec=args.crop_left_sec,
            crop_right_sec=args.crop_right_sec,
        )
    else:
        results = evaluate_full(
            pred_dir=args.pred_dir,
            ref_dir=args.ref_dir,
            rec_ids=rec_ids,
            manifest_durations=manifest_durations,
            collar=args.collar,
        )

    summary = build_summary(results, args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if args.out_csv is not None:
        write_csv(args.out_csv, results)

    for row in results:
        print(
            f"{row['recording']:12s} "
            f"DER={row['DER']:.2f}% "
            f"FA={row['FA']:.2f}% "
            f"MISS={row['MISS']:.2f}% "
            f"CER={row['CER']:.2f}% "
            f"units={row['units_evaluated']}"
        )


if __name__ == "__main__":
    main()

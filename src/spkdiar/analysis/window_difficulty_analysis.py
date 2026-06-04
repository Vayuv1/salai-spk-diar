"""
window_difficulty_analysis.py

Quantify how diarization error changes with 10-second window difficulty on the
held-out evaluation set. The analysis keeps absolute error seconds so speech-
normalized rates are only computed where the denominator is valid.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

from spkdiar.analysis.ieee_style import IEEE_DOUBLE_COL, SYSTEM_STYLES, apply_ieee_style, save_fig

DEFAULT_SYSTEMS = {
    "pretrained": Path("results/repro/sortformer_pretrained_eval4_rerun/pred_rttm"),
    "streaming": Path("results/repro/sortformer_streaming10_eval4_rerun/pred_rttm"),
    "finetuned": Path("results/repro/sortformer_finetuned_eval4_rerun/pred_rttm"),
}


def is_controller(raw_speaker_id: str, rec_id: str) -> bool:
    prefix = rec_id + "_"
    bare = raw_speaker_id[len(prefix):] if raw_speaker_id.startswith(prefix) else raw_speaker_id
    return "-" in bare


def load_annotation(rttm_path: Path) -> Annotation:
    annotation = Annotation(uri=rttm_path.stem)
    with open(rttm_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            end = start + float(parts[4])
            annotation[Segment(start, end)] = parts[7]
    return annotation


def clipped_tracks(annotation: Annotation, start: float, end: float) -> list[tuple[float, float, str]]:
    tracks = []
    for seg, _, speaker in annotation.itertracks(yield_label=True):
        cs = max(start, seg.start)
        ce = min(end, seg.end)
        if ce > cs:
            tracks.append((cs, ce, speaker))
    tracks.sort(key=lambda item: (item[0], item[1], item[2]))
    return tracks


def union_duration(segments: list[tuple[float, float]]) -> float:
    if not segments:
        return 0.0
    segments = sorted(segments)
    total = 0.0
    cur_start, cur_end = segments[0]
    for start, end in segments[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return total


def turn_changes(tracks: list[tuple[float, float, str]]) -> int:
    if not tracks:
        return 0
    changes = 0
    prev_speaker = tracks[0][2]
    for _start, _end, speaker in tracks[1:]:
        if speaker != prev_speaker:
            changes += 1
        prev_speaker = speaker
    return changes


def dominant_role(tracks: list[tuple[float, float, str]], rec_id: str) -> str:
    controller = 0.0
    pilot = 0.0
    for start, end, speaker in tracks:
        if is_controller(speaker, rec_id):
            controller += end - start
        else:
            pilot += end - start
    if controller == pilot:
        return "mixed"
    return "controller" if controller > pilot else "pilot"


def speaker_bin(count: int) -> str:
    return str(count) if count < 4 else "4+"


def turn_bin(count: int) -> str:
    return str(count) if count < 5 else "5+"


def speech_bin(fraction: float) -> str:
    bins = [
        (0.0, 0.10, "0-10%"),
        (0.10, 0.25, "10-25%"),
        (0.25, 0.50, "25-50%"),
        (0.50, 0.75, "50-75%"),
        (0.75, 1.01, "75-100%"),
    ]
    for low, high, label in bins:
        if low <= fraction < high:
            return label
    return "75-100%"


def summarise_weighted(rows: list[dict], key: str) -> list[dict]:
    buckets: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "total_ref_sec": 0.0,
            "fa_sec": 0.0,
            "miss_sec": 0.0,
            "cer_sec": 0.0,
        }
    )
    for row in rows:
        total_ref_sec = float(row["total_ref_sec"])
        if total_ref_sec <= 0.0:
            continue
        bucket = buckets[(row["system"], row[key])]
        bucket["count"] += 1
        bucket["total_ref_sec"] += total_ref_sec
        bucket["fa_sec"] += float(row["fa_sec"])
        bucket["miss_sec"] += float(row["miss_sec"])
        bucket["cer_sec"] += float(row["cer_sec"])

    summary = []
    for (system, bucket_name), values in buckets.items():
        count = int(values["count"])
        total_ref_sec = values["total_ref_sec"]
        summary.append({
            "system": system,
            "bucket": bucket_name,
            "count": count,
            "total_ref_sec": total_ref_sec,
            "weighted_DER": 100.0 * (
                values["fa_sec"] + values["miss_sec"] + values["cer_sec"]
            ) / total_ref_sec,
            "weighted_FA": 100.0 * values["fa_sec"] / total_ref_sec,
            "weighted_MISS": 100.0 * values["miss_sec"] / total_ref_sec,
            "weighted_CER": 100.0 * values["cer_sec"] / total_ref_sec,
        })
    return summary


def summarise_window_fa(rows: list[dict], key: str) -> list[dict]:
    buckets: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "fa_sec": 0.0, "any_fa": 0.0}
    )
    for row in rows:
        bucket = buckets[(row["system"], row[key])]
        bucket["count"] += 1
        fa_sec = float(row["fa_sec"])
        bucket["fa_sec"] += fa_sec
        bucket["any_fa"] += 1.0 if fa_sec > 0.0 else 0.0

    summary = []
    for (system, bucket_name), values in buckets.items():
        count = int(values["count"])
        summary.append({
            "system": system,
            "bucket": bucket_name,
            "count": count,
            "mean_fa_sec_per_window": values["fa_sec"] / count,
            "pct_windows_with_fa": 100.0 * values["any_fa"] / count,
        })
    return summary


def summarise_silent_windows(rows: list[dict]) -> list[dict]:
    silent_rows = [row for row in rows if float(row["speech_sec"]) == 0.0]
    return summarise_window_fa(silent_rows, key="system_silent_bucket")


def plot_summary(
    speaker_summary: list[dict],
    turn_summary: list[dict],
    speech_fa_summary: list[dict],
    out_path: Path,
) -> None:
    apply_ieee_style()
    fig, axes = plt.subplots(1, 3, figsize=(IEEE_DOUBLE_COL, 2.5), sharey=False)

    plots = [
        (speaker_summary, ["1", "2", "3", "4+"], "Distinct Speakers", "weighted_CER"),
        (turn_summary, ["0", "1", "2", "3", "4", "5+"], "Speaker Changes", "weighted_CER"),
        (
            speech_fa_summary,
            ["0-10%", "10-25%", "25-50%", "50-75%", "75-100%"],
            "Speech Occupancy",
            "mean_fa_sec_per_window",
        ),
    ]

    for ax, (summary, order, xlabel, metric_key) in zip(axes, plots):
        for system in ("pretrained", "streaming", "finetuned"):
            style = SYSTEM_STYLES[system]
            data = {
                row["bucket"]: row[metric_key]
                for row in summary
                if row["system"] == system
            }
            y = [data.get(bucket, float("nan")) for bucket in order]
            display_name = {
                "pretrained": "Vanilla",
                "streaming": "Streaming",
                "finetuned": "Fine-tuned",
            }[system]
            ax.plot(
                range(len(order)),
                y,
                label=display_name,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.0,
                markersize=4,
            )
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", linestyle=":", linewidth=0.5)

    axes[0].set_ylabel("CER (%)")
    axes[1].set_ylabel("CER (%)")
    axes[2].set_ylabel("Mean FA (s / 10 s window)")
    axes[0].legend(loc="upper left", framealpha=1.0)
    fig.tight_layout(pad=0.4, w_pad=0.8)
    save_fig(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze error as a function of window difficulty.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifests/windowed_10s_5s.jsonl"))
    parser.add_argument("--rec-ids", type=str, default="dca_d1_1,dca_d2_2,dfw_a1_1,log_id_1")
    parser.add_argument("--collar", type=float, default=0.25)
    parser.add_argument("--out-dir", type=Path, default=Path("results/paper_figures"))
    args = parser.parse_args()

    rec_ids = [item.strip() for item in args.rec_ids.split(",")]
    gt_cache: dict[str, Annotation] = {}
    rows: list[dict] = []

    with open(args.manifest, "r", encoding="utf-8") as handle:
        entries = [json.loads(line) for line in handle if line.strip()]

    for entry in entries:
        uniq_id = entry["uniq_id"]
        rec_id = uniq_id.rsplit("-", 2)[0]
        if rec_id not in rec_ids:
            continue
        offset = float(entry["offset"])
        duration = float(entry["duration"])
        end = offset + duration

        if rec_id not in gt_cache:
            gt_cache[rec_id] = load_annotation(Path(entry["rttm_filepath"]))

        gt_tracks = clipped_tracks(gt_cache[rec_id], offset, end)
        gt_intervals = [(start, stop) for start, stop, _speaker in gt_tracks]
        speech_sec = union_duration(gt_intervals)
        difficulty = {
            "uniq_id": uniq_id,
            "recording": rec_id,
            "offset_sec": offset,
            "duration_sec": duration,
            "n_ref_speakers": len({speaker for _s, _e, speaker in gt_tracks}),
            "n_turn_changes": turn_changes(gt_tracks),
            "speech_sec": speech_sec,
            "speech_fraction": speech_sec / duration if duration else 0.0,
            "dominant_role": dominant_role(gt_tracks, rec_id),
            "speaker_bin": speaker_bin(len({speaker for _s, _e, speaker in gt_tracks})),
            "turn_bin": turn_bin(turn_changes(gt_tracks)),
            "speech_bin": speech_bin(speech_sec / duration if duration else 0.0),
        }

        ref = gt_cache[rec_id]
        uem = Timeline([Segment(offset, end)])
        for system, pred_dir in DEFAULT_SYSTEMS.items():
            hyp_path = pred_dir / f"{uniq_id}.rttm"
            if not hyp_path.exists():
                continue
            hyp = load_annotation(hyp_path)
            hyp_tracks = clipped_tracks(hyp, offset, end)
            hyp_intervals = [(start, stop) for start, stop, _speaker in hyp_tracks]
            hyp_speech_sec = union_duration(hyp_intervals)
            metric = DiarizationErrorRate(collar=args.collar, skip_overlap=False)
            detail = metric(ref, hyp, uem=uem, detailed=True)
            total = float(detail["total"])
            fa_sec = float(detail["false alarm"])
            miss_sec = float(detail["missed detection"])
            cer_sec = float(detail["confusion"])
            # Silent reference windows have no valid speech-normalized denominator.
            # In that case, treat the hypothesis speech duration as absolute FA time.
            if total == 0.0 and speech_sec == 0.0:
                fa_sec = hyp_speech_sec
            fa = 100.0 * fa_sec / total if total else float("nan")
            miss = 100.0 * miss_sec / total if total else float("nan")
            cer = 100.0 * cer_sec / total if total else float("nan")
            rows.append({
                **difficulty,
                "system": system,
                "total_ref_sec": total,
                "hyp_speech_sec": hyp_speech_sec,
                "fa_sec": fa_sec,
                "miss_sec": miss_sec,
                "cer_sec": cer_sec,
                "DER": fa + miss + cer if total else float("nan"),
                "FA": fa,
                "MISS": miss,
                "CER": cer,
                "system_silent_bucket": "silent_windows",
            })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "window_difficulty_metrics.csv"
    json_path = args.out_dir / "window_difficulty_summary.json"

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    speaker_summary = summarise_weighted(rows, "speaker_bin")
    turn_summary = summarise_weighted(rows, "turn_bin")
    speech_fa_summary = summarise_window_fa(rows, "speech_bin")
    silent_window_summary = summarise_silent_windows(rows)
    summary = {
        "speaker_weighted": speaker_summary,
        "turn_weighted": turn_summary,
        "speech_bin_window_fa": speech_fa_summary,
        "silent_window_fa": silent_window_summary,
    }
    json_path.write_text(json.dumps(summary, indent=2))

    plot_summary(
        speaker_summary,
        turn_summary,
        speech_fa_summary,
        args.out_dir / "fig_window_difficulty",
    )


if __name__ == "__main__":
    main()

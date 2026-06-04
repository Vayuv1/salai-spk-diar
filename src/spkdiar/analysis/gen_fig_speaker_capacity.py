"""
gen_fig_speaker_capacity.py

Visualize the mismatch between the four-speaker model capacity and the actual
number of labeled speakers per window in the training and evaluation manifests.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyannote.core import Annotation, Segment

from spkdiar.analysis.ieee_style import IEEE_SINGLE_COL, apply_ieee_style, save_fig


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


def count_window_speakers(annotation: Annotation, start: float, end: float) -> int:
    speakers = set()
    for seg, _, speaker in annotation.itertracks(yield_label=True):
        if min(end, seg.end) > max(start, seg.start):
            speakers.add(speaker)
    return len(speakers)


def histogram_for_manifest(manifest_path: Path, rec_ids: set[str] | None = None) -> Counter:
    hist = Counter()
    cache: dict[Path, Annotation] = {}
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            uniq_id = entry["uniq_id"]
            rec_id = uniq_id.rsplit("-", 2)[0]
            if rec_ids and rec_id not in rec_ids:
                continue
            rttm_path = Path(entry["rttm_filepath"])
            if rttm_path not in cache:
                cache[rttm_path] = load_annotation(rttm_path)
            count = count_window_speakers(
                cache[rttm_path],
                float(entry["offset"]),
                float(entry["offset"]) + float(entry["duration"]),
            )
            hist[min(count, 8)] += 1
    return hist


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot speaker-count histograms per manifest.")
    parser.add_argument("--train-manifest", type=Path, default=Path("data/processed/manifests/finetune_train.jsonl"))
    parser.add_argument("--eval-manifest", type=Path, default=Path("data/processed/manifests/finetune_eval.jsonl"))
    parser.add_argument("--test-manifest", type=Path, default=Path("data/processed/manifests/windowed_10s_5s.jsonl"))
    parser.add_argument("--test-rec-ids", type=str, default="dca_d1_1,dca_d2_2,dfw_a1_1,log_id_1")
    parser.add_argument("--out-dir", type=Path, default=Path("results/paper_figures"))
    args = parser.parse_args()

    test_rec_ids = {item.strip() for item in args.test_rec_ids.split(",")}
    # Legend labels align with the partition names used in Table I.
    histograms = {
        "Fine-tuning train": histogram_for_manifest(args.train_manifest),
        "Fine-tuning val.": histogram_for_manifest(args.eval_manifest),
        "Evaluation": histogram_for_manifest(args.test_manifest, rec_ids=test_rec_ids),
    }

    summary = {
        split: {
            str(bin_idx if bin_idx < 8 else "8+"): count
            for bin_idx, count in sorted(hist.items())
        }
        for split, hist in histograms.items()
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "speaker_capacity_histogram.json").write_text(json.dumps(summary, indent=2))

    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 2.8))
    labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8+"]
    x = range(len(labels))
    width = 0.24
    styles = [
        ("Fine-tuning train", -width, "#1f77b4", ""),
        ("Fine-tuning val.", 0.0, "#d62728", "//"),
        ("Evaluation", width, "#2ca02c", ".."),
    ]
    max_percent = 0.0
    for name, shift, color, hatch in styles:
        counts = [histograms[name].get(i if i < 8 else 8, 0) for i in range(9)]
        total = sum(counts)
        values = [(count / total) * 100.0 if total else 0.0 for count in counts]
        max_percent = max(max_percent, max(values, default=0.0))
        ax.bar(
            [idx + shift for idx in x],
            values,
            width=width,
            color=color,
            edgecolor="black",
            hatch=hatch,
            linewidth=0.6,
            label=name,
        )

    ax.axvline(4.5, color="black", linestyle=":", linewidth=0.8)
    ax.text(4.55, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] else 1.0, "Above 4-spk capacity",
            fontsize=7.5, rotation=90, va="top")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Distinct Labeled Speakers per Window")
    ax.set_ylabel("Windows (%)")
    upper = max(5.0, ((max_percent + 4.9) // 5) * 5)
    ax.set_ylim(0, upper)
    ax.set_yticks(list(range(0, int(upper) + 1, 5)))
    ax.legend(loc="upper right", framealpha=1.0)
    ax.grid(axis="y", linestyle=":", linewidth=0.5)
    fig.tight_layout(pad=0.4)
    save_fig(fig, args.out_dir / "fig_speaker_capacity")


if __name__ == "__main__":
    main()

"""
make_finetune_manifest.py

Create 90-second windowed manifests for Sortformer fine-tuning on ATC0R data.

Split:
  Train (12 recordings): all RTTM recordings except the four held-out ones
  Eval  ( 4 recordings): dca_d1_1, dca_d2_2, log_id_1, dfw_a1_1

Windows: 90s duration, 45s shift.  Only windows with ≥80s of actual audio are
included (matching the model's min_duration=80 training filter).

Usage:
    uv run python -m spkdiar.data.make_finetune_manifest \
        --audio-dir data/atc0r/audio \
        --rttm-dir data/processed/rttm \
        --out-dir data/processed/manifests
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

WINDOW_DUR  = 90.0   # seconds
WINDOW_SHIFT = 45.0  # seconds
MIN_DUR      = 80.0  # minimum usable window length

EVAL_RECS = {"dca_d1_1", "dca_d2_2", "log_id_1", "dfw_a1_1"}


def get_audio_duration(audio_path: Path) -> float:
    """Use ffprobe to get the duration of an audio file in seconds."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def make_windows(
    rec_id: str,
    audio_path: Path,
    rttm_path: Path,
    duration: float,
    window_dur: float,
    shift: float,
    min_dur: float,
) -> list[dict]:
    entries = []
    start = 0.0
    while start < duration:
        actual_dur = min(window_dur, duration - start)
        if actual_dur < min_dur:
            break
        start_ms  = int(round(start * 1000))
        dur_ms    = int(round(window_dur * 1000))
        uniq_id   = f"{rec_id}-{start_ms}-{dur_ms}"
        entries.append({
            "audio_filepath": str(audio_path.resolve()),
            "offset":         start,
            "duration":       window_dur,
            "label":          "infer",
            "rttm_filepath":  str(rttm_path.resolve()),
            "uniq_id":        uniq_id,
            "num_spks":       4,
        })
        start += shift
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, default=Path("data/atc0r/audio"))
    parser.add_argument("--rttm-dir",  type=Path, default=Path("data/processed/rttm"))
    parser.add_argument("--out-dir",   type=Path, default=Path("data/processed/manifests"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Enumerate recordings that have both an RTTM and an audio file
    rttm_files = sorted(args.rttm_dir.glob("*.rttm"))
    train_entries: list[dict] = []
    eval_entries:  list[dict] = []

    for rttm_path in rttm_files:
        rec_id     = rttm_path.stem
        audio_path = args.audio_dir / f"{rec_id}.mp3"
        if not audio_path.exists():
            print(f"  SKIP {rec_id}: audio not found")
            continue

        duration = get_audio_duration(audio_path)
        windows  = make_windows(
            rec_id, audio_path, rttm_path,
            duration, WINDOW_DUR, WINDOW_SHIFT, MIN_DUR,
        )
        if not windows:
            print(f"  SKIP {rec_id}: no usable windows (dur={duration:.1f}s)")
            continue

        bucket = eval_entries if rec_id in EVAL_RECS else train_entries
        bucket.extend(windows)
        split = "eval" if rec_id in EVAL_RECS else "train"
        print(f"  {split:5s}  {rec_id:15s}  dur={duration:7.1f}s  windows={len(windows)}")

    train_path = args.out_dir / "finetune_train.jsonl"
    eval_path  = args.out_dir / "finetune_eval.jsonl"

    with open(train_path, "w") as f:
        for entry in train_entries:
            f.write(json.dumps(entry) + "\n")

    with open(eval_path, "w") as f:
        for entry in eval_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\nTrain manifest: {train_path}  ({len(train_entries)} windows)")
    print(f"Eval  manifest: {eval_path}  ({len(eval_entries)} windows)")


if __name__ == "__main__":
    main()

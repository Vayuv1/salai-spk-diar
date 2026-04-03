"""
make_manifest.py

Create NeMo-compatible JSONL manifests from ATC0R STM files and audio.

Produces two manifests:
  1. Full-recording manifest (one entry per recording) — for full-session evaluation
  2. Windowed manifest (overlapping windows) — for Sortformer inference

Usage:
    uv run python -m spkdiar.data.make_manifest \
        --stm-dir data/atc0r/stm \
        --audio-dir data/atc0r/audio \
        --rttm-dir data/processed/rttm \
        --out-dir data/processed/manifests \
        --window-sec 10.0 \
        --shift-sec 5.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spkdiar.data.stm_parser import parse_stm_dir


def find_audio_file(audio_dir: Path, rec_id: str) -> Path | None:
    """Find the audio file for a recording ID, trying common extensions."""
    for ext in (".mp3", ".wav", ".flac", ".MP3", ".WAV"):
        candidate = audio_dir / f"{rec_id}{ext}"
        if candidate.is_file():
            return candidate
    return None


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds."""
    try:
        import soundfile as sf
        info = sf.info(str(audio_path))
        if info.frames and info.samplerate:
            return info.frames / info.samplerate
    except Exception:
        pass

    try:
        import librosa
        return librosa.get_duration(path=str(audio_path))
    except Exception as e:
        raise RuntimeError(f"Cannot determine duration of {audio_path}: {e}") from e


def make_full_manifest(
    stm_dir: Path,
    audio_dir: Path,
    rttm_dir: Path,
    out_path: Path,
) -> int:
    """Create a full-recording manifest (one entry per recording).

    Returns the number of entries written.
    """
    recordings = parse_stm_dir(stm_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped = 0

    with out_path.open("w", encoding="utf-8") as f:
        for rec in recordings:
            audio_path = find_audio_file(audio_dir, rec.rec_id)
            if audio_path is None:
                print(f"  WARNING: no audio for {rec.rec_id}, skipping")
                n_skipped += 1
                continue

            rttm_path = rttm_dir / f"{rec.rec_id}.rttm"
            if not rttm_path.is_file():
                print(f"  WARNING: no RTTM for {rec.rec_id}, skipping")
                n_skipped += 1
                continue

            # Use STM-derived duration (latest end time) as recording duration
            # This may be less than the actual audio duration (silence at end)
            duration = rec.duration

            entry = {
                "audio_filepath": str(audio_path.resolve()),
                "rttm_filepath": str(rttm_path.resolve()),
                "offset": 0.0,
                "duration": round(duration, 3),
                "uniq_id": rec.rec_id,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Full manifest: {n_written} recordings written, {n_skipped} skipped → {out_path}")
    return n_written


def make_windowed_manifest(
    stm_dir: Path,
    audio_dir: Path,
    rttm_dir: Path,
    out_path: Path,
    window_sec: float = 10.0,
    shift_sec: float = 5.0,
    min_tail_sec: float = 2.0,
    rec_ids: list[str] | None = None,
    max_duration_sec: float | None = None,
) -> int:
    """Create a windowed manifest with overlapping windows.

    Args:
        stm_dir: Directory with STM files.
        audio_dir: Directory with audio files.
        rttm_dir: Directory with RTTM files.
        out_path: Output JSONL path.
        window_sec: Window length in seconds.
        shift_sec: Window shift (hop) in seconds.
        min_tail_sec: Discard tail windows shorter than this.
        rec_ids: If provided, only process these recording IDs.
        max_duration_sec: If provided, only window the first N seconds of each recording.

    Returns the number of entries written.
    """
    recordings = parse_stm_dir(stm_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0

    with out_path.open("w", encoding="utf-8") as f:
        for rec in recordings:
            if rec_ids and rec.rec_id not in rec_ids:
                continue

            audio_path = find_audio_file(audio_dir, rec.rec_id)
            if audio_path is None:
                print(f"  WARNING: no audio for {rec.rec_id}, skipping")
                continue

            rttm_path = rttm_dir / f"{rec.rec_id}.rttm"
            if not rttm_path.is_file():
                print(f"  WARNING: no RTTM for {rec.rec_id}, skipping")
                continue

            total_dur = rec.duration
            if max_duration_sec is not None:
                total_dur = min(total_dur, max_duration_sec)

            n_rec = 0

            if total_dur < window_sec:
                # Recording shorter than one window
                if total_dur >= min_tail_sec:
                    uid = f"{rec.rec_id}-0-{int(round(total_dur * 1000))}"
                    entry = {
                        "audio_filepath": str(audio_path.resolve()),
                        "rttm_filepath": str(rttm_path.resolve()),
                        "offset": 0.0,
                        "duration": round(total_dur, 3),
                        "uniq_id": uid,
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    n_rec += 1
            else:
                # Full windows
                n_full = int((total_dur - window_sec) / shift_sec) + 1
                for i in range(n_full):
                    start = i * shift_sec
                    uid = (
                        f"{rec.rec_id}-"
                        f"{int(round(start * 1000))}-"
                        f"{int(round(window_sec * 1000))}"
                    )
                    entry = {
                        "audio_filepath": str(audio_path.resolve()),
                        "rttm_filepath": str(rttm_path.resolve()),
                        "offset": round(start, 3),
                        "duration": round(window_sec, 3),
                        "uniq_id": uid,
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    n_rec += 1

                # Tail window
                tail_start = n_full * shift_sec
                tail_len = total_dur - tail_start
                if tail_len >= min_tail_sec:
                    tail_dur = min(window_sec, tail_len)
                    uid = (
                        f"{rec.rec_id}-"
                        f"{int(round(tail_start * 1000))}-"
                        f"{int(round(tail_dur * 1000))}"
                    )
                    entry = {
                        "audio_filepath": str(audio_path.resolve()),
                        "rttm_filepath": str(rttm_path.resolve()),
                        "offset": round(tail_start, 3),
                        "duration": round(tail_dur, 3),
                        "uniq_id": uid,
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    n_rec += 1

            n_total += n_rec
            print(f"  {rec.rec_id}: {n_rec} windows")

    print(f"Windowed manifest: {n_total} entries → {out_path}")
    return n_total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create NeMo manifests from ATC0R STM files."
    )
    parser.add_argument("--stm-dir", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--rttm-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--shift-sec", type=float, default=5.0)
    parser.add_argument("--min-tail-sec", type=float, default=2.0)
    parser.add_argument(
        "--rec-ids", type=str, default=None,
        help="Comma-separated recording IDs to include (default: all).",
    )
    parser.add_argument(
        "--max-duration", type=float, default=None,
        help="Only window the first N seconds of each recording.",
    )
    args = parser.parse_args()

    rec_ids = None
    if args.rec_ids:
        rec_ids = [r.strip() for r in args.rec_ids.split(",")]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Full manifest
    full_path = args.out_dir / "full_manifest.jsonl"
    make_full_manifest(args.stm_dir, args.audio_dir, args.rttm_dir, full_path)

    # Windowed manifest
    tag = f"{int(args.window_sec)}s_{int(args.shift_sec)}s"
    windowed_path = args.out_dir / f"windowed_{tag}.jsonl"
    make_windowed_manifest(
        stm_dir=args.stm_dir,
        audio_dir=args.audio_dir,
        rttm_dir=args.rttm_dir,
        out_path=windowed_path,
        window_sec=args.window_sec,
        shift_sec=args.shift_sec,
        min_tail_sec=args.min_tail_sec,
        rec_ids=rec_ids,
        max_duration_sec=args.max_duration,
    )


if __name__ == "__main__":
    main()

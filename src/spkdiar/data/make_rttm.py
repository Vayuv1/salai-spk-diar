"""
make_rttm.py

Convert ATC0R STM files to RTTM format for diarization evaluation.

RTTM format (NIST standard, 10 fields):
    SPEAKER <rec_id> <channel> <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>

Usage:
    uv run python -m spkdiar.data.make_rttm \
        --stm-dir data/atc0r/stm \
        --out-dir data/processed/rttm
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spkdiar.data.stm_parser import Recording, parse_stm_dir, parse_stm_file


def recording_to_rttm(rec: Recording) -> list[str]:
    """Convert a Recording to RTTM lines.

    Speaker IDs are prefixed with the recording ID to ensure uniqueness
    across recordings (matching the convention from the old atc0_make_rttm.py).
    """
    lines = []
    for cue in rec.cues:
        duration = cue.end - cue.start
        if duration <= 0:
            continue

        # Prefix speaker with recording ID for global uniqueness
        spk_id = f"{rec.rec_id}_{cue.speaker}"

        # 10-field RTTM: TYPE FILE CHAN TBEG TDUR ORTH STYPE SPKR CONF SLAT
        line = (
            f"SPEAKER {rec.rec_id} 1 "
            f"{cue.start:.3f} {duration:.3f} "
            f"<NA> <NA> {spk_id} <NA> <NA>"
        )
        lines.append(line)

    return lines


def write_rttm(rec: Recording, out_dir: Path) -> Path:
    """Write RTTM file for a single recording. Returns the output path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{rec.rec_id}.rttm"

    lines = recording_to_rttm(rec)
    with out_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ATC0R STM files to RTTM for diarization evaluation."
    )
    parser.add_argument(
        "--stm-dir", type=Path, required=True,
        help="Directory containing .stm files.",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="Directory to write .rttm files.",
    )
    parser.add_argument(
        "--stm-file", type=Path, default=None,
        help="Process a single STM file instead of a directory.",
    )
    args = parser.parse_args()

    if args.stm_file:
        rec = parse_stm_file(args.stm_file)
        out_path = write_rttm(rec, args.out_dir)
        print(f"Wrote {len(rec.cues)} segments to {out_path}")
        print(f"  Recording: {rec.rec_id}")
        print(f"  Duration: {rec.duration:.1f}s")
        print(f"  Speakers: {rec.n_speakers} ({', '.join(rec.speakers[:5])}...)")
    else:
        recordings = parse_stm_dir(args.stm_dir)
        print(f"Found {len(recordings)} STM files in {args.stm_dir}")
        for rec in recordings:
            out_path = write_rttm(rec, args.out_dir)
            print(
                f"  {rec.rec_id}: {len(rec.cues)} segments, "
                f"{rec.n_speakers} speakers, {rec.duration:.1f}s → {out_path.name}"
            )

    print("Done.")


if __name__ == "__main__":
    main()

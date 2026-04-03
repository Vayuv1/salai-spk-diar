"""
prep_all.py

One-command data preparation: STM → RTTM → Manifests.

Usage:
    uv run python -m spkdiar.data.prep_all \
        --stm-dir data/atc0r/stm \
        --audio-dir data/atc0r/audio
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spkdiar.data.make_rttm import write_rttm
from spkdiar.data.make_manifest import (make_full_manifest,
                                         make_windowed_manifest)
from spkdiar.data.stm_parser import parse_stm_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Full data prep pipeline: STM → RTTM → manifests.")
    parser.add_argument("--stm-dir", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument(
        "--out-root", type=Path, default=Path("data/processed"),
        help="Root output directory (default: data/processed).",
    )
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--shift-sec", type=float, default=5.0)
    parser.add_argument("--rec-ids", type=str, default=None)
    parser.add_argument("--max-duration", type=float, default=None)
    args = parser.parse_args()

    rttm_dir = args.out_root / "rttm"
    manifest_dir = args.out_root / "manifests"

    rec_ids = None
    if args.rec_ids:
        rec_ids = [r.strip() for r in args.rec_ids.split(",")]

    # Step 1: Parse STM files
    print("=" * 60)
    print("STEP 1: Parsing STM files")
    print("=" * 60)
    recordings = parse_stm_dir(args.stm_dir)
    print(f"Found {len(recordings)} recordings\n")

    for rec in recordings:
        print(
            f"  {rec.rec_id}: {len(rec.cues)} cues, "
            f"{rec.n_speakers} speakers, {rec.duration:.1f}s"
        )

    # Step 2: Generate RTTM files
    print(f"\n{'=' * 60}")
    print("STEP 2: Generating RTTM files")
    print("=" * 60)
    for rec in recordings:
        out_path = write_rttm(rec, rttm_dir)
        print(f"  {rec.rec_id} → {out_path.name}")

    # Step 3: Generate manifests
    print(f"\n{'=' * 60}")
    print("STEP 3: Generating manifests")
    print("=" * 60)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    full_path = manifest_dir / "full_manifest.jsonl"
    make_full_manifest(args.stm_dir, args.audio_dir, rttm_dir, full_path)

    tag = f"{int(args.window_sec)}s_{int(args.shift_sec)}s"
    windowed_path = manifest_dir / f"windowed_{tag}.jsonl"
    make_windowed_manifest(
        stm_dir=args.stm_dir,
        audio_dir=args.audio_dir,
        rttm_dir=rttm_dir,
        out_path=windowed_path,
        window_sec=args.window_sec,
        shift_sec=args.shift_sec,
        rec_ids=rec_ids,
        max_duration_sec=args.max_duration,
    )

    # Summary
    print(f"\n{'=' * 60}")
    print("DONE — Data preparation complete")
    print("=" * 60)
    print(f"  RTTM files:        {rttm_dir}/")
    print(f"  Full manifest:     {full_path}")
    print(f"  Windowed manifest: {windowed_path}")
    print()
    print("Next: run an experiment, e.g.:")
    print(f"  uv run python -m spkdiar.inference.run_sortformer \\")
    print(f"      --manifest {windowed_path} \\")
    print(f"      --model-path <path-to-.nemo>")


if __name__ == "__main__":
    main()

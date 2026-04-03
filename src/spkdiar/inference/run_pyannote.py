"""
run_pyannote.py

Run pyannote.audio 3.x speaker diarization pipeline on ATC0R recordings.

Pyannote uses a clustering-based approach (segmentation → embeddings → clustering),
fundamentally different from Sortformer's end-to-end approach. If pyannote does not
exhibit the lock-up pattern, it confirms the issue is specific to end-to-end architectures.

Requires a HuggingFace token with access to pyannote gated models:
    export HF_TOKEN=hf_xxxxx
    # Or: huggingface-cli login

Usage:
    uv run python -m spkdiar.inference.run_pyannote \
        --manifest data/processed/manifests/full_manifest.jsonl \
        --out-dir results/pyannote \
        --hf-token hf_xxxxx

    # Process specific recordings, first 350s each:
    uv run python -m spkdiar.inference.run_pyannote \
        --manifest data/processed/manifests/full_manifest.jsonl \
        --out-dir results/pyannote \
        --rec-ids dca_d1_1,log_id_1 \
        --max-duration 350
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def run_pyannote_diarization(
    manifest_path: Path,
    out_dir: Path,
    hf_token: str | None = None,
    rec_ids: list[str] | None = None,
    max_duration: float | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> None:
    """Run pyannote diarization on recordings from a NeMo manifest."""
    from pyannote.audio import Pipeline
    from pyannote.metrics.diarization import DiarizationErrorRate
    from pyannote.core import Annotation, Segment

    out_dir.mkdir(parents=True, exist_ok=True)
    rttm_dir = out_dir / "pred_rttm"
    rttm_dir.mkdir(parents=True, exist_ok=True)

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "HuggingFace token required for pyannote. "
            "Set HF_TOKEN env var or pass --hf-token."
        )

    # Load the pretrained pipeline
    log.info("Loading pyannote pipeline (pyannote/speaker-diarization-3.1)...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )

    import torch
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        log.info("Pipeline moved to CUDA")

    # Read manifest
    entries = []
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            uid = entry.get("uniq_id", "")

            if rec_ids and uid not in rec_ids:
                continue

            entries.append(entry)

    log.info(f"Processing {len(entries)} recordings")

    # DER evaluation
    der_metric = DiarizationErrorRate(collar=0.25)

    for entry in entries:
        uid = entry["uniq_id"]
        audio_path = entry["audio_filepath"]
        rttm_path = entry.get("rttm_filepath")
        duration = float(entry.get("duration", 0))

        if max_duration and duration > max_duration:
            duration = max_duration

        log.info(f"Processing {uid} ({duration:.1f}s)...")

        # Run pyannote
        params = {}
        if min_speakers:
            params["min_speakers"] = min_speakers
        if max_speakers:
            params["max_speakers"] = max_speakers

        import torchaudio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Trim to max_duration if needed
        if max_duration:
            max_samples = int(max_duration * sample_rate)
            waveform = waveform[:, :max_samples]

        diarization = pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            **params,
        )

        # Write predicted RTTM
        pred_rttm_path = rttm_dir / f"{uid}.rttm"
        with open(pred_rttm_path, "w") as f:
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                f.write(
                    f"SPEAKER {uid} 1 {turn.start:.3f} {turn.duration:.3f} "
                    f"<NA> <NA> {speaker} <NA> <NA>\n"
                )

        n_speakers = len(diarization.labels())
        log.info(f"  {uid}: {n_speakers} speakers detected")

        # Compute DER if reference available
        if rttm_path and Path(rttm_path).is_file():
            reference = Annotation(uri=uid)
            with open(rttm_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 8 or parts[0] != "SPEAKER":
                        continue
                    ref_uid = parts[1]
                    if ref_uid != uid:
                        continue
                    start = float(parts[3])
                    dur = float(parts[4])
                    spk = parts[7]
                    # Trim reference to max_duration too
                    if max_duration and start >= max_duration:
                        continue
                    end = min(start + dur, max_duration) if max_duration else start + dur
                    reference[Segment(start, end)] = spk

            hypothesis = Annotation(uri=uid)
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                hypothesis[Segment(turn.start, turn.end)] = speaker

            file_der = der_metric(reference, hypothesis, detailed=True)
            log.info(
                f"  DER: {file_der['diarization error rate']:.4f} "
                f"(FA={file_der['false alarm']:.4f}, "
                f"MISS={file_der['missed detection']:.4f}, "
                f"CONF={file_der['confusion']:.4f})"
            )

    # Overall DER
    overall = abs(der_metric)
    log.info(f"\nOverall DER: {overall:.4f}")

    # Save summary
    summary_path = out_dir / "summary.json"
    summary = {
        "system": "pyannote-3.1",
        "overall_der": float(overall),
        "n_recordings": len(entries),
        "collar": 0.25,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary saved to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pyannote.audio speaker diarization pipeline."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/pyannote"))
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--rec-ids", type=str, default=None)
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    args = parser.parse_args()

    rec_ids = None
    if args.rec_ids:
        rec_ids = [r.strip() for r in args.rec_ids.split(",")]

    run_pyannote_diarization(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        hf_token=args.hf_token,
        rec_ids=rec_ids,
        max_duration=args.max_duration,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )


if __name__ == "__main__":
    main()

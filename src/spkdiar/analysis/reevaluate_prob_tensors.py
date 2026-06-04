"""
reevaluate_prob_tensors.py

Rebuild RTTMs and DER metrics from saved Sortformer probability tensors using
custom postprocessing thresholds.

This is intended for calibration studies where rerunning the neural model would
be wasteful. The script mirrors the postprocessing/evaluation path used by the
offline and streaming inference entrypoints.
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.metrics.der import score_labels
from nemo.collections.asr.parts.utils.speaker_utils import (
    audio_rttm_map,
    timestamps_to_pyannote_object,
)
from nemo.collections.asr.parts.utils.vad_utils import (
    load_postprocessing_from_yaml,
    predlist_to_timestamps,
)

from spkdiar.utils.eval_export import build_metric_summary, save_metric_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def filter_manifest(
    manifest_path: Path,
    rec_ids: list[str] | None = None,
    exclude_rec_ids: list[str] | None = None,
    max_offset: float | None = None,
) -> Path:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    kept = 0
    total = 0

    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total += 1
            entry = json.loads(line)
            uniq_id = entry["uniq_id"]
            parts = uniq_id.rsplit("-", 2)
            rec_id = parts[0] if len(parts) >= 3 else uniq_id

            if rec_ids and rec_id not in rec_ids:
                continue
            if exclude_rec_ids and rec_id in exclude_rec_ids:
                continue
            if max_offset is not None and float(entry.get("offset", 0.0)) > max_offset:
                continue

            tmp.write(line + "\n")
            kept += 1

    tmp.close()
    log.info("Filtered manifest: %d/%d entries kept -> %s", kept, total, tmp.name)
    return Path(tmp.name)


def reevaluate_prob_tensors(
    manifest_path: Path,
    prob_dir: Path,
    out_dir: Path,
    collar: float = 0.25,
    onset: float | None = None,
    offset: float | None = None,
    bypass_postprocessing: bool = True,
    stem: str = "eval_metrics",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rttm_dir = out_dir / "pred_rttm"
    rttm_dir.mkdir(parents=True, exist_ok=True)

    infer_audio_rttm_dict = audio_rttm_map(str(manifest_path))
    preds_list: list[torch.Tensor] = []
    missing: list[str] = []

    for uniq_id in infer_audio_rttm_dict:
        npy_path = prob_dir / f"{uniq_id}.npy"
        if not npy_path.exists():
            missing.append(uniq_id)
            continue
        arr = np.load(npy_path)
        preds_list.append(torch.from_numpy(arr).unsqueeze(0))

    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} probability tensors in {prob_dir}; "
            f"examples: {missing[:5]}"
        )

    postprocessing_cfg = load_postprocessing_from_yaml(None)
    cfg_vad_params = OmegaConf.structured(postprocessing_cfg)
    if onset is not None:
        cfg_vad_params.onset = onset
    if offset is not None:
        cfg_vad_params.offset = offset
    effective_bypass_postprocessing = bypass_postprocessing
    if (onset is not None or offset is not None) and bypass_postprocessing:
        log.info("Threshold overrides requested; disabling bypass_postprocessing so overrides take effect.")
        effective_bypass_postprocessing = False

    total_speaker_timestamps = predlist_to_timestamps(
        batch_preds_list=preds_list,
        audio_rttm_map_dict=infer_audio_rttm_dict,
        cfg_vad_params=cfg_vad_params,
        unit_10ms_frame_count=8,
        bypass_postprocessing=effective_bypass_postprocessing,
    )

    all_hyps, all_refs, all_uems = [], [], []
    for sample_idx, (uniq_id, audio_rttm_values) in enumerate(infer_audio_rttm_dict.items()):
        speaker_timestamps = total_speaker_timestamps[sample_idx]
        all_hyps, all_refs, all_uems = timestamps_to_pyannote_object(
            speaker_timestamps,
            uniq_id,
            audio_rttm_values,
            all_hyps,
            all_refs,
            all_uems,
            str(rttm_dir),
        )

    metric_result = score_labels(
        AUDIO_RTTM_MAP=infer_audio_rttm_dict,
        all_reference=all_refs,
        all_hypothesis=all_hyps,
        all_uem=all_uems,
        collar=collar,
        ignore_overlap=False,
    )

    if not isinstance(metric_result, tuple):
        raise RuntimeError("Expected score_labels to return (metric, mapping, itemized_errors).")

    metric_obj, _mapping, itemized_errors = metric_result
    der, cer, fa, miss = itemized_errors
    summary = build_metric_summary(metric_obj, collar=collar, ignore_overlap=False)
    summary["postprocessing"] = {
        "onset": float(cfg_vad_params.onset),
        "offset": float(cfg_vad_params.offset),
        "bypass_postprocessing": bool(effective_bypass_postprocessing),
    }
    save_metric_summary(summary, out_dir, stem=stem)

    return {
        "der": float(der),
        "cer": float(cer),
        "fa": float(fa),
        "miss": float(miss),
        "n_predictions": len(preds_list),
        "summary_path": str(out_dir / f"{stem}.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate saved Sortformer probability tensors with custom thresholds."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prob-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--collar", type=float, default=0.25)
    parser.add_argument("--onset", type=float, default=None)
    parser.add_argument("--offset", type=float, default=None)
    parser.add_argument("--rec-ids", type=str, default=None)
    parser.add_argument("--exclude-rec-ids", type=str, default=None)
    parser.add_argument("--max-offset", type=float, default=None)
    parser.add_argument("--stem", type=str, default="eval_metrics")
    parser.add_argument("--disable-bypass-postprocessing", action="store_true")
    args = parser.parse_args()

    manifest = args.manifest
    tmp_manifest = None
    rec_ids = [item.strip() for item in args.rec_ids.split(",")] if args.rec_ids else None
    exclude_rec_ids = (
        [item.strip() for item in args.exclude_rec_ids.split(",")]
        if args.exclude_rec_ids else None
    )

    if rec_ids or exclude_rec_ids or args.max_offset is not None:
        tmp_manifest = filter_manifest(
            manifest,
            rec_ids=rec_ids,
            exclude_rec_ids=exclude_rec_ids,
            max_offset=args.max_offset,
        )
        manifest = tmp_manifest

    try:
        result = reevaluate_prob_tensors(
            manifest_path=manifest,
            prob_dir=args.prob_dir,
            out_dir=args.out_dir,
            collar=args.collar,
            onset=args.onset,
            offset=args.offset,
            bypass_postprocessing=not args.disable_bypass_postprocessing,
            stem=args.stem,
        )
        log.info("Results: %s", result)
    finally:
        if tmp_manifest and tmp_manifest.exists():
            tmp_manifest.unlink()


if __name__ == "__main__":
    main()

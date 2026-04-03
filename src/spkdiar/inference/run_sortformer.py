"""
run_sortformer.py

Run offline Sortformer diarization on a windowed manifest.

Outputs:
  - Per-window probability tensors (.npy) for visualization
  - Predicted RTTM files for DER evaluation
  - DER scores printed to console

Usage:
    uv run python -m spkdiar.inference.run_sortformer \
        --manifest data/processed/manifests/windowed_10s_5s.jsonl \
        --model-path models/diar_sortformer_4spk-v1.nemo \
        --out-dir results/sortformer_offline

    # Quick test on first 350s of specific recordings:
    uv run python -m spkdiar.inference.run_sortformer \
        --manifest data/processed/manifests/windowed_10s_5s.jsonl \
        --model-path models/diar_sortformer_4spk-v1.nemo \
        --out-dir results/sortformer_offline \
        --rec-ids dca_d1_1,dca_d1_3,log_id_1 \
        --max-offset 350
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import lightning.pytorch as pl
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.metrics.der import score_labels
from nemo.collections.asr.parts.utils.speaker_utils import (
    audio_rttm_map,
    timestamps_to_pyannote_object,
)
from nemo.collections.asr.parts.utils.vad_utils import (
    load_postprocessing_from_yaml,
    predlist_to_timestamps,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

seed_everything(42)
torch.backends.cudnn.deterministic = True


def filter_manifest(
    manifest_path: Path,
    rec_ids: list[str] | None = None,
    max_offset: float | None = None,
) -> Path:
    """Filter a manifest by recording IDs and max offset.

    Returns path to a temporary filtered manifest.
    """
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    n_kept = 0
    n_total = 0

    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            entry = json.loads(line)

            # Filter by recording ID (extract from uniq_id)
            if rec_ids:
                uid = entry.get("uniq_id", "")
                parts = uid.rsplit("-", 2)
                base_id = parts[0] if len(parts) >= 3 else uid
                if base_id not in rec_ids:
                    continue

            # Filter by max offset
            if max_offset is not None:
                offset = float(entry.get("offset", 0.0))
                if offset > max_offset:
                    continue

            tmp.write(line + "\n")
            n_kept += 1

    tmp.close()
    log.info(f"Filtered manifest: {n_kept}/{n_total} entries kept → {tmp.name}")
    return Path(tmp.name)


def run_inference(
    model_path: str,
    manifest_path: Path,
    out_dir: Path,
    precision: str = "bf16-mixed",
    batch_size: int = 1,
    num_workers: int = 0,
    collar: float = 0.25,
    bypass_postprocessing: bool = True,
    save_probs: bool = True,
) -> dict:
    """Run Sortformer offline inference.

    Returns a dict with DER components.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    prob_dir = out_dir / "prob_tensors"
    rttm_dir = out_dir / "pred_rttm"

    if save_probs:
        prob_dir.mkdir(parents=True, exist_ok=True)
    rttm_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    log.info(f"Loading model from {model_path}")
    map_location = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = [0] if torch.cuda.is_available() else 1

    if model_path.endswith(".ckpt"):
        diar_model = SortformerEncLabelModel.load_from_checkpoint(
            checkpoint_path=model_path, map_location=map_location, strict=False
        )
    elif model_path.endswith(".nemo"):
        diar_model = SortformerEncLabelModel.restore_from(
            restore_path=model_path, map_location=map_location
        )
    else:
        raise ValueError(f"Model path must end with .ckpt or .nemo, got: {model_path}")

    # --- Configure test dataset ---
    diar_model._cfg.test_ds.manifest_filepath = str(manifest_path)
    diar_model._cfg.test_ds.batch_size = batch_size
    diar_model._cfg.test_ds.pin_memory = False
    diar_model._cfg.test_ds.num_workers = num_workers

    OmegaConf.set_struct(diar_model._cfg, False)
    diar_model._cfg.test_ds.use_lhotse = True
    diar_model._cfg.test_ds.use_bucketing = False
    diar_model._cfg.test_ds.drop_last = False
    diar_model._cfg.test_ds.batch_duration = 100000
    OmegaConf.set_struct(diar_model._cfg, True)

    # --- Setup trainer ---
    trainer = pl.Trainer(devices=devices, accelerator=accelerator, precision=precision)
    diar_model.set_trainer(trainer)

    if torch.cuda.is_bf16_supported() and precision.startswith("bf16"):
        diar_model = diar_model.to(dtype=torch.bfloat16).eval()
    else:
        diar_model = diar_model.eval()

    diar_model.setup_test_data(test_data_config=diar_model._cfg.test_ds)

    # --- Run inference ---
    log.info("Running inference...")
    infer_audio_rttm_dict = audio_rttm_map(str(manifest_path))

    with torch.inference_mode(), torch.autocast(
        device_type=diar_model.device.type, dtype=diar_model.dtype
    ):
        diar_model.test_batch()

    preds_list = diar_model.preds_total_list
    log.info(f"Inference complete: {len(preds_list)} predictions")

    # --- Save probability tensors ---
    if save_probs and len(preds_list) == len(infer_audio_rttm_dict):
        log.info(f"Saving probability tensors to {prob_dir}")
        for prob_tensor, uniq_id in zip(preds_list, infer_audio_rttm_dict.keys()):
            arr = prob_tensor.squeeze(0).to(torch.float32).cpu().numpy()
            np.save(prob_dir / f"{uniq_id}.npy", arr)

    # --- Convert predictions to RTTM and evaluate ---
    log.info("Converting predictions to RTTM segments...")
    postprocessing_cfg = load_postprocessing_from_yaml(None)
    all_hyps, all_refs, all_uems = [], [], []
    cfg_vad_params = OmegaConf.structured(postprocessing_cfg)

    total_speaker_timestamps = predlist_to_timestamps(
        batch_preds_list=preds_list,
        audio_rttm_map_dict=infer_audio_rttm_dict,
        cfg_vad_params=cfg_vad_params,
        unit_10ms_frame_count=8,
        bypass_postprocessing=bypass_postprocessing,
    )

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

    # --- Compute DER ---
    log.info("Computing DER...")
    metric = score_labels(
        AUDIO_RTTM_MAP=infer_audio_rttm_dict,
        all_reference=all_refs,
        all_hypothesis=all_hyps,
        all_uem=all_uems,
        collar=collar,
        ignore_overlap=False,
    )

    if isinstance(metric, tuple):
        der = abs(metric[0])
    else:
        der = abs(metric)
    log.info(f"DER: {der:.4f}")

    return {"der": der, "n_predictions": len(preds_list)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run offline Sortformer diarization inference."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/sortformer_offline"))
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--collar", type=float, default=0.25)
    parser.add_argument("--no-save-probs", action="store_true", help="Skip saving probability tensors")
    parser.add_argument(
        "--rec-ids", type=str, default=None,
        help="Comma-separated recording IDs to process (default: all).",
    )
    parser.add_argument(
        "--max-offset", type=float, default=None,
        help="Only process windows with offset <= this value (seconds).",
    )
    args = parser.parse_args()

    # Filter manifest if needed
    manifest = args.manifest
    tmp_manifest = None
    if args.rec_ids or args.max_offset:
        rec_ids = [r.strip() for r in args.rec_ids.split(",")] if args.rec_ids else None
        tmp_manifest = filter_manifest(manifest, rec_ids=rec_ids, max_offset=args.max_offset)
        manifest = tmp_manifest

    try:
        results = run_inference(
            model_path=args.model_path,
            manifest_path=manifest,
            out_dir=args.out_dir,
            precision=args.precision,
            batch_size=args.batch_size,
            collar=args.collar,
            save_probs=not args.no_save_probs,
        )
        log.info(f"Results: {results}")
    finally:
        # Clean up temp manifest
        if tmp_manifest and tmp_manifest.exists():
            tmp_manifest.unlink()


if __name__ == "__main__":
    main()

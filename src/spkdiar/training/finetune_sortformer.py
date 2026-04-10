"""
finetune_sortformer.py

Fine-tune diar_sortformer_4spk-v1.nemo on ATC0R data.

Strategy:
  - Freeze the NEST Fast Conformer encoder (115M params) — it extracts generic
    acoustic features and was trained on broad data; fine-tuning it on our small
    ATC set would cause catastrophic forgetting.
  - Fine-tune the 18-layer Transformer encoder (8M) + Sortformer output modules
    (0.14M) — these are the speaker tracking layers most sensitive to domain.
  - Use lr=1e-5 (10× lower than original), InverseSquareRootAnnealing,
    warmup=100 steps to protect the pretrained weights early in training.

CEC 599 deliverable 2d.

Usage:
    uv run python -m spkdiar.training.finetune_sortformer \\
        --model-path models/diar_sortformer_4spk-v1.nemo \\
        --train-manifest data/processed/manifests/finetune_train.jsonl \\
        --eval-manifest data/processed/manifests/finetune_eval.jsonl \\
        --out-dir results/finetune \\
        --max-steps 1000 \\
        --lr 1e-5
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf, open_dict
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# Suppress NeMo's verbose config dumps on load
os.environ.setdefault("NEMO_TESTING", "0")


def freeze_encoder(model: torch.nn.Module) -> None:
    """Freeze the NEST Fast Conformer encoder; leave Transformer + heads trainable."""
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.encoder.eval()   # also freeze BN/dropout statistics

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.encoder.parameters())
    log.info(f"Encoder frozen ({frozen/1e6:.2f}M params). Trainable: {trainable/1e6:.2f}M params")


def build_train_cfg(base_cfg, manifest: str, batch_size: int) -> object:
    cfg = OmegaConf.structured(OmegaConf.to_container(base_cfg, resolve=True))
    with open_dict(cfg):
        cfg.manifest_filepath = manifest
        cfg.batch_size        = batch_size
        cfg.num_workers       = min(8, os.cpu_count() or 4)
        cfg.shuffle           = True
        cfg.pin_memory        = True
    return cfg


def build_val_cfg(base_cfg, manifest: str, batch_size: int) -> object:
    cfg = OmegaConf.structured(OmegaConf.to_container(base_cfg, resolve=True))
    with open_dict(cfg):
        cfg.manifest_filepath = manifest
        cfg.batch_size        = batch_size
        cfg.num_workers       = min(4, os.cpu_count() or 4)
        cfg.shuffle           = False
    return cfg


def patch_optim(model, lr: float, warmup_steps: int, max_steps: int) -> None:
    """Update optimizer and scheduler config in-place using OmegaConf open_dict."""
    with open_dict(model.cfg):
        model.cfg.optim.lr                = lr
        model.cfg.optim.sched.warmup_steps = warmup_steps
        # InverseSquareRootAnnealing doesn't use max_steps but set it anyway
        if hasattr(model.cfg.optim.sched, "max_steps"):
            model.cfg.optim.sched.max_steps = max_steps
    log.info(f"Optim: adamw lr={lr:.2e}, warmup={warmup_steps} steps")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",     type=Path,  default=Path("models/diar_sortformer_4spk-v1.nemo"))
    parser.add_argument("--train-manifest", type=str,   default="data/processed/manifests/finetune_train.jsonl")
    parser.add_argument("--eval-manifest",  type=str,   default="data/processed/manifests/finetune_eval.jsonl")
    parser.add_argument("--out-dir",        type=Path,  default=Path("results/finetune"))
    parser.add_argument("--max-steps",      type=int,   default=1000)
    parser.add_argument("--lr",             type=float, default=1e-5)
    parser.add_argument("--batch-size",     type=int,   default=4)
    parser.add_argument("--warmup-steps",   type=int,   default=100)
    parser.add_argument("--val-interval",   type=int,   default=200)
    parser.add_argument("--ckpt-interval",  type=int,   default=200)
    parser.add_argument("--no-cuda",        action="store_true")
    args = parser.parse_args()

    ckpt_dir = args.out_dir / "checkpoints"
    log_dir  = args.out_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ model
    log.info(f"Loading model from {args.model_path}")
    from nemo.collections.asr.models import SortformerEncLabelModel
    model = SortformerEncLabelModel.restore_from(
        restore_path=str(args.model_path),
        map_location="cpu",
    )

    # ------------------------------------------------------------------ freeze
    freeze_encoder(model)

    # ------------------------------------------------------------------ optim
    patch_optim(model, args.lr, args.warmup_steps, args.max_steps)

    # ------------------------------------------------------------------ data
    log.info("Setting up training data ...")
    train_cfg = build_train_cfg(model.cfg.train_ds, args.train_manifest, args.batch_size)
    model.setup_training_data(train_cfg)

    log.info("Setting up validation data ...")
    val_cfg = build_val_cfg(model.cfg.validation_ds, args.eval_manifest, args.batch_size)
    model.setup_validation_data(val_cfg)

    # ------------------------------------------------------------------ trainer
    device = "cpu" if args.no_cuda or not torch.cuda.is_available() else "gpu"
    precision = "bf16-mixed" if device == "gpu" else "32-true"

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="sortformer-atc-step{step:05d}",
        every_n_train_steps=args.ckpt_interval,
        save_top_k=-1,      # keep all interval checkpoints
        save_last=True,
    )

    csv_logger = CSVLogger(save_dir=str(log_dir), name="finetune")

    trainer = pl.Trainer(
        accelerator=device,
        devices=1,
        precision=precision,
        max_steps=args.max_steps,
        val_check_interval=args.val_interval,
        check_val_every_n_epoch=None,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_cb],
        logger=csv_logger,
        enable_progress_bar=True,
    )

    log.info(
        f"Training for {args.max_steps} steps | "
        f"device={device} | precision={precision} | "
        f"batch_size={args.batch_size} | lr={args.lr:.2e}"
    )

    # ------------------------------------------------------------------ fit
    trainer.fit(model)

    # ------------------------------------------------------------------ save
    out_nemo = Path("models") / "diar_sortformer_4spk-v1-atc.nemo"
    out_nemo.parent.mkdir(parents=True, exist_ok=True)
    model.save_to(str(out_nemo))
    log.info(f"Fine-tuned model saved: {out_nemo}")


if __name__ == "__main__":
    main()

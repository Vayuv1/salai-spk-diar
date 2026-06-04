"""
sweep_sortformer_checkpoints.py

Evaluate all interval checkpoints from a fine-tuning run on the held-out test
manifest and export a single summary table.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from spkdiar.inference.run_sortformer import filter_manifest, run_inference

STEP_RE = re.compile(r"stepstep=(\d+)\.ckpt$")


def discover_checkpoints(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    items: list[tuple[int, Path]] = []
    for path in sorted(checkpoint_dir.glob("*.ckpt")):
        match = STEP_RE.search(path.name)
        if not match:
            continue
        items.append((int(match.group(1)), path))
    if not items:
        raise FileNotFoundError(f"No interval checkpoints found in {checkpoint_dir}")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all fine-tuning checkpoints on a held-out manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rec-ids", type=str, required=True)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--collar", type=float, default=0.25)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rec_ids = [item.strip() for item in args.rec_ids.split(",")]
    checkpoints = discover_checkpoints(args.checkpoint_dir)
    tmp_manifest = filter_manifest(args.manifest, rec_ids=rec_ids)
    rows: list[dict] = []

    try:
        for step, ckpt_path in checkpoints:
            run_dir = args.out_dir / f"step_{step:05d}"
            metrics_path = run_dir / "eval_metrics.json"
            if args.force or not metrics_path.exists():
                run_inference(
                    model_path=str(ckpt_path),
                    manifest_path=tmp_manifest,
                    out_dir=run_dir,
                    precision=args.precision,
                    batch_size=args.batch_size,
                    collar=args.collar,
                    save_probs=False,
                )
            summary = json.loads(metrics_path.read_text())
            rows.append({
                "step": step,
                "checkpoint": str(ckpt_path.resolve()),
                "DER": summary["overall"]["DER"],
                "FA": summary["overall"]["FA"],
                "MISS": summary["overall"]["MISS"],
                "CER": summary["overall"]["CER"],
                "metrics_json": str(metrics_path.resolve()),
            })
    finally:
        if tmp_manifest.exists():
            tmp_manifest.unlink()

    rows.sort(key=lambda item: item["step"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = args.out_dir / "checkpoint_sweep_summary.json"
    summary_csv = args.out_dir / "checkpoint_sweep_summary.csv"
    summary_json.write_text(json.dumps(rows, indent=2))

    with open(summary_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "checkpoint", "DER", "FA", "MISS", "CER", "metrics_json"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

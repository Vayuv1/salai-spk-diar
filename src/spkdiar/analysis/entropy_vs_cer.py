"""
entropy_vs_cer.py

For each Sortformer windowed inference window in a specified offset range,
compute:
  1. Layer-17 mean attention entropy from the Transformer (via forward hooks)
  2. Per-window CER by comparing the existing offline Sortformer pred RTTM
     against the GT RTTM clipped to [window_start, window_end]

Then scatter-plot entropy vs CER with a linear trend line and print Pearson r / p.

Usage:
    uv run python -m spkdiar.analysis.entropy_vs_cer \
        --rec-id dca_d2_2 \
        --offset-start 2880 --offset-end 2960 \
        --model-path models/diar_sortformer_4spk-v1.nemo \
        --audio-dir data/atc0r/audio \
        --gt-rttm-dir data/processed/rttm \
        --pred-rttm-dir results/sortformer_offline/pred_rttm \
        --prob-dir results/sortformer_offline/prob_tensors \
        --out-dir results/entropy_cer \
        --plot-dir results/plots
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate
from scipy import stats as scipy_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

WINDOW_DUR  = 10.0      # seconds
FRAME_STEP  = 0.08      # 80ms per Sortformer frame
SAMPLE_RATE = 16_000
COLLAR      = 0.25


# ---------------------------------------------------------------------------
# Reuse AttentionCapture + entropy from attention_entropy.py
# ---------------------------------------------------------------------------

class AttentionCapture:
    def __init__(self, model: torch.nn.Module) -> None:
        self._hooks: list = []
        self.weights: dict[int, list[torch.Tensor]] = {}
        for i, block in enumerate(model.transformer_encoder.layers):
            self.weights[i] = []
            hook = block.first_sub_layer.attn_dropout.register_forward_hook(
                self._make_hook(i)
            )
            self._hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            self.weights[layer_idx].append(inputs[0].detach().float().cpu())
        return hook

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def clear(self) -> None:
        for k in self.weights:
            self.weights[k].clear()


def layer17_entropy(
    model: torch.nn.Module,
    capture: AttentionCapture,
    audio: torch.Tensor,
    device: torch.device,
    eps: float = 1e-9,
) -> float:
    """Run one window; return layer-17 mean entropy (nats)."""
    capture.clear()
    audio = audio.to(device)
    length = torch.tensor([audio.shape[1]], dtype=torch.long, device=device)
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _ = model(audio, length)
    tensors = capture.weights.get(17, [])
    if not tensors:
        return float("nan")
    attn = torch.cat(tensors, dim=0).float()   # (B, H, T, T)
    H = -(attn * torch.log(attn + eps)).sum(dim=-1)   # (B, H, T_q)
    return float(H.mean().item())


# ---------------------------------------------------------------------------
# RTTM parsing → pyannote Annotation
# ---------------------------------------------------------------------------

def rttm_to_annotation(rttm_path: Path, t_start: float, t_end: float) -> Annotation:
    """Parse an RTTM file into a pyannote Annotation, clipped to [t_start, t_end]."""
    ann = Annotation()
    if not rttm_path.exists():
        return ann
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9 or parts[0] != "SPEAKER":
                continue
            seg_start = float(parts[3])
            dur       = float(parts[4])
            spk       = parts[7]
            seg_end   = seg_start + dur
            # Clip to window
            clipped_start = max(seg_start, t_start)
            clipped_end   = min(seg_end, t_end)
            if clipped_end <= clipped_start:
                continue
            ann[Segment(clipped_start, clipped_end)] = spk
    return ann


# ---------------------------------------------------------------------------
# Per-window CER
# ---------------------------------------------------------------------------

def compute_window_cer(
    pred_rttm: Path,
    gt_rttm: Path,
    t_start: float,
    t_end: float,
    collar: float = COLLAR,
) -> dict | None:
    """Compute DER components for one window.

    Returns None if reference has zero speech (skip silentwindows).
    Returns dict with keys: cer, fa, miss, der, total_ref.
    """
    hyp = rttm_to_annotation(pred_rttm, t_start, t_end)
    ref = rttm_to_annotation(gt_rttm,  t_start, t_end)

    uem = Timeline([Segment(t_start, t_end)])
    metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
    result = metric(ref, hyp, uem=uem, detailed=True)

    total = result["total"]
    if total <= 0.0:
        return None   # silent window — skip

    return {
        "cer":       result["confusion"]      / total,
        "fa":        result["false alarm"]    / total,
        "miss":      result["missed detection"] / total,
        "der":       result["diarization error rate"],
        "total_ref": total,
    }


# ---------------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------------

def plot_scatter(
    entropy_vals: list[float],
    cer_vals:     list[float],
    window_ids:   list[str],
    pearson_r:    float,
    pearson_p:    float,
    out_path:     Path,
) -> None:
    x = np.array(entropy_vals)
    y = np.array(cer_vals)

    # Linear trend
    slope, intercept, _, _, _ = scipy_stats.linregress(x, y)
    x_line = np.linspace(x.min() - 0.05, x.max() + 0.05, 200)
    y_line = slope * x_line + intercept

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(x, y, color="#2166ac", s=55, alpha=0.80, edgecolors="white",
               linewidths=0.6, zorder=3)

    # Trend line
    p_label = f"p = {pearson_p:.3f}" if pearson_p >= 0.001 else f"p = {pearson_p:.2e}"
    ax.plot(x_line, y_line, color="#d6604d", linewidth=1.5, linestyle="--", zorder=2,
            label=f"Linear fit  r = {pearson_r:+.3f},  {p_label}")

    # Annotate a few extreme points
    sorted_idx = np.argsort(y)[::-1]
    for rank, idx in enumerate(sorted_idx[:3]):
        ax.annotate(
            window_ids[idx].split("-")[2],   # just the offset in ms
            xy=(x[idx], y[idx]),
            xytext=(6, 4), textcoords="offset points",
            fontsize=6, color="#444444",
        )

    ax.set_xlabel("Layer-17 attention entropy (nats)", fontsize=10)
    ax.set_ylabel("Per-window CER", fontsize=10)
    ax.set_title(
        "Attention Entropy vs Speaker Confusion — dca_d2_2\n"
        "Lower entropy → more concentrated attention → higher CER?",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)

    # Add max-entropy reference line
    T = int(WINDOW_DUR / FRAME_STEP)
    ax.axvline(math.log(T), color="#aaaaaa", linewidth=0.8, linestyle=":",
               alpha=0.7, label=f"Uniform max (ln {T})")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rec-id",        default="dca_d2_2")
    parser.add_argument("--offset-start",  type=float, default=2880.0)
    parser.add_argument("--offset-end",    type=float, default=2960.0)
    parser.add_argument("--window-dur",    type=float, default=WINDOW_DUR)
    parser.add_argument("--window-shift",  type=float, default=5.0)
    parser.add_argument("--model-path",    type=Path,
                        default=Path("models/diar_sortformer_4spk-v1.nemo"))
    parser.add_argument("--audio-dir",     type=Path, default=Path("data/atc0r/audio"))
    parser.add_argument("--gt-rttm-dir",   type=Path, default=Path("data/processed/rttm"))
    parser.add_argument("--pred-rttm-dir", type=Path,
                        default=Path("results/sortformer_offline/pred_rttm"))
    parser.add_argument("--prob-dir",      type=Path,
                        default=Path("results/sortformer_offline/prob_tensors"))
    parser.add_argument("--out-dir",       type=Path, default=Path("results/entropy_cer"))
    parser.add_argument("--plot-dir",      type=Path, default=Path("results/plots"))
    parser.add_argument("--no-cuda",       action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # ------------------------------------------------------------------ model
    log.info(f"Loading Sortformer from {args.model_path}")
    from nemo.collections.asr.models import SortformerEncLabelModel
    model = SortformerEncLabelModel.restore_from(
        restore_path=str(args.model_path), map_location=device
    )
    model.eval()
    model.streaming_mode = False
    capture = AttentionCapture(model)

    # ------------------------------------------------------------------ windows
    audio_path = args.audio_dir / f"{args.rec_id}.mp3"
    gt_rttm    = args.gt_rttm_dir / f"{args.rec_id}.rttm"

    # Enumerate all windows whose start falls in [offset_start, offset_end)
    windows = []
    for p in sorted(args.prob_dir.glob(f"{args.rec_id}-*.npy")):
        stem   = p.stem
        parts  = stem.split("-")
        start_ms = int(parts[-2])
        dur_ms   = int(parts[-1])
        start_s  = start_ms / 1000.0
        end_s    = start_s + dur_ms / 1000.0
        if start_s >= args.offset_start and start_s < args.offset_end:
            windows.append((start_s, end_s, stem))
    windows.sort()
    log.info(f"Found {len(windows)} windows in [{args.offset_start}, {args.offset_end})s")

    # ------------------------------------------------------------------ per-window loop
    results = []
    for start_s, end_s, stem in windows:
        start_ms = int(start_s * 1000)
        dur_ms   = int(args.window_dur * 1000)

        # 1. CER from existing pred RTTM
        pred_rttm_path = args.pred_rttm_dir / f"{stem}.rttm"
        cer_data = compute_window_cer(pred_rttm_path, gt_rttm, start_s, end_s)
        if cer_data is None:
            log.info(f"  {stem}: no reference speech — skipping")
            continue

        # 2. Attention entropy — load audio slice, run model
        audio, _ = librosa.load(
            str(audio_path), sr=SAMPLE_RATE, offset=start_s,
            duration=args.window_dur, mono=True
        )
        audio_t = torch.from_numpy(audio).unsqueeze(0)
        h17 = layer17_entropy(model, capture, audio_t, device)

        log.info(
            f"  {stem}: H17={h17:.4f}  CER={cer_data['cer']:.4f}  "
            f"FA={cer_data['fa']:.4f}  MISS={cer_data['miss']:.4f}  "
            f"ref={cer_data['total_ref']:.1f}s"
        )
        results.append({
            "window_id": stem,
            "start_s":   start_s,
            "end_s":     end_s,
            "h17":       h17,
            **cer_data,
        })

    capture.remove()

    if len(results) < 3:
        log.error(f"Only {len(results)} usable windows — need at least 3 for correlation")
        return

    # ------------------------------------------------------------------ statistics
    entropy_vals = [r["h17"]  for r in results]
    cer_vals     = [r["cer"]  for r in results]
    window_ids   = [r["window_id"] for r in results]

    r_val, p_val = scipy_stats.pearsonr(entropy_vals, cer_vals)

    log.info(f"\n{'Window':>35}  {'H17':>8}  {'CER':>8}  {'FA':>8}  {'MISS':>8}")
    log.info("-" * 80)
    for row in results:
        log.info(
            f"  {row['window_id']:>33}  {row['h17']:>8.4f}  "
            f"{row['cer']:>8.4f}  {row['fa']:>8.4f}  {row['miss']:>8.4f}"
        )
    log.info(f"\nPearson r = {r_val:+.4f}   p = {p_val:.4f}   n = {len(results)}")
    log.info(f"(Negative r expected: lower entropy → higher CER)")

    # ------------------------------------------------------------------ save
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / f"{args.rec_id}_entropy_cer.json"
    with open(out_json, "w") as f:
        json.dump({
            "rec_id": args.rec_id,
            "offset_range": [args.offset_start, args.offset_end],
            "collar": COLLAR,
            "pearson_r": r_val,
            "pearson_p": p_val,
            "n_windows": len(results),
            "windows": results,
        }, f, indent=2)
    log.info(f"Data saved: {out_json}")

    # ------------------------------------------------------------------ plot
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    plot_scatter(
        entropy_vals, cer_vals, window_ids, r_val, p_val,
        args.plot_dir / "entropy_vs_cer_scatter.png",
    )


if __name__ == "__main__":
    main()

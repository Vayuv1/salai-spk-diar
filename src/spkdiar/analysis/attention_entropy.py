"""
attention_entropy.py

Extract per-layer Shannon entropy of Transformer self-attention weights from
Sortformer's 18-layer TransformerEncoder during inference.

Architecture context
--------------------
SortformerEncLabelModel has two attention stacks:
  encoder.layers.{0..17}.self_attn          RelPositionMultiHeadAttention
      → Fast Conformer encoder, limited-context attention. Ignored here.
  transformer_encoder.layers.{0..17}.first_sub_layer  MultiHeadAttention
      → Full self-attention over the 125-frame window. THIS is what we hook.

Hooking mechanism
-----------------
NeMo's MultiHeadAttention.forward() computes:
    attention_probs = torch.softmax(attention_scores, dim=-1)  # (B, H, T, T)
    attention_probs = self.attn_dropout(attention_probs)
We hook attn_dropout: the first input tensor is attention_probs before dropout,
shape (B, n_heads, T, T), in float32 regardless of model bf16 dtype.

Usage:
    uv run python -m spkdiar.analysis.attention_entropy \
        --model-path models/diar_sortformer_4spk-v1.nemo \
        --audio-path data/atc0r/audio/dca_d1_1.mp3 \
        --out-dir results/attention_entropy
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
import matplotlib.patches as mpatches
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000   # Sortformer input sample rate
WINDOW_DUR  = 10.0     # seconds
N_LAYERS    = 18       # TransformerEncoder depth
N_HEADS     = 8        # attention heads per layer


# ---------------------------------------------------------------------------
# Attention hook
# ---------------------------------------------------------------------------

class AttentionCapture:
    """Registers hooks on every attn_dropout in transformer_encoder and
    accumulates raw attention weight matrices per layer."""

    def __init__(self, model: torch.nn.Module) -> None:
        self._hooks: list = []
        # Maps layer_index → list of captured (B, H, T, T) float32 tensors
        self.weights: dict[int, list[torch.Tensor]] = {}

        for i, block in enumerate(model.transformer_encoder.layers):
            self.weights[i] = []
            # attn_dropout input[0] is post-softmax attention before dropout
            hook = block.first_sub_layer.attn_dropout.register_forward_hook(
                self._make_hook(i)
            )
            self._hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            # inputs[0]: (B, n_heads, T, T) — may be bf16; cast to float32
            self.weights[layer_idx].append(inputs[0].detach().float().cpu())
        return hook

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def clear(self) -> None:
        for k in self.weights:
            self.weights[k].clear()


# ---------------------------------------------------------------------------
# Entropy computation
# ---------------------------------------------------------------------------

def entropy_per_head(attn: torch.Tensor, eps: float = 1e-9) -> np.ndarray:
    """Per-head mean Shannon entropy.

    Args:
        attn: (B, n_heads, T_q, T_k) float32 — softmax attention weights.
    Returns:
        (n_heads,) float64 array — entropy for each head, averaged over
        batch and query positions.  Units: nats.
    """
    # H[b, h, q] = -sum_k A[b,h,q,k] * log(A[b,h,q,k] + eps)
    H = -(attn * torch.log(attn + eps)).sum(dim=-1)  # (B, H, T_q)
    # mean over batch and query → (H,)
    return H.mean(dim=(0, 2)).numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_window(audio_path: Path, offset: float, duration: float) -> torch.Tensor:
    """Load a single audio window as a (1, n_samples) float32 tensor at 16 kHz."""
    audio, _ = librosa.load(
        str(audio_path),
        sr=SAMPLE_RATE,
        offset=offset,
        duration=duration,
        mono=True,
    )
    return torch.from_numpy(audio).unsqueeze(0)  # (1, T)


# ---------------------------------------------------------------------------
# Inference with hooks
# ---------------------------------------------------------------------------

def run_window(
    model: torch.nn.Module,
    capture: AttentionCapture,
    audio: torch.Tensor,
    device: torch.device,
) -> tuple[dict[int, float], dict[int, np.ndarray]]:
    """Run one window through the model.

    Returns:
        mean_entropy: {layer_idx: mean over heads}
        head_entropy: {layer_idx: (n_heads,) array, one value per head}
    """
    capture.clear()

    audio = audio.to(device)
    length = torch.tensor([audio.shape[1]], dtype=torch.long, device=device)

    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _ = model(audio, length)

    mean_entropy: dict[int, float] = {}
    head_entropy: dict[int, np.ndarray] = {}

    for layer_idx, tensors in capture.weights.items():
        if not tensors:
            log.warning(f"Layer {layer_idx}: no attention captured")
            mean_entropy[layer_idx] = float("nan")
            head_entropy[layer_idx] = np.full(N_HEADS, float("nan"))
            continue
        attn = torch.cat(tensors, dim=0)   # (B, H, T, T)
        h_per = entropy_per_head(attn)     # (n_heads,)
        head_entropy[layer_idx] = h_per
        mean_entropy[layer_idx] = float(h_per.mean())

    return mean_entropy, head_entropy


# ---------------------------------------------------------------------------
# Publication-quality figure
# ---------------------------------------------------------------------------

def plot_entropy_publication(
    windows: list[dict],        # each: {label, mean, heads, color, marker, linestyle}
    out_path: Path,
    max_entropy: float,
    T_frames: int,
    rec_id: str = "",
) -> None:
    """Generate the publication-quality attention entropy comparison figure.

    Args:
        windows: list of window dicts, each with keys:
            label     str
            mean      dict[int, float]   per-layer mean entropy
            heads     dict[int, ndarray] per-layer (n_heads,) entropy
            color     str hex
            fill      str hex  (lighter shade for ±1 std band)
            marker    str matplotlib marker code
            linestyle str
        out_path: save path
        max_entropy: ln(T_frames) theoretical maximum
        T_frames: number of frames per window
    """
    layers = sorted(windows[0]["mean"].keys())
    x = np.array(layers)

    fig, ax = plt.subplots(figsize=(8, 5))

    # ---- Output-layer shaded region (layers 14–17) ----
    ax.axvspan(13.5, 17.5, alpha=0.07, color="#bbbbbb", zorder=0)
    ax.text(15.5, max_entropy - 0.04, "Output\nlayers",
            ha="center", va="top", fontsize=7, color="#666666",
            fontstyle="italic")

    # ---- Horizontal uniform-max reference ----
    ax.axhline(max_entropy, color="#888888", linewidth=0.9, linestyle="--",
               alpha=0.8, zorder=1, label=f"Maximum (uniform)  ln({T_frames}) = {max_entropy:.3f}")

    # ---- Per-window curves with ±1σ bands ----
    for w in windows:
        mean_vals = np.array([w["mean"][l] for l in layers])
        std_vals  = np.array([w["heads"][l].std() for l in layers])

        ax.fill_between(x, mean_vals - std_vals, mean_vals + std_vals,
                        alpha=0.18, color=w["color"], zorder=2)
        ax.plot(x, mean_vals,
                color=w["color"], marker=w["marker"], markersize=5,
                linewidth=1.6, linestyle=w.get("linestyle", "-"),
                label=w["label"], zorder=3)

    # ---- Annotate Δ at layer 17 ----
    # Find the two primary windows (first good, first bad)
    primary_good = windows[0]
    primary_bad  = windows[1]
    g17 = primary_good["mean"][17]
    b17 = primary_bad["mean"][17]
    delta = g17 - b17

    # Double-headed arrow
    ax.annotate(
        "", xy=(17, b17), xytext=(17, g17),
        arrowprops=dict(arrowstyle="<->", color="#333333",
                        lw=1.2, shrinkA=2, shrinkB=2),
        zorder=4,
    )
    # Label to the right of the arrow
    ax.text(17.2, (g17 + b17) / 2, f"Δ = {delta:.2f} nats",
            va="center", ha="left", fontsize=8, color="#333333", zorder=4)

    # ---- Formatting ----
    ax.set_xlim(-0.6, 19.8)   # extra room for Δ annotation at layer 17
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax.set_xlabel("Transformer Layer", fontsize=10)
    ax.set_ylabel("Attention Entropy (nats)", fontsize=10)
    title_rec = f" ({rec_id})" if rec_id else ""
    ax.set_title(f"Attention Entropy Collapse in Sortformer on ATC Audio{title_rec}", fontsize=11)

    # y-axis: don't start at 0, zoom into the interesting range
    all_vals = [w["mean"][l] for w in windows for l in layers]
    y_min = min(all_vals) - 0.15
    y_max = max_entropy + 0.08
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

    ax.legend(fontsize=8, loc="lower left", framealpha=0.85,
              handlelength=1.8, borderpad=0.7)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figure saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Sortformer Transformer attention entropy.")
    parser.add_argument("--model-path", type=Path, default=Path("models/diar_sortformer_4spk-v1.nemo"))
    parser.add_argument("--audio-path", type=Path, default=Path("data/atc0r/audio/dca_d1_1.mp3"))
    parser.add_argument("--good-offset", type=float, default=55.0)
    parser.add_argument("--bad-offset",  type=float, default=95.0)
    parser.add_argument("--good2-offset", type=float, default=60.0,
                        help="Second 'good' window offset for replication check")
    parser.add_argument("--bad2-offset",  type=float, default=115.0,
                        help="Second 'bad' window offset (deep lock-up region)")
    parser.add_argument("--window-dur", type=float, default=WINDOW_DUR)
    parser.add_argument("--out-dir", type=Path, default=Path("results/attention_entropy"))
    parser.add_argument("--plot-dir", type=Path, default=Path("results/plots"))
    parser.add_argument("--plot-name", type=str, default="attention_entropy_comparison.png",
                        help="Output plot filename (saved inside --plot-dir)")
    parser.add_argument("--rec-id", type=str, default="",
                        help="Recording ID for plot title (inferred from audio-path if empty)")
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    if not args.rec_id:
        args.rec_id = args.audio_path.stem

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    log.info(f"Device: {device}")

    # --- Load model ---
    log.info(f"Loading model from {args.model_path}")
    from nemo.collections.asr.models import SortformerEncLabelModel
    model = SortformerEncLabelModel.restore_from(
        restore_path=str(args.model_path),
        map_location=device,
    )
    model.eval()
    model.streaming_mode = False
    log.info(f"Model loaded — {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # --- Register hooks ---
    capture = AttentionCapture(model)
    log.info(f"Hooks registered on {len(capture.weights)} Transformer layers")

    # --- Window specs ---
    T_frames    = int(args.window_dur / 0.08)
    max_entropy = math.log(T_frames)

    window_specs = [
        dict(offset=args.good_offset,  role="good",
             label=f"Window at {args.good_offset:.0f}s — low CER (correct tracking)",
             color="#2166ac", fill="#a6cee3", marker="o"),
        dict(offset=args.bad_offset,   role="bad",
             label=f"Window at {args.bad_offset:.0f}s — high CER (speaker confusion)",
             color="#d6604d", fill="#fdbf8e", marker="^"),
        dict(offset=args.good2_offset, role="good2",
             label=f"Window at {args.good2_offset:.0f}s — replication (low CER)",
             color="#4393c3", fill="#c6dbef", marker="o", linestyle="--"),
        dict(offset=args.bad2_offset,  role="bad2",
             label=f"Window at {args.bad2_offset:.0f}s — replication (high CER)",
             color="#f4a582", fill="#fee0d2", marker="^", linestyle="--"),
    ]

    log.info(f"Loading audio: {args.audio_path}")
    results = []
    for spec in window_specs:
        log.info(f"  Running window at offset={spec['offset']}s ...")
        audio = load_window(args.audio_path, spec["offset"], args.window_dur)
        mean_ent, head_ent = run_window(model, capture, audio, device)
        results.append({**spec, "mean": mean_ent, "heads": head_ent})
        log.info(f"    Layer-17 entropy: {mean_ent[17]:.4f} nats "
                 f"(std across heads: {head_ent[17].std():.4f})")

    capture.remove()

    # --- Layer-17 summary for all four windows ---
    log.info("\n=== Layer-17 entropy (all four windows) ===")
    log.info(f"{'Window':>45}  {'H_17 (nats)':>12}  {'std':>8}")
    for r in results:
        log.info(f"  {r['label']:>43}  {r['mean'][17]:>12.4f}  {r['heads'][17].std():>8.4f}")

    log.info(f"\n  Max uniform entropy (ln {T_frames}): {max_entropy:.4f}")
    log.info(f"  Primary gap (good−bad) at layer 17: "
             f"{results[0]['mean'][17] - results[1]['mean'][17]:.4f} nats")
    log.info(f"  Replication gap (good2−bad2) at layer 17: "
             f"{results[2]['mean'][17] - results[3]['mean'][17]:.4f} nats")

    # --- Save JSON ---
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "entropy_data.json"
    json_data = {
        "model": str(args.model_path),
        "audio": str(args.audio_path),
        "max_uniform_entropy_nats": max_entropy,
        "n_frames_per_window": T_frames,
        "frame_shift_sec": 0.08,
        "n_transformer_layers": N_LAYERS,
        "n_attention_heads": N_HEADS,
        "windows": [
            {
                "role": r["role"],
                "offset_sec": r["offset"],
                "duration_sec": args.window_dur,
                "label": r["label"],
                "entropy_per_layer_mean": {str(k): float(v) for k, v in r["mean"].items()},
                "entropy_per_layer_per_head": {
                    str(k): r["heads"][k].tolist() for k in r["heads"]
                },
                "entropy_per_layer_std": {
                    str(k): float(r["heads"][k].std()) for k in r["heads"]
                },
            }
            for r in results
        ],
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    log.info(f"\nEntropy data saved: {json_path}")

    # --- Generate publication figure ---
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.plot_dir / args.plot_name
    plot_entropy_publication(results, plot_path, max_entropy, T_frames, rec_id=args.rec_id)


if __name__ == "__main__":
    main()

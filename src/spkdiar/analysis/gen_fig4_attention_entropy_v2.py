"""
gen_fig4_attention_entropy_v2.py

Fig 4 (v2): Layer-wise attention entropy — pretrained vs fine-tuned Sortformer.
Four curves on one axes:
  • Pretrained correct tracking (55 s) — solid blue, circles
  • Pretrained speaker lock-up   (95 s) — solid red,  triangles
  • Fine-tuned  correct tracking (55 s) — dashed blue, circles
  • Fine-tuned  speaker lock-up  (95 s) — dashed red,  triangles

±1 std bands are drawn for the two pretrained curves only (to avoid clutter);
fine-tuned bands are omitted and only the mean lines are shown.

Entropy is extracted live from the fine-tuned model and cached to
  results/attention_entropy/entropy_data_finetuned.json

The pretrained data is read from the existing
  results/attention_entropy/entropy_data.json

Output: results/paper_figures/fig4_attention_entropy_v2.{pdf,png}

Usage:
    uv run python -m spkdiar.analysis.gen_fig4_attention_entropy_v2 \
        [--finetuned-model models/diar_sortformer_4spk-v1-atc.nemo] \
        [--force-rerun]  # re-extract even if cached JSON exists
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

from spkdiar.analysis.ieee_style import IEEE_SINGLE_COL, apply_ieee_style, save_fig

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
WINDOW_DUR  = 10.0
N_LAYERS    = 18
N_HEADS     = 8

PRETRAINED_JSON  = Path("results/attention_entropy/entropy_data.json")
FINETUNED_JSON   = Path("results/attention_entropy/entropy_data_finetuned.json")
FINETUNED_MODEL  = Path("models/diar_sortformer_4spk-v1-atc.nemo")
AUDIO_PATH       = Path("data/atc0r/audio/dca_d1_1.mp3")
GOOD_OFFSET      = 55.0   # correct-tracking window (matches pretrained run)
BAD_OFFSET       = 95.0   # lock-up window


# ---------------------------------------------------------------------------
# Attention hook (copy from attention_entropy.py to avoid circular import)
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


def entropy_per_head(attn: torch.Tensor, eps: float = 1e-9) -> np.ndarray:
    H = -(attn * torch.log(attn + eps)).sum(dim=-1)
    return H.mean(dim=(0, 2)).numpy().astype(np.float64)


def load_window(audio_path: Path, offset: float, duration: float) -> torch.Tensor:
    audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE,
                             offset=offset, duration=duration, mono=True)
    return torch.from_numpy(audio).unsqueeze(0)


def run_window(model, capture: AttentionCapture, audio: torch.Tensor, device: torch.device):
    capture.clear()
    audio = audio.to(device)
    length = torch.tensor([audio.shape[1]], dtype=torch.long, device=device)
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _ = model(audio, length)
    mean_ent: dict[int, float] = {}
    head_ent: dict[int, np.ndarray] = {}
    for layer_idx, tensors in capture.weights.items():
        if not tensors:
            mean_ent[layer_idx] = float("nan")
            head_ent[layer_idx] = np.full(N_HEADS, float("nan"))
            continue
        attn = torch.cat(tensors, dim=0)
        h_per = entropy_per_head(attn)
        head_ent[layer_idx] = h_per
        mean_ent[layer_idx] = float(h_per.mean())
    return mean_ent, head_ent


# ---------------------------------------------------------------------------
# Extract finetuned entropy and cache
# ---------------------------------------------------------------------------

def extract_finetuned_entropy(
    model_path: Path,
    audio_path: Path,
    out_json: Path,
) -> dict:
    """Load fine-tuned model, run two windows, save JSON, return data dict."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f"Loading fine-tuned model: {model_path}")
    from nemo.collections.asr.models import SortformerEncLabelModel
    model = SortformerEncLabelModel.restore_from(
        restore_path=str(model_path),
        map_location=device,
    )
    model.eval()
    model.streaming_mode = False
    log.info(f"Fine-tuned model loaded — {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    capture = AttentionCapture(model)
    T_frames    = int(WINDOW_DUR / 0.08)
    max_entropy = math.log(T_frames)

    specs = [
        dict(offset=GOOD_OFFSET, role="good",
             label=f"Fine-tuned correct tracking ({GOOD_OFFSET:.0f}s)"),
        dict(offset=BAD_OFFSET,  role="bad",
             label=f"Fine-tuned speaker lock-up ({BAD_OFFSET:.0f}s)"),
    ]

    results = []
    for spec in specs:
        log.info(f"  Running fine-tuned window at offset={spec['offset']}s ...")
        audio = load_window(audio_path, spec["offset"], WINDOW_DUR)
        mean_ent, head_ent = run_window(model, capture, audio, device)
        results.append({**spec, "mean": mean_ent, "heads": head_ent})
        log.info(f"    Layer-17 H = {mean_ent[17]:.4f} nats  (std={head_ent[17].std():.4f})")

    capture.remove()

    out_json.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "model": str(model_path),
        "audio": str(audio_path),
        "max_uniform_entropy_nats": max_entropy,
        "n_frames_per_window": T_frames,
        "frame_shift_sec": 0.08,
        "n_transformer_layers": N_LAYERS,
        "n_attention_heads": N_HEADS,
        "windows": [
            {
                "role": r["role"],
                "offset_sec": r["offset"],
                "duration_sec": WINDOW_DUR,
                "label": r["label"],
                "entropy_per_layer_mean": {str(k): float(v) for k, v in r["mean"].items()},
                "entropy_per_layer_std":  {str(k): float(r["heads"][k].std()) for k in r["heads"]},
                "entropy_per_layer_per_head": {
                    str(k): r["heads"][k].tolist() for k in r["heads"]
                },
            }
            for r in results
        ],
    }
    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)
    log.info(f"Fine-tuned entropy saved: {out_json}")
    return data


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def make_fig4_v2(
    pre_data: dict,
    ft_data:  dict,
    out_dir:  Path,
) -> None:
    apply_ieee_style()

    max_entropy = pre_data["max_uniform_entropy_nats"]
    T_frames    = pre_data["n_frames_per_window"]

    # Index windows by role
    pre_by_role = {w["role"]: w for w in pre_data["windows"]}
    ft_by_role  = {w["role"]: w for w in ft_data["windows"]}

    pre_good = pre_by_role["good"]
    pre_bad  = pre_by_role["bad"]
    ft_good  = ft_by_role["good"]
    ft_bad   = ft_by_role["bad"]

    layers = sorted(int(k) for k in pre_good["entropy_per_layer_mean"].keys())
    x = np.array(layers)

    def means(w):
        return np.array([w["entropy_per_layer_mean"][str(l)] for l in layers])

    def stds(w):
        return np.array([w["entropy_per_layer_std"][str(l)] for l in layers])

    pre_good_mean = means(pre_good);  pre_good_std = stds(pre_good)
    pre_bad_mean  = means(pre_bad);   pre_bad_std  = stds(pre_bad)
    ft_good_mean  = means(ft_good)
    ft_bad_mean   = means(ft_bad)

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 3.0))

    # Uniform-max reference line
    ax.axhline(max_entropy, color="#999999", linewidth=0.6, linestyle=":",
               label=f"Uniform max  ln({T_frames}) = {max_entropy:.2f} nats", zorder=1)

    # Output-layer shading (layers 14–17)
    ax.axvspan(13.5, 17.5, alpha=0.06, color="#888888", zorder=0)
    ax.text(15.5, max_entropy - 0.07, "Output\nlayers",
            ha="center", va="top", fontsize=6, color="#666666", fontstyle="italic")

    # Pretrained ±1σ bands
    ax.fill_between(x, pre_good_mean - pre_good_std, pre_good_mean + pre_good_std,
                    alpha=0.12, color="#1f77b4", zorder=2)
    ax.fill_between(x, pre_bad_mean  - pre_bad_std,  pre_bad_mean  + pre_bad_std,
                    alpha=0.12, color="#d62728", zorder=2)

    # Pretrained curves (solid)
    ax.plot(x, pre_good_mean, color="#1f77b4", linestyle="-", marker="o",
            markersize=3.5, linewidth=1.0, markerfacecolor="white",
            markeredgewidth=0.8, zorder=3,
            label=f"Pre. correct ({GOOD_OFFSET:.0f}s)  H₁₇={pre_good_mean[17]:.2f}")
    ax.plot(x, pre_bad_mean, color="#d62728", linestyle="-", marker="^",
            markersize=3.5, linewidth=1.0, markerfacecolor="white",
            markeredgewidth=0.8, zorder=3,
            label=f"Pre. lock-up  ({BAD_OFFSET:.0f}s)   H₁₇={pre_bad_mean[17]:.2f}")

    # Fine-tuned curves (dashed, no band)
    ax.plot(x, ft_good_mean, color="#1f77b4", linestyle="--", marker="o",
            markersize=3.5, linewidth=1.0, markerfacecolor="#1f77b4",
            markeredgewidth=0.8, zorder=3,
            label=f"FT  correct ({GOOD_OFFSET:.0f}s)  H₁₇={ft_good_mean[17]:.2f}")
    ax.plot(x, ft_bad_mean, color="#d62728", linestyle="--", marker="^",
            markersize=3.5, linewidth=1.0, markerfacecolor="#d62728",
            markeredgewidth=0.8, zorder=3,
            label=f"FT  lock-up  ({BAD_OFFSET:.0f}s)   H₁₇={ft_bad_mean[17]:.2f}")

    # Δ annotation (pretrained gap at layer 17)
    pre_delta = float(pre_good_mean[17] - pre_bad_mean[17])
    ax.annotate(
        "", xy=(17, pre_bad_mean[17]), xytext=(17, pre_good_mean[17]),
        arrowprops=dict(arrowstyle="<->", color="#333333",
                        lw=0.9, shrinkA=2, shrinkB=2),
        zorder=4,
    )
    ax.text(17.3, (pre_good_mean[17] + pre_bad_mean[17]) / 2,
            f"Δ={pre_delta:.2f}", va="center", ha="left",
            fontsize=6.5, color="#333333", zorder=4)

    ax.set_xlim(-0.5, 19.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=7)
    ax.set_xlabel("Transformer Layer", fontsize=9)
    ax.set_ylabel("Attention Entropy (nats)", fontsize=9)

    all_vals = np.concatenate([pre_good_mean, pre_bad_mean, ft_good_mean, ft_bad_mean])
    ax.set_ylim(all_vals.min() - 0.15, max_entropy + 0.10)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))

    ax.legend(fontsize=6.5, loc="lower left", framealpha=0.85,
              handlelength=1.8, borderpad=0.5, labelspacing=0.3)

    fig.tight_layout(pad=0.4)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, out_dir / "fig4_attention_entropy_v2")
    log.info(f"Fig4 v2 saved to {out_dir / 'fig4_attention_entropy_v2'}.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned-model", type=Path, default=FINETUNED_MODEL)
    parser.add_argument("--audio-path",      type=Path, default=AUDIO_PATH)
    parser.add_argument("--pretrained-json", type=Path, default=PRETRAINED_JSON)
    parser.add_argument("--finetuned-json",  type=Path, default=FINETUNED_JSON)
    parser.add_argument("--out-dir",         type=Path, default=Path("results/paper_figures"))
    parser.add_argument("--force-rerun",     action="store_true",
                        help="Re-extract fine-tuned entropy even if cache exists")
    args = parser.parse_args()

    # Load pretrained data
    log.info(f"Loading pretrained entropy: {args.pretrained_json}")
    pre_data = json.load(open(args.pretrained_json))

    # Extract or load fine-tuned data
    if args.finetuned_json.exists() and not args.force_rerun:
        log.info(f"Loading cached fine-tuned entropy: {args.finetuned_json}")
        ft_data = json.load(open(args.finetuned_json))
    else:
        ft_data = extract_finetuned_entropy(
            args.finetuned_model, args.audio_path, args.finetuned_json
        )

    # Report comparison
    pre_by_role = {w["role"]: w for w in pre_data["windows"]}
    ft_by_role  = {w["role"]: w for w in ft_data["windows"]}
    log.info("\n=== Layer-17 entropy comparison ===")
    for role in ("good", "bad"):
        pre_h = pre_by_role[role]["entropy_per_layer_mean"]["17"]
        ft_h  = ft_by_role[role]["entropy_per_layer_mean"]["17"]
        log.info(f"  {role:4s}  pretrained={pre_h:.4f}  finetuned={ft_h:.4f}  Δ={ft_h-pre_h:+.4f}")

    make_fig4_v2(pre_data, ft_data, args.out_dir)


if __name__ == "__main__":
    main()

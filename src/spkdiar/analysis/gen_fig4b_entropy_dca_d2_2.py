"""
gen_fig4b_entropy_dca_d2_2.py

Extract per-layer Transformer attention entropy from dca_d2_2 for both the
pretrained and fine-tuned Sortformer models, using the two windows identified
from the entropy-vs-CER scatter:
  - High-CER window:  2935000 ms offset  (CER = 0.46)
  - Low-CER window:   2955000 ms offset  (CER = 0.002)

Generates a 4-curve IEEE single-column figure (3.5 × 3.0 in, 600 DPI):
  • Pretrained low-CER  (2955 s) — solid blue, open circles
  • Pretrained high-CER (2935 s) — solid red, open triangles
  • Fine-tuned  low-CER (2955 s) — dashed blue, filled circles
  • Fine-tuned  high-CER(2935 s) — dashed red, filled triangles

±1 std bands drawn for pretrained curves only.

Output:
  results/attention_entropy/entropy_data_dca_d2_2_pretrained.json
  results/attention_entropy/entropy_data_dca_d2_2_finetuned.json
  results/paper_figures/fig4b_entropy_dca_d2_2.{pdf,png}

Usage:
    uv run python -m spkdiar.analysis.gen_fig4b_entropy_dca_d2_2 [--force-rerun]
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

AUDIO_PATH = Path("data/atc0r/audio/dca_d2_2.mp3")
DEFAULT_PRETRAINED_MODEL = Path("models/diar_sortformer_4spk-v1.nemo")
DEFAULT_FINETUNED_MODEL = Path("models/diar_sortformer_4spk-v1-atc-1ksteps-rerun.nemo")

# Windows: (offset_sec, role_label, cer)
LOW_CER_OFFSET  = 2955.0   # CER = 0.002
HIGH_CER_OFFSET = 2935.0   # CER = 0.46

OUT_ENTROPY_DIR = Path("results/attention_entropy")
OUT_FIGURE_DIR  = Path("results/paper_figures")


# ---------------------------------------------------------------------------
# Attention hook
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

    def _make_hook(self, idx: int):
        def hook(module, inputs, output):
            self.weights[idx].append(inputs[0].detach().float().cpu())
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


def run_window(
    model: torch.nn.Module,
    capture: AttentionCapture,
    audio: torch.Tensor,
    device: torch.device,
) -> tuple[dict[int, float], dict[int, np.ndarray]]:
    capture.clear()
    audio  = audio.to(device)
    length = torch.tensor([audio.shape[1]], dtype=torch.long, device=device)
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _ = model(audio, length)
    mean_ent: dict[int, float]       = {}
    head_ent: dict[int, np.ndarray]  = {}
    for layer_idx, tensors in capture.weights.items():
        if not tensors:
            mean_ent[layer_idx] = float("nan")
            head_ent[layer_idx] = np.full(N_HEADS, float("nan"))
            continue
        attn  = torch.cat(tensors, dim=0)
        h_per = entropy_per_head(attn)
        head_ent[layer_idx] = h_per
        mean_ent[layer_idx] = float(h_per.mean())
    return mean_ent, head_ent


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_entropy(
    model_path: Path,
    audio_path: Path,
    offsets: list[dict],   # list of {offset, role, label}
    out_json: Path,
    device: torch.device,
) -> dict:
    """Load model, run entropy hooks on specified windows, save + return data."""
    log.info(f"Loading model: {model_path}")
    from nemo.collections.asr.models import SortformerEncLabelModel
    model = SortformerEncLabelModel.restore_from(
        restore_path=str(model_path),
        map_location=device,
    )
    model.eval()
    model.streaming_mode = False
    log.info(f"  {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    capture     = AttentionCapture(model)
    T_frames    = int(WINDOW_DUR / 0.08)
    max_entropy = math.log(T_frames)
    results     = []

    for spec in offsets:
        log.info(f"  Window at {spec['offset']:.0f} s ({spec['role']}) ...")
        audio = load_window(audio_path, spec["offset"], WINDOW_DUR)
        mean_ent, head_ent = run_window(model, capture, audio, device)
        results.append({**spec, "mean": mean_ent, "heads": head_ent})
        log.info(f"    H17 = {mean_ent[17]:.4f} nats  std = {head_ent[17].std():.4f}")

    capture.remove()

    out_json.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "model": str(model_path),
        "audio": str(audio_path),
        "max_uniform_entropy_nats": max_entropy,
        "n_frames_per_window": T_frames,
        "windows": [
            {
                "role": r["role"],
                "offset_sec": r["offset"],
                "label": r["label"],
                "cer": r["cer"],
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
    log.info(f"  Saved: {out_json}")
    return data


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(pre_data: dict, ft_data: dict, out_dir: Path) -> None:
    apply_ieee_style()

    max_entropy = pre_data["max_uniform_entropy_nats"]
    T_frames    = pre_data["n_frames_per_window"]

    pre_by_role = {w["role"]: w for w in pre_data["windows"]}
    ft_by_role  = {w["role"]: w for w in ft_data["windows"]}

    pre_low  = pre_by_role["low_cer"]
    pre_high = pre_by_role["high_cer"]
    ft_low   = ft_by_role["low_cer"]
    ft_high  = ft_by_role["high_cer"]

    layers = sorted(int(k) for k in pre_low["entropy_per_layer_mean"].keys())
    x = np.array(layers)

    def means(w) -> np.ndarray:
        return np.array([w["entropy_per_layer_mean"][str(l)] for l in layers])

    def stds(w) -> np.ndarray:
        return np.array([w["entropy_per_layer_std"][str(l)] for l in layers])

    pre_low_mean  = means(pre_low);   pre_low_std  = stds(pre_low)
    pre_high_mean = means(pre_high);  pre_high_std = stds(pre_high)
    ft_low_mean   = means(ft_low)
    ft_high_mean  = means(ft_high)

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 3.0))

    # Uniform-max reference
    ax.axhline(max_entropy, color="#999999", linewidth=0.6, linestyle=":",
               label=f"Uniform max  ln({T_frames}) = {max_entropy:.2f} nats", zorder=1)

    # Output-layer shading
    ax.axvspan(13.5, 17.5, alpha=0.06, color="#888888", zorder=0)
    ax.text(15.5, max_entropy - 0.07, "Output\nlayers",
            ha="center", va="top", fontsize=6, color="#666666", fontstyle="italic")

    # Pretrained ±1σ bands
    ax.fill_between(x, pre_low_mean  - pre_low_std,  pre_low_mean  + pre_low_std,
                    alpha=0.12, color="#1f77b4", zorder=2)
    ax.fill_between(x, pre_high_mean - pre_high_std, pre_high_mean + pre_high_std,
                    alpha=0.12, color="#d62728", zorder=2)

    # Pretrained curves (solid, open markers)
    ax.plot(x, pre_low_mean, color="#1f77b4", linestyle="-", marker="o",
            markersize=3.5, linewidth=1.0, markerfacecolor="white",
            markeredgewidth=0.8, zorder=3,
            label=f"Pre. low-CER  ({LOW_CER_OFFSET:.0f} s)   H₁₇={pre_low_mean[17]:.2f}")
    ax.plot(x, pre_high_mean, color="#d62728", linestyle="-", marker="^",
            markersize=3.5, linewidth=1.0, markerfacecolor="white",
            markeredgewidth=0.8, zorder=3,
            label=f"Pre. high-CER ({HIGH_CER_OFFSET:.0f} s)  H₁₇={pre_high_mean[17]:.2f}")

    # Fine-tuned curves (dashed, filled markers)
    ax.plot(x, ft_low_mean, color="#1f77b4", linestyle="--", marker="o",
            markersize=3.5, linewidth=1.0, markerfacecolor="#1f77b4",
            markeredgewidth=0.8, zorder=3,
            label=f"FT  low-CER  ({LOW_CER_OFFSET:.0f} s)   H₁₇={ft_low_mean[17]:.2f}")
    ax.plot(x, ft_high_mean, color="#d62728", linestyle="--", marker="^",
            markersize=3.5, linewidth=1.0, markerfacecolor="#d62728",
            markeredgewidth=0.8, zorder=3,
            label=f"FT  high-CER ({HIGH_CER_OFFSET:.0f} s)  H₁₇={ft_high_mean[17]:.2f}")

    # Δ annotation: pretrained gap at layer 17
    pre_delta = float(pre_low_mean[17] - pre_high_mean[17])
    ax.annotate(
        "", xy=(17, pre_high_mean[17]), xytext=(17, pre_low_mean[17]),
        arrowprops=dict(arrowstyle="<->", color="#333333",
                        lw=0.9, shrinkA=2, shrinkB=2),
        zorder=4,
    )
    ax.text(17.3, (pre_low_mean[17] + pre_high_mean[17]) / 2,
            f"Δ={pre_delta:+.2f}", va="center", ha="left",
            fontsize=6.5, color="#333333", zorder=4)

    ax.set_xlim(-0.5, 19.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=7)
    ax.set_xlabel("Transformer Layer", fontsize=9)
    ax.set_ylabel("Attention Entropy (nats)", fontsize=9)

    all_vals = np.concatenate([pre_low_mean, pre_high_mean, ft_low_mean, ft_high_mean])
    ax.set_ylim(all_vals.min() - 0.15, max_entropy + 0.10)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))

    ax.legend(fontsize=6.5, loc="lower left", framealpha=0.85,
              handlelength=1.8, borderpad=0.5, labelspacing=0.3)

    fig.tight_layout(pad=0.4)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, out_dir / "fig4b_entropy_dca_d2_2")
    log.info(f"Figure saved: {out_dir / 'fig4b_entropy_dca_d2_2'}.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-rerun", action="store_true",
                        help="Re-extract even if cached JSONs exist")
    parser.add_argument("--pretrained-model", type=Path, default=DEFAULT_PRETRAINED_MODEL)
    parser.add_argument("--finetuned-model", type=Path, default=DEFAULT_FINETUNED_MODEL)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    offsets = [
        dict(offset=LOW_CER_OFFSET,  role="low_cer",
             label=f"dca_d2_2  {LOW_CER_OFFSET:.0f} s  (low CER = 0.002)",
             cer=0.002),
        dict(offset=HIGH_CER_OFFSET, role="high_cer",
             label=f"dca_d2_2  {HIGH_CER_OFFSET:.0f} s  (high CER = 0.46)",
             cer=0.46),
    ]

    pre_json = OUT_ENTROPY_DIR / "entropy_data_dca_d2_2_pretrained.json"
    ft_json  = OUT_ENTROPY_DIR / "entropy_data_dca_d2_2_finetuned.json"

    if pre_json.exists() and not args.force_rerun:
        log.info(f"Loading cached pretrained entropy: {pre_json}")
        pre_data = json.load(open(pre_json))
    else:
        pre_data = extract_entropy(args.pretrained_model, AUDIO_PATH, offsets, pre_json, device)

    if ft_json.exists() and not args.force_rerun:
        log.info(f"Loading cached finetuned entropy: {ft_json}")
        ft_data = json.load(open(ft_json))
    else:
        ft_data = extract_entropy(args.finetuned_model, AUDIO_PATH, offsets, ft_json, device)

    # ---- Print layer-17 summary ----
    pre_by_role = {w["role"]: w for w in pre_data["windows"]}
    ft_by_role  = {w["role"]: w for w in ft_data["windows"]}

    print("\n" + "=" * 62)
    print(f"{'Condition':<35}  {'H₁₇ (nats)':>10}  {'std':>7}")
    print("-" * 62)
    for role, label in [
        ("low_cer",  f"Pretrained  low-CER  ({LOW_CER_OFFSET:.0f} s)"),
        ("high_cer", f"Pretrained  high-CER ({HIGH_CER_OFFSET:.0f} s)"),
        ("low_cer",  f"Fine-tuned  low-CER  ({LOW_CER_OFFSET:.0f} s)"),
        ("high_cer", f"Fine-tuned  high-CER ({HIGH_CER_OFFSET:.0f} s)"),
    ]:
        src = pre_by_role if "Pretrained" in label else ft_by_role
        h   = src[role]["entropy_per_layer_mean"]["17"]
        s   = src[role]["entropy_per_layer_std"]["17"]
        print(f"  {label:<33}  {h:>10.4f}  {s:>7.4f}")
    print("-" * 62)

    pre_gap = (pre_by_role["low_cer"]["entropy_per_layer_mean"]["17"]
               - pre_by_role["high_cer"]["entropy_per_layer_mean"]["17"])
    ft_gap  = (ft_by_role["low_cer"]["entropy_per_layer_mean"]["17"]
               - ft_by_role["high_cer"]["entropy_per_layer_mean"]["17"])
    print(f"  Pretrained gap (low − high):              {pre_gap:>+10.4f}")
    print(f"  Fine-tuned  gap (low − high):             {ft_gap:>+10.4f}")
    print("=" * 62 + "\n")

    make_figure(pre_data, ft_data, OUT_FIGURE_DIR)


if __name__ == "__main__":
    main()

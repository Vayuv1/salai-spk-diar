"""
gen_fig4_entropy.py

Fig 4: Layer-wise attention entropy comparison — correct-tracking window (55 s,
solid + circles) vs high-confusion window (95 s, dashed + triangles) from
dca_d1_1. Shows ±1 std band across the 8 attention heads. Single column
(3.5 × 2.5 in).

Output: results/paper_figures/fig4_attention_entropy.{pdf,png}

Usage:
    uv run python -m spkdiar.analysis.gen_fig4_entropy
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spkdiar.analysis.ieee_style import (
    IEEE_SINGLE_COL, apply_ieee_style, save_fig,
)

ENTROPY_JSON = Path("results/attention_entropy/entropy_data.json")


def main(
    json_path: Path = ENTROPY_JSON,
    out_dir:   Path = Path("results/paper_figures"),
) -> None:
    apply_ieee_style()

    data = json.load(open(json_path))
    max_entropy = data["max_uniform_entropy_nats"]
    T_frames    = data["n_frames_per_window"]

    # Select 'good' (55 s) and 'bad' (95 s) windows — primary pair
    windows_by_role = {w["role"]: w for w in data["windows"]}
    good = windows_by_role["good"]
    bad  = windows_by_role["bad"]

    layers = sorted(int(k) for k in good["entropy_per_layer_mean"].keys())
    x      = np.array(layers)

    good_mean = np.array([good["entropy_per_layer_mean"][str(l)] for l in layers])
    bad_mean  = np.array([bad["entropy_per_layer_mean"][str(l)]  for l in layers])
    good_std  = np.array([good["entropy_per_layer_std"][str(l)]  for l in layers])
    bad_std   = np.array([bad["entropy_per_layer_std"][str(l)]   for l in layers])

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 2.5))

    # Uniform-max reference
    ax.axhline(max_entropy, color="#999999", linewidth=0.7, linestyle=":",
               label=f"Uniform max  ln({T_frames}) = {max_entropy:.2f} nats")

    # Output-layer shading (layers 14–17)
    ax.axvspan(13.5, 17.5, alpha=0.06, color="#888888", zorder=0)
    ax.text(15.5, max_entropy - 0.06, "Output\nlayers",
            ha="center", va="top", fontsize=6, color="#666666", fontstyle="italic")

    # Bands (±1σ)
    ax.fill_between(x, good_mean - good_std, good_mean + good_std,
                    alpha=0.15, color="#1f77b4", zorder=1)
    ax.fill_between(x, bad_mean  - bad_std,  bad_mean  + bad_std,
                    alpha=0.15, color="#d62728", zorder=1)

    # Curves
    ax.plot(x, good_mean, color="#1f77b4", linestyle="-", marker="o",
            markersize=3.5, linewidth=1.0, markerfacecolor="white",
            markeredgewidth=0.8,
            label=f"Correct tracking (55 s)  H₁₇ = {good_mean[17]:.2f}")
    ax.plot(x, bad_mean,  color="#d62728", linestyle="--", marker="^",
            markersize=3.5, linewidth=1.0, markerfacecolor="white",
            markeredgewidth=0.8,
            label=f"Speaker confusion (95 s)  H₁₇ = {bad_mean[17]:.2f}")

    # Δ annotation at layer 17
    delta = float(good_mean[17] - bad_mean[17])
    ax.annotate(
        "", xy=(17, bad_mean[17]), xytext=(17, good_mean[17]),
        arrowprops=dict(arrowstyle="<->", color="#333333",
                        lw=0.9, shrinkA=2, shrinkB=2),
        zorder=4,
    )
    ax.text(17.3, (good_mean[17] + bad_mean[17]) / 2,
            f"Δ = {delta:.2f}", va="center", ha="left",
            fontsize=7, color="#333333", zorder=4)

    ax.set_xlim(-0.5, 19.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=7)
    ax.set_xlabel("Transformer Layer", fontsize=9)
    ax.set_ylabel("Attention Entropy (nats)", fontsize=9)

    all_vals = np.concatenate([good_mean, bad_mean])
    ax.set_ylim(all_vals.min() - 0.15, max_entropy + 0.10)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))

    ax.legend(fontsize=7, loc="lower left", framealpha=0.85,
              handlelength=1.8, borderpad=0.5)

    fig.tight_layout(pad=0.4)
    save_fig(fig, out_dir / "fig4_attention_entropy")


if __name__ == "__main__":
    main()

"""
gen_fig3_embeddings.py

Fig 3: Speaker embedding cosine similarity distributions — dca_d1_1 full (32
speakers), 16 kHz TitaNet-Large embeddings. KDE curves for intra-speaker
(solid) vs inter-speaker (dashed) similarity. Single column (3.5 × 3.0 in).

Output: results/paper_figures/fig3_embedding_similarity.{pdf,png}

Usage:
    uv run python -m spkdiar.analysis.gen_fig3_embeddings
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from spkdiar.analysis.ieee_style import (
    IEEE_SINGLE_COL, apply_ieee_style, save_fig,
)

NPZ_PATH = Path("results/speaker_embeddings/dca_d1_1_full/dca_d1_1_16k_embeddings.npz")


def load_similarities(npz_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute all intra- and inter-speaker cosine similarities from NPZ.

    Returns:
        intra_sims : (N_intra,) array
        inter_sims : (N_inter,) array
        stats      : dict with mean_intra, mean_inter, margin
    """
    d = np.load(str(npz_path), allow_pickle=True)
    speakers = list(d.keys())

    all_embs: list[np.ndarray] = []
    all_labels: list[str]      = []
    for spk in speakers:
        embs = d[spk].astype(np.float32)   # (n_cues, 192)
        all_embs.append(embs)
        all_labels.extend([spk] * len(embs))

    E      = np.vstack(all_embs)           # (N, 192)
    norms  = np.linalg.norm(E, axis=1, keepdims=True)
    E_norm = E / (norms + 1e-9)

    # Full pairwise cosine similarity
    sim_matrix = E_norm @ E_norm.T         # (N, N)

    labels = np.array(all_labels)
    N      = len(labels)
    ui, uj = np.triu_indices(N, k=1)       # upper triangle indices

    same_spk   = (labels[ui] == labels[uj])
    intra_sims = sim_matrix[ui, uj][same_spk].astype(float)
    inter_sims = sim_matrix[ui, uj][~same_spk].astype(float)

    stats = {
        "mean_intra": float(intra_sims.mean()),
        "mean_inter": float(inter_sims.mean()),
        "margin":     float(intra_sims.mean() - inter_sims.mean()),
        "n_intra":    len(intra_sims),
        "n_inter":    len(inter_sims),
    }
    return intra_sims, inter_sims, stats


def main(
    npz_path: Path = NPZ_PATH,
    out_dir:  Path = Path("results/paper_figures"),
) -> None:
    apply_ieee_style()

    intra_sims, inter_sims, stats = load_similarities(npz_path)

    # KDE
    x = np.linspace(-0.3, 1.05, 600)
    kde_intra = gaussian_kde(intra_sims, bw_method="silverman")
    kde_inter = gaussian_kde(inter_sims, bw_method="silverman")

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 3.0))

    ax.plot(x, kde_intra(x), color="#1f77b4", linestyle="-",  linewidth=1.2,
            label=f"Intra-speaker  (μ = {stats['mean_intra']:.3f})")
    ax.plot(x, kde_inter(x), color="#d62728", linestyle="--", linewidth=1.2,
            label=f"Inter-speaker  (μ = {stats['mean_inter']:.3f})")

    # Mean lines
    ax.axvline(stats["mean_intra"], color="#1f77b4", linewidth=0.8,
               linestyle=":", alpha=0.8)
    ax.axvline(stats["mean_inter"], color="#d62728", linewidth=0.8,
               linestyle=":", alpha=0.8)

    # Separability margin annotation
    y_arrow = kde_intra(stats["mean_intra"]) * 0.55
    ax.annotate(
        "", xy=(stats["mean_intra"], y_arrow),
        xytext=(stats["mean_inter"], y_arrow),
        arrowprops=dict(arrowstyle="<->", color="#333333",
                        lw=0.9, shrinkA=0, shrinkB=0),
    )
    ax.text(
        (stats["mean_intra"] + stats["mean_inter"]) / 2,
        y_arrow + 0.05,
        f"Margin = {stats['margin']:.3f}",
        ha="center", va="bottom", fontsize=7.5, color="#333333",
    )

    ax.set_xlabel("Cosine Similarity", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_xlim(-0.25, 1.05)
    ax.set_ylim(bottom=0)

    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.85,
              handlelength=1.5, borderpad=0.5)
    ax.text(
        0.98, 0.96,
        f"dca_d1_1 (32 speakers, 389 cues)\n16 kHz TitaNet-Large",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=6.5, color="#555555",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
    )

    fig.tight_layout(pad=0.4)
    save_fig(fig, out_dir / "fig3_embedding_similarity")


if __name__ == "__main__":
    main()

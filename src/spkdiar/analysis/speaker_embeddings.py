"""
speaker_embeddings.py

Extract TitaNet-Large speaker embeddings from ATC0R audio at 8 kHz and 16 kHz
and analyze the speaker separability at each rate.

CEC 599 deliverable 2c — Speaker embedding analysis at 8/16 kHz.

The 8 kHz condition loads audio at 8 kHz then resamples to 16 kHz before
feeding to TitaNet.  The audio content is band-limited to 4 kHz (matching VHF
radio), even though the sample rate fed to the model is 16 kHz.  If the
separability margin is similar across both conditions, it means the embedding
collapse is inherent to the ATC domain and not an artifact of sample rate.

Usage:
    uv run python -m spkdiar.analysis.speaker_embeddings \
        --stm data/atc0r/stm/dca_d1_1.stm \
        --audio data/atc0r/audio/dca_d1_1.mp3 \
        --max-duration 350 \
        --out-dir results/speaker_embeddings \
        --plot-dir results/plots
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

MODEL_TAG   = "nvidia/speakerverification_en_titanet_large"
NATIVE_SR   = 16_000   # TitaNet's expected sample rate
MIN_DUR_SEC = 0.5      # skip cues shorter than this


# ---------------------------------------------------------------------------
# STM parsing
# ---------------------------------------------------------------------------

def parse_stm(stm_path: Path, max_duration: float | None = None) -> list[dict]:
    """Parse ATC0R STM into a list of cue dicts.

    STM format per line:
        <rec_id> <ch> <quality>><speaker>><listener> <start> <end> <transcript...>

    Quality level 4 cues and cues shorter than MIN_DUR_SEC are excluded.
    """
    cues = []
    with open(stm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            ql_spk_lst = parts[2].split(">")
            if len(ql_spk_lst) < 3:
                continue
            quality = int(ql_spk_lst[0])
            if quality == 4:
                continue
            speaker = ql_spk_lst[1]
            start   = float(parts[3])
            end     = float(parts[4])
            if max_duration is not None:
                if start >= max_duration:
                    continue
                end = min(end, max_duration)
            dur = end - start
            if dur < MIN_DUR_SEC:
                continue
            cues.append({"speaker": speaker, "start": start, "end": end, "dur": dur})
    return cues


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_segment(
    audio_path: Path,
    start: float,
    duration: float,
    load_sr: int,
) -> torch.Tensor:
    """Load one segment at load_sr, then resample to NATIVE_SR for TitaNet.

    Returns (1, n_samples_at_16k) float32 tensor.
    """
    audio, _ = librosa.load(
        str(audio_path), sr=load_sr, offset=start, duration=duration, mono=True
    )
    if load_sr != NATIVE_SR:
        audio = librosa.resample(audio, orig_sr=load_sr, target_sr=NATIVE_SR)
    return torch.from_numpy(audio).unsqueeze(0)   # (1, T)


def get_embedding(
    model,
    audio: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Run TitaNet forward and return the 192-dim speaker embedding as float32 numpy."""
    audio   = audio.to(device)
    length  = torch.tensor([audio.shape[1]], dtype=torch.long, device=device)
    with torch.no_grad():
        _, embs = model(input_signal=audio, input_signal_length=length)
    return embs.squeeze(0).float().cpu().numpy()   # (192,)


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def compute_similarities(
    embeddings_by_spk: dict[str, np.ndarray],
) -> dict:
    """Compute intra- and inter-speaker cosine similarities.

    Args:
        embeddings_by_spk: {speaker_id: (N, 192) array}

    Returns dict with keys: intra_pairs, inter_pairs, mean_intra, mean_inter, margin
    """
    intra_sims, inter_sims = [], []

    speakers = list(embeddings_by_spk.keys())

    # Intra: pairs within the same speaker
    for spk, embs in embeddings_by_spk.items():
        if embs.shape[0] < 2:
            continue
        for i, j in combinations(range(embs.shape[0]), 2):
            s = cosine_sim(embs[i], embs[j])
            if not np.isnan(s):
                intra_sims.append(s)

    # Inter: pairs across different speakers
    for s1, s2 in combinations(speakers, 2):
        e1 = embeddings_by_spk[s1]
        e2 = embeddings_by_spk[s2]
        for i in range(e1.shape[0]):
            for j in range(e2.shape[0]):
                s = cosine_sim(e1[i], e2[j])
                if not np.isnan(s):
                    inter_sims.append(s)

    mean_intra = float(np.mean(intra_sims)) if intra_sims else float("nan")
    mean_inter = float(np.mean(inter_sims)) if inter_sims else float("nan")
    margin     = mean_intra - mean_inter if not (np.isnan(mean_intra) or np.isnan(mean_inter)) else float("nan")

    return {
        "n_intra_pairs": len(intra_sims),
        "n_inter_pairs": len(inter_sims),
        "intra_sims":    intra_sims,
        "inter_sims":    inter_sims,
        "mean_intra":    mean_intra,
        "mean_inter":    mean_inter,
        "margin":        margin,
    }


# ---------------------------------------------------------------------------
# Figure A: similarity distributions
# ---------------------------------------------------------------------------

def plot_similarity_distributions(
    stats_16k: dict,
    stats_8k: dict,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    titles = ["16 kHz (native)", "8 kHz → 16 kHz (band-limited)"]
    all_stats = [stats_16k, stats_8k]

    for ax, stats, title in zip(axes, all_stats, titles):
        intra = np.array(stats["intra_sims"])
        inter = np.array(stats["inter_sims"])

        bins = np.linspace(-0.1, 1.0, 35)
        ax.hist(inter, bins=bins, alpha=0.65, color="#d6604d", label="Inter-speaker", density=True)
        ax.hist(intra, bins=bins, alpha=0.65, color="#2166ac", label="Intra-speaker", density=True)

        # Vertical lines at means
        ax.axvline(stats["mean_intra"], color="#2166ac", linewidth=1.4, linestyle="--")
        ax.axvline(stats["mean_inter"], color="#d6604d", linewidth=1.4, linestyle="--")

        # Margin annotation
        ax.annotate(
            "", xy=(stats["mean_intra"], 3.5), xytext=(stats["mean_inter"], 3.5),
            arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.2),
        )
        mid = (stats["mean_intra"] + stats["mean_inter"]) / 2
        ax.text(mid, 3.8, f"margin\n{stats['margin']:+.3f}",
                ha="center", va="bottom", fontsize=8, color="#333333")

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Cosine similarity", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlim(-0.15, 1.05)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        "TitaNet-Large Speaker Embedding Similarity — dca_d1_1 (350 s)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure B: t-SNE
# ---------------------------------------------------------------------------

def plot_tsne(
    embs_by_spk_16k: dict[str, np.ndarray],
    embs_by_spk_8k: dict[str, np.ndarray],
    out_path: Path,
    n_label_speakers: int = 5,
    title_suffix: str = "",
) -> None:
    from matplotlib.lines import Line2D

    all_speakers = sorted(set(embs_by_spk_16k) | set(embs_by_spk_8k))
    n_spk = len(all_speakers)

    # Use tab20 for up to 20 speakers, then fall back to a continuous colormap
    if n_spk <= 20:
        cmap = plt.get_cmap("tab20")
        spk_color = {spk: cmap(i / max(n_spk - 1, 1)) for i, spk in enumerate(all_speakers)}
    else:
        cmap = plt.get_cmap("gist_rainbow")
        spk_color = {spk: cmap(i / (n_spk - 1)) for i, spk in enumerate(all_speakers)}

    # Top-N speakers by cue count — labeled in both panels
    counts_16k = {spk: embs.shape[0] for spk, embs in embs_by_spk_16k.items()}
    top_speakers = set(sorted(counts_16k, key=lambda s: -counts_16k[s])[:n_label_speakers])

    # Legend only when ≤ 15 speakers (beyond that it's unreadable)
    show_legend = n_spk <= 15

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    titles    = ["16 kHz (native)", "8 kHz → 16 kHz (band-limited)"]
    all_dicts = [embs_by_spk_16k, embs_by_spk_8k]

    for ax, embs_by_spk, title in zip(axes, all_dicts, titles):
        all_embs, all_labels = [], []
        for spk, embs in embs_by_spk.items():
            for e in embs:
                all_embs.append(e)
                all_labels.append(spk)
        X = np.stack(all_embs)   # (N, 192)
        n = X.shape[0]

        # Perplexity: sklearn requires perplexity < n_samples; scale with dataset size
        perplexity = min(30, max(5, n // 10))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                    max_iter=1000, init="pca")
        X2 = tsne.fit_transform(X)   # (N, 2)

        label_positions: dict[str, list] = defaultdict(list)
        for (x, y), spk in zip(X2, all_labels):
            ax.scatter(x, y, color=spk_color[spk], s=18, alpha=0.75,
                       edgecolors="none", zorder=3)
            label_positions[spk].append((x, y))

        # Label top-N speakers at their centroid
        for spk in top_speakers:
            if spk not in label_positions:
                continue
            pts = np.array(label_positions[spk])
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            ax.text(cx, cy, spk, fontsize=7, ha="center", va="center",
                    fontweight="bold", color="black",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75, ec="none"))

        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        # Light border instead of full spines
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color("#cccccc")

        # Annotation: N points, N speakers
        ax.text(0.01, 0.01, f"{n} cues · {n_spk} speakers",
                transform=ax.transAxes, fontsize=7, color="#666666", va="bottom")

    # Legend only for small speaker sets; otherwise a note
    if show_legend:
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=spk_color[spk],
                   markersize=7, label=spk)
            for spk in all_speakers
        ]
        ncol = min(n_spk, 8)
        fig.legend(handles=legend_handles, loc="lower center", ncol=ncol,
                   fontsize=7, framealpha=0.85, title="Speaker", title_fontsize=7,
                   borderpad=0.5, handletextpad=0.4)
        bottom_margin = 0.08 + 0.04 * ((n_spk - 1) // ncol)
    else:
        fig.text(0.5, 0.01,
                 f"Colors cycle through {n_spk} speakers · top-{n_label_speakers} by cue count labeled",
                 ha="center", fontsize=8, color="#555555")
        bottom_margin = 0.05

    rec_label = title_suffix or ""
    fig.suptitle(
        f"t-SNE of TitaNet-Large Embeddings{rec_label}\n"
        f"Points are individual speaker cues (192-dim embeddings)",
        fontsize=10, fontweight="bold",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout(rect=[0, bottom_margin, 1, 1])
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(
    cues: list[dict],
    embs_by_spk_16k: dict[str, np.ndarray],
    stats_16k: dict,
    stats_8k: dict,
) -> None:
    n_speakers  = len(embs_by_spk_16k)
    n_cues      = sum(e.shape[0] for e in embs_by_spk_16k.values())

    header = f"{'Metric':<40} {'16 kHz':>12} {'8 kHz':>12}"
    sep    = "-" * len(header)
    rows   = [
        ("Speakers",              f"{n_speakers}",             f"{n_speakers}"),
        ("Cues (≥0.5 s)",         f"{n_cues}",                 f"{n_cues}"),
        ("Intra-spk pairs",       f"{stats_16k['n_intra_pairs']}", f"{stats_8k['n_intra_pairs']}"),
        ("Inter-spk pairs",       f"{stats_16k['n_inter_pairs']}", f"{stats_8k['n_inter_pairs']}"),
        ("Mean intra-spk sim",    f"{stats_16k['mean_intra']:.4f}", f"{stats_8k['mean_intra']:.4f}"),
        ("Mean inter-spk sim",    f"{stats_16k['mean_inter']:.4f}", f"{stats_8k['mean_inter']:.4f}"),
        ("Separability margin",   f"{stats_16k['margin']:.4f}", f"{stats_8k['margin']:.4f}"),
    ]

    log.info("\n" + sep)
    log.info(f"  TitaNet-Large Embedding Analysis — dca_d1_1 (first 350 s)")
    log.info(sep)
    log.info(header)
    log.info(sep)
    for label, v16, v8 in rows:
        log.info(f"  {label:<38} {v16:>12} {v8:>12}")
    log.info(sep)

    # Per-speaker breakdown
    log.info("\n  Per-speaker cue counts:")
    for spk, embs in sorted(embs_by_spk_16k.items(), key=lambda x: -x[1].shape[0]):
        log.info(f"    {spk:<20} {embs.shape[0]} cues")
    log.info("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TitaNet speaker embedding analysis at 8/16 kHz.")
    parser.add_argument("--stm",          type=Path, default=Path("data/atc0r/stm/dca_d1_1.stm"))
    parser.add_argument("--audio",        type=Path, default=Path("data/atc0r/audio/dca_d1_1.mp3"))
    parser.add_argument("--max-duration", type=float, default=350.0)
    parser.add_argument("--out-dir",      type=Path, default=Path("results/speaker_embeddings"))
    parser.add_argument("--plot-dir",     type=Path, default=Path("results/plots"))
    parser.add_argument("--no-cuda",      action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    log.info(f"Device: {device}")

    # --- Load TitaNet ---
    log.info(f"Loading {MODEL_TAG} ...")
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    model = EncDecSpeakerLabelModel.from_pretrained(MODEL_TAG)
    model.eval().to(device)
    log.info(f"Model loaded — {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # --- Parse STM ---
    cues = parse_stm(args.stm, max_duration=args.max_duration)
    log.info(f"Parsed {len(cues)} cues from {args.stm.name} (max_dur={args.max_duration}s)")

    # --- Extract embeddings at both sample rates ---
    embs_by_spk_16k: dict[str, list[np.ndarray]] = defaultdict(list)
    embs_by_spk_8k:  dict[str, list[np.ndarray]] = defaultdict(list)

    for i, cue in enumerate(cues):
        spk, start, dur = cue["speaker"], cue["start"], cue["dur"]
        log.info(f"  [{i+1:2d}/{len(cues)}] {spk:20s}  {start:.1f}s  ({dur:.2f}s)")

        for load_sr, store in [(NATIVE_SR, embs_by_spk_16k), (8000, embs_by_spk_8k)]:
            audio = extract_segment(args.audio, start, dur, load_sr)
            emb   = get_embedding(model, audio, device)
            store[spk].append(emb)

    # Convert lists → stacked arrays
    embs_16k = {spk: np.stack(v) for spk, v in embs_by_spk_16k.items()}
    embs_8k  = {spk: np.stack(v) for spk, v in embs_by_spk_8k.items()}

    # --- Compute similarities ---
    stats_16k = compute_similarities(embs_16k)
    stats_8k  = compute_similarities(embs_8k)

    # --- Print summary ---
    print_summary(cues, embs_16k, stats_16k, stats_8k)

    # --- Save embeddings ---
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rec_id = args.stm.stem   # e.g. dca_d1_1 or log_id_1
    np.savez(args.out_dir / f"{rec_id}_16k_embeddings.npz", **{k: v for k, v in embs_16k.items()})
    np.savez(args.out_dir / f"{rec_id}_8k_embeddings.npz",  **{k: v for k, v in embs_8k.items()})
    log.info(f"Embeddings saved to {args.out_dir}/")

    # --- Save stats JSON (without the raw sim lists to keep it compact) ---
    def stats_compact(s: dict) -> dict:
        return {k: v for k, v in s.items() if k not in ("intra_sims", "inter_sims")}

    with open(args.out_dir / "similarity_stats.json", "w") as f:
        json.dump({"16k": stats_compact(stats_16k), "8k": stats_compact(stats_8k)}, f, indent=2)
    log.info(f"Stats saved to {args.out_dir}/similarity_stats.json")

    # --- Figures ---
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    plot_similarity_distributions(
        stats_16k, stats_8k,
        args.plot_dir / "embedding_similarity_distributions.png",
    )
    plot_tsne(
        embs_16k, embs_8k,
        args.plot_dir / "embedding_tsne.png",
    )


if __name__ == "__main__":
    main()

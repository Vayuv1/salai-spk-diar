"""
plot_timeline.py

Timeline comparison of four speaker diarization systems on ATC0R recordings.
Produces a single figure with 5 horizontal lanes sharing the same x-axis.

Lanes (top to bottom):
  GT                  — ground truth RTTM bars, colored by role (controller/pilot)
  Sortformer (offline) — per-window probability curves, overlapping windows shown
  Sortformer (stream) — same format, streaming model
  LS-EEND             — full-recording probability tensor, active speakers only
  Pyannote 3.1        — hypothesis RTTM bars

For DASC 2026 paper.

Usage:
    uv run python -m spkdiar.analysis.plot_timeline \
        --rec-id dca_d1_1 --t-start 55 --t-end 150
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------
# Sortformer / LS-EEND speaker curve colors (ColorBrewer Set1)
SPK_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#999999"]

CONTROLLER_COLOR = "#e6550d"   # orange
PILOT_COLORS = [
    "#3182bd",  # blue
    "#31a354",  # green
    "#756bb1",  # purple
    "#636363",  # grey
    "#de2d26",  # red
    "#08519c",  # dark blue
    "#006d2c",  # dark green
    "#54278f",  # dark purple
]
LANE_BG = ["#ffffff", "#f5f5f5", "#ffffff", "#f5f5f5", "#ffffff"]


# ---------------------------------------------------------------------------
# RTTM parsing
# ---------------------------------------------------------------------------

def load_rttm(path: Path, t_start: float, t_end: float) -> dict[str, list[tuple[float, float]]]:
    """Parse RTTM into {speaker: [(start, end), ...]} clipped to [t_start, t_end]."""
    segments: dict[str, list[tuple[float, float]]] = {}
    if not path.exists():
        return segments
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            dur = float(parts[4])
            spk = parts[7]
            end = start + dur
            if end <= t_start or start >= t_end:
                continue
            seg_start = max(start, t_start)
            seg_end = min(end, t_end)
            segments.setdefault(spk, []).append((seg_start, seg_end))
    return segments


# ---------------------------------------------------------------------------
# Sortformer windowed prob tensors
# ---------------------------------------------------------------------------

def load_sortformer_windows(
    prob_dir: Path,
    rec_id: str,
    t_start: float,
    t_end: float,
    frame_step: float = 0.08,
) -> list[tuple[float, np.ndarray]]:
    """Load all windows whose time range overlaps [t_start, t_end].

    Returns list of (window_start_sec, probs) where probs is (T, 4) float32.
    Sorted by window start time.
    """
    if not prob_dir.exists():
        return []
    windows = []
    for path in sorted(prob_dir.glob(f"{rec_id}-*.npy")):
        stem = path.stem  # e.g. dca_d1_1-55000-10000
        parts = stem.split("-")
        # rec_id may contain hyphens; parse from the right: last two segments are start_ms, dur_ms
        start_ms = int(parts[-2])
        dur_ms = int(parts[-1])
        win_start = start_ms / 1000.0
        win_end = win_start + dur_ms / 1000.0
        if win_end <= t_start or win_start >= t_end:
            continue
        probs = np.load(str(path)).astype(np.float32)  # (T, 4)
        windows.append((win_start, probs))
    windows.sort(key=lambda x: x[0])
    return windows


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_empty(ax: plt.Axes, msg: str = "data not available") -> None:
    ax.text(
        0.5, 0.5, msg,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=9, color="#888888", style="italic",
    )


def _draw_rttm_bars(
    ax: plt.Axes,
    segments: dict[str, list[tuple[float, float]]],
    speaker_colors: dict[str, str],
    t_start: float,
    t_end: float,
    bar_height: float = 0.7,
    label_speakers: bool = False,
) -> None:
    """Draw horizontal bars for each speaker in its own sub-row."""
    speakers = sorted(segments.keys())
    n = len(speakers)
    if n == 0:
        _draw_empty(ax)
        return
    ax.set_ylim(0, n)
    ax.set_yticks([i + 0.5 for i in range(n)])
    ax.set_yticklabels(
        [_format_spk_label(s) for s in speakers],
        fontsize=7,
    )
    for i, spk in enumerate(speakers):
        color = speaker_colors.get(spk, "#aaaaaa")
        for seg_start, seg_end in segments[spk]:
            ax.barh(
                i + 0.5,
                seg_end - seg_start,
                left=seg_start,
                height=bar_height,
                color=color,
                alpha=0.85,
                linewidth=0,
            )


def _format_spk_label(spk_id: str) -> str:
    """Strip rec_id prefix and shorten for display."""
    # dca_d1_1_D1-1 → D1-1
    parts = spk_id.split("_", 3)
    if len(parts) >= 4:
        return parts[3]
    elif len(parts) == 3:
        # e.g. dca_d1_1 → rec_id has 3 parts, speaker is after
        return parts[2]
    return spk_id


def _is_controller(spk_id: str, rec_id: str) -> bool:
    """Controller IDs contain a hyphen after stripping the rec_id prefix."""
    prefix = rec_id + "_"
    raw = spk_id[len(prefix):] if spk_id.startswith(prefix) else spk_id
    return "-" in raw


def _assign_gt_colors(
    segments: dict[str, list[tuple[float, float]]],
    rec_id: str,
) -> dict[str, str]:
    colors: dict[str, str] = {}
    pilot_idx = 0
    for spk in sorted(segments.keys()):
        if _is_controller(spk, rec_id):
            colors[spk] = CONTROLLER_COLOR
        else:
            colors[spk] = PILOT_COLORS[pilot_idx % len(PILOT_COLORS)]
            pilot_idx += 1
    return colors


def _draw_prob_curves(
    ax: plt.Axes,
    windows: list[tuple[float, np.ndarray]],
    t_start: float,
    t_end: float,
    frame_step: float,
    threshold: float | None = None,
    alpha: float = 0.6,
) -> None:
    """Plot probability curves for all speaker columns across overlapping windows."""
    if not windows:
        _draw_empty(ax)
        return

    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1"], fontsize=7)

    for win_start, probs in windows:
        T, n_spk = probs.shape
        win_dur = T * frame_step
        t = win_start + np.arange(T) * frame_step
        # Use only the centre 5 s of each 10 s window so each timepoint is
        # covered by exactly one window and overlapping-window jitter disappears.
        center_start = win_start + win_dur / 4.0   # T+2.5s for a 10s window
        center_end   = win_start + win_dur * 3 / 4.0  # T+7.5s
        mask = (t >= max(t_start, center_start)) & (t <= min(t_end, center_end))
        if not mask.any():
            continue
        for s in range(n_spk):
            ax.plot(
                t[mask],
                probs[mask, s],
                color=SPK_COLORS[s % len(SPK_COLORS)],
                linewidth=0.8,
                alpha=alpha,
            )

    if threshold is not None:
        ax.axhline(threshold, color="#555555", linewidth=0.6, linestyle="--", alpha=0.7)

    # Build legend (one entry per speaker index)
    n_spk = windows[0][1].shape[1] if windows else 0
    handles = [
        mpatches.Patch(color=SPK_COLORS[s % len(SPK_COLORS)], label=f"spk{s}")
        for s in range(n_spk)
    ]
    ax.legend(handles=handles, fontsize=6, loc="upper right",
              ncol=n_spk, framealpha=0.7, handlelength=1.0)


def _draw_lseend_curves(
    ax: plt.Axes,
    prob_path: Path,
    t_start: float,
    t_end: float,
    frame_step: float = 0.1,
    threshold: float | None = None,
    active_threshold_tanh: float = -0.5,
) -> None:
    """Plot LS-EEND probability curves for active speakers in [t_start, t_end]."""
    if not prob_path.exists():
        _draw_empty(ax)
        return

    probs_tanh = np.load(str(prob_path)).astype(np.float32)  # (T, 9)
    T, n_cols = probs_tanh.shape
    t = np.arange(T) * frame_step  # absolute time in seconds

    mask = (t >= t_start) & (t <= t_end)
    if not mask.any():
        _draw_empty(ax, "no frames in time range")
        return

    t_slice = t[mask]
    # col 0 = silence; cols 1-8 = speakers
    sil_prob = (probs_tanh[mask, 0] + 1.0) / 2.0   # (T_slice,) in [0,1]
    spk_tanh = probs_tanh[mask, 1:]                  # (T_slice, 8)
    # Convert tanh → [0, 1] for visual consistency with Sortformer sigmoid outputs
    spk_prob = (spk_tanh + 1.0) / 2.0  # (T_slice, 8)

    # Determine which speakers are active in this window
    active_mask = spk_tanh.max(axis=0) > active_threshold_tanh  # (8,)
    active_indices = np.where(active_mask)[0]

    if len(active_indices) == 0:
        _draw_empty(ax, "no active speakers in range")
        return

    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1"], fontsize=7)

    # Silence column: light grey dashed line drawn first (below speaker curves)
    ax.plot(t_slice, sil_prob, color="#aaaaaa", linewidth=0.8, linestyle="--",
            alpha=0.8, zorder=1)

    handles = [mpatches.Patch(color="#aaaaaa", label="silence",
                               linestyle="--", fill=False)]
    # Re-create as a Line2D proxy so the dashed style shows in the legend
    import matplotlib.lines as mlines
    sil_handle = mlines.Line2D([], [], color="#aaaaaa", linewidth=0.8,
                                linestyle="--", label="silence")

    spk_handles = []
    for s in active_indices:
        color = SPK_COLORS[s % len(SPK_COLORS)]
        ax.plot(t_slice, spk_prob[:, s], color=color, linewidth=0.9, alpha=0.8,
                zorder=2)
        spk_handles.append(mpatches.Patch(color=color, label=f"spk{s+1}"))

    if threshold is not None:
        ax.axhline(threshold, color="#555555", linewidth=0.6, linestyle="--", alpha=0.7)

    ax.legend(handles=[sil_handle] + spk_handles, fontsize=6, loc="upper right",
              ncol=1 + len(active_indices), framealpha=0.7, handlelength=1.2)


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_timeline(
    rec_id: str,
    t_start: float,
    t_end: float,
    rttm_dir: Path,
    sortformer_offline_dir: Path,
    sortformer_streaming_dir: Path,
    lseend_dir: Path,
    pyannote_dir: Path,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load all data (best-effort) ---
    gt_rttm = rttm_dir / f"{rec_id}.rttm"
    gt_segs = load_rttm(gt_rttm, t_start, t_end)
    gt_colors = _assign_gt_colors(gt_segs, rec_id)

    sf_offline_windows = load_sortformer_windows(
        sortformer_offline_dir / "prob_tensors", rec_id, t_start, t_end, frame_step=0.08
    )

    sf_streaming_windows = load_sortformer_windows(
        sortformer_streaming_dir / "prob_tensors", rec_id, t_start, t_end, frame_step=0.08
    )

    lseend_prob_path = lseend_dir / "prob_tensors" / f"{rec_id}.npy"
    lseend_meta_path = lseend_dir / "prob_tensors" / f"{rec_id}_meta.json"
    lseend_threshold = None
    if lseend_meta_path.exists():
        with open(lseend_meta_path) as f:
            meta = json.load(f)
            lseend_threshold = meta.get("threshold_used")

    pyannote_rttm = pyannote_dir / f"{rec_id}.rttm"
    pyannote_segs = load_rttm(pyannote_rttm, t_start, t_end)
    pyannote_colors = {spk: SPK_COLORS[i % len(SPK_COLORS)]
                       for i, spk in enumerate(sorted(pyannote_segs.keys()))}

    # --- Figure layout ---
    # GT and Pyannote heights scale with speaker count; prob-curve lanes are fixed
    n_gt_spk = max(len(gt_segs), 1)
    n_py_spk = max(len(pyannote_segs), 1)
    # height ratios proportional to sub-rows, min 1.5 inches per lane
    SUB_ROW_H = 0.55   # inches per sub-row
    CURVE_H   = 1.5    # inches for prob-curve lanes

    heights = [
        max(n_gt_spk * SUB_ROW_H, 1.2),  # GT
        CURVE_H,                           # Sortformer offline
        CURVE_H,                           # Sortformer streaming
        CURVE_H,                           # LS-EEND
        max(n_py_spk * SUB_ROW_H, 1.2),  # Pyannote
    ]
    fig_height = sum(heights) + 0.8  # top + bottom margin
    fig, axes = plt.subplots(
        5, 1,
        figsize=(16, fig_height),
        sharex=True,
        gridspec_kw={"height_ratios": heights, "hspace": 0.08},
    )

    lane_labels = [
        "GT",
        "Sortformer\n(offline)",
        "Sortformer\n(streaming)",
        "LS-EEND",
        "Pyannote 3.1",
    ]

    for i, ax in enumerate(axes):
        ax.set_facecolor(LANE_BG[i % len(LANE_BG)])
        ax.set_xlim(t_start, t_end)
        # Lane label on left side
        ax.set_ylabel(lane_labels[i], fontsize=9, fontweight="bold",
                      rotation=0, ha="right", va="center", labelpad=6)
        ax.tick_params(axis="y", which="both", length=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    # --- Lane 0: Ground Truth ---
    ax = axes[0]
    if gt_segs:
        _draw_rttm_bars(ax, gt_segs, gt_colors, t_start, t_end)
        # Legend: controller vs pilots
        ctrl_patch = mpatches.Patch(color=CONTROLLER_COLOR, label="Controller")
        pilot_patch = mpatches.Patch(color=PILOT_COLORS[0], label="Pilot(s)")
        ax.legend(handles=[ctrl_patch, pilot_patch], fontsize=7,
                  loc="upper right", framealpha=0.8)
    else:
        _draw_empty(ax)
    ax.set_title(
        f"{rec_id}  |  t = {t_start}–{t_end} s",
        fontsize=10, fontweight="bold", loc="left", pad=4,
    )

    # --- Lane 1: Sortformer offline ---
    ax = axes[1]
    _draw_prob_curves(ax, sf_offline_windows, t_start, t_end, frame_step=0.08,
                      threshold=0.5, alpha=0.65)

    # --- Lane 2: Sortformer streaming ---
    ax = axes[2]
    _draw_prob_curves(ax, sf_streaming_windows, t_start, t_end, frame_step=0.08,
                      threshold=0.5, alpha=0.65)

    # --- Lane 3: LS-EEND ---
    ax = axes[3]
    _draw_lseend_curves(ax, lseend_prob_path, t_start, t_end, frame_step=0.1,
                        threshold=None)

    # --- Lane 4: Pyannote ---
    ax = axes[4]
    if pyannote_segs:
        _draw_rttm_bars(ax, pyannote_segs, pyannote_colors, t_start, t_end)
    else:
        _draw_empty(ax)

    # --- Shared x-axis ---
    ax = axes[-1]
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.tick_params(axis="x", which="major", labelsize=8)
    ax.tick_params(axis="x", which="minor", length=3)

    fig.align_ylabels(axes)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout(rect=[0.07, 0, 1, 1])

    out_path = out_dir / f"{rec_id}_timeline_{int(t_start)}-{int(t_end)}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Timeline comparison plot for all diarization systems.")
    p.add_argument("--rec-id", default="dca_d1_1")
    p.add_argument("--t-start", type=float, default=55.0)
    p.add_argument("--t-end", type=float, default=150.0)
    p.add_argument("--rttm-dir", type=Path, default=Path("data/processed/rttm"))
    p.add_argument("--sortformer-offline-dir", type=Path, default=Path("results/sortformer_offline"))
    p.add_argument("--sortformer-streaming-dir", type=Path, default=Path("results/sortformer_streaming_10s"))
    p.add_argument("--lseend-dir", type=Path, default=Path("results/lseend"))
    p.add_argument("--pyannote-dir", type=Path, default=Path("results/pyannote/pred_rttm"))
    p.add_argument("--out-dir", type=Path, default=Path("results/plots"))
    args = p.parse_args()

    plot_timeline(
        rec_id=args.rec_id,
        t_start=args.t_start,
        t_end=args.t_end,
        rttm_dir=args.rttm_dir,
        sortformer_offline_dir=args.sortformer_offline_dir,
        sortformer_streaming_dir=args.sortformer_streaming_dir,
        lseend_dir=args.lseend_dir,
        pyannote_dir=args.pyannote_dir,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()

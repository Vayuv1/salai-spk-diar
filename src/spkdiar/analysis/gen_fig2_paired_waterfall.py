"""
gen_fig2_paired_waterfall.py

Fig 2 (paired waterfall): 3 consecutive 10s windows from dca_d2_2 (starting
at 2910 s, 2915 s, 2920 s), pretrained vs fine-tuned side-by-side.

Layout: 3 rows × 2 columns
  • Left  column: pretrained Sortformer
  • Right column: fine-tuned Sortformer
  Each cell contains:
    - thin GT speaker bar at the top (y ∈ [1.05, 1.35], broken_barh)
    - 4 speaker-slot probability curves (y ∈ [0, 1])
    - 0.5 threshold dashed line
  Shared x-axis within each row (10 s span, center-cropped 5 s displayed).
  Column titles (top) and row labels (left) are added.

Output: results/paper_figures/fig2_paired_waterfall.{pdf,png}

Usage:
    uv run python -m spkdiar.analysis.gen_fig2_paired_waterfall
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from spkdiar.analysis.ieee_style import (
    IEEE_DOUBLE_COL, apply_ieee_style, save_fig,
    ATC_STYLE, PILOT_STYLE, SPK_STYLES,
)

REC_ID     = "dca_d2_2"
WINDOW_DUR = 10.0
FRAME_STEP = 0.08
CENTER_PAD = 2.5
CROP_FRAME_START = int(CENTER_PAD / FRAME_STEP)          # 31
CROP_FRAME_END   = int((WINDOW_DUR - CENTER_PAD) / FRAME_STEP)  # 93

GT_RTTM_DIR   = Path("data/processed/rttm")
PRETRAIN_DIR  = Path("results/sortformer_offline/prob_tensors")
FINETUNE_DIR  = Path("results/sortformer_finetuned/prob_tensors")

# Three consecutive windows to display
WINDOW_STARTS_SEC = [2910.0, 2915.0, 2920.0]

# Controller heuristic: hyphen in bare speaker ID (after stripping rec_id prefix)
def is_controller(raw_id: str) -> bool:
    prefix = REC_ID + "_"
    bare   = raw_id[len(prefix):] if raw_id.startswith(prefix) else raw_id
    return "-" in bare


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gt_segments(rttm_path: Path, t_start: float, t_end: float) -> list:
    """Return [(start, end, bare_label, is_ctrl), ...]."""
    segs = []
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9 or parts[0] != "SPEAKER":
                continue
            seg_s = float(parts[3])
            seg_e = seg_s + float(parts[4])
            cs = max(seg_s, t_start)
            ce = min(seg_e, t_end)
            if ce <= cs:
                continue
            raw_id  = parts[7]
            bare    = raw_id.split("_", 1)[1] if "_" in raw_id else raw_id
            is_ctrl = is_controller(raw_id)
            segs.append((cs, ce, bare, is_ctrl))
    return segs


def load_prob_tensor(tensor_dir: Path, win_start_sec: float) -> np.ndarray | None:
    """Load center-cropped (62, 4) prob array for one window."""
    start_ms = int(round(win_start_sec * 1000))
    fname    = tensor_dir / f"{REC_ID}-{start_ms}-10000.npy"
    if not fname.exists():
        return None
    probs = np.load(str(fname))          # (125, 4)
    return probs[CROP_FRAME_START:CROP_FRAME_END]   # (62, 4)


def frame_times(win_start_sec: float) -> np.ndarray:
    """Absolute time array for the center-cropped frames."""
    return (
        win_start_sec + CENTER_PAD
        + np.arange(CROP_FRAME_END - CROP_FRAME_START) * FRAME_STEP
        + FRAME_STEP / 2
    )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(out_dir: Path) -> None:
    apply_ieee_style()

    gt_rttm = GT_RTTM_DIR / f"{REC_ID}.rttm"

    n_rows   = len(WINDOW_STARTS_SEC)
    n_cols   = 2
    col_labs = ["Pretrained", "Fine-tuned"]
    tensor_dirs = [PRETRAIN_DIR, FINETUNE_DIR]

    # Height ratios: each row = [GT_bar_height, prob_curve_height]
    # We flatten into one axis per cell and manage y-range ourselves.
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(IEEE_DOUBLE_COL, 5.0),
        gridspec_kw={"hspace": 0.35, "wspace": 0.08},
    )

    # Unique speakers in this segment (for GT color coding)
    all_gt_segs = load_gt_segments(gt_rttm, WINDOW_STARTS_SEC[0], WINDOW_STARTS_SEC[-1] + WINDOW_DUR)
    spk_order = []
    for _, _, bare, _ in all_gt_segs:
        if bare not in spk_order:
            spk_order.append(bare)

    # ---- Build one cell ----
    def draw_cell(ax: plt.Axes, row: int, col: int) -> None:
        win_s    = WINDOW_STARTS_SEC[row]
        win_e    = win_s + WINDOW_DUR
        # Center-crop display span
        disp_s   = win_s + CENTER_PAD
        disp_e   = win_e - CENTER_PAD
        t_arr    = frame_times(win_s)

        # ---- GT bars (shown at elevated y positions) ----
        # y band: [1.08, 1.30] in data coords; probability curves fill [0, 1]
        GT_Y_BOT = 1.10
        GT_HEIGHT = 0.18

        gt_segs = load_gt_segments(gt_rttm, disp_s, disp_e)
        seen = set()
        for cs, ce, bare, is_ctrl in gt_segs:
            style = ATC_STYLE if is_ctrl else PILOT_STYLE
            lbl   = None
            if bare not in seen and row == 0 and col == 0:
                lbl = f"{'CTR' if is_ctrl else ''}{bare}"
                seen.add(bare)
            ax.broken_barh(
                [(cs, ce - cs)], (GT_Y_BOT, GT_HEIGHT),
                facecolors=style["color"],
                hatch=style.get("hatch", ""),
                edgecolor="white" if style.get("hatch", "") else style["color"],
                linewidth=0.2,
            )
            # Label the speaker name inside the bar (left edge)
            ax.text(cs + 0.05, GT_Y_BOT + GT_HEIGHT / 2, bare,
                    va="center", ha="left", fontsize=5, color="white",
                    clip_on=True)

        # Divider line between GT and probs
        ax.axhline(1.07, color="#cccccc", linewidth=0.5, linestyle="-", zorder=1)

        # ---- Probability curves ----
        probs = load_prob_tensor(tensor_dirs[col], win_s)
        if probs is not None:
            for slot in range(4):
                st = SPK_STYLES[slot]
                ax.plot(
                    t_arr, probs[:, slot],
                    color=st["color"], linestyle=st["linestyle"],
                    linewidth=0.85, label=f"Slot {slot}",
                )

        # 0.5 threshold
        ax.axhline(0.5, color="#bbbbbb", linewidth=0.4, linestyle=":", zorder=0)

        # ---- Axes formatting ----
        ax.set_xlim(disp_s, disp_e)
        ax.set_ylim(-0.08, GT_Y_BOT + GT_HEIGHT + 0.05)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["0", "0.5", "1"], fontsize=6.5)

        # X-ticks: major every 1s, labels only on bottom row
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
        if row < n_rows - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (s)", fontsize=8)
            for lbl in ax.get_xticklabels():
                lbl.set_fontsize(7)

        # Y-label only on leftmost column
        if col == 0:
            win_s_int = int(win_s)
            ax.set_ylabel(f"{win_s_int}–{win_s_int+10} s", fontsize=7,
                          rotation=0, labelpad=36, va="center")

        # Suppress right/top spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Column title on top row
        if row == 0:
            ax.set_title(col_labs[col], fontsize=9, pad=4)

        # Slot legend on bottom-right cell only
        if row == n_rows - 1 and col == n_cols - 1:
            handles = [
                mpatches.Patch(facecolor=SPK_STYLES[s]["color"],
                               label=f"Slot {s}",
                               linestyle=SPK_STYLES[s]["linestyle"])
                for s in range(4)
            ]
            ax.legend(handles=handles, fontsize=6, loc="upper right",
                      framealpha=0.85, ncol=2, handlelength=1.2,
                      borderpad=0.4, columnspacing=0.6)

    # ---- Fill all cells ----
    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes[row][col]
            draw_cell(ax, row, col)

    # ---- GT legend (controller / pilot) ----
    ctrl_patch  = mpatches.Patch(facecolor=ATC_STYLE["color"],  hatch="//",
                                  edgecolor="white", label="Controller (ATC)")
    pilot_patch = mpatches.Patch(facecolor=PILOT_STYLE["color"], hatch="",
                                  edgecolor="white", label="Pilot")
    axes[0][0].legend(handles=[ctrl_patch, pilot_patch],
                      fontsize=6.5, loc="upper left", framealpha=0.85,
                      handlelength=1.0, borderpad=0.4)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout(pad=0.3)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, out_dir / "fig2_paired_waterfall")


if __name__ == "__main__":
    make_figure(Path("results/paper_figures"))

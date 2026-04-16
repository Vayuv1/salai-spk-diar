"""
gen_fig2_finetune_timeline.py

Fig 2: Before/after fine-tuning timeline comparison for dca_d2_2, t=2900–2960 s.

Three lanes (shared x-axis):
  Lane 1 — GT speaker bars (controller: orange+hatch, pilots: blue, no hatch)
  Lane 2 — Pretrained Sortformer prob curves (4 speaker slots, distinct linestyles)
  Lane 3 — Fine-tuned Sortformer prob curves

Prob tensors use center-5s crop (frames 31–93 of each 125-frame window) to
eliminate overlap jitter at window boundaries.

Output: results/paper_figures/fig2_finetune_comparison.{pdf,png}

Usage:
    uv run python -m spkdiar.analysis.gen_fig2_finetune_timeline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from spkdiar.analysis.ieee_style import (
    IEEE_DOUBLE_COL, apply_ieee_style, save_fig,
    ATC_STYLE, PILOT_STYLE, SPK_STYLES,
)

REC_ID     = "dca_d2_2"
T_START    = 2900.0
T_END      = 2960.0
WINDOW_DUR = 10.0
FRAME_STEP = 0.08   # seconds per Sortformer frame
CENTER_PAD = 2.5    # center-crop: keep [start+2.5, start+7.5]
# Frames covering [2.5s, 7.5s] within a 10s window at 80ms resolution
CROP_FRAME_START = int(CENTER_PAD / FRAME_STEP)     # 31
CROP_FRAME_END   = int((WINDOW_DUR - CENTER_PAD) / FRAME_STEP)  # 93


# ---------------------------------------------------------------------------
# RTTM parsing
# ---------------------------------------------------------------------------

def load_gt_segments(rttm_path: Path, t_start: float, t_end: float):
    """Return list of (start, end, speaker_id, is_controller)."""
    segments = []
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9 or parts[0] != "SPEAKER":
                continue
            seg_s  = float(parts[3])
            seg_e  = seg_s + float(parts[4])
            raw_id = parts[7]
            # Clip to display window
            cs = max(seg_s, t_start)
            ce = min(seg_e, t_end)
            if ce <= cs:
                continue
            # Strip rec_id prefix to get bare speaker label
            bare = raw_id.split("_", 1)[1] if "_" in raw_id else raw_id
            is_ctrl = "-" in bare
            segments.append((cs, ce, bare, is_ctrl))
    return segments


# ---------------------------------------------------------------------------
# Prob tensor stitching
# ---------------------------------------------------------------------------

def stitch_prob_tensors(
    tensor_dir: Path,
    rec_id: str,
    t_start: float,
    t_end: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (times, probs) arrays from center-cropped window tensors.

    times : (N,)  absolute time of each frame midpoint
    probs : (N, 4) speaker probability for each of 4 slots
    """
    times_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []

    # Enumerate all 5s-shifted windows whose start falls in [t_start, t_end)
    win_start = t_start
    while win_start < t_end:
        start_ms = int(round(win_start * 1000))
        fname    = tensor_dir / f"{rec_id}-{start_ms}-10000.npy"
        if fname.exists():
            probs = np.load(str(fname))   # (125, 4)
            crop  = probs[CROP_FRAME_START:CROP_FRAME_END]  # (62, 4)
            # Absolute time for each cropped frame midpoint
            frame_times = (
                win_start + CENTER_PAD
                + np.arange(len(crop)) * FRAME_STEP
                + FRAME_STEP / 2
            )
            times_list.append(frame_times)
            probs_list.append(crop)
        win_start += 5.0  # 5s shift between windows

    if not times_list:
        return np.array([]), np.zeros((0, 4))

    return np.concatenate(times_list), np.concatenate(probs_list, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    gt_rttm_dir:  Path = Path("data/processed/rttm"),
    pretrain_dir: Path = Path("results/sortformer_offline/prob_tensors"),
    finetune_dir: Path = Path("results/sortformer_finetuned/prob_tensors"),
    out_dir:      Path = Path("results/paper_figures"),
) -> None:
    apply_ieee_style()

    gt_rttm = gt_rttm_dir / f"{REC_ID}.rttm"
    gt_segs = load_gt_segments(gt_rttm, T_START, T_END)

    t_pre, p_pre = stitch_prob_tensors(pretrain_dir, REC_ID, T_START, T_END)
    t_ft,  p_ft  = stitch_prob_tensors(finetune_dir, REC_ID, T_START, T_END)

    fig, axes = plt.subplots(
        3, 1,
        figsize=(IEEE_DOUBLE_COL, 4.5),
        sharex=True,
        gridspec_kw={"hspace": 0.08, "height_ratios": [1.0, 1.5, 1.5]},
    )

    # ---- Lane 1: GT ----
    ax_gt = axes[0]
    seen_ctrl  = False
    seen_pilot = False
    yticks_gt: list[tuple[float, str]] = []
    y_pos = 0.0

    # Collect unique speakers in display order
    spk_order: list[str] = []
    for _, _, bare, _ in gt_segs:
        if bare not in spk_order:
            spk_order.append(bare)

    for spk in spk_order:
        y_pos_spk = y_pos
        for cs, ce, bare, is_ctrl in gt_segs:
            if bare != spk:
                continue
            style = ATC_STYLE if is_ctrl else PILOT_STYLE
            label = "Controller" if (is_ctrl and not seen_ctrl) else (
                    "Pilot"      if (not is_ctrl and not seen_pilot) else None)
            if is_ctrl:
                seen_ctrl = True
            else:
                seen_pilot = True
            ax_gt.broken_barh(
                [(cs, ce - cs)], (y_pos_spk, 0.7),
                facecolors=style["color"],
                hatch=style["hatch"],
                edgecolor="white",
                linewidth=0.3,
                label=label,
            )
        yticks_gt.append((y_pos_spk + 0.35, spk))
        y_pos += 0.85

    ax_gt.set_ylim(-0.1, y_pos)
    ax_gt.set_yticks([t for t, _ in yticks_gt])
    ax_gt.set_yticklabels([s for _, s in yticks_gt], fontsize=5.5)
    ax_gt.set_ylabel("GT", fontsize=8, rotation=0, labelpad=22, va="center")
    ax_gt.legend(fontsize=7, loc="upper right", framealpha=0.85,
                 ncol=2, handlelength=1.0, borderpad=0.4)

    # ---- Lane 2: Pretrained ----
    ax_pre = axes[1]
    if len(t_pre):
        for slot in range(4):
            st = SPK_STYLES[slot]
            ax_pre.plot(
                t_pre, p_pre[:, slot],
                color=st["color"], linestyle=st["linestyle"],
                linewidth=0.9, label=f"Slot {slot}",
            )
    ax_pre.set_ylim(-0.05, 1.10)
    ax_pre.set_yticks([0, 0.5, 1.0])
    ax_pre.set_yticklabels(["0", "0.5", "1"], fontsize=7)
    ax_pre.set_ylabel("Pretrained\nProb.", fontsize=8, rotation=0, labelpad=30, va="center")
    ax_pre.legend(fontsize=7, loc="upper right", framealpha=0.85,
                  ncol=4, handlelength=1.2, borderpad=0.4, columnspacing=0.7)
    ax_pre.axhline(0.5, color="#aaaaaa", linewidth=0.4, linestyle=":", zorder=1)

    # ---- Lane 3: Fine-tuned ----
    ax_ft = axes[2]
    if len(t_ft):
        for slot in range(4):
            st = SPK_STYLES[slot]
            ax_ft.plot(
                t_ft, p_ft[:, slot],
                color=st["color"], linestyle=st["linestyle"],
                linewidth=0.9, label=f"Slot {slot}",
            )
    ax_ft.set_ylim(-0.05, 1.10)
    ax_ft.set_yticks([0, 0.5, 1.0])
    ax_ft.set_yticklabels(["0", "0.5", "1"], fontsize=7)
    ax_ft.set_ylabel("Fine-tuned\nProb.", fontsize=8, rotation=0, labelpad=30, va="center")
    ax_ft.legend(fontsize=7, loc="upper right", framealpha=0.85,
                 ncol=4, handlelength=1.2, borderpad=0.4, columnspacing=0.7)
    ax_ft.axhline(0.5, color="#aaaaaa", linewidth=0.4, linestyle=":", zorder=1)

    ax_ft.set_xlabel("Time (s)", fontsize=9)
    ax_ft.set_xlim(T_START, T_END)
    ax_ft.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax_ft.xaxis.set_minor_locator(plt.MultipleLocator(5))

    # Shared spine cleanup
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout(pad=0.3)

    save_fig(fig, out_dir / "fig2_finetune_comparison")


if __name__ == "__main__":
    main()

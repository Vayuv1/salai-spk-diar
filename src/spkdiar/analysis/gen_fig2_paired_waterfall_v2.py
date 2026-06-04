"""
gen_fig2_paired_waterfall_v2.py

Fig 2 (paired waterfall, v2): Best 3 consecutive 10-second windows from
dca_d2_2 that maximise the pretrained-minus-finetuned CER gap.
Selected windows: 2935 s, 2940 s, 2945 s
  - 2935 s: Pre CER = 0.519,  FT CER = 0.063  (Δ = +0.456) ← highest-gap window
  - 2940 s: Pre CER = 0.000,  FT CER = 0.000  (Δ =  0.000) — 2-speaker exchange
  - 2945 s: Pre CER = 0.000,  FT CER = 0.000  (Δ =  0.000) — controller dominates

Layout: 3 rows × 2 columns
  Left column  = pretrained Sortformer
  Right column = fine-tuned Sortformer
  Each cell:
    • GT speaker bar at top (20% lane height, broken_barh; controller orange+hatch,
      pilot solid blue); speaker label inside bar
    • 4 slot probability curves (linewidth=1.5, distinct linestyle per slot)
    • 0.5 threshold dashed line
    • CER annotation in upper-left corner of prob area

Shared x-axis per row (center 5 s of each 10 s window displayed).
Column titles top; row labels left; per-panel CER in corner.

Output: results/paper_figures/fig2_paired_waterfall.{pdf,png}

Usage:
    uv run python -m spkdiar.analysis.gen_fig2_paired_waterfall_v2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

from spkdiar.analysis.ieee_style import (
    IEEE_DOUBLE_COL, apply_ieee_style, save_fig,
    ATC_STYLE, PILOT_STYLE, SPK_STYLES,
)

REC_ID     = "dca_d2_2"
WINDOW_DUR = 10.0
FRAME_STEP = 0.08
CENTER_PAD = 2.5
CROP_S     = int(CENTER_PAD / FRAME_STEP)
CROP_E     = int((WINDOW_DUR - CENTER_PAD) / FRAME_STEP)

COLLAR     = 0.25

# Best 3 consecutive windows (pretrained−finetuned CER gap: +0.46, 0, 0)
WINDOW_STARTS_SEC = [2935.0, 2940.0, 2945.0]

GT_RTTM_DIR = Path("data/processed/rttm")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_controller(raw_id: str) -> bool:
    prefix = REC_ID + "_"
    bare   = raw_id[len(prefix):] if raw_id.startswith(prefix) else raw_id
    return "-" in bare


def rttm_to_ann(path: Path) -> Annotation:
    ann = Annotation()
    if not path.exists():
        return ann
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 9 or p[0] != "SPEAKER":
                continue
            s = float(p[3]); e = s + float(p[4])
            ann[Segment(s, e)] = p[7]
    return ann


def clip_ann(ann: Annotation, t0: float, t1: float) -> Annotation:
    out = Annotation()
    for seg, _, spk in ann.itertracks(yield_label=True):
        cs, ce = max(seg.start, t0), min(seg.end, t1)
        if ce > cs:
            out[Segment(cs, ce)] = spk
    return out


def load_gt_segments(rttm_path: Path, t_start: float, t_end: float) -> list:
    segs = []
    with open(rttm_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 9 or p[0] != "SPEAKER":
                continue
            seg_s = float(p[3]); seg_e = seg_s + float(p[4])
            cs = max(seg_s, t_start); ce = min(seg_e, t_end)
            if ce <= cs:
                continue
            raw_id  = p[7]
            bare    = raw_id.split("_", 1)[1] if "_" in raw_id else raw_id
            segs.append((cs, ce, bare, is_controller(raw_id)))
    return segs


def load_prob_crop(tensor_dir: Path, win_start_sec: float) -> np.ndarray | None:
    start_ms = int(round(win_start_sec * 1000))
    f = tensor_dir / f"{REC_ID}-{start_ms}-10000.npy"
    if not f.exists():
        return None
    return np.load(str(f))[CROP_S:CROP_E]   # (62, 4)


def frame_times(win_start_sec: float) -> np.ndarray:
    return (
        win_start_sec + CENTER_PAD
        + np.arange(CROP_E - CROP_S) * FRAME_STEP
        + FRAME_STEP / 2
    )


def compute_window_cer(gt_ann: Annotation, pred_rttm: Path,
                       win_start_sec: float) -> float:
    cs = win_start_sec + CENTER_PAD
    ce = win_start_sec + WINDOW_DUR - CENTER_PAD
    ref = clip_ann(gt_ann, cs, ce)
    hyp = clip_ann(rttm_to_ann(pred_rttm), cs, ce)
    total = sum(s.end - s.start for s, _ in ref.itertracks())
    if total <= 0:
        return float("nan")
    uem    = Timeline([Segment(cs, ce)])
    metric = DiarizationErrorRate(collar=COLLAR, skip_overlap=False)
    r      = metric(ref, hyp, uem=uem, detailed=True)
    return r["confusion"] / r["total"] if r["total"] > 0 else 0.0


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paired waterfall figure.")
    parser.add_argument(
        "--pretrained-prob-dir",
        type=Path,
        default=Path("results/repro/sortformer_pretrained_eval4_rerun/prob_tensors"),
    )
    parser.add_argument(
        "--finetuned-prob-dir",
        type=Path,
        default=Path("results/repro/sortformer_finetuned_eval4_rerun/prob_tensors"),
    )
    parser.add_argument(
        "--pretrained-rttm-dir",
        type=Path,
        default=Path("results/repro/sortformer_pretrained_eval4_rerun/pred_rttm"),
    )
    parser.add_argument(
        "--finetuned-rttm-dir",
        type=Path,
        default=Path("results/repro/sortformer_finetuned_eval4_rerun/pred_rttm"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/paper_figures"),
    )
    return parser.parse_args()


def make_figure(
    out_dir: Path,
    pretrained_prob_dir: Path,
    finetuned_prob_dir: Path,
    pretrained_rttm_dir: Path,
    finetuned_rttm_dir: Path,
) -> None:
    apply_ieee_style()

    gt_rttm = GT_RTTM_DIR / f"{REC_ID}.rttm"
    gt_ann  = rttm_to_ann(gt_rttm)

    n_rows = len(WINDOW_STARTS_SEC)
    n_cols = 2
    col_labels  = ["Pretrained", "Fine-tuned"]
    tensor_dirs = [pretrained_prob_dir, finetuned_prob_dir]
    col_keys    = ["pretrained", "finetuned"]
    pred_dirs   = {
        "pretrained": pretrained_rttm_dir,
        "finetuned": finetuned_rttm_dir,
    }

    # Pre-compute CER for all cells
    cer_table: dict[tuple, float] = {}
    for row, win_s in enumerate(WINDOW_STARTS_SEC):
        start_ms = int(round(win_s * 1000))
        for col, key in enumerate(col_keys):
            pred_f = pred_dirs[key] / f"{REC_ID}-{start_ms}-10000.rttm"
            cer_table[(row, col)] = compute_window_cer(gt_ann, pred_f, win_s)

    # Height ratios: GT bar = 1 unit, prob lane = 4 units per row
    height_ratios = []
    for _ in range(n_rows):
        height_ratios.extend([1, 4])

    fig = plt.figure(figsize=(IEEE_DOUBLE_COL, 4.5))
    gs  = fig.add_gridspec(
        n_rows * 2, n_cols,
        hspace=0.0,
        wspace=0.08,
        height_ratios=height_ratios,
    )

    # Helper: draw one GT lane (spans both columns at a given gridspec row)
    def draw_gt_lane(gt_row: int, win_s: float) -> None:
        ax_gt = fig.add_subplot(gs[gt_row, :])   # span both columns
        disp_s = win_s + CENTER_PAD
        disp_e = win_s + WINDOW_DUR - CENTER_PAD
        segs   = load_gt_segments(gt_rttm, disp_s, disp_e)
        seen_ctrl = seen_pilot = False
        for cs, ce, bare, is_ctrl in segs:
            style = ATC_STYLE if is_ctrl else PILOT_STYLE
            lbl   = None
            if is_ctrl and not seen_ctrl:
                lbl = "Controller"; seen_ctrl = True
            elif not is_ctrl and not seen_pilot:
                lbl = "Pilot"; seen_pilot = True
            ax_gt.broken_barh(
                [(cs, ce - cs)], (0.1, 0.8),
                facecolors=style["color"],
                hatch=style.get("hatch", ""),
                edgecolor="white",
                linewidth=0.3,
                label=lbl,
            )
            ax_gt.text(cs + 0.05, 0.5, bare, va="center", ha="left",
                       fontsize=5.5, color="white", clip_on=True)
        ax_gt.set_xlim(disp_s, disp_e)
        ax_gt.set_ylim(0, 1)
        ax_gt.axis("off")
        # Small "GT" label on the left
        ax_gt.text(disp_s - 0.25, 0.5, "GT", va="center", ha="right",
                   fontsize=6, color="#444444", transform=ax_gt.transData)
        # GT legend on first row only
        if gt_row == 0:
            ctrl_p  = mpatches.Patch(facecolor=ATC_STYLE["color"],  hatch="//",
                                     edgecolor="white", linewidth=0.3, label="Controller")
            pilot_p = mpatches.Patch(facecolor=PILOT_STYLE["color"], hatch="",
                                     edgecolor="white", linewidth=0.3, label="Pilot")
            ax_gt.legend(handles=[ctrl_p, pilot_p], fontsize=6.5,
                         loc="upper right", framealpha=0.9,
                         ncol=2, handlelength=1.0, borderpad=0.4,
                         columnspacing=0.6)

    # Helper: draw one prob lane
    def draw_prob_lane(prob_row: int, col: int, win_s: float) -> None:
        ax = fig.add_subplot(gs[prob_row, col])
        disp_s = win_s + CENTER_PAD
        disp_e = win_s + WINDOW_DUR - CENTER_PAD
        t_arr  = frame_times(win_s)

        probs = load_prob_crop(tensor_dirs[col], win_s)
        if probs is not None:
            for slot in range(4):
                st = SPK_STYLES[slot]
                ax.plot(t_arr, probs[:, slot],
                        color=st["color"], linestyle=st["linestyle"],
                        linewidth=1.5, zorder=3)

        ax.axhline(0.5, color="#bbbbbb", linewidth=0.5, linestyle=":", zorder=1)

        # CER annotation
        cer_val = cer_table[(prob_row // 2, col)]
        if not np.isnan(cer_val):
            cer_str = f"CER={cer_val:.2f}"
            bg_col  = "#ffe0e0" if cer_val > 0.1 else "#e0ffe0"
            ax.text(0.02, 0.96, cer_str, transform=ax.transAxes,
                    ha="left", va="top", fontsize=7,
                    color="#222222",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=bg_col,
                              edgecolor="none", alpha=0.9))

        ax.set_xlim(disp_s, disp_e)
        ax.set_ylim(-0.05, 1.10)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["0", "", "1"], fontsize=7)

        row_idx = prob_row // 2
        # X-axis: ticks only on bottom row
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
        if row_idx < n_rows - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (s)", fontsize=8)
            for lbl in ax.get_xticklabels():
                lbl.set_fontsize(7)

        # Y-label: row window range on leftmost column
        if col == 0:
            win_s_int = int(win_s)
            ax.set_ylabel(f"{win_s_int}–{win_s_int+10} s",
                          fontsize=7, rotation=0, labelpad=40, va="center")

        # Column title on top row
        if row_idx == 0:
            ax.set_title(col_labels[col], fontsize=9, pad=3)

        # Slot legend — bottom-right cell only
        if row_idx == n_rows - 1 and col == n_cols - 1:
            handles = [
                mpatches.Patch(
                    facecolor=SPK_STYLES[s]["color"],
                    label=f"Slot {s}",
                )
                for s in range(4)
            ]
            ax.legend(handles=handles, fontsize=6, loc="lower right",
                      framealpha=0.85, ncol=2, handlelength=0.9,
                      borderpad=0.4, columnspacing=0.6)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ---- Build all lanes ----
    for row, win_s in enumerate(WINDOW_STARTS_SEC):
        gt_row   = row * 2
        prob_row = row * 2 + 1
        draw_gt_lane(gt_row, win_s)
        for col in range(n_cols):
            draw_prob_lane(prob_row, col, win_s)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout(pad=0.3, h_pad=0.1, w_pad=0.08)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, out_dir / "fig2_paired_waterfall")


if __name__ == "__main__":
    args = parse_args()
    make_figure(
        out_dir=args.out_dir,
        pretrained_prob_dir=args.pretrained_prob_dir,
        finetuned_prob_dir=args.finetuned_prob_dir,
        pretrained_rttm_dir=args.pretrained_rttm_dir,
        finetuned_rttm_dir=args.finetuned_rttm_dir,
    )

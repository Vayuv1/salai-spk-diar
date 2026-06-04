"""
gen_fig2_rttm_comparison.py

Fig 2: RTTM comparison timeline — dca_d2_2, 2900–2960 s.
Three horizontal lanes on a shared time axis:
  Lane 1 — Ground Truth  (multi-speaker broken_barh; controller orange+hatch,
            pilots in blue shades; one sub-row per speaker)
  Lane 2 — Pretrained Sortformer  (stitched from per-window RTTMs, center-5s crop;
            one sub-row per predicted slot)
  Lane 3 — Fine-tuned Sortformer  (same)

Predicted RTTM files are per-window with independent speaker_0/1/2/3 labels.
Stitching applies center-crop [start+2.5, start+7.5] s per window, producing a
continuous binary diarization timeline. Speaker-slot labels are kept as-is
(speaker_0 = slot 0, etc.) so the lock-up pattern is directly visible:
pretrained collapses to speaker_0 for most windows; finetuned shows
alternation between speaker_0 and speaker_1.

Each speaker/slot is assigned a distinct color AND hatch pattern for
grayscale-safe printing.

Output: results/paper_figures/fig2_rttm_comparison.{pdf,png}

Usage:
    uv run python -m spkdiar.analysis.gen_fig2_rttm_comparison
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from spkdiar.analysis.ieee_style import IEEE_DOUBLE_COL, apply_ieee_style, save_fig

REC_ID     = "dca_d2_2"
T_START    = 2900.0
T_END      = 2960.0
WINDOW_DUR = 10.0
CENTER_PAD = 2.5

GT_RTTM    = Path("data/processed/rttm/dca_d2_2.rttm")
PRE_DIR    = Path("results/sortformer_offline/pred_rttm")
FT_DIR     = Path("results/sortformer_finetuned/pred_rttm")

# Controller identification: hyphen in speaker ID after stripping rec_id prefix
def is_controller(raw_id: str) -> bool:
    prefix = REC_ID + "_"
    bare   = raw_id[len(prefix):] if raw_id.startswith(prefix) else raw_id
    return "-" in bare


# ---------------------------------------------------------------------------
# RTTM parsing
# ---------------------------------------------------------------------------

def parse_rttm(path: Path) -> list[tuple[float, float, str]]:
    """Return [(start, end, speaker), ...] sorted by start."""
    segs = []
    if not path.exists():
        return segs
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 9 or p[0] != "SPEAKER":
                continue
            s = float(p[3]); e = s + float(p[4])
            segs.append((s, e, p[7]))
    segs.sort()
    return segs


def stitch_pred_rttm(pred_dir: Path, rec_id: str,
                     t_start: float, t_end: float) -> list[tuple[float, float, str]]:
    """Center-crop each per-window RTTM and return stitched segments in [t_start,t_end]."""
    segs = []
    win = t_start
    while win < t_end:
        ms    = int(round(win * 1000))
        fname = pred_dir / f"{rec_id}-{ms}-10000.rttm"
        cs    = win + CENTER_PAD
        ce    = win + WINDOW_DUR - CENTER_PAD
        for s, e, spk in parse_rttm(fname):
            cs2 = max(s, cs); ce2 = min(e, ce)
            if ce2 > cs2:
                segs.append((cs2, ce2, spk))
        win += 5.0
    segs.sort()
    return segs


def clip_segs(segs: list, t0: float, t1: float) -> list:
    out = []
    for s, e, spk in segs:
        cs, ce = max(s, t0), min(e, t1)
        if ce > cs:
            out.append((cs, ce, spk))
    return out


# ---------------------------------------------------------------------------
# Style assignment
# ---------------------------------------------------------------------------

# Ground-truth speakers: controller gets ATC orange+hatch, each pilot gets a
# distinct blue shade with its own hatch so the plot is grayscale-safe.
CTRL_COLOR = "#E8702A"
CTRL_HATCH = "//"

# Pilot palette: progressively darker blues, distinct hatches
PILOT_COLORS = ["#4C72B0", "#2171B5", "#08519C", "#08306B", "#41B6C4", "#1D91C0"]
PILOT_HATCHES = ["", "\\\\", "..", "xx", "||", "--"]

# Predicted slot colors and hatches (4 slots; grayscale-safe)
SLOT_COLORS  = ["#E8702A", "#4C72B0", "#2CA02C", "#9467BD"]
SLOT_HATCHES = ["//",      "",        "..",      "xx"     ]


def assign_gt_styles(segs: list) -> dict[str, dict]:
    """Assign color/hatch to each unique GT speaker."""
    speakers = list(dict.fromkeys(spk for _, _, spk in segs))
    styles   = {}
    pilot_idx = 0
    for spk in speakers:
        if is_controller(spk):
            styles[spk] = {"color": CTRL_COLOR, "hatch": CTRL_HATCH,
                           "label": spk.replace(f"{REC_ID}_", "")}
        else:
            idx = pilot_idx % len(PILOT_COLORS)
            styles[spk] = {"color": PILOT_COLORS[idx], "hatch": PILOT_HATCHES[idx],
                           "label": spk.replace(f"{REC_ID}_", "")}
            pilot_idx += 1
    return styles


def assign_slot_styles() -> dict[str, dict]:
    return {
        f"speaker_{i}": {"color": SLOT_COLORS[i], "hatch": SLOT_HATCHES[i],
                          "label": f"Slot {i}"}
        for i in range(4)
    }


# ---------------------------------------------------------------------------
# Lane drawing helper
# ---------------------------------------------------------------------------

BAR_HEIGHT = 0.65
BAR_PAD    = 0.10

def draw_lane(
    ax: plt.Axes,
    segs: list[tuple[float, float, str]],
    styles: dict[str, dict],
    spk_order: list[str],
) -> float:
    """Draw broken_barh segments into ax.

    Returns total lane height (y extent from 0).
    Each speaker in spk_order occupies its own sub-row.
    """
    y = 0.0
    seen_labels: set[str] = set()

    for spk in spk_order:
        st = styles.get(spk, {"color": "#888888", "hatch": "", "label": spk})
        lbl_text = st["label"]
        lbl      = lbl_text if lbl_text not in seen_labels else None
        if lbl:
            seen_labels.add(lbl_text)

        for s, e, sp in segs:
            if sp != spk:
                continue
            ax.broken_barh(
                [(s, e - s)], (y, BAR_HEIGHT),
                facecolors=st["color"],
                hatch=st["hatch"],
                edgecolor="white",
                linewidth=0.25,
                label=lbl,
            )
            lbl = None  # only label the first bar

        # Compact speaker name inside first bar of this sub-row
        first = next(((s, e) for s, e, sp in segs if sp == spk), None)
        if first:
            mid = (first[0] + first[1]) / 2
            ax.text(
                min(mid, T_END - 0.5), y + BAR_HEIGHT / 2,
                lbl_text, va="center", ha="center",
                fontsize=5, color="white", clip_on=True,
            )

        y += BAR_HEIGHT + BAR_PAD

    return y


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def make_figure(out_dir: Path) -> None:
    apply_ieee_style()

    # ---- Load data ----
    gt_segs  = clip_segs(parse_rttm(GT_RTTM),              T_START, T_END)
    pre_segs = stitch_pred_rttm(PRE_DIR, REC_ID, T_START, T_END)
    ft_segs  = stitch_pred_rttm(FT_DIR,  REC_ID, T_START, T_END)

    # Speaker orders (appearance order)
    gt_spk_order  = list(dict.fromkeys(spk for _, _, spk in gt_segs))
    slot_order    = [f"speaker_{i}" for i in range(4)]
    pre_slots     = [s for s in slot_order if any(sp == s for _, _, sp in pre_segs)]
    ft_slots      = [s for s in slot_order if any(sp == s for _, _, sp in ft_segs)]

    gt_styles   = assign_gt_styles(gt_segs)
    slot_styles = assign_slot_styles()

    # ---- Compute lane heights ----
    n_gt_rows  = len(gt_spk_order)
    n_pre_rows = len(pre_slots)
    n_ft_rows  = len(ft_slots)

    gt_height  = n_gt_rows  * (BAR_HEIGHT + BAR_PAD)
    pre_height = n_pre_rows * (BAR_HEIGHT + BAR_PAD)
    ft_height  = n_ft_rows  * (BAR_HEIGHT + BAR_PAD)

    # Inter-lane gap
    LANE_GAP = 0.35

    # ---- Create figure with height ratios proportional to lane heights ----
    fig, axes = plt.subplots(
        3, 1,
        figsize=(IEEE_DOUBLE_COL, 3.5),
        sharex=True,
        gridspec_kw={
            "hspace": 0.12,
            "height_ratios": [gt_height, pre_height, ft_height],
        },
    )

    ax_gt, ax_pre, ax_ft = axes

    # ---- Draw lanes ----
    draw_lane(ax_gt,  gt_segs,  gt_styles,   gt_spk_order)
    draw_lane(ax_pre, pre_segs, slot_styles, pre_slots)
    draw_lane(ax_ft,  ft_segs,  slot_styles, ft_slots)

    # ---- Formatting ----
    for ax, label, height in [
        (ax_gt,  "Ground\nTruth",   gt_height),
        (ax_pre, "Pretrained",      pre_height),
        (ax_ft,  "Fine-tuned",      ft_height),
    ]:
        ax.set_xlim(T_START, T_END)
        ax.set_ylim(-0.05, height + 0.05)
        ax.set_yticks([])
        ax.set_ylabel(label, fontsize=8, rotation=0, labelpad=42, va="center")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)

    ax_ft.set_xlabel("Time (s)", fontsize=9)
    ax_ft.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax_ft.xaxis.set_minor_locator(plt.MultipleLocator(5))
    for lbl in ax_ft.get_xticklabels():
        lbl.set_fontsize(8)

    # ---- Legends ----
    # GT: controller + pilot type (not every speaker)
    ctrl_patch  = mpatches.Patch(facecolor=CTRL_COLOR, hatch="//",
                                 edgecolor="white", linewidth=0.3, label="Controller")
    pilot_patch = mpatches.Patch(facecolor=PILOT_COLORS[0], hatch="",
                                 edgecolor="white", linewidth=0.3, label="Pilot")
    ax_gt.legend(handles=[ctrl_patch, pilot_patch], fontsize=7,
                 loc="upper right", framealpha=0.9,
                 ncol=2, handlelength=1.0, borderpad=0.4)

    # Pred: slot colors (same for both pred lanes, put on ft lane)
    slot_handles = [
        mpatches.Patch(facecolor=SLOT_COLORS[i], hatch=SLOT_HATCHES[i],
                       edgecolor="white", linewidth=0.3, label=f"Slot {i}")
        for i in range(4)
    ]
    ax_ft.legend(handles=slot_handles, fontsize=7, loc="upper right",
                 framealpha=0.9, ncol=4, handlelength=0.9,
                 borderpad=0.4, columnspacing=0.6)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout(pad=0.3)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, out_dir / "fig2_rttm_comparison")


if __name__ == "__main__":
    make_figure(Path("results/paper_figures"))

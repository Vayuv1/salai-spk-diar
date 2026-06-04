"""
gen_fig2_lockup_comparison_v4.py

Fig 2 (v4): Speaker lock-up comparison — dca_d1_1, 105–120 s.
Single column (3.5 × 4.0 in), 600 DPI.

Window rationale
----------------
At 105–120 s, dca_d1_1 contains the clearest single-window lock-up event:
  • D1-1 (controller) speaks at 106.85–109.54 s → both models correctly track on Slot 0.
  • DAL209 (pilot) speaks at 109.99–112.13 s:
      - Pretrained: Slot 0 snaps back to 1.0 (lock-up — assigns the new pilot
        to the same slot as the controller).
      - Fine-tuned: Slot 0 drops to ~0.002, Slot 1 rises to ~0.8 (correct
        speaker change detected).
  • 112–120 s is clean silence: both models correctly output near-zero.

Only Slot 0 (solid) and Slot 1 (dashed) are shown — Slots 2/3 stay below 0.12
throughout this window and would clutter the display.

Slot legend placed in the lower-left of each prob panel (silence region, 112–120 s,
curves at 0) so it never obscures any active curve.

Three panels, shared x-axis, height ratio 2.5 : 3.5 : 3.5:
  Panel 1 (GT)         — broken_barh, one sub-row per speaker.
                          Controller D1-1: orange, diagonal hatch.
                          Pilot DAL209: blue, no hatch.
  Panel 2 (Pretrained) — Slot 0 solid orange lw=1.5, Slot 1 dashed blue lw=1.5.
  Panel 3 (Fine-tuned) — same format.

Output: results/paper_figures/fig2_lockup_comparison_v4.{pdf,png}

Usage:
    uv run python -m spkdiar.analysis.gen_fig2_lockup_comparison_v4
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from spkdiar.analysis.ieee_style import IEEE_SINGLE_COL, apply_ieee_style, save_fig

REC_ID     = "dca_d1_1"
DISP_START = 105.0
DISP_END   = 120.0
WINDOW_DUR = 10.0
FRAME_STEP = 0.08
CENTER_PAD = 2.5
CROP_S     = int(CENTER_PAD / FRAME_STEP)                    # 31
CROP_E     = int((WINDOW_DUR - CENTER_PAD) / FRAME_STEP)     # 93

GT_RTTM  = Path("data/processed/rttm/dca_d1_1.rttm")
PRE_DIR  = Path("results/sortformer_offline/prob_tensors")
FT_DIR   = Path("results/sortformer_finetuned/prob_tensors")

# Windows whose center-crops cover [105, 120]:
#   win 100 → crop [102.5, 107.5] → visible [105.0, 107.5]
#   win 105 → crop [107.5, 112.5] → full  ← KEY window
#   win 110 → crop [112.5, 117.5] → full  (clean silence)
#   win 115 → crop [117.5, 122.5] → visible [117.5, 120.0]
WIN_STARTS = [100.0, 105.0, 110.0, 115.0]

# Only two slots are informative in this window
N_SLOTS = 2
SLOT_STYLES = [
    {"color": "#E8702A", "linestyle": "-",  "label": "Slot 0 (controller)"},
    {"color": "#4C72B0", "linestyle": "--", "label": "Slot 1 (pilot)"},
]

CTRL_STYLE  = {"color": "#E8702A", "hatch": "//", "lw": 0.3}
PILOT_STYLE = {"color": "#4C72B0", "hatch": "",   "lw": 0.3}

BAR_H   = 0.65
BAR_PAD = 0.15


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def is_controller(raw_id: str) -> bool:
    prefix = REC_ID + "_"
    bare   = raw_id[len(prefix):] if raw_id.startswith(prefix) else raw_id
    return "-" in bare


def load_gt_segments(t0: float, t1: float) -> list[tuple]:
    prefix = REC_ID + "_"
    segs   = []
    with open(GT_RTTM) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 9 or p[0] != "SPEAKER":
                continue
            s = float(p[3]); e = s + float(p[4])
            cs, ce = max(s, t0), min(e, t1)
            if ce <= cs:
                continue
            raw  = p[7]
            bare = raw[len(prefix):] if raw.startswith(prefix) else raw
            segs.append((cs, ce, bare, is_controller(raw)))
    return segs


def build_prob_curve(tensor_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Stitch center-cropped frames from WIN_STARTS; return (times, probs[:, :N_SLOTS])."""
    times_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []

    for win_s in WIN_STARTS:
        ms = int(round(win_s * 1000))
        f  = tensor_dir / f"{REC_ID}-{ms}-10000.npy"
        if not f.exists():
            continue
        tensor = np.load(str(f))
        crop_idx   = np.arange(CROP_S, CROP_E)
        frame_times = win_s + (crop_idx + 0.5) * FRAME_STEP
        crop_probs  = tensor[CROP_S:CROP_E, :N_SLOTS]

        mask = (frame_times >= DISP_START) & (frame_times <= DISP_END)
        if not mask.any():
            continue
        times_list.append(frame_times[mask])
        probs_list.append(crop_probs[mask])

    if not times_list:
        return np.array([]), np.zeros((0, N_SLOTS))

    times = np.concatenate(times_list)
    probs = np.concatenate(probs_list, axis=0)
    order = np.argsort(times)
    return times[order], probs[order]


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(out_dir: Path) -> None:
    apply_ieee_style()

    gt_segs = load_gt_segments(DISP_START, DISP_END)
    t_pre, p_pre = build_prob_curve(PRE_DIR)
    t_ft,  p_ft  = build_prob_curve(FT_DIR)

    # Unique speakers in appearance order
    spk_order = list(dict.fromkeys(bare for _, _, bare, _ in gt_segs))
    spk_ctrl  = {bare: ctrl for _, _, bare, ctrl in gt_segs}

    # ---- Layout ----
    fig, axes = plt.subplots(
        3, 1,
        figsize=(IEEE_SINGLE_COL, 4.0),
        sharex=True,
        gridspec_kw={"hspace": 0.06, "height_ratios": [2.5, 3.5, 3.5]},
    )
    ax_gt, ax_pre, ax_ft = axes

    # ---- Panel 1: GT speaker bars ----
    for row_idx, bare in enumerate(spk_order):
        y    = row_idx * (BAR_H + BAR_PAD)
        ctrl = spk_ctrl[bare]
        st   = CTRL_STYLE if ctrl else PILOT_STYLE

        for cs, ce, b, _ in gt_segs:
            if b != bare:
                continue
            ax_gt.broken_barh(
                [(cs, ce - cs)], (y, BAR_H),
                facecolors=st["color"],
                hatch=st["hatch"],
                edgecolor="white",
                linewidth=st["lw"],
            )

        # Speaker label: centre of the first (and typically only) bar
        bar_segs = [(cs, ce) for cs, ce, b, _ in gt_segs if b == bare]
        if bar_segs:
            cs0, ce0 = bar_segs[0]
            bar_w = ce0 - cs0
            x_txt = (cs0 + ce0) / 2 if bar_w >= 0.8 else cs0 + 0.1
            ha    = "center"           if bar_w >= 0.8 else "left"
            ax_gt.text(x_txt, y + BAR_H / 2, bare, ha=ha, va="center",
                       fontsize=6.5, color="white", clip_on=True,
                       fontweight="bold")

    total_gt_h = len(spk_order) * (BAR_H + BAR_PAD)
    ax_gt.set_ylim(-0.08, total_gt_h + 0.08)
    ax_gt.set_yticks([])
    ax_gt.set_ylabel("GT", fontsize=8, rotation=0, labelpad=20, va="center")
    ax_gt.spines["left"].set_visible(False)

    # GT legend — upper left, before any speech starts (105–106.5 s is silent)
    ctrl_p  = mpatches.Patch(facecolor=CTRL_STYLE["color"],  hatch="//",
                              edgecolor="white", linewidth=0.3, label="Controller")
    pilot_p = mpatches.Patch(facecolor=PILOT_STYLE["color"], hatch="",
                              edgecolor="white", linewidth=0.3, label="Pilot")
    ax_gt.legend(handles=[ctrl_p, pilot_p], fontsize=7,
                 loc="upper right", framealpha=0.9,
                 ncol=2, handlelength=1.0, borderpad=0.4, columnspacing=0.5)

    # ---- Panels 2 & 3: Probability curves ----
    for ax, t_arr, p_arr, lane_label in [
        (ax_pre, t_pre, p_pre, "Pretrained"),
        (ax_ft,  t_ft,  p_ft,  "Fine-tuned"),
    ]:
        ax.axhline(0.5, color="#cccccc", linewidth=0.5, linestyle=":", zorder=1)

        if len(t_arr):
            for slot in range(N_SLOTS):
                st = SLOT_STYLES[slot]
                ax.plot(t_arr, p_arr[:, slot],
                        color=st["color"],
                        linestyle=st["linestyle"],
                        linewidth=1.5,
                        zorder=3)

        ax.set_ylim(-0.05, 1.12)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["0", "0.5", "1"], fontsize=7)
        ax.set_ylabel(lane_label, fontsize=8, rotation=0, labelpad=36, va="center")
        ax.spines["left"].set_visible(False)

    # ---- Slot legend: lower left of pretrained panel ----
    # Curves are at 0 for t > 112 s — legend sits in that silence region.
    slot_handles = [
        plt.Line2D([0], [0],
                   color=SLOT_STYLES[s]["color"],
                   linestyle=SLOT_STYLES[s]["linestyle"],
                   linewidth=1.5,
                   label=SLOT_STYLES[s]["label"])
        for s in range(N_SLOTS)
    ]
    ax_pre.legend(
        handles=slot_handles,
        fontsize=7,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.22),  # lift above the near-zero dashed curve
        framealpha=0.9,
        ncol=1,
        handlelength=1.4,
        borderpad=0.5,
        handletextpad=0.5,
    )

    # ---- Shared x-axis ----
    ax_ft.set_xlim(DISP_START, DISP_END)
    ax_ft.xaxis.set_major_locator(plt.MultipleLocator(3.0))
    ax_ft.xaxis.set_minor_locator(plt.MultipleLocator(1.0))
    ax_ft.set_xlabel("Time (s)", fontsize=9)
    for lbl in ax_ft.get_xticklabels():
        lbl.set_fontsize(8)

    # ---- Spine cleanup ----
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout(pad=0.4)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, out_dir / "fig2_lockup_comparison_v4")


if __name__ == "__main__":
    make_figure(Path("results/paper_figures"))

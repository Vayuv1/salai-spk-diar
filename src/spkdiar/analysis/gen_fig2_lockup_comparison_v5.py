"""
gen_fig2_lockup_comparison_v5.py

Fig 2 (v5): Speaker lock-up comparison — dca_d1_1, three event windows.
Double column (7.16 × 4.0 in), 600 DPI.

Three event windows selected by maximum pretrained-minus-finetuned slot-1 gap
on dca_d1_1:
  Window A — 105–115 s  (D1-1 → DAL209,  gap=0.778)
  Window B — 264–274 s  (D1-1 → AAL1581, gap=0.656)
  Window C — 299–309 s  (N99G → D1-1,    gap=0.944)  ← strongest example

Layout: 3 rows × 3 columns (one column per window), shared x within each column.
  Row 1 (GT)         — broken_barh, controller orange+hatch, pilot blue.
  Row 2 (Pretrained) — Slot 0 solid orange, Slot 1 dashed blue.
  Row 3 (Fine-tuned) — same format.

Column titles show actual time range. Row labels on left column.
Slot legend in lower-right of rightmost fine-tuned panel.

Output: results/paper_figures/fig2_lockup_comparison_v5.{pdf,png}

Usage:
    uv run python -m spkdiar.analysis.gen_fig2_lockup_comparison_v5
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from spkdiar.analysis.ieee_style import IEEE_DOUBLE_COL, apply_ieee_style, save_fig

REC_ID     = "dca_d1_1"
WINDOW_DUR = 10.0
FRAME_STEP = 0.08
CENTER_PAD = 2.5
CROP_S     = int(CENTER_PAD / FRAME_STEP)              # 31
CROP_E     = int((WINDOW_DUR - CENTER_PAD) / FRAME_STEP)  # 93

GT_RTTM  = Path("data/processed/rttm/dca_d1_1.rttm")
PRE_DIR  = Path("results/sortformer_offline/prob_tensors")
FT_DIR   = Path("results/sortformer_finetuned/prob_tensors")

# Three event windows. Each entry: display range and the stitching windows
# needed to cover it (center crops must span disp_start→disp_end).
WINDOWS = [
    {
        "disp_start": 105.0,
        "disp_end":   115.0,
        "win_starts": [100.0, 105.0, 110.0],
        "col_label":  "105–115 s",
    },
    {
        "disp_start": 264.0,
        "disp_end":   274.0,
        "win_starts": [260.0, 265.0, 270.0],
        "col_label":  "264–274 s",
    },
    {
        "disp_start": 299.0,
        "disp_end":   309.0,
        "win_starts": [295.0, 300.0, 305.0],
        "col_label":  "299–309 s",
    },
]

N_SLOTS = 2
SLOT_STYLES = [
    {"color": "#E8702A", "linestyle": "-",  "label": "Slot 0"},
    {"color": "#4C72B0", "linestyle": "--", "label": "Slot 1"},
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


def build_prob_curve(tensor_dir: Path, win_starts: list[float],
                     disp_start: float, disp_end: float) -> tuple[np.ndarray, np.ndarray]:
    """Stitch center-cropped frames; return (times, probs[:, :N_SLOTS])."""
    times_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []

    for win_s in win_starts:
        ms = int(round(win_s * 1000))
        f  = tensor_dir / f"{REC_ID}-{ms}-10000.npy"
        if not f.exists():
            continue
        tensor = np.load(str(f))
        crop_idx    = np.arange(CROP_S, CROP_E)
        frame_times = win_s + (crop_idx + 0.5) * FRAME_STEP
        crop_probs  = tensor[CROP_S:CROP_E, :N_SLOTS]

        mask = (frame_times >= disp_start) & (frame_times <= disp_end)
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

    n_cols = len(WINDOWS)
    n_rows = 3  # GT, Pre, FT

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(IEEE_DOUBLE_COL, 4.0),
        gridspec_kw={
            "hspace": 0.06,
            "wspace": 0.10,
            "height_ratios": [2.0, 3.5, 3.5],
        },
    )
    # axes shape: (3, 3)

    # Share x within each column (between rows)
    for col in range(n_cols):
        axes[1, col].sharex(axes[0, col])
        axes[2, col].sharex(axes[0, col])

    # ---- Draw each column ----
    for col, win in enumerate(WINDOWS):
        ds  = win["disp_start"]
        de  = win["disp_end"]
        ws  = win["win_starts"]

        ax_gt  = axes[0, col]
        ax_pre = axes[1, col]
        ax_ft  = axes[2, col]

        # -- GT panel --
        gt_segs   = load_gt_segments(ds, de)
        spk_order = list(dict.fromkeys(bare for _, _, bare, _ in gt_segs))
        spk_ctrl  = {bare: ctrl for _, _, bare, ctrl in gt_segs}

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
            # Speaker label: centre of first bar
            bar_segs = [(cs, ce) for cs, ce, b, _ in gt_segs if b == bare]
            if bar_segs:
                cs0, ce0 = bar_segs[0]
                bar_w = ce0 - cs0
                x_txt = (cs0 + ce0) / 2 if bar_w >= 0.6 else cs0 + 0.1
                ha    = "center"           if bar_w >= 0.6 else "left"
                ax_gt.text(x_txt, y + BAR_H / 2, bare, ha=ha, va="center",
                           fontsize=5.5, color="white", clip_on=True,
                           fontweight="bold")

        total_gt_h = max(len(spk_order), 1) * (BAR_H + BAR_PAD)
        ax_gt.set_xlim(ds, de)
        ax_gt.set_ylim(-0.08, total_gt_h + 0.08)
        ax_gt.set_yticks([])
        ax_gt.spines["left"].set_visible(False)
        ax_gt.tick_params(bottom=False, labelbottom=False)

        # Column title on top of GT panel
        ax_gt.set_title(win["col_label"], fontsize=8, pad=3)

        # GT legend: only on leftmost column, upper right (silence region on right side)
        if col == 0:
            ctrl_p  = mpatches.Patch(facecolor=CTRL_STYLE["color"],  hatch="//",
                                     edgecolor="white", linewidth=0.3, label="Controller")
            pilot_p = mpatches.Patch(facecolor=PILOT_STYLE["color"], hatch="",
                                     edgecolor="white", linewidth=0.3, label="Pilot")
            ax_gt.legend(handles=[ctrl_p, pilot_p], fontsize=6.5,
                         loc="upper right", framealpha=0.9,
                         ncol=2, handlelength=1.0, borderpad=0.4, columnspacing=0.5)

        # -- Prob panels (Pre and FT) --
        t_pre, p_pre = build_prob_curve(PRE_DIR, ws, ds, de)
        t_ft,  p_ft  = build_prob_curve(FT_DIR,  ws, ds, de)

        for ax, t_arr, p_arr, row_label in [
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
                            linewidth=1.3,
                            zorder=3)

            ax.set_ylim(-0.05, 1.12)
            ax.set_yticks([0, 0.5, 1.0])

            # Y-tick labels and row label only on leftmost column
            if col == 0:
                ax.set_yticklabels(["0", "0.5", "1"], fontsize=7)
                ax.set_ylabel(row_label, fontsize=8, rotation=0,
                              labelpad=36, va="center")
            else:
                ax.set_yticklabels([])
                ax.tick_params(left=False)

            ax.spines["left"].set_visible(False)

        # GT row label (leftmost column only)
        if col == 0:
            ax_gt.set_ylabel("GT", fontsize=8, rotation=0, labelpad=20, va="center")

        # X-axis ticks: shown only on bottom row; major every 2s, minor every 1s
        for ax in [ax_gt, ax_pre]:
            ax.tick_params(bottom=False, labelbottom=False)

        ax_ft.xaxis.set_major_locator(plt.MultipleLocator(2.0))
        ax_ft.xaxis.set_minor_locator(plt.MultipleLocator(1.0))
        ax_ft.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
        for lbl in ax_ft.get_xticklabels():
            lbl.set_fontsize(7)
        if col == 1:  # center column carries the shared x-label
            ax_ft.set_xlabel("Time (s)", fontsize=9)

        # Spine cleanup
        for ax in [ax_gt, ax_pre, ax_ft]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    # ---- Slot legend: lower-right of rightmost fine-tuned panel, lifted ----
    slot_handles = [
        plt.Line2D([0], [0],
                   color=SLOT_STYLES[s]["color"],
                   linestyle=SLOT_STYLES[s]["linestyle"],
                   linewidth=1.3,
                   label=SLOT_STYLES[s]["label"])
        for s in range(N_SLOTS)
    ]
    axes[2, -1].legend(
        handles=slot_handles,
        fontsize=7,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.22),
        framealpha=0.9,
        ncol=1,
        handlelength=1.4,
        borderpad=0.5,
        handletextpad=0.5,
    )

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout(pad=0.4)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, out_dir / "fig2_lockup_comparison_v5")


if __name__ == "__main__":
    make_figure(Path("results/paper_figures"))

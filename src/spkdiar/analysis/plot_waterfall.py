#!/usr/bin/env python3
"""
plot_waterfall.py

Publication-quality waterfall comparison plots for DASC 2026.

Each grid shows N consecutive 10 s windows stacked vertically (earliest
at top, latest at bottom — natural reading order).  Within every window
lane the vertical space is partitioned as:

  ┌──────────────────────────────────────────────────┐  ← lane top
  │  GT speaker bars  (ATC = orange, pilots = blue/  │
  │  green shades; speaker ID with hyphen = ATC)     │
  ├──────────────────────────────────────────────────┤
  │  Sortformer Offline   — per-speaker prob curves  │
  ├──────────────────────────────────────────────────┤
  │  Sortformer Streaming — per-speaker prob curves  │
  └──────────────────────────────────────────────────┘  ← lane bottom

Outputs (saved to --out-dir):
  {rec_id}_grid_NN.png          one PNG per grid
  {rec_id}_waterfall_all.png    all grids stacked into a single tall figure

Usage:
    uv run python -m spkdiar.analysis.plot_waterfall \\
        --rec-id dca_d1_1 \\
        --offline-dir  results/sortformer_offline/prob_tensors \\
        --streaming-dir results/sortformer_streaming_10s/prob_tensors \\
        --rttm-dir data/processed/rttm \\
        --out-dir results/plots \\
        --offset-min 55 \\
        --num-grids 3 \\
        --windows-per-grid 6
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Publication style defaults
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
})

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FRAME_SHIFT_SEC = 0.08   # 80 ms Sortformer output frame step

# GT coloring
ATC_COLOR    = "#E8702A"           # vivid orange
PILOT_COLORS = [
    "#1f77b4",   # steel blue
    "#2ca02c",   # medium green
    "#17becf",   # cyan
    "#8c564b",   # brown
    "#7f7f7f",   # grey
    "#bcbd22",   # yellow-green
]

# Speaker probability curve colors (4 Sortformer output columns)
SPK_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]  # red, blue, green, purple
SPK_LABELS = ["Spk 1", "Spk 2", "Spk 3", "Spk 4"]

# Lane proportions  (fractions of lane height = 1.0)
MARGIN_TOP    = 0.04
MARGIN_BOT    = 0.04
GT_FRAC       = 0.22   # GT band
SEP_PX        = 0.015  # thin separator (in lane units)
# remaining after GT + margins + 2 separators split equally between 2 sublanes
_USABLE       = 1.0 - MARGIN_TOP - MARGIN_BOT - GT_FRAC - 3 * SEP_PX
SUBLANE_FRAC  = _USABLE / 2.0   # ≈ 0.335 each


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    start: float
    end:   float
    spk:   str


def parse_uniq_id(stem: str) -> tuple[str, float, float]:
    """'dca_d1_1-60000-10000' → ('dca_d1_1', 60.0, 10.0)."""
    parts = stem.split("-")
    dur_ms   = int(parts.pop())
    start_ms = int(parts.pop())
    rec_id   = "-".join(parts)
    return rec_id, start_ms / 1000.0, dur_ms / 1000.0


def load_rttm(rttm_path: Path, rec_id: str) -> list[Segment]:
    segs = []
    with open(rttm_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 9 or p[0] != "SPEAKER":
                continue
            if p[1] != rec_id:
                continue
            start = float(p[3])
            dur   = float(p[4])
            spk   = p[7]
            segs.append(Segment(start=start, end=start + dur, spk=spk))
    return segs


def infer_role(spk_id: str, rec_id: str) -> str:
    """Return 'ATC' if the speaker label (after stripping rec_id prefix) contains
    a hyphen, e.g. D1-1, F1-2.  Otherwise 'PILOT'."""
    label = spk_id
    prefix = rec_id + "_"
    if label.startswith(prefix):
        label = label[len(prefix):]
    return "ATC" if "-" in label else "PILOT"


def build_gt_color_map(segs: list[Segment], rec_id: str) -> dict[str, str]:
    """Assign a fixed color to each unique speaker across the recording."""
    all_spks = sorted(set(s.spk for s in segs))
    pilot_idx = 0
    cmap: dict[str, str] = {}
    for spk in all_spks:
        if infer_role(spk, rec_id) == "ATC":
            cmap[spk] = ATC_COLOR
        else:
            cmap[spk] = PILOT_COLORS[pilot_idx % len(PILOT_COLORS)]
            pilot_idx += 1
    return cmap


def get_windows(prob_dir: Path, rec_id: str, offset_min: float) -> list[Path]:
    """Return .npy files for rec_id, sorted by start time, filtered by offset_min."""
    entries = []
    for f in prob_dir.glob(f"{rec_id}-*.npy"):
        try:
            _, start_s, _ = parse_uniq_id(f.stem)
        except (ValueError, IndexError):
            continue
        if start_s >= offset_min:
            entries.append((start_s, f))
    entries.sort(key=lambda x: x[0])
    return [f for _, f in entries]


# ---------------------------------------------------------------------------
# Core drawing routine
# ---------------------------------------------------------------------------

def _draw_grid(
    ax: plt.Axes,
    window_files: list[Path],
    gt_segs: list[Segment],
    gt_cmap: dict[str, str],
    offline_dir:   Path,
    streaming_dir: Path,
    rec_id: str,
    grid_label: str = "",
) -> None:
    """Draw one grid (list of windows) onto ax."""
    n = len(window_files)
    if n == 0:
        ax.text(0.5, 0.5, "no windows", transform=ax.transAxes, ha="center")
        return

    _, first_start, _ = parse_uniq_id(window_files[0].stem)
    _, last_start, last_dur = parse_uniq_id(window_files[-1].stem)
    grid_end = last_start + last_dur

    ax.set_xlim(first_start, grid_end)
    ax.set_ylim(-0.5, n + 0.5)

    # x-axis ticks every 5 s
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.grid(axis="x", which="major", linestyle="--", linewidth=0.5, alpha=0.5, color="#888")
    ax.grid(axis="x", which="minor", linestyle=":",  linewidth=0.3, alpha=0.3, color="#aaa")

    ax.set_xlabel("Time (s)", labelpad=3)
    ax.set_ylabel("Window", labelpad=4)
    if grid_label:
        ax.set_title(grid_label, fontsize=10, pad=4)

    y_ticks, y_labels = [], []
    sublane_label_added = {"offline": False, "streaming": False}

    # Windows stacked top→bottom: y_level = n-1 for first window, 0 for last
    for rank, wf in enumerate(window_files):
        y_level = (n - 1) - rank          # earliest at top (highest y value)
        _, win_start, win_dur = parse_uniq_id(wf.stem)
        win_end = win_start + win_dur

        y_ticks.append(y_level)
        y_labels.append(f"{win_start:.0f}–{win_end:.0f} s")

        lane_top = y_level + 0.5 - MARGIN_TOP
        lane_bot = y_level - 0.5 + MARGIN_BOT

        # ── GT band ──────────────────────────────────────────────────────
        gt_top = lane_top
        gt_bot = lane_top - GT_FRAC

        win_segs = [s for s in gt_segs if s.start < win_end and s.end > win_start]
        drawn_spks: set[str] = set()
        for seg in win_segs:
            s = max(seg.start, win_start)
            e = min(seg.end,   win_end)
            if e <= s:
                continue
            color = gt_cmap.get(seg.spk, "#aaaaaa")
            # label only once per speaker across the whole figure (handled in legend)
            ax.broken_barh(
                [(s, e - s)],
                (gt_bot, GT_FRAC),
                facecolors=color,
                alpha=0.82,
                linewidth=0,
                label=seg.spk if seg.spk not in drawn_spks else "_nolegend_",
            )
            drawn_spks.add(seg.spk)

        # separator below GT
        sep1 = gt_bot - SEP_PX
        ax.hlines(sep1, win_start, win_end, colors="#555", linewidths=0.6, alpha=0.6)

        # ── Offline sublane ───────────────────────────────────────────────
        off_top = sep1
        off_bot = off_top - SUBLANE_FRAC

        # faint background tint for offline sublane
        ax.fill_between([win_start, win_end], [off_bot, off_bot], [off_top, off_top],
                        color="#e8f0fb", alpha=0.45, linewidth=0, zorder=0)

        _draw_prob_sublane(ax, offline_dir / wf.name, win_start, off_bot, off_top)
        sublane_label_added["offline"] = True

        # label at left edge of sublane
        ax.text(
            win_start + 0.15,
            (off_top + off_bot) / 2,
            "Offline",
            ha="left", va="center", fontsize=6.5, color="#1a4a8a",
            fontstyle="italic", fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.5),
            zorder=5,
        )

        sep2 = off_bot - SEP_PX
        ax.hlines(sep2, win_start, win_end, colors="#555", linewidths=0.6, alpha=0.4)

        # ── Streaming sublane ─────────────────────────────────────────────
        str_top = sep2
        str_bot = str_top - SUBLANE_FRAC

        # faint background tint for streaming sublane
        ax.fill_between([win_start, win_end], [str_bot, str_bot], [str_top, str_top],
                        color="#fdf0e0", alpha=0.45, linewidth=0, zorder=0)

        _draw_prob_sublane(ax, streaming_dir / wf.name, win_start, str_bot, str_top)
        sublane_label_added["streaming"] = True

        ax.text(
            win_start + 0.15,
            (str_top + str_bot) / 2,
            "Streaming",
            ha="left", va="center", fontsize=6.5, color="#7a3000",
            fontstyle="italic", fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.5),
            zorder=5,
        )

        # lane boundary
        ax.hlines(y_level - 0.5 + MARGIN_BOT / 2, win_start, win_end,
                  colors="#bbb", linewidths=0.4, alpha=0.6)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=7.5)

    # ── Legend ────────────────────────────────────────────────────────────
    # Part 1: GT speakers (deduplicated)
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, mpatches.Patch] = {}
    for h, l in zip(handles, labels):
        if l and not l.startswith("_") and l not in seen:
            seen[l] = h

    # Shorten speaker label: strip rec_id prefix
    prefix = rec_id + "_"
    legend_items = [
        (mpatches.Patch(facecolor=h.get_facecolor()[0] if hasattr(h, "get_facecolor") else h.get_color(),
                        alpha=0.82, label=l[len(prefix):] if l.startswith(prefix) else l),
         l[len(prefix):] if l.startswith(prefix) else l)
        for l, h in seen.items()
    ]

    # Part 2: model speaker curve colors
    spk_handles = [
        mpatches.Patch(facecolor=SPK_COLORS[i], label=SPK_LABELS[i])
        for i in range(len(SPK_COLORS))
    ]

    # Part 3: sublane type markers
    sublane_handles = [
        mpatches.Patch(facecolor="#dddddd", edgecolor="#555", linewidth=0.8,
                       label="■ Offline sublane"),
        mpatches.Patch(facecolor="#dddddd", edgecolor="#555", linewidth=0.8,
                       label="■ Streaming sublane"),
    ]

    all_handles = [p for p, _ in legend_items] + spk_handles
    all_labels  = [l for _, l in legend_items] + SPK_LABELS[:len(SPK_COLORS)]

    leg = ax.legend(
        all_handles, all_labels,
        loc="upper right",
        ncol=2,
        fontsize=7.5,
        framealpha=0.88,
        edgecolor="#888",
        title="GT speakers | Model outputs",
        title_fontsize=7.5,
        borderpad=0.5,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=0.8,
    )
    leg.get_frame().set_linewidth(0.5)


def _draw_prob_sublane(
    ax: plt.Axes,
    npy_path: Path,
    win_start: float,
    bot: float,
    top: float,
    label: Optional[str] = None,
) -> None:
    """Plot probability curves for one sublane (offline or streaming)."""
    if not npy_path.exists():
        ax.text(
            win_start + 0.1, (bot + top) / 2,
            f"missing: {npy_path.name}", fontsize=5, color="red", va="center"
        )
        return

    probs = np.load(npy_path)   # (T, 4)
    T, C = probs.shape
    t = np.arange(T) * FRAME_SHIFT_SEC + win_start
    scale = top - bot

    for c in range(C):
        ax.plot(
            t,
            probs[:, c] * scale + bot,
            color=SPK_COLORS[c % len(SPK_COLORS)],
            linewidth=1.3,
            alpha=0.88,
            solid_capstyle="round",
        )

    # sublane baseline
    ax.hlines(bot, t[0], t[-1], colors="#999", linewidths=0.5, alpha=0.5)
    # fill under each curve lightly
    for c in range(C):
        ax.fill_between(
            t,
            bot,
            probs[:, c] * scale + bot,
            color=SPK_COLORS[c % len(SPK_COLORS)],
            alpha=0.08,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication waterfall comparison plots (offline vs streaming).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rec-id",        default="dca_d1_1")
    parser.add_argument("--offline-dir",   type=Path,
                        default=Path("results/sortformer_offline/prob_tensors"))
    parser.add_argument("--streaming-dir", type=Path,
                        default=Path("results/sortformer_streaming_10s/prob_tensors"))
    parser.add_argument("--rttm-dir",      type=Path,
                        default=Path("data/processed/rttm"))
    parser.add_argument("--out-dir",       type=Path,
                        default=Path("results/plots"))
    parser.add_argument("--offset-min",    type=float, default=55.0,
                        help="Only include windows with start >= this time (s)")
    parser.add_argument("--windows-per-grid", type=int, default=6)
    parser.add_argument("--num-grids",     type=int, default=3)
    parser.add_argument("--fig-width",     type=float, default=16.0,
                        help="Figure width in inches")
    parser.add_argument("--lane-height",   type=float, default=1.15,
                        help="Height per window lane in inches")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    windows = get_windows(args.offline_dir, args.rec_id, args.offset_min)
    if not windows:
        print(f"No windows found for {args.rec_id} with offset >= {args.offset_min}s "
              f"in {args.offline_dir}")
        return

    need = args.windows_per_grid * args.num_grids
    if len(windows) < need:
        print(f"Warning: only {len(windows)} windows available, "
              f"need {need} for {args.num_grids} grids × {args.windows_per_grid}. "
              f"Reducing num-grids.")
        args.num_grids = len(windows) // args.windows_per_grid
        if args.num_grids == 0:
            print("Not enough windows. Exiting.")
            return

    rttm_path = args.rttm_dir / f"{args.rec_id}.rttm"
    gt_segs = load_rttm(rttm_path, args.rec_id) if rttm_path.exists() else []
    gt_cmap = build_gt_color_map(gt_segs, args.rec_id)

    print(f"Recording  : {args.rec_id}")
    print(f"GT segments: {len(gt_segs)}")
    print(f"GT speakers: {', '.join(sorted(gt_cmap))}")
    print(f"Windows    : {len(windows)} available, plotting {args.num_grids} grids "
          f"× {args.windows_per_grid} windows")

    # ── Per-grid figures ────────────────────────────────────────────────────
    fig_height = args.lane_height * args.windows_per_grid + 1.0  # +1 for title/xlabel

    overall_figs: list[plt.Figure] = []

    for g in range(args.num_grids):
        batch = windows[g * args.windows_per_grid : (g + 1) * args.windows_per_grid]

        _, t0, _     = parse_uniq_id(batch[0].stem)
        _, t1, td    = parse_uniq_id(batch[-1].stem)
        grid_title   = (
            f"{args.rec_id}  |  Grid {g + 1}/{args.num_grids}: "
            f"{t0:.0f}–{t1 + td:.0f} s  "
            f"(offline vs streaming Sortformer, collar=0.25 s)"
        )

        fig, ax = plt.subplots(figsize=(args.fig_width, fig_height))
        _draw_grid(ax, batch, gt_segs, gt_cmap,
                   args.offline_dir, args.streaming_dir,
                   args.rec_id, grid_title)

        out = args.out_dir / f"{args.rec_id}_grid_{g + 1:02d}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"  Grid {g + 1}: {out}")
        overall_figs.append(out)

    # ── Overall stacked figure ──────────────────────────────────────────────
    total_h = args.lane_height * args.windows_per_grid * args.num_grids + 1.5 * args.num_grids
    fig_all, axes = plt.subplots(
        nrows=args.num_grids, ncols=1,
        figsize=(args.fig_width, total_h),
        squeeze=False,
    )
    fig_all.suptitle(
        f"{args.rec_id}  —  Sortformer Offline vs Streaming 10 s latency\n"
        f"(GT: orange = ATC controller, blue/green = pilots; "
        f"curves: red/blue/green/purple = Spk 1–4)",
        fontsize=9, y=1.002,
    )

    for g in range(args.num_grids):
        batch = windows[g * args.windows_per_grid : (g + 1) * args.windows_per_grid]
        _, t0, _  = parse_uniq_id(batch[0].stem)
        _, t1, td = parse_uniq_id(batch[-1].stem)
        grid_title = f"Grid {g + 1}: {t0:.0f}–{t1 + td:.0f} s"

        _draw_grid(axes[g, 0], batch, gt_segs, gt_cmap,
                   args.offline_dir, args.streaming_dir,
                   args.rec_id, grid_title)

    fig_all.tight_layout(h_pad=1.5)
    out_all = args.out_dir / f"{args.rec_id}_waterfall_all.png"
    fig_all.savefig(out_all)
    plt.close(fig_all)
    print(f"  Overall  : {out_all}")
    print("Done.")


if __name__ == "__main__":
    main()

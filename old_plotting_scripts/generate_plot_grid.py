#!/usr/bin/env python3
"""
generate_plot_grid.py

Generates a "waterfall" grid of diarization probability plots,
similar to the example image 'image002.png'.

It produces one image per "grid" (row) AND one combined "overall" image.
It supports an optional ground-truth (RTTM) layer.
"""

import argparse
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import math

import matplotlib.pyplot as plt
import numpy as np


# --- Helper functions copied from plot_diarization_analysis.py ---
# We need these to parse RTTMs and window IDs

@dataclass
class RttmSegment:
    rec_id: str
    start: float
    dur: float
    spk_id: str

    @property
    def end(self) -> float:
        return self.start + self.dur


def read_rttm(rttm_path: Path, base_rec_id: str) -> List[RttmSegment]:
    """Parses an RTTM file into a list of Segment objects."""
    print(f"--- Reading RTTM: {rttm_path.name} for rec_id {base_rec_id} ---")
    segments = []
    with rttm_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line.startswith("SPEAKER"):
                continue

            parts = line.split()
            if len(parts) < 8:
                continue

            try:
                rec_id = parts[1]
                # Only parse segments for the recording we care about
                if rec_id != base_rec_id:
                    continue
                start = float(parts[3])
                dur = float(parts[4])
                spk_id = parts[7]
                segments.append(
                    RttmSegment(rec_id=rec_id, start=start, dur=dur,
                                spk_id=spk_id)
                )
            except (ValueError, IndexError):
                continue
    print(f"--- Found {len(segments)} valid segments for {base_rec_id} ---")
    return segments


def parse_uniq_id(uniq_id: str) -> Tuple[str, float, float]:
    """Parses 'dca_d1_1-10000-20000' -> ('dca_d1_1', 10.0, 20.0)"""
    try:
        parts = uniq_id.split("-")
        dur_ms = int(parts.pop())
        start_ms = int(parts.pop())
        base_rec_id = "-".join(parts)
        return base_rec_id, start_ms / 1000.0, dur_ms / 1000.0
    except Exception:
        raise ValueError(f"Could not parse uniq_id: {uniq_id}")


# --- New Main Plotting Logic ---

def get_window_files(prob_dir: Path, base_id: str) -> List[Path]:
    """Finds all .npy files for a base_id and sorts them by time."""
    files = []
    for f in prob_dir.glob(f"{base_id}-*.npy"):
        try:
            _, start_ms, _ = parse_uniq_id(f.stem)
            files.append((start_ms, f))
        except ValueError:
            print(f"Warning: Skipping file with bad name: {f.name}")

    # Sort by start time
    files.sort(key=lambda x: x[0])
    return [f for _, f in files]


def _draw_single_grid(
        ax: plt.Axes,
        window_files: List[Path],
        all_gt_segments: Optional[List[RttmSegment]],
        overlay_dirs: List[Path] = None,
):
    """
    Stacked rows per case (run dir) with GT bars colored:
      - ATC = orange
      - Pilots = blue/green shades

    Legend entries:
      - GT speaker ids (from RTTM bars)
      - ONE entry per case (case dir name), not per speaker

    Layout fix:
      Each window lane is centered in [y_level-0.5, y_level+0.5] so the last lane
      never gets clipped from below when stacking multiple sub-rows.
    """
    if not window_files:
        ax.text(0.5, 0.5, "No windows to plot", transform=ax.transAxes, ha="center")
        return

    # ----- Time range for this grid (no right pad) -----
    first_id = window_files[0].stem
    last_id  = window_files[-1].stem
    _, grid_start_sec, _        = parse_uniq_id(first_id)
    _, last_win_start, last_dur = parse_uniq_id(last_id)
    grid_end_sec = last_win_start + last_dur
    ax.set_xlim(grid_start_sec, grid_end_sec)

    # ----- Y range: one lane per window (centered) -----
    num_windows = len(window_files)
    ax.set_ylim(-0.5, num_windows + 0.5)

    y_labels, y_ticks = [], []

    # ----- Colors & constants -----
    spk_cmap        = plt.get_cmap("Set1")   # model Spk1..Spk4
    frame_shift_sec = 0.08
    top_margin      = 0.05
    gt_band_height  = 0.30

    cases      = overlay_dirs or []
    num_cases  = max(1, len(cases))
    case_names = [p.name if isinstance(p, Path) else str(p) for p in cases]

    # ----- ATC vs Pilot helpers for GT coloring -----
    has_gt = bool(all_gt_segments)
    ATC_COLOR     = "#FF7F0E"  # vivid orange
    PILOT_PALETTE = ["#1f77b4", "#2ca02c", "#17becf", "#2ca25f", "#66c2a4", "#a6bddb"]

    import re
    CTRL_TOKENS = {"atc","tower","twr","ground","gnd","approach","app","departure","dep",
                   "center","ctr","clearance","clr","delivery","del","control"}

    def infer_role(spk_id: str) -> str:
        s = (spk_id or "").lower()
        tok = s.split("_")[-1]  # avoid file-level prefixes (e.g., 'log_f1_4_...')
        if any(t in tok for t in CTRL_TOKENS): return "ATC"
        if re.fullmatch(r"f1-\d+", tok):       return "ATC"
        if re.fullmatch(r"n\d{3,6}[a-z]?", tok):           return "PILOT"
        if re.fullmatch(r"[a-z]{2,3}\d{2,4}", tok):        return "PILOT"
        if re.search(r"\d", tok):                           return "PILOT"
        return "ATC" if "atc" in tok else "PILOT"

    if has_gt:
        gt_speakers = sorted(list(set(seg.spk_id for seg in all_gt_segments)))
        gt_spk2idx  = {spk_id: i for i, spk_id in enumerate(gt_speakers)}
    else:
        gt_speakers, gt_spk2idx = [], {}

    # ----- Draw windows (top lane = earliest window) -----
    for y_level, prob_file in enumerate(reversed(window_files)):
        uniq_id = prob_file.stem
        _, win_start, win_dur = parse_uniq_id(uniq_id)
        win_end = win_start + win_dur

        y_labels.append(f"{win_start:.1f}s - {win_end:.1f}s")
        y_ticks.append(y_level)

        # Define the lane strictly within [y_level-0.5, y_level+0.5]
        lane_top = y_level + 0.5 - top_margin
        lane_bot = y_level - 0.5 + top_margin

        # --- A) Ground truth band at top of lane ---
        if has_gt:
            gt_top = lane_top
            gt_bot = gt_top - gt_band_height

            segs = [seg for seg in all_gt_segments if (seg.start < win_end and seg.end > win_start)]
            spks_in_win = sorted(list(set(seg.spk_id for seg in segs)))
            for spk_id in spks_in_win:
                xranges = []
                for seg in segs:
                    if seg.spk_id != spk_id: continue
                    s = max(seg.start, win_start)
                    e = min(seg.end,   win_end)
                    if e > s:
                        xranges.append((s, e - s))
                if not xranges:
                    continue

                role = infer_role(spk_id)
                if role == "ATC":
                    color = ATC_COLOR
                else:
                    i = gt_spk2idx.get(spk_id, 0)
                    color = PILOT_PALETTE[i % len(PILOT_PALETTE)]

                ax.broken_barh(
                    xranges, (gt_bot, gt_band_height),
                    facecolors=color, alpha=0.75,
                    label=spk_id
                )

        # Separator under GT band
        sep_y = lane_top - gt_band_height
        ax.hlines(y=sep_y, xmin=win_start, xmax=win_end, colors="k", alpha=0.25, linewidth=0.6)

        # --- B) Stacked case rows: fill the remaining space down to lane_bot ---
        usable = sep_y - lane_bot
        sublane_h = usable / max(1, num_cases)

        for case_idx, case_dir in enumerate(cases):
            sub_top = sep_y - case_idx * sublane_h
            sub_bot = sub_top - sublane_h + 0.02
            base    = sub_bot
            scale   = (sub_top - sub_bot)

            curr_file = case_dir / f"{uniq_id}.npy"
            if curr_file.exists():
                probs = np.load(curr_file)  # [T, C]
                T, C = probs.shape
                t_axis = np.arange(T) * frame_shift_sec + win_start

                for spk in range(C):
                    lbl = case_names[case_idx] if (y_level == num_windows-1 and spk == 0) else None
                    ax.plot(
                        t_axis,
                        probs[:, spk] * scale + base,
                        color=spk_cmap(spk), linewidth=1.7, alpha=0.95,
                        label=lbl
                    )

            # sublane baseline
            ax.hlines(y=sub_bot, xmin=win_start, xmax=win_end, colors="k", alpha=0.35, linewidth=0.6)

        # faint bottom border for the whole lane
        ax.hlines(y=lane_bot, xmin=win_start, xmax=win_end, colors="k", alpha=0.15, linewidth=0.4)

    # ----- Axes cosmetics -----
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_ylabel("Window Time Range")
    ax.set_xlabel("Absolute Time (seconds)")
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    # ----- Legend (previous behavior: de-dup by label) -----
    handles, labels = ax.get_legend_handles_labels()
    by_label = {lab: h for lab, h in zip(labels, handles) if lab}
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize="small")


def main():
    # --- replace the CLI args in main() ---
    parser = argparse.ArgumentParser(
        description="Generate a grid of waterfall plots (supports multi-run overlays).")
    parser.add_argument("--base-id", required=True,
                        help="Base recording ID (e.g., 'log_f1_1')")
    # NEW: comma-separated list of prob dirs, first is baseline
    parser.add_argument("--prob-dirs", required=True,
                        help="Comma-separated dirs of .npy tensors (e.g. run0,run+3ms,run-3ms)")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="Directory to save the output plot PNGs")
    parser.add_argument("--rttm-dir", type=Path, default=None,
                        help="[Optional] Directory with ground truth RTTMs")
    parser.add_argument("--windows-per-grid", type=int, default=6,
                        help="Number of windows to stack in one grid")
    parser.add_argument("--num-grids", type=int, default=3,
                        help="Number of grids to create")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    prob_dirs = [Path(p.strip()) for p in args.prob_dirs.split(",")]
    for p in prob_dirs:
        if not p.exists():
            raise FileNotFoundError(f"Prob dir not found: {p}")

    # Use the FIRST dir to enumerate available windows and time order
    all_window_files = get_window_files(prob_dirs[0], args.base_id)
    if not all_window_files:
        print(
            f"Error: No .npy files found for '{args.base_id}' in {prob_dirs[0]}")
        return

    total_windows_needed = args.windows_per_grid * args.num_grids
    if len(all_window_files) < total_windows_needed:
        print(
            f"Error: You asked for {args.num_grids} grids of {args.windows_per_grid} windows "
            f"({total_windows_needed} total), but only {len(all_window_files)} "
            f".npy files were found for '{args.base_id}'.")
        print("Please reduce --num-grids or --windows-per-grid.")
        return

    # 2. Load Ground Truth (if provided)
    all_gt_segments = None

    if args.rttm_dir:
        gt_path = args.rttm_dir / f"{args.base_id}.rttm"
        if gt_path.exists():
            all_gt_segments = read_rttm(gt_path, args.base_id)
        else:
            print(f"Warning: --rttm-dir provided, but {gt_path} not found.")

    # 3. Create the 'Overall' figure
    # We make one stacked plot for each grid
    fig_overall, axes_overall = plt.subplots(
        nrows=args.num_grids,
        ncols=1,
        figsize=(20, 6 * args.num_grids),  # 20" wide, 6" tall per grid
        squeeze=False  # Ensures axes_overall is always a 2D array
    )
    fig_overall.suptitle(f"Overall Analysis for {args.base_id}", fontsize=16,
                         y=1.02)

    # 4. Loop through each grid
    file_idx = 0
    for i in range(args.num_grids):
        grid_num = i + 1

        # Get the slice of files for this grid
        grid_window_files = all_window_files[
                            file_idx: file_idx + args.windows_per_grid]
        file_idx += args.windows_per_grid

        if not grid_window_files:
            continue  # Should be prevented by our earlier check, but good to be safe

        print(f"--- Processing Grid {grid_num}/{args.num_grids} "
              f"({len(grid_window_files)} windows) ---")

        # --- A: Create and save the INDIVIDUAL grid image ---
        fig_single, ax_single = plt.subplots(figsize=(20, 8))
        _draw_single_grid(ax_single, grid_window_files, all_gt_segments, overlay_dirs=prob_dirs)

        first_win_t = parse_uniq_id(grid_window_files[0].stem)[1]
        last_win_t, last_win_dur = parse_uniq_id(grid_window_files[-1].stem)[1:]

        ax_single.set_title(
            f"Grid {grid_num}: {first_win_t:.1f}s to {last_win_t + last_win_dur:.1f}s")

        out_path_single = args.out_dir / f"{args.base_id}_grid_{grid_num:02d}.png"
        fig_single.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_single.savefig(out_path_single)
        plt.close(fig_single)
        print(f"Saved single grid plot to: {out_path_single}")

        # --- B: Draw this same grid on the OVERALL image ---
        ax_for_overall = axes_overall[i, 0]
        _draw_single_grid(ax_for_overall, grid_window_files, all_gt_segments, overlay_dirs=prob_dirs)
        ax_for_overall.set_title(
            f"Grid {grid_num}: {first_win_t:.1f}s to {last_win_t + last_win_dur:.1f}s")

    # 5. Save the 'Overall' figure
    out_path_overall = args.out_dir / f"{args.base_id}_grid_OVERALL.png"
    fig_overall.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig_overall.savefig(out_path_overall)
    plt.close(fig_overall)
    print(f"Saved OVERALL plot to: {out_path_overall}")
    print("Done.")


if __name__ == "__main__":
    main()
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
):
    """
    Draws one 'grid' (one waterfall plot) onto a single matplotlib Axes.
    This is the core logic that reproduces the layout of 'image002.png'.
    """

    # 1. --- Determine Time and Y-Axis ranges ---
    if not window_files:
        ax.text(0.5, 0.5, "No windows to plot", transform=ax.transAxes,
                ha="center")
        return

    # Get time range of this grid
    first_id = window_files[0].stem
    last_id = window_files[-1].stem

    _, grid_start_sec, first_win_dur = parse_uniq_id(first_id)
    _, last_win_start, last_win_dur = parse_uniq_id(last_id)
    grid_end_sec = last_win_start + last_win_dur

    # Set X-axis to absolute time
    ax.set_xlim(grid_start_sec, grid_end_sec)

    # Set Y-axis for staggered windows
    # We plot each window at a different Y-level
    num_windows = len(window_files)
    ax.set_ylim(-0.5, num_windows + 0.5)

    y_labels = []
    y_ticks = []

    # 2. --- Prepare Colors and GT Speaker Mapping ---
    model_colors = plt.get_cmap("Set1")
    frame_shift_sec = 0.08  # 0.01s * 8 subsampling

    has_ground_truth = bool(all_gt_segments)
    gt_speaker_set = set()
    gt_spk_to_color_idx = {}

    # *** COLORMAP FIX ***
    # Use a continuous colormap to get many unique colors
    gt_colors = plt.get_cmap("gist_rainbow")
    num_gt_speakers = 0

    if has_ground_truth:
        # Get all unique speaker IDs from all segments
        gt_speaker_set = sorted(
            list(set(seg.spk_id for seg in all_gt_segments)))
        num_gt_speakers = len(gt_speaker_set)
        # Create a consistent mapping from spk_id to a color index
        gt_spk_to_color_idx = {spk_id: i for i, spk_id in
                               enumerate(gt_speaker_set)}

    # 3. --- Loop and plot each window ---
    for y_level, prob_file in enumerate(reversed(window_files)):
        # We reverse so the *first* window is at the *top* (high Y value)

        uniq_id = prob_file.stem
        _, win_start, win_dur = parse_uniq_id(uniq_id)
        win_end = win_start + win_dur

        y_labels.append(f"{win_start:.1f}s - {win_end:.1f}s")
        y_ticks.append(y_level)

        # --- A: Plot Ground Truth (Optional) ---
        if has_ground_truth:
            # Plot GT in its own "sub-panel" from y=[y_level + 0.1] to y=[y_level + 0.4]
            gt_y_base = y_level + 0.1
            gt_height = 0.3

            # Filter segments for this window
            window_segments = []
            for seg in all_gt_segments:
                if seg.start < win_end and seg.end > win_start:
                    window_segments.append(seg)

            # Find unique speakers *in this window*
            gt_speakers_in_window = sorted(
                list(set(seg.spk_id for seg in window_segments)))

            # Plot one speaker at a time
            for spk_id in gt_speakers_in_window:
                xranges = []
                # Find segments for *this* speaker
                for seg in window_segments:
                    if seg.spk_id == spk_id:
                        # Clip to window boundaries for plotting
                        plot_start = max(seg.start, win_start)
                        plot_end = min(seg.end, win_end)
                        if plot_end > plot_start:
                            xranges.append((plot_start, plot_end - plot_start))

                if not xranges:
                    continue

                # Get the consistent color index for this speaker
                color_idx = gt_spk_to_color_idx.get(spk_id, 0)

                # *** COLORMAP FIX ***
                # Normalize the index to get a value between 0.0 and 1.0
                if num_gt_speakers > 1:
                    color_norm = color_idx / (num_gt_speakers - 1)
                else:
                    color_norm = 0.5  # Default color if only 1 speaker

                # Get the unique color from the continuous colormap
                color = gt_colors(color_norm)

                # Plot all bars for this speaker
                ax.broken_barh(
                    xranges,
                    (gt_y_base, gt_height),  # (y_base, height)
                    facecolors=color, alpha=0.6,
                    label=spk_id  # Add label for the legend
                )

                # Text labels are removed as requested

        # --- B: Plot Model Probabilities ---
        prob_tensor = np.load(prob_file)
        num_frames, num_speakers = prob_tensor.shape

        # Create absolute time axis for this tensor
        time_axis = np.arange(num_frames) * frame_shift_sec + win_start

        # Adjust scale and base position based on whether GT is present
        if has_ground_truth:
            # Plot probs in the *lower* sub-panel
            # from y=[y_level - 0.45] to y=[y_level + 0.05]
            prob_scale = 0.5  # Scale 0-1 to 0-0.5
            prob_base = y_level - 0.45
        else:
            # Original logic: use the full space
            # from y=[y_level - 0.45] to y=[y_level + 0.45]
            prob_scale = 0.9  # Scale 0-1 to 0-0.9
            prob_base = y_level - 0.45

        # Plot each of the 4 speakers
        for i in range(num_speakers):
            ax.plot(
                time_axis,
                prob_tensor[:, i] * prob_scale + prob_base,
                # Use new scale and base
                color=model_colors(i),
                alpha=0.7,
                label=f"Model Spk {i + 1}"  # Add label for the legend
            )

    # 4. --- Finalize Axes ---
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_ylabel("Window Time Range")
    ax.set_xlabel("Absolute Time (seconds)")
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    # --- Create a single legend ---
    # Get all unique handles and labels from the plot
    handles, labels = ax.get_legend_handles_labels()
    # Create a dictionary to de-duplicate labels
    by_label = dict(zip(labels, handles))
    # Plot the de-duplicated legend
    ax.legend(by_label.values(), by_label.keys(), loc="upper right",
              fontsize="small")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a grid of waterfall plots.")
    parser.add_argument("--base-id", required=True,
                        help="Base recording ID (e.g., 'dca_d1_1')")
    parser.add_argument("--prob-dir", required=True, type=Path,
                        help="Directory containing the .npy probability tensors")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="Directory to save the output plot PNGs")
    parser.add_argument("--rttm-dir", type=Path, default=None,
                        help="[Optional] Directory with ground truth RTTMs")
    parser.add_argument("--windows-per-grid", type=int, default=6,
                        help="Number of windows to stack in one grid (your 'x')")
    parser.add_argument("--num-grids", type=int, default=3,
                        help="Number of grids to create (your 'y')")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Find all available window files, sorted by time
    all_window_files = get_window_files(args.prob_dir, args.base_id)
    if not all_window_files:
        print(
            f"Error: No .npy files found for '{args.base_id}' in {args.prob_dir}")
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
        _draw_single_grid(ax_single, grid_window_files, all_gt_segments)

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
        _draw_single_grid(ax_for_overall, grid_window_files, all_gt_segments)
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
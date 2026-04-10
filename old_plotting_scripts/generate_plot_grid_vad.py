#!/usr/bin/env python3
"""
generate_plot_grid_vad.py

Generate waterfall-style grids for one recording with:
  - Ground-truth RTTM speaker bars (ATC vs pilots colored)
  - Sortformer-derived VAD band   (max over 4 speaker probs)
  - External VAD band            (from run_external_vad.py)
  - 4-speaker probability curves

Layout and overall figure structure are modeled after the original
generate_plot_grid.py so that:
  * Each grid row shows only its own time span (xlim per grid)
  * The OVERALL figure stacks grids vertically, no horizontal scrolling
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# RTTM helpers
# ---------------------------------------------------------------------

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
    """Parse RTTM for a single recording id."""
    print(f"--- Reading RTTM: {rttm_path.name} for rec_id={base_rec_id} ---")
    segments: List[RttmSegment] = []
    with rttm_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("SPEAKER"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                rec_id = parts[1]
                if rec_id != base_rec_id:
                    continue
                start = float(parts[3])
                dur = float(parts[4])
                spk_id = parts[7]
            except (ValueError, IndexError):
                continue
            segments.append(RttmSegment(rec_id=rec_id, start=start, dur=dur, spk_id=spk_id))
    print(f"--- Found {len(segments)} RTTM segments for {base_rec_id} ---")
    return segments


# ---------------------------------------------------------------------
# Window / file helpers
# ---------------------------------------------------------------------

def parse_uniq_id(uniq_id: str) -> Tuple[str, float, float]:
    """
    Parse uniq_id 'log_f1_4-10000-10000' -> ('log_f1_4', 10.0, 10.0)
    """
    parts = uniq_id.split("-")
    if len(parts) < 3:
        raise ValueError(f"Could not parse uniq_id: {uniq_id}")
    dur_ms = int(parts[-1])
    start_ms = int(parts[-2])
    base_rec_id = "-".join(parts[:-2])
    return base_rec_id, start_ms / 1000.0, dur_ms / 1000.0


def get_window_files(prob_dir: Path, base_id: str) -> List[Path]:
    """
    List all <base_id>-*.npy in prob_dir, sorted by window start time.
    """
    files: List[Tuple[float, Path]] = []
    for f in prob_dir.glob(f"{base_id}-*.npy"):
        try:
            _, start_sec, _ = parse_uniq_id(f.stem)
            files.append((start_sec, f))
        except ValueError:
            print(f"Warning: skipping bad file name: {f.name}")
    files.sort(key=lambda t: t[0])
    return [f for _, f in files]


# ---------------------------------------------------------------------
# Core drawing for one grid (one subplot)
# ---------------------------------------------------------------------

def _draw_single_grid(
    ax: plt.Axes,
    window_files: List[Path],
    all_gt_segments: Optional[List[RttmSegment]],
    probs_dir: Path,
    vad_silero_dir: Optional[Path],
    vad_nemo_dir: Optional[Path],
    frame_step_sec: float,
) -> Tuple[float, float]:

    """
    Draw a single grid (a group of consecutive windows) into ax.

    Returns:
        (grid_start_sec, grid_end_sec) for this grid.
    """
    if not window_files:
        ax.text(0.5, 0.5, "No windows", ha="center", va="center",
                transform=ax.transAxes)
        return 0.0, 0.0

    # Determine the time range for this grid
    first_id = window_files[0].stem
    last_id = window_files[-1].stem
    _, grid_start_sec, _ = parse_uniq_id(first_id)
    _, last_start, last_dur = parse_uniq_id(last_id)
    grid_end_sec = last_start + last_dur

    # Y range: one lane per window, centered at y_level
    num_windows = len(window_files)
    ax.set_ylim(-0.5, num_windows + 0.5)

    y_labels: List[str] = []
    y_ticks: List[float] = []

    spk_cmap = plt.get_cmap("Set1")

    # vertical structure inside each lane
    # smaller margin so we get more vertical space for curves
    top_margin = 0.05
    lane_height = 1.0 - 2 * top_margin

    # Make all four bands roughly the same height
    gt_frac = 0.25
    vad1_frac = 0.25  # Silero
    vad2_frac = 0.25  # NeMo
    prob_frac = 0.25  # Sortformer speaker probs

    ATC_COLOR = "#FF7F0E"
    PILOT_PALETTE = ["#1f77b4", "#2ca02c", "#17becf", "#2ca25f", "#66c2a4", "#a6bddb"]

    import re as _re
    CTRL_TOKENS = {
        "atc", "tower", "twr", "ground", "gnd", "approach", "app",
        "departure", "dep", "center", "ctr", "clearance", "clr",
        "delivery", "del", "control",
    }

    def infer_role(spk_id: str) -> str:
        s = (spk_id or "").lower()
        tok = s.split("_")[-1]
        if any(t in tok for t in CTRL_TOKENS):
            return "ATC"
        if _re.fullmatch(r"f1-\d+", tok):
            return "ATC"
        if _re.fullmatch(r"n\d{3,6}[a-z]?", tok):
            return "PILOT"
        return "PILOT"

    has_gt = bool(all_gt_segments)
    if has_gt:
        gt_speakers = sorted({seg.spk_id for seg in all_gt_segments})
        gt_spk2idx: Dict[str, int] = {s: i for i, s in enumerate(gt_speakers)}
    else:
        gt_speakers = []
        gt_spk2idx = {}

    # Walk windows in chronological order but map earliest to top lane
    for lane_idx, prob_file in enumerate(window_files):
        uniq_id = prob_file.stem
        _, win_start, win_dur = parse_uniq_id(uniq_id)
        win_end = win_start + win_dur

        # invert to have earliest window at top
        y_level = num_windows - 1 - lane_idx

        y_center = float(y_level)
        lane_top = y_center + 0.5 - top_margin
        lane_bot = y_center - 0.5 + top_margin

        gt_top = lane_top
        gt_bot = gt_top - gt_frac * lane_height
        vad1_top = gt_bot
        vad1_bot = vad1_top - vad1_frac * lane_height
        vad2_top = vad1_bot
        vad2_bot = vad2_top - vad2_frac * lane_height
        prob_top = vad2_bot
        prob_bot = lane_bot

        y_ticks.append(y_center)
        y_labels.append(f"{win_start:.1f}s - {win_end:.1f}s")

        # ------------------------------------------------------------------
        # A) Ground truth band (RTTM bars)
        # ------------------------------------------------------------------
        if has_gt:
            segs = [
                seg for seg in all_gt_segments
                if (seg.start < win_end and seg.end > win_start)
            ]
            for seg in segs:
                seg_start = max(seg.start, win_start)
                seg_end = min(seg.end, win_end)
                if seg_end <= seg_start:
                    continue
                spk_id = seg.spk_id
                role = infer_role(spk_id)
                if role == "ATC":
                    color = ATC_COLOR
                else:
                    idx = gt_spk2idx.get(spk_id, 0)
                    color = PILOT_PALETTE[idx % len(PILOT_PALETTE)]

                ax.broken_barh(
                    [(seg_start, seg_end - seg_start)],
                    (gt_bot, gt_top - gt_bot),
                    facecolors=color,
                    alpha=0.8,
                    label=spk_id,
                )

        # line separating GT from rest
        ax.hlines(gt_bot, win_start, win_end, colors="k", alpha=0.3, linewidth=0.6)

        # ------------------------------------------------------------------
        # B) Load probs and build Sortformer VAD
        # ------------------------------------------------------------------
        probs_path = probs_dir / f"{uniq_id}.npy"
        if not probs_path.is_file():
            continue
        probs = np.load(probs_path)
        if probs.ndim == 1:
            probs = probs[:, None]
        T, C = probs.shape
        t_axis = np.arange(T, dtype=np.float32) * frame_step_sec + win_start

        sortformer_vad = probs.max(axis=1)  # [T]

        # We still have full Sortformer speaker probs in `probs` for the last band.

        # ------------------------------------------------------------------
        # C) Silero and NeMo VAD tracks
        # ------------------------------------------------------------------
        silero_vad = None
        nemo_vad = None

        # Silero VAD probabilities
        if vad_silero_dir is not None:
            silero_path = vad_silero_dir / f"{uniq_id}.npy"
            if silero_path.is_file():
                silero_vad = np.load(silero_path).astype(np.float32)
                if silero_vad.ndim > 1:
                    silero_vad = silero_vad.squeeze()
                if len(silero_vad) != T:
                    idx = np.linspace(
                        0, max(1, len(silero_vad) - 1), num=T, dtype=np.float32
                    )
                    silero_vad = np.interp(
                        idx, np.arange(len(silero_vad)), silero_vad
                    )

        # NeMo VAD probabilities
        if vad_nemo_dir is not None:
            nemo_path = vad_nemo_dir / f"{uniq_id}.npy"
            if nemo_path.is_file():
                nemo_vad = np.load(nemo_path).astype(np.float32)
                if nemo_vad.ndim > 1:
                    nemo_vad = nemo_vad.squeeze()
                if len(nemo_vad) != T:
                    idx = np.linspace(
                        0, max(1, len(nemo_vad) - 1), num=T, dtype=np.float32
                    )
                    nemo_vad = np.interp(
                        idx, np.arange(len(nemo_vad)), nemo_vad
                    )

        # ------------------------------------------------------------------
        # D) Silero VAD band (scaled for visibility)
        # We scale by a constant factor so tiny raw probs (~1e-3) become visible.
        # The legend explicitly says "×1000" so it is not confused with 0–1.
        # ------------------------------------------------------------------
        if silero_vad is not None:
            v = silero_vad.astype(np.float32)

            scale = 1000.0  # adjust if you ever need a different factor
            v_scaled = v * scale

            # clamp to [0, 1] for plotting inside the lane
            v_scaled = np.clip(v_scaled, 0.0, 1.0)

            y_vad1 = vad1_bot + v_scaled * (vad1_top - vad1_bot)
            if lane_idx == 0:
                label = f"Silero VAD (×{int(scale)}, raw prob)"
            else:
                label = None

            ax.plot(
                t_axis,
                y_vad1,
                color="black",
                linewidth=1.5,
                label=label,
            )

        # ------------------------------------------------------------------
        # E) NeMo VAD band
        # ------------------------------------------------------------------
        if nemo_vad is not None:
            y_vad2 = vad2_bot + nemo_vad * (vad2_top - vad2_bot)
            ax.plot(
                t_axis,
                y_vad2,
                color="tab:orange",  # or any other distinct color
                linewidth=1.5,
                label="NeMo VAD" if lane_idx == 0 else None,
            )

        # separators between VAD bands
        ax.hlines(vad1_bot, win_start, win_end, colors="gray", alpha=0.25, linewidth=0.5)
        ax.hlines(vad2_bot, win_start, win_end, colors="gray", alpha=0.25, linewidth=0.5)

        # ------------------------------------------------------------------
        # F) Speaker probability curves band
        # ------------------------------------------------------------------
        base = prob_bot
        scale = max(prob_top - prob_bot, 1e-6)

        for spk in range(C):
            label = f"Spk{spk+1}" if lane_idx == 0 else None
            curve = base + probs[:, spk] * scale
            ax.plot(
                t_axis,
                curve,
                color=spk_cmap(spk),
                linewidth=1.5,
                alpha=0.95,
                label=label,
            )

        # lane bottom border (for panel separation)
        ax.hlines(lane_bot, win_start, win_end, colors="k", alpha=0.2, linewidth=0.5)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_ylabel("Window Time Range")
    ax.set_xlabel("Absolute Time (seconds)")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # dedup legend labels
    handles, labels = ax.get_legend_handles_labels()
    by_label: Dict[str, object] = {}
    for h, lab in zip(handles, labels):
        if not lab:
            continue
        if lab not in by_label:
            by_label[lab] = h
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize="small")

    # xlim for THIS grid only
    ax.set_xlim(grid_start_sec, grid_end_sec)
    return grid_start_sec, grid_end_sec


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate waterfall VAD grids for one recording."
    )
    parser.add_argument("--base_id", required=True,
                        help="Base recording ID (e.g., 'log_f1_4').")
    parser.add_argument("--probs_dir", type=Path, required=True,
                        help="Directory with <uniq_id>.npy [T,4] prob tensors.")
    parser.add_argument("--vad_silero_dir", type=Path, required=True,
                        help="Directory with <uniq_id>.npy Silero VAD arrays.")
    parser.add_argument("--vad_nemo_dir", type=Path, required=True,
                        help="Directory with <uniq_id>.npy NeMo VAD arrays.")

    parser.add_argument("--rttm_dir", type=Path, default=None,
                        help="Directory with '<base-id>.rttm'.")
    parser.add_argument("--out_dir", type=Path, required=True,
                        help="Directory to save PNGs.")
    parser.add_argument("--windows_per_grid", type=int, default=6,
                        help="Number of windows per grid row.")
    parser.add_argument("--num_grids", type=int, default=5,
                        help="Number of grid rows to generate.")
    parser.add_argument("--frame_step_sec", type=float, default=0.08,
                        help="Frame step (seconds) used when computing probs.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Enumerate all windows from probs_dir
    all_window_files = get_window_files(args.probs_dir, args.base_id)
    if not all_window_files:
        print(f"Error: no .npy windows for base-id={args.base_id} in {args.probs_dir}")
        return

    total_needed = args.windows_per_grid * args.num_grids
    if len(all_window_files) < total_needed:
        print(
            f"Warning: requested {args.num_grids} grids × {args.windows_per_grid} "
            f"windows ({total_needed}), but only {len(all_window_files)} windows exist."
        )
        args.num_grids = max(
            1,
            (len(all_window_files) + args.windows_per_grid - 1) // args.windows_per_grid,
        )
        total_needed = args.windows_per_grid * args.num_grids

    # Load RTTM if provided
    all_gt_segments: Optional[List[RttmSegment]] = None
    if args.rttm_dir:
        rttm_path = args.rttm_dir / f"{args.base_id}.rttm"
        if rttm_path.is_file():
            all_gt_segments = read_rttm(rttm_path, args.base_id)
        else:
            print(f"Warning: RTTM file not found: {rttm_path}")

    # OVERALL stacked figure
    fig_overall, axes_overall = plt.subplots(
        nrows=args.num_grids,
        ncols=1,
        figsize=(20, 8 * args.num_grids),
        squeeze=False,
        sharex=False,   # important: each grid has its own xlim
    )
    fig_overall.suptitle(f"Overall Analysis for {args.base_id}", fontsize=16, y=0.99)

    file_idx = 0
    for grid_idx in range(args.num_grids):
        grid_num = grid_idx + 1
        grid_window_files = all_window_files[
            file_idx: file_idx + args.windows_per_grid
        ]
        file_idx += args.windows_per_grid
        if not grid_window_files:
            continue

        # Per-grid figure (individual image)
        fig_single, ax_single = plt.subplots(figsize=(20, 20))
        grid_start, grid_end = _draw_single_grid(
            ax_single,
            grid_window_files,
            all_gt_segments,
            probs_dir=args.probs_dir,
            vad_silero_dir=args.vad_silero_dir,
            vad_nemo_dir=args.vad_nemo_dir,
            frame_step_sec=args.frame_step_sec,
        )

        ax_single.set_title(
            f"Grid {grid_num}: {grid_start:.1f}s to {grid_end:.1f}s"
        )

        out_single = args.out_dir / f"{args.base_id}_grid_{grid_num:02d}_VAD.png"
        fig_single.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_single.savefig(out_single)
        plt.close(fig_single)
        print(f"Saved single grid plot to: {out_single}")

        # Same grid into OVERALL figure
        ax_overall = axes_overall[grid_idx, 0]
        grid_start_ov, grid_end_ov = _draw_single_grid(
            ax_overall,
            grid_window_files,
            all_gt_segments,
            probs_dir=args.probs_dir,
            vad_silero_dir=args.vad_silero_dir,
            vad_nemo_dir=args.vad_nemo_dir,
            frame_step_sec=args.frame_step_sec,
        )

        ax_overall.set_title(
            f"Grid {grid_num}: {grid_start_ov:.1f}s to {grid_end_ov:.1f}s"
        )

    out_overall = args.out_dir / f"{args.base_id}_grid_OVERALL_VAD.png"
    fig_overall.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig_overall.savefig(out_overall)
    plt.close(fig_overall)
    print(f"Saved OVERALL plot to: {out_overall}")
    print("Done.")


if __name__ == "__main__":
    main()

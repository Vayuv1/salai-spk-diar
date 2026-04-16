"""
role_cer_analysis.py

Per-role CER analysis for the 4 eval recordings.

Note on methodology:
  Per-window predicted RTTMs have independent speaker labels (speaker_0,
  speaker_1 ...) in each window — no global stitching.  Running role-filtered
  DER per-window gives artifactually low numbers because per-window optimal
  Hungarian matching trivially resolves single-role frames.

  Valid approach: classify each 10 s window by its dominant GT role (controller
  or pilot), compute full per-window CER (no role filtering), and aggregate.
  This gives the mean CER experienced in controller-dominated vs pilot-dominated
  acoustic contexts, which is the relevant diagnostic for the paper.

  Additionally, report GT role composition (speech duration fraction) per
  recording to contextualise the aggregate DER numbers.

Outputs:
  - Console table
  - results/paper_figures/role_cer_table.json

Usage:
    uv run python -m spkdiar.analysis.role_cer_analysis
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

EVAL_RECS   = ["dca_d1_1", "dca_d2_2", "dfw_a1_1", "log_id_1"]
GT_RTTM_DIR = Path("data/processed/rttm")
COLLAR      = 0.25
WINDOW_DUR  = 10.0
CENTER_PAD  = 2.5   # center-crop: keep [start+2.5, start+7.5]

SYSTEMS = {
    "pretrained": Path("results/sortformer_offline/pred_rttm"),
    "finetuned":  Path("results/sortformer_finetuned/pred_rttm"),
}


# ---------------------------------------------------------------------------
# RTTM helpers
# ---------------------------------------------------------------------------

def is_controller(raw_speaker_id: str, rec_id: str) -> bool:
    """Controller IDs contain a hyphen after stripping the {rec_id}_ prefix."""
    prefix = rec_id + "_"
    bare   = raw_speaker_id[len(prefix):] if raw_speaker_id.startswith(prefix) else raw_speaker_id
    return "-" in bare


def rttm_to_annotation(rttm_path: Path) -> Annotation:
    ann = Annotation()
    if not rttm_path.exists():
        return ann
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9 or parts[0] != "SPEAKER":
                continue
            seg_s = float(parts[3])
            seg_e = seg_s + float(parts[4])
            ann[Segment(seg_s, seg_e)] = parts[7]
    return ann


def clip_annotation(ann: Annotation, t_start: float, t_end: float) -> Annotation:
    out = Annotation()
    for seg, _, spk in ann.itertracks(yield_label=True):
        cs = max(seg.start, t_start)
        ce = min(seg.end,   t_end)
        if ce > cs:
            out[Segment(cs, ce)] = spk
    return out


# ---------------------------------------------------------------------------
# GT role composition for one recording
# ---------------------------------------------------------------------------

def role_composition(gt_ann: Annotation, rec_id: str) -> dict:
    """Return total speech duration split by controller/pilot."""
    ctrl_dur  = 0.0
    pilot_dur = 0.0
    for seg, _, spk in gt_ann.itertracks(yield_label=True):
        dur = seg.end - seg.start
        if is_controller(spk, rec_id):
            ctrl_dur  += dur
        else:
            pilot_dur += dur
    total = ctrl_dur + pilot_dur
    return {
        "controller_sec":      ctrl_dur,
        "pilot_sec":           pilot_dur,
        "total_sec":           total,
        "controller_fraction": ctrl_dur / total if total > 0 else 0.0,
        "pilot_fraction":      pilot_dur / total if total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Per-window role classification
# ---------------------------------------------------------------------------

def window_dominant_role(
    gt_ann:  Annotation,
    rec_id:  str,
    t_start: float,
    t_end:   float,
) -> str | None:
    """Return 'controller', 'pilot', or None (< 1s total speech in window)."""
    ctrl  = 0.0
    pilot = 0.0
    for seg, _, spk in gt_ann.itertracks(yield_label=True):
        cs = max(seg.start, t_start)
        ce = min(seg.end,   t_end)
        if ce <= cs:
            continue
        if is_controller(spk, rec_id):
            ctrl  += ce - cs
        else:
            pilot += ce - cs
    if ctrl + pilot < 0.5:
        return None
    return "controller" if ctrl >= pilot else "pilot"


# ---------------------------------------------------------------------------
# Per-recording analysis
# ---------------------------------------------------------------------------

def evaluate_recording(
    rec_id:       str,
    gt_rttm_path: Path,
    pred_dir:     Path,
) -> dict:
    """
    For every window in pred_dir for rec_id:
      1. Classify window by dominant GT role.
      2. Compute per-window CER (full DER with center-crop, no role filtering).
      3. Accumulate confusion / total per role.

    Returns dict with:
      role_composition: speech duration stats from GT
      controller:  {confusion, total, cer, n_windows, cer_values}
      pilot:       {confusion, total, cer, n_windows, cer_values}
    """
    gt_ann = rttm_to_annotation(gt_rttm_path)
    comp   = role_composition(gt_ann, rec_id)

    accum = {
        "controller": {"confusion": 0.0, "total": 0.0, "n_windows": 0, "cer_values": []},
        "pilot":      {"confusion": 0.0, "total": 0.0, "n_windows": 0, "cer_values": []},
    }

    pattern   = f"{rec_id}-*-10000.rttm"
    win_files = sorted(pred_dir.glob(pattern))
    if not win_files:
        log.warning(f"No pred RTTM files for {rec_id} in {pred_dir}")
        return {"role_composition": comp, **accum}

    metric = DiarizationErrorRate(collar=COLLAR, skip_overlap=False)

    for rttm_file in win_files:
        stem     = rttm_file.stem
        parts    = stem.split("-")
        start_ms = int(parts[-2])
        start_s  = start_ms / 1000.0
        end_s    = start_s + WINDOW_DUR

        # Center-crop
        cs  = start_s + CENTER_PAD
        ce  = end_s   - CENTER_PAD
        uem = Timeline([Segment(cs, ce)])

        # Classify window
        role = window_dominant_role(gt_ann, rec_id, cs, ce)
        if role is None:
            continue

        ref  = clip_annotation(gt_ann, cs, ce)
        hyp  = clip_annotation(rttm_to_annotation(rttm_file), cs, ce)

        total = sum((s.end - s.start) for s, _ in ref.itertracks())
        if total <= 0.0:
            continue

        result   = metric(ref, hyp, uem=uem, detailed=True)
        conf     = float(result.get("confusion", 0.0))
        tot      = float(result.get("total", 0.0))
        win_cer  = conf / tot if tot > 0 else 0.0

        accum[role]["confusion"] += conf
        accum[role]["total"]     += tot
        accum[role]["n_windows"] += 1
        accum[role]["cer_values"].append(win_cer)

    # Compute aggregate CER for each role
    for role in accum:
        tot = accum[role]["total"]
        accum[role]["cer"]        = accum[role]["confusion"] / tot if tot > 0 else 0.0
        accum[role]["median_cer"] = float(np.median(accum[role]["cer_values"])) if accum[role]["cer_values"] else 0.0
        accum[role]["mean_cer"]   = float(np.mean(accum[role]["cer_values"]))   if accum[role]["cer_values"] else 0.0

    return {"role_composition": comp, **accum}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    gt_rttm_dir: Path = GT_RTTM_DIR,
    out_dir:     Path = Path("results/paper_figures"),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {}

    for system_name, pred_dir in SYSTEMS.items():
        log.info(f"\n=== System: {system_name} ===")
        results[system_name] = {}

        for rec_id in EVAL_RECS:
            gt_rttm = gt_rttm_dir / f"{rec_id}.rttm"
            r = evaluate_recording(rec_id, gt_rttm, pred_dir)
            results[system_name][rec_id] = r
            comp = r["role_composition"]
            log.info(
                f"  {rec_id:12s}  "
                f"ctrl={comp['controller_fraction']:.1%} of speech  "
                f"ctrl CER={r['controller']['cer']:.4f} ({r['controller']['n_windows']} windows)  "
                f"pilot CER={r['pilot']['cer']:.4f} ({r['pilot']['n_windows']} windows)"
            )

    # ---- Summary table ----
    hdr = (f"{'Rec':12s}  {'System':12s}  "
           f"{'Ctrl%':>7s}  {'Ctrl CER':>9s}  {'Ctrl n':>7s}  "
           f"{'Pilot%':>7s}  {'Pilot CER':>9s}  {'Pilot n':>7s}")
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for rec_id in EVAL_RECS:
        for system_name in SYSTEMS:
            r    = results[system_name][rec_id]
            comp = r["role_composition"]
            print(
                f"{rec_id:12s}  {system_name:12s}  "
                f"{comp['controller_fraction']:>7.1%}  "
                f"{r['controller']['cer']:>9.4f}  {r['controller']['n_windows']:>7d}  "
                f"{comp['pilot_fraction']:>7.1%}  "
                f"{r['pilot']['cer']:>9.4f}  {r['pilot']['n_windows']:>7d}"
            )
    print("=" * len(hdr))

    # Flatten cer_values lists for JSON serialisation
    for sys in results:
        for rec in results[sys]:
            for role in ("controller", "pilot"):
                results[sys][rec][role].pop("cer_values", None)

    out_json = out_dir / "role_cer_table.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()

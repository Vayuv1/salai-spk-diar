"""
run_lseend.py

Run LS-EEND inference on ATC0R recordings.

LS-EEND uses a Conformer + Retention architecture (no softmax self-attention).
This script compares against Sortformer to determine whether the lock-up pattern
is architecture-specific (attention entropy collapse) or domain-general.

Prerequisite: download the CALLHOME-finetuned checkpoint from Google Drive and
place it at models/lseend_callhome.ckpt:
  https://drive.google.com/file/d/1W8nYAB6YoEKMM5KZX-apVADvHaYc2Fre/view

Model inputs:
  - Audio resampled to 8 kHz
  - 23-dim log-mel features, STFT frame_size=200 → fft=256, hop=80 (10 ms/frame)
  - Cumulative mean normalization (logmel23_cummn)
  - Subsampling by 10 → 100 ms per output frame
  - Context splicing: 7 frames each side → 345-dim input
  - Sequence padded to multiple of recurrent_chunk_size=500

Checkpoint format: PyTorch Lightning .ckpt
  - Keys prefixed with "model." → stripped when loading into bare nn.Module

Output columns (max_nspks=9 with CALLHOME config max_speakers=7):
  col 0: silence  |  cols 1-7: speakers  |  col 8: none

Feature normalization note:
  ATC VHF radio audio produces logmel features with std ≈ 0.31 after cumulative
  mean normalization, vs. ≈ 1.0 for CALLHOME telephone data.  At std=0.31 the
  model saturates to 100% silence; at std=1.0 it produces meaningful activations.
  --feat-normalize (on by default) rescales each recording's features so that
  std = --target-std (default 1.0) before feeding the model.

Usage:
    uv run python -m spkdiar.inference.run_lseend \
        --model-path models/lseend_callhome.ckpt \
        --audio-dir data/atc0r/audio \
        --rttm-dir data/processed/rttm \
        --out-dir results/lseend \
        --rec-ids dca_d1_1,dca_d1_3,log_id_1 \
        --max-duration 350 \
        --threshold 0.99
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Make LS-EEND importable without installing it
# ---------------------------------------------------------------------------
_LSEEND_ROOT = Path(__file__).resolve().parents[3] / "external" / "FS-EEND" / "LS-EEND"
if str(_LSEEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_LSEEND_ROOT))

from nnet.model.onl_conformer_retention_enc_1dcnn_tfm_retention_enc_linear_non_autoreg_pos_enc_l2norm_emb_loss_mask import (  # noqa: E501
    OnlineConformerRetentionDADiarization,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model configuration (matches spk_onl_conformer_retention_enc_dec_nonautoreg_callhome_infer.yaml)
# ---------------------------------------------------------------------------
MODEL_CFG = dict(
    n_units=256,
    n_heads=4,
    enc_n_layers=4,
    dec_n_layers=2,
    dropout=0.1,
    max_seqlen=10000,
    recurrent_chunk_size=500,
    feed_forward_expansion_factor=4,
    conv_expansion_factor=2,
    conv_kernel_size=16,
    half_step_residual=True,
    conv_delay=9,
    mask_delay=0,
)

# Feature hyper-parameters
SAMPLE_RATE = 8000
FRAME_SIZE = 200       # samples (win_length); fft_size rounds up to 256
HOP_LENGTH = 80        # samples → 10 ms per STFT frame
N_MELS = 23
SUBSAMPLING = 10       # → 100 ms per output frame
CONTEXT_SIZE = 7       # frames on each side → input_dim = (2*7+1)*23 = 345
MAX_SPEAKERS = 7       # CALLHOME config: max_speakers
MAX_NSPKS = MAX_SPEAKERS + 2  # silence col + N speaker cols + none col = 9

FRAME_SHIFT_SEC = (HOP_LENGTH / SAMPLE_RATE) * SUBSAMPLING  # 0.1 s

# Binarization: model outputs raw logits (BCE loss), default threshold 0.99.
# ATC audio requires a high threshold because the CALLHOME-trained model is not
# calibrated for ATC's sparse speech pattern; the bimodal logit distribution
# (frames are either ~-0.9999 or ~+0.9993 with a sharp boundary near p90)
# means near-integer thresholds are needed to suppress noise floor detections.
LOGIT_THRESHOLD = 0.99

# Feature normalization: rescale logmel features so that per-recording std ≈
# TARGET_STD before feeding the model.  ATC audio has std ≈ 0.31 after
# cumulative mean norm; CALLHOME training data had std ≈ 1.0.  Without this
# rescaling the model saturates to 100% silence on ATC audio.
FEAT_NORMALIZE = True
TARGET_STD = 1.0

COLLAR = 0.25  # seconds, consistent with Sortformer evaluation


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _stft_frames(audio: np.ndarray) -> np.ndarray:
    """Compute STFT magnitude, matching feature.py stft() exactly.

    feature.py rounds win_length up to the next power of two to get fft_size,
    then drops the last frame when len(audio) % hop_length == 0 (librosa
    center=True pads symmetrically and produces one spurious extra frame at
    exact multiples of hop_length).

    Returns: (T, fft_size//2+1) float32 magnitude array.
    """
    fft_size = 1 << (FRAME_SIZE - 1).bit_length()  # 200 → 256
    frames = librosa.stft(
        audio.astype(np.float32),
        n_fft=fft_size,
        win_length=FRAME_SIZE,
        hop_length=HOP_LENGTH,
        center=True,
    ).T  # (T, F)
    if len(audio) % HOP_LENGTH == 0:
        frames = frames[:-1]
    return np.abs(frames).astype(np.float32)  # (T, 129)


def _logmel23_cummn(
    mag: np.ndarray,
    normalize: bool = FEAT_NORMALIZE,
    target_std: float = TARGET_STD,
) -> np.ndarray:
    """23-dim log-mel + cumulative mean norm from STFT magnitude.

    Replicates the 'logmel23_cummn' branch of feature.py transform().
    n_fft is inferred from mag.shape[1] as 2*(bins-1), matching the original.

    normalize: if True, rescale result so that per-recording std = target_std.
        ATC VHF audio has std ≈ 0.31 after cummn, vs. ≈ 1.0 for CALLHOME;
        without rescaling the model saturates to 100% silence on ATC audio.
    """
    n_fft = 2 * (mag.shape[1] - 1)  # 2*(129-1) = 256
    mel_basis = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=n_fft, n_mels=N_MELS)
    logmel = np.log10(np.maximum(np.dot(mag ** 2, mel_basis.T), 1e-10)).astype(np.float32)

    cumsum = np.cumsum(logmel, axis=0)
    idx = np.arange(1, logmel.shape[0] + 1)
    cummean = cumsum / idx[:, None]
    logmel = (logmel - cummean).astype(np.float32)  # (T, 23)  — zero mean by construction

    log.info(
        f"  [feat] logmel BEFORE norm: mean={logmel.mean():.4f}  std={logmel.std():.4f}"
        f"  min={logmel.min():.4f}  max={logmel.max():.4f}  normalize={normalize}"
    )

    if normalize:
        s = float(logmel.std())
        if s > 1e-6:
            scale = target_std / s
            logmel = (logmel * scale).astype(np.float32)
            log.info(
                f"  [feat] logmel AFTER  norm: scale={scale:.4f}"
                f"  std={logmel.std():.4f}  min={logmel.min():.4f}  max={logmel.max():.4f}"
            )
        else:
            log.info(f"  [feat] logmel std={s:.2e} too small — normalization skipped")

    return logmel


def _splice(Y: np.ndarray) -> np.ndarray:
    """Concatenate ±CONTEXT_SIZE neighbouring frames → (T, (2*C+1)*F).

    Matches feature.py splice() exactly.
    """
    Y_pad = np.pad(Y, [(CONTEXT_SIZE, CONTEXT_SIZE), (0, 0)], "constant")
    Y_spliced = np.lib.stride_tricks.as_strided(
        Y_pad,
        shape=(Y.shape[0], Y.shape[1] * (2 * CONTEXT_SIZE + 1)),
        strides=(Y_pad.strides[0], Y_pad.strides[1]),
        writeable=False,
    )
    return Y_spliced.copy()


def extract_features(
    audio_path: Path,
    max_duration: float | None,
    normalize: bool = FEAT_NORMALIZE,
    target_std: float = TARGET_STD,
) -> tuple[torch.Tensor, int]:
    """Load audio, resample to 8 kHz, compute model input features.

    Pipeline (matches diarization_dataset.py exactly):
        STFT magnitude → logmel23_cummn [+ optional std norm] → splice → subsample
                                                                  ↑ splice FIRST (10ms frames)
                                                                             ↑ subsample LAST

    normalize: rescale logmel (23-dim) to std≈target_std BEFORE splicing.
        Must be applied at this stage — after splicing the 345-dim array would
        still have the same std (splice is just concatenation), but applying it
        at the raw 23-dim stage is the correct insertion point matching the
        training pipeline's normalisation intent.

    Returns:
        feat: (T_sub, 345) float32 tensor  (345 = 23 × 15 context frames)
        n_frames: number of output frames
    """
    audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    if max_duration is not None:
        audio = audio[: int(max_duration * SAMPLE_RATE)]

    log.info(f"  [feat] audio: samples={len(audio)}  sr={SAMPLE_RATE}  duration={len(audio)/SAMPLE_RATE:.1f}s")

    mag     = _stft_frames(audio)                                         # (T_stft, 129)
    logmel  = _logmel23_cummn(mag, normalize=normalize, target_std=target_std)  # (T_stft, 23)
    spliced = _splice(logmel)                                             # (T_stft, 345)  ← splice FIRST
    feat_np = spliced[::SUBSAMPLING]                                      # (T_sub,  345)  ← subsample LAST

    log.info(
        f"  [feat] final: shape={feat_np.shape}"
        f"  mean={feat_np.mean():.4f}  std={feat_np.std():.4f}"
        f"  min={feat_np.min():.4f}  max={feat_np.max():.4f}"
    )

    feat = torch.from_numpy(feat_np)
    return feat, feat_np.shape[0]


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, device: torch.device) -> OnlineConformerRetentionDADiarization:
    """Instantiate model and load a PyTorch Lightning checkpoint."""
    input_dim = (2 * CONTEXT_SIZE + 1) * N_MELS  # 345
    model = OnlineConformerRetentionDADiarization(
        n_speakers=None,  # unused in .test()
        in_size=input_dim,
        **MODEL_CFG,
    )

    log.info(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # Unwrap PL checkpoint: weights may be under a "state_dict" key or at the top level.
    raw: dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # Strip "model." prefix if present (added by PL's SpeakerDiarization wrapper).
    # Auto-detect: check the first tensor key.
    first_key = next(k for k, v in raw.items() if isinstance(v, torch.Tensor))
    if first_key.startswith("model."):
        prefix = "model."
        state = {k[len(prefix):]: v for k, v in raw.items()
                 if isinstance(v, torch.Tensor) and k.startswith(prefix)}
    else:
        state = {k: v for k, v in raw.items() if isinstance(v, torch.Tensor)}

    log.info(f"Checkpoint keys found: {len(state)}")

    missing, unexpected = model.load_state_dict(state, strict=True)
    # strict=True raises on mismatch; the lines below are a belt-and-suspenders
    # check in case load_state_dict is ever called with strict=False.
    assert not missing, f"{len(missing)} missing keys: {missing[:5]}"
    assert not unexpected, f"{len(unexpected)} unexpected keys: {unexpected[:5]}"

    log.info(f"Loaded {len(state)} keys, 0 missing, 0 unexpected")

    model.to(device)
    model.eval()
    log.info(f"Model loaded  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model


# ---------------------------------------------------------------------------
# Inference → RTTM
# ---------------------------------------------------------------------------

def infer_recording(
    model: OnlineConformerRetentionDADiarization,
    feat: torch.Tensor,
    n_frames: int,
    device: torch.device,
    threshold: float = LOGIT_THRESHOLD,
) -> np.ndarray:
    """Run model.test() and return binary (T, MAX_SPEAKERS) activity matrix.

    Returns shape (n_frames, MAX_SPEAKERS) with 0/1 values.
    """
    feat = feat.to(device).unsqueeze(0)  # (1, T, 345)
    with torch.no_grad():
        outputs, _embs, _attractors = model.test(
            [feat.squeeze(0)],  # model expects list of (T, D) tensors
            [n_frames],
            max_nspks=MAX_NSPKS,
        )
    # outputs[0]: (n_frames, MAX_NSPKS=9)
    # col 0 = silence, cols 1..MAX_SPEAKERS = speakers, col MAX_NSPKS-1 = none
    logits = outputs[0].cpu().float()  # (T, 9)
    probs = torch.tanh(logits).numpy().astype(np.float32)  # (T, 9) full prob tensor
    spk_logits = logits[:, 1 : MAX_SPEAKERS + 1]  # (T, 7) speaker columns only

    log.info(f"  [infer] logits shape={logits.shape}  threshold={threshold}")
    log.info(f"  [infer] col0(sil): mean={logits[:,0].mean():.4f}  min={logits[:,0].min():.4f}  max={logits[:,0].max():.4f}")
    for i in range(MAX_SPEAKERS):
        col = spk_logits[:, i]
        pct = 100.0 * float((col > threshold).float().mean())
        log.info(f"  [infer] spk{i}: mean={col.mean():.4f}  min={col.min():.4f}  max={col.max():.4f}  pct_active={pct:.1f}%")

    activity = (spk_logits > threshold).numpy().astype(np.int32)
    return activity, probs  # (T, 7), (T, 9)


def activity_to_rttm(
    activity: np.ndarray,
    rec_id: str,
    frame_shift: float = FRAME_SHIFT_SEC,
    min_duration: float = 0.1,
) -> list[tuple[str, float, float]]:
    """Convert binary (T, S) activity matrix to list of (speaker, start, end) tuples."""
    T, S = activity.shape
    segments = []
    for s in range(S):
        col = activity[:, s]
        # pad to detect edge transitions
        padded = np.pad(col, (1, 1), constant_values=0)
        changes = np.where(np.diff(padded.astype(int)))[0]
        for onset, offset in zip(changes[::2], changes[1::2]):
            dur = (offset - onset) * frame_shift
            if dur >= min_duration:
                start = onset * frame_shift
                segments.append((f"{rec_id}_spk{s:02d}", start, start + dur))
    return segments


def write_rttm(segments: list[tuple[str, float, float]], rec_id: str, path: Path) -> None:
    with open(path, "w") as f:
        for spk, start, end in segments:
            f.write(f"SPEAKER {rec_id} 1 {start:.3f} {end - start:.3f} <NA> <NA> {spk} <NA> <NA>\n")


# ---------------------------------------------------------------------------
# DER evaluation (pyannote.metrics)
# ---------------------------------------------------------------------------

def load_rttm_annotation(rttm_path: Path, max_duration: float | None = None):
    """Parse an RTTM file into a pyannote.core.Annotation.

    max_duration: if set, discard any segment that starts at or after this
        time and clip the end of segments that extend beyond it.  Must match
        the --max-duration used for inference so that reference and hypothesis
        cover the same time span.  Omitting this produces a full-recording
        reference, making MISS ≈ 100% whenever the hypothesis covers only a
        short prefix.
    """
    from pyannote.core import Annotation, Segment

    ann = Annotation()
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            dur = float(parts[4])
            spk = parts[7]
            if max_duration is not None:
                if start >= max_duration:
                    continue
                dur = min(dur, max_duration - start)
            ann[Segment(start, start + dur)] = spk
    return ann


def evaluate_der(
    hyp_rttm: Path,
    ref_rttm: Path,
    rec_id: str,
    collar: float = COLLAR,
    max_duration: float | None = None,
) -> dict:
    """Compute DER between hypothesis and reference, both clipped to max_duration.

    max_duration must match the --max-duration used for inference.  Without it,
    the reference covers the full recording while the hypothesis covers only a
    prefix, so MISS is inflated to nearly 100% by the uncovered tail.
    """
    from pyannote.core import Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate

    metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
    hyp = load_rttm_annotation(hyp_rttm, max_duration=max_duration)
    ref = load_rttm_annotation(ref_rttm, max_duration=max_duration)

    # Supply an explicit UEM so pyannote evaluates exactly [0, max_duration).
    # Without this it approximates the UEM as the union of ref+hyp extents,
    # which can differ when one is shorter than the other.
    if max_duration is not None:
        uem = Timeline([Segment(0.0, max_duration)])
        der = metric(ref, hyp, uem=uem, detailed=True)
    else:
        der = metric(ref, hyp, detailed=True)

    total = der["total"]
    return {
        "rec_id": rec_id,
        "DER": der["diarization error rate"],
        "FA": der["false alarm"] / total if total > 0 else 0.0,
        "MISS": der["missed detection"] / total if total > 0 else 0.0,
        "CER": der["confusion"] / total if total > 0 else 0.0,
        "total_ref_speech": total,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run LS-EEND inference on ATC0R recordings.")
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Path to CALLHOME-finetuned LS-EEND checkpoint (.ckpt)")
    parser.add_argument("--audio-dir", type=Path, default=Path("data/atc0r/audio"),
                        help="Directory containing .sph or .wav audio files")
    parser.add_argument("--rttm-dir", type=Path, default=Path("data/processed/rttm"),
                        help="Directory containing ground-truth RTTM files")
    parser.add_argument("--out-dir", type=Path, default=Path("results/lseend"))
    parser.add_argument("--rec-ids", type=str, default="dca_d1_1,dca_d1_3,log_id_1",
                        help="Comma-separated recording IDs to process")
    parser.add_argument("--max-duration", type=float, default=350.0,
                        help="Truncate recordings to this many seconds (default: 350 s)")
    parser.add_argument("--threshold", type=float, default=LOGIT_THRESHOLD,
                        help="Logit threshold for binarization (default: 0.99). "
                             "ATC audio requires a high threshold because the bimodal "
                             "logit distribution has a sharp boundary near p90.")
    parser.add_argument("--no-feat-normalize", action="store_true",
                        help="Disable per-recording feature std normalization. "
                             "Without normalization, ATC audio produces std≈0.31 vs "
                             "training distribution std≈1.0, causing 100%% silence output.")
    parser.add_argument("--target-std", type=float, default=TARGET_STD,
                        help="Target std for feature normalization (default: 1.0)")
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    log.info(f"Device: {device}")
    normalize = not args.no_feat_normalize
    log.info(f"Feature normalization: {'on (target_std={})'.format(args.target_std) if normalize else 'off'}")

    if not args.model_path.exists():
        log.error(
            f"Checkpoint not found: {args.model_path}\n"
            "Download the CALLHOME-finetuned checkpoint from:\n"
            "  https://drive.google.com/file/d/1W8nYAB6YoEKMM5KZX-apVADvHaYc2Fre/view\n"
            "and save it to models/lseend_callhome.ckpt"
        )
        sys.exit(1)

    rttm_out_dir = args.out_dir / "pred_rttm"
    rttm_out_dir.mkdir(parents=True, exist_ok=True)
    prob_tensors_dir = args.out_dir / "prob_tensors"
    prob_tensors_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model_path, device)

    rec_ids = [r.strip() for r in args.rec_ids.split(",")]
    all_results = []

    for rec_id in rec_ids:
        # Find audio file
        audio_path = None
        for ext in [".wav", ".mp3", ".sph", ".flac"]:
            p = args.audio_dir / f"{rec_id}{ext}"
            if p.exists():
                audio_path = p
                break
        if audio_path is None:
            log.warning(f"Audio not found for {rec_id} in {args.audio_dir}, skipping")
            continue

        ref_rttm = args.rttm_dir / f"{rec_id}.rttm"
        if not ref_rttm.exists():
            log.warning(f"Reference RTTM not found: {ref_rttm}, skipping DER for {rec_id}")

        log.info(f"Processing {rec_id} ({audio_path.name})")
        feat, n_frames = extract_features(audio_path, max_duration=args.max_duration,
                                          normalize=normalize, target_std=args.target_std)
        duration = n_frames * FRAME_SHIFT_SEC
        log.info(f"  Features: {n_frames} frames ({duration:.1f} s), input_dim={feat.shape[1]}")

        activity, probs = infer_recording(model, feat, n_frames, device, threshold=args.threshold)
        log.info(f"  Active speakers detected: {activity.any(axis=0).sum()}")

        # Save full probability tensor (T, 9) as float32
        prob_path = prob_tensors_dir / f"{rec_id}.npy"
        np.save(str(prob_path), probs)
        meta = {
            "frame_step_sec": 0.1,
            "n_frames": int(probs.shape[0]),
            "n_cols": 9,
            "col_0": "silence",
            "cols_1_8": "speakers",
            "threshold_used": args.threshold,
            "feat_normalize": normalize,
        }
        meta_path = prob_tensors_dir / f"{rec_id}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        log.info(f"  Prob tensor saved: {prob_path}  shape={probs.shape}  dtype={probs.dtype}")

        segments = activity_to_rttm(activity, rec_id)
        hyp_rttm = rttm_out_dir / f"{rec_id}.rttm"
        write_rttm(segments, rec_id, hyp_rttm)
        log.info(f"  RTTM written: {hyp_rttm}  ({len(segments)} segments)")

        if ref_rttm.exists():
            result = evaluate_der(hyp_rttm, ref_rttm, rec_id,
                                  max_duration=args.max_duration)
            log.info(
                f"  {rec_id}: DER={result['DER']:.4f}  "
                f"FA={result['FA']:.4f}  MISS={result['MISS']:.4f}  CER={result['CER']:.4f}"
                f"  (ref_speech={result['total_ref_speech']:.1f}s)"
            )
            all_results.append(result)

    if all_results:
        mean_der = np.mean([r["DER"] for r in all_results])
        mean_fa = np.mean([r["FA"] for r in all_results])
        mean_miss = np.mean([r["MISS"] for r in all_results])
        mean_cer = np.mean([r["CER"] for r in all_results])
        log.info(
            f"\n=== LS-EEND Aggregate ({len(all_results)} recordings) ===\n"
            f"  DER : {mean_der:.4f}\n"
            f"  FA  : {mean_fa:.4f}\n"
            f"  MISS: {mean_miss:.4f}\n"
            f"  CER : {mean_cer:.4f}\n"
            f"(collar={COLLAR}s, max_duration={args.max_duration}s, threshold={args.threshold})"
        )


if __name__ == "__main__":
    main()

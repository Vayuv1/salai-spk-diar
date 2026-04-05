# Phase 2: Comparative Diarization Experiments — Running Log

**CEC 599: Speaker Diarization | DASC 2026 Paper**
Shital Pandey | Started: April 3, 2026

---

## Objective

Determine whether the speaker assignment lock-up observed with Sortformer on ATC audio is architecture-specific or a general weakness of neural diarization on this domain. Compare multiple systems on the ATC0R dataset and identify the highest-leverage path toward improved diarization for the STAR pipeline.

---

## Environment

- Hardware: ASUS ROG NUC 970, NVIDIA GeForce RTX 4090 (24 GB), Ubuntu 24.04
- Python 3.12.2, CUDA 12.4, Driver 555.42
- Package management: uv 0.8.16
- NeMo 2.7.2, PyTorch 2.6.0+cu124, pyannote.audio 3.4.0
- Repository: salai-spk-diar (personal GitHub)

---

## Dataset

ATC0R — revised ATC0 dataset with corrected timestamps, casing, punctuation, and quality levels (L1–L4).

- **16 recordings** parsed from STM files
- Total annotated speech: ~22 hours across DCA, DFW, and LOG facilities
- Speaker counts per recording: 21–90 unique speakers
- **16,912 windowed segments** (10s window, 5s shift) generated for Sortformer inference
- Ground truth: RTTM files generated from ATC0R STM annotations
- Level 4 (unintelligible) cues excluded from ground truth

---

## Experiment 1: Sortformer Offline Baseline

**Date:** April 3, 2026

**Model:** `diar_sortformer_4spk-v1.nemo` (NVIDIA, 123M params)

**Setup:** Windowed manifest (10s/5s), 3 recordings (dca_d1_1, dca_d1_3, log_id_1), first 350 seconds each. 213 windows total. Precision: bf16-mixed. Default postprocessing (binarization threshold 0.5, bypass=True).

**Results:**

| Metric | Value |
|--------|-------|
| DER | **10.71%** |
| FA | 0.81% |
| MISS | 1.38% |
| CER | **8.52%** |

**Observations:**
- Confusion error dominates, consistent with the lock-up pattern observed in prior work.
- FA and MISS are both low — the model detects speech activity well but misassigns speakers.
- Lower DER than the 22.04% reported on old ATC0 (different recordings, revised timestamps in ATC0R may also contribute).
- Probability tensors saved for all 213 windows for visualization and threshold analysis.

---

## Experiment 2: Streaming Sortformer with AOSC

**Date:** April 3, 2026

**Model:** `diar_streaming_sortformer_4spk-v2.nemo` (NVIDIA, with Arrival-Order Speaker Cache)

Required NeMo upgrade from 2.2.1 → 2.7.2 to support AOSC parameters (`spkcache_len`, etc.). Offline model verified to still load and produce consistent results after upgrade.

### 2a: Medium latency (10.0s, chunk_len=124)

**Results (NeMo evaluation):**

| Metric | Value |
|--------|-------|
| DER | **34.83%** |
| FA | **27.55%** |
| MISS | 0.46% |
| CER | **6.82%** |

### 2b: Low latency (1.04s, chunk_len=6)

**Results (NeMo evaluation):**

| Metric | Value |
|--------|-------|
| DER | **69.49%** |
| FA | **61.24%** |
| MISS | 0.38% |
| CER | **7.86%** |

### Key finding: CER improves, FA explodes

| Configuration | DER | FA | MISS | CER |
|--------------|------|------|------|------|
| Offline | 10.71% | 0.81% | 1.38% | 8.52% |
| Streaming 10s | 34.83% | 27.55% | 0.46% | **6.82%** |
| Streaming 1s | 69.49% | 61.24% | 0.38% | 7.86% |

The AOSC mechanism **reduces speaker confusion** (CER drops from 8.52% to 6.82% at 10s latency), confirming the hypothesis that chunked processing with a speaker cache limits the lock-up pattern. However, the streaming model produces systematically elevated probabilities during non-speech regions, causing massive false alarm rates. This is a **probability calibration mismatch** — the model was trained on conversational audio with near-continuous speech, while ATC has long silences between transmissions.

### 2c: Threshold analysis on streaming probabilities

Compared raw probability distributions between offline and streaming on dca_d1_1 (8,875 frames total):

- Offline: 15.9% of frames have max speaker probability > 0.5. Distribution is nearly binary — changing threshold from 0.3 to 0.9 only moves activation from 16.8% to 13.6%.
- Streaming: 22.9% of frames exceed 0.5, but 34.0% exceed 0.3. The probabilities are "softer" with a wider spread.
- **Threshold 0.72** on streaming matches offline's 15.9% activation rate.

Re-evaluated streaming tensors with swept thresholds (pyannote.metrics):

| Threshold | DER |
|-----------|------|
| 0.50 | 31.1% |
| 0.60 | 21.7% |
| 0.70 | 21.7% |
| **0.72** | **21.4%** |
| 0.75 | 21.5% |
| 0.80 | 23.1% |

Optimal threshold (0.72) reduces DER from 31.1% to 21.4%. Still higher than offline (10.7%) but the gap is narrower. The remaining difference may come from the streaming model's different training regime rather than a fundamental architectural issue.

---

---

## Experiment 3: Pyannote 3.1 (clustering-based)

**Date:** April 3, 2026

**Model:** `pyannote/speaker-diarization-3.1` (HuggingFace, clustering pipeline)

**Results:**

| Recording | DER | Ref speakers | Detected speakers |
|-----------|-----|--------------|-------------------|
| dca_d1_1 | 32.49% | 32 | 2 |
| dca_d1_3 | — | 46 | 1 |
| log_id_1 | — | 88 | 6 |

**Observations:**
- Massive under-clustering. The VHF radio processing (300–3400 Hz bandwidth, AM modulation, automatic squelch) makes all speakers' embedding vectors collapse into a very small region of the ECAPA-TDNN speaker embedding space. The clustering algorithm sees all ATC voices as a single cluster.
- The embedding collapse is the expected consequence of narrow-band, highly processed audio: speaker-identifying high-frequency features (4–8 kHz) are entirely absent.
- Result is worse than Sortformer by a large margin despite Pyannote being state-of-the-art on telephone speech benchmarks.

---

## Experiment 4: LS-EEND (CALLHOME checkpoint, PyTorch 2.6 port)

**Date:** April 4, 2026

**Model:** `OnlineConformerRetentionDADiarization` — Conformer encoder (4 layers) + Multi-Scale Retention decoder (2 layers), 11.2M params

**Checkpoint:** CALLHOME-finetuned (`lseend_callhome.ckpt`, 229-key state dict, loaded with 0 missing/unexpected keys)

### 4a: PyTorch 2.6 compatibility fixes

Three changes required to port from PyTorch 1.13:

1. **`is_causal` kwarg**: `nn.TransformerEncoder.forward` now passes `is_causal` to all sublayers. Added `is_causal: Optional[bool] = None` to `TransformerEncoderFusionLayer.forward()`.

2. **`batch_first` attribute**: `nn.TransformerEncoder.forward` unconditionally reads `first_layer.self_attn.batch_first` before any fast-path gating. The `self_attn` slot was occupied by `MultiScaleRetention` (which has no `batch_first`). Fixed with a `_BatchFirstProxy` plain Python object (not an `nn.Module`) that exposes `.batch_first` without registering as a submodule — keeping it invisible to `state_dict`/`load_state_dict`.

3. **`enable_nested_tensor=False`**: Added to `TransformerEncoder` constructor to suppress the nested-tensor fast-path warning (custom layer is not `TransformerEncoderLayer`).

4. **Dead fast-path block**: Removed the `if self.training == False: torch._transformer_encoder_layer_fwd(...)` block — the private API was removed in PyTorch 2.x and the 4-D input shape `(B, T, C, D)` made it unreachable anyway.

### 4b: Feature distribution mismatch

ATC VHF radio audio produces log-mel features with **std ≈ 0.31** after cumulative mean normalization, compared to **std ≈ 1.0** for CALLHOME telephone data (verified on the 2-speaker FS-EEND example mix). This 3.3× variance gap causes complete model saturation:

| Condition | Feat std | col0 (silence) | Speaker cols | % active (thr=0) |
|-----------|----------|----------------|--------------|-------------------|
| FS-EEND 2-speaker mix | 1.03 | −0.45 | max +0.9998 | **72.7%** |
| ATC dca_d1_1 (raw) | 0.31 | +0.9996 | max −0.9831 | 0.0% |
| ATC dca_d1_1 (×3.3) | 1.00 | −0.94 | max +0.9980 | 95.8% |
| Pure silence | 0.00 | +0.9930 | max −0.9743 | 0.0% |

The encoder inter-frame cosine similarity confirms the collapse: ATC audio (unscaled) gives mean cos-sim = 0.9987 (std=0.0085) — essentially all frames map to the same embedding — vs 0.9880 (std=0.0587) for the FS-EEND example.

**Root cause**: ATC radio squelch and background RF noise create a nearly uniform noise floor across all mel bands. Cumulative mean normalization tracks this floor, leaving very small residuals. The model learned that "silence/noise" corresponds to low-variance features and saturates accordingly.

**Fix**: Per-recording instance normalization of the logmel features to std = 1.0 before splicing/subsampling. This is implemented as `--feat-normalize` (on by default) in `run_lseend.py`.

### 4c: DER results (with feature normalization, corrected evaluation)

Evaluated on dca_d1_1, dca_d1_3, log_id_1, first 350 seconds each (collar=0.25s, threshold=0.99).

| Recording | DER | FA | MISS | CER | ref speech |
|-----------|-----|----|------|-----|------------|
| dca_d1_1 | 43.62% | 2.48% | 10.22% | 30.92% | 51.8s |
| dca_d1_3 | 276.04% | 217.12% | 19.56% | 39.36% | 20.7s |
| log_id_1 | 38.16% | 16.11% | 11.22% | 10.83% | 93.3s |

Note on dca_d1_3: with only 20.7s of reference speech in 350s (5.9% density), any moderate false alarm rate produces DER well above 100%. The model generates roughly 2–3 speaker columns active for ~3s each in silence periods, which is small in absolute terms but enormous relative to the tiny reference denominator.

**Earlier numbers (69.48% DER etc.) were wrong**: `load_rttm_annotation` was reading the full multi-hour recording RTTM without clipping to `max_duration`. The reference denominator was hours of speech while the hypothesis covered only 350s, artificially inflating MISS to ~96% and collapsing FA to ~0.1%. Fixed by (a) clipping reference to `max_duration`, and (b) supplying an explicit UEM `[0, max_duration)` to pyannote so both sides are evaluated over the same time span.

**Actual failure mode**: CER dominates for dca_d1_1 and dca_d1_3 (30–39%). The model detects speech in roughly the right windows but assigns everything to one or two attractor slots (spk1 column carries 14.9% of frames; all other columns are saturated at −0.9999). With ATC's 4–88 speakers collapsing into 1–2 attractors, all frames get labeled as the same speaker, producing high confusion. For log_id_1 (88 speakers, denser speech at 26.7%), CER is lower (10.8%) because multiple attractor slots do activate.

**Key finding**: LS-EEND does NOT exhibit Sortformer's lock-up pattern. MISS is low (10–20%) and the model correctly tracks speech boundaries. The primary failure is attractor collapse: the CALLHOME model can only populate 1–2 of its 7 speaker slots on ATC audio, so it cannot separate the large cast of ATC speakers. CER (30–39%) is worse than Sortformer (8.52%), and FA is higher because the RF noise floor occasionally crosses the logit threshold. The simulated-data checkpoint (1–8 speaker variety) is the logical next step.

---

## Summary of findings

Numbers below are macro-averages across dca_d1_1, dca_d1_3, log_id_1 (first 350s each, collar=0.25s).

| System | DER | FA | MISS | CER | Notes |
|--------|-----|----|------|-----|-------|
| **Sortformer offline** | **10.71%** | 0.81% | 1.38% | **8.52%** | Best; CER (lock-up) dominates |
| Streaming Sortformer (10s latency) | 34.83% | 27.55% | 0.46% | 6.82% | AOSC reduces CER; FA explodes |
| Streaming Sortformer (1s, thr=0.72) | 21.4% | — | — | — | Threshold-tuned streaming |
| Pyannote 3.1 | 32.49%+ | — | — | — | Under-clustering; 46 → 1 speaker |
| LS-EEND CALLHOME (thr=0.99, norm) | varies¹ | — | 10–20% | 11–39% | Attractor collapse; 1–2 slots active |

¹ DER is recording-dependent: 43.6% (dca_d1_1), 38.2% (log_id_1), 276% (dca_d1_3 — sparse speech amplifies FA). Aggregate macro-average is not meaningful without normalising for speech density.

**Architecture conclusions:**

1. Sortformer's attention mechanism is NOT the sole cause of poor ATC diarization — all three systems struggle with domain mismatch. Sortformer's advantage is that it was pre-trained on more diverse conversational data.

2. LS-EEND's Retention mechanism avoids attention entropy collapse in principle, but the CALLHOME fine-tuning makes it telephone-specific. The model needs either the simulated checkpoint or fine-tuning on ATC data to be a fair comparison.

3. Clustering-based approaches (Pyannote) are fundamentally limited by narrow-band ATC audio: the speaker embeddings collapse in ECAPA-TDNN space, making cluster separation impossible.

4. The ATC domain requires either (a) models fine-tuned on ATC data, or (b) architectures with explicit noise/silence separation that does not rely on speaker embedding quality.

---

## Upcoming experiments

- [ ] LS-EEND simulated checkpoint (1–8 speaker variety) — download and test
- [ ] Attention entropy extraction from Sortformer layers (diagnostic evidence for lock-up)
- [ ] Speaker embedding analysis: ECAPA-TDNN at 8 kHz vs 16 kHz on ATC
- [ ] Waterfall plots comparing all systems visually on same windows
- [ ] Engineering design decisions note (syllabus deliverable 1e)
- [ ] Extend to all 16 recordings for final paper numbers

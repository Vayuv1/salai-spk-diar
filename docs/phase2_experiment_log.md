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

## Summary of findings so far

1. **Sortformer offline on ATC0R** produces DER of 10.7% with CER as the dominant error (8.52%), confirming speaker confusion (lock-up) as the primary problem.

2. **Streaming Sortformer reduces speaker confusion** (CER 6.82% vs 8.52%), supporting the hypothesis that the AOSC's per-speaker acoustic anchors and chunked processing provide partial self-correction against the lock-up.

3. **False alarm is the streaming bottleneck**, caused by probability calibration mismatch between conversational training data and ATC's sparse speech pattern. Threshold tuning (0.72 instead of 0.5) partially mitigates this.

4. **Next:** Run pyannote (clustering-based) and potentially NeMo MSDD on the same recordings to determine whether end-to-end architectures are fundamentally problematic for ATC, or whether the issue is specific to Sortformer's attention mechanism.

---

## Upcoming experiments

- [ ] Pyannote 3.1 on dca_d1_1, dca_d1_3, log_id_1 (first 350s)
- [ ] NeMo MSDD clustering pipeline on same recordings
- [ ] LS-EEND on same recordings (FS-EEND codebase adaptation)
- [ ] Attention entropy extraction from Sortformer layers
- [ ] Waterfall plots comparing all systems visually
- [ ] Extend to all 16 recordings for final paper numbers

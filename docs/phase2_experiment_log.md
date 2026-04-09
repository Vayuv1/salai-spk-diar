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

**Diagnostic conclusions (attention entropy):**

5. Layer-17 attention entropy is not a reliable predictor of per-window CER (r = +0.076, p = 0.789, n = 15 on dca_d2_2). Entropy values differ between high-CER and low-CER windows, but the direction reverses between recordings and the magnitude tracks acoustic complexity (speaker count, transition density) rather than model error directly. Entropy analysis provides descriptive characterization of how Sortformer processes ATC audio — it does not establish a causal mechanism for the lock-up pattern.

---

## Visual Analysis: Cross-System Timeline Comparison

**Script:** `src/spkdiar/analysis/plot_timeline.py`
**Figure:** `results/plots/dca_d1_1_timeline_55-150.png`
**Window:** dca_d1_1, 55–150 s (covers the first dense exchange block plus a long silence, then a three-speaker cluster at 93–112 s)

The timeline shows five horizontal lanes on a shared time axis: ground truth (GT), Sortformer offline, Sortformer streaming, LS-EEND, and Pyannote 3.1. Sortformer and LS-EEND lanes show probability curves (per-speaker sigmoid or tanh-to-[0,1] converted outputs). GT and Pyannote lanes show RTTM bars with one sub-row per speaker. Sortformer windows are rendered using only the centre 5 s of each 10 s window so each time point is covered by exactly one window.

### Observations by system

**GT**: Three speakers appear in the 55–150 s window — controller D1-1 (orange) and two pilots, DAL209 and AAL1581 (blue/green). Speech is sparse: short bursts at 63–69 s, then a dense exchange at 93–112 s with three distinct speakers alternating at 3–5 s intervals. Roughly 50 s of silence separates the two clusters.

**Sortformer offline**: The lock-up pattern is visually unambiguous. Speaker 0 (red) dominates the entire 93–120 s block, maintaining a near-1.0 probability even as the GT shows DAL209 and AAL1581 speaking at 93–96 s and 101–103 s. Speaker 1 (blue) has a few brief activations but never cleanly alternates with speaker 0. The model correctly identifies that speech is occurring — curves snap between 0 and 1 with sharp transitions that align well with GT segment boundaries — but the identity assignment freezes after the first few frames of the burst. This is the attention entropy collapse in practice: once the model commits to a speaker assignment within a 10 s window, it does not recover.

**Sortformer streaming**: The speaker alternation is noticeably better than offline. Speaker 0 (red) and speaker 1 (blue) exchange across the 93–112 s cluster in a pattern that loosely matches the GT turn-taking structure, which is the expected benefit of AOSC's explicit speaker cache. However, the baseline probabilities during the 55–90 s silence are elevated to 0.05–0.25 for multiple channels simultaneously, rather than collapsing cleanly to zero. This elevated noise floor is the calibration mismatch that causes the 27.55% FA rate: the model was trained on near-continuous speech and never learned to produce confident near-zero outputs during the long silences that characterise ATC radio.

**LS-EEND**: The grey dashed silence column (col 0) correctly tracks speech activity throughout — it drops toward zero at every GT speech segment and rises during silence. The speech boundary detection is working. The failure is in speaker separation: essentially all detected speech collapses to a single attractor (spk1, blue), with spk2 (green) activating only briefly at the boundary of the 93 s cluster. The CALLHOME model populated its attractor slots from a maximum of 6–8 simultaneously-active telephone speakers; with ATC's sequential single-speaker turns and narrow-band audio, the cross-speaker attention in the decoder finds no pressure to populate a second attractor. The result is 30–39% CER despite correct VAD — a different failure mode from Sortformer's lock-up, but equally severe.

**Pyannote 3.1**: Two sub-rows appear in the Pyannote lane (SPEAKER\_00, SPEAKER\_01), compared to three GT speakers. SPEAKER\_01 accounts for the majority of detected segments, with SPEAKER\_00 appearing only once in the 93–96 s region. Many GT segments are present in the Pyannote output with roughly correct boundaries, but two of the three speakers are merged into SPEAKER\_01. This is the embedding collapse described in Experiment 3: the ECAPA-TDNN embeddings for all ATC speakers land in a tight cluster in speaker space, and agglomerative clustering below the similarity threshold merges them. Pyannote does not exhibit lock-up — its decisions are independent per segment — but the upstream embedding failure makes correct clustering impossible.

### Cross-system interpretation

All four systems fail, but each fails in a distinct way that reflects the interaction between architecture and the ATC domain:

| System | VAD quality | Speaker ID quality | Failure mechanism |
|--------|-------------|-------------------|-------------------|
| Sortformer offline | Good (FA 0.8%, MISS 1.4%) | Poor (CER 8.5%) | Attention lock-up within window |
| Sortformer streaming | Over-active (FA 27.6%) | Best CER (6.8%) | Calibration mismatch in silence |
| LS-EEND | Good (silence curve tracks GT) | Poor (CER 31–39%) | Attractor collapse to 1–2 slots |
| Pyannote 3.1 | Moderate | Very poor | Embedding collapse pre-clustering |

Sortformer offline remains the best system by DER (10.71%) because its VAD is accurate and CER, while problematic, is substantially lower than the attractor collapse seen in LS-EEND. The timeline also suggests that the lock-up is not a permanent state — speaker 0 does drop to near zero at some segment boundaries, meaning the window-level reset partially mitigates persistence. Fine-tuning on ATC data or adding explicit noise-floor suppression are the highest-leverage interventions for any of these architectures.

**Waterfall plots** (per-window probability heatmaps across the full 350 s run) are available at `src/spkdiar/analysis/plot_waterfall.py` and provide a complementary view showing how the probability assignment evolves across all 35 windows rather than a single continuous time slice.

---

## Phase 3.1: Attention Entropy Analysis — Descriptive Characterization of ATC Audio

**Date:** April 7–9, 2026
**Script:** `src/spkdiar/analysis/attention_entropy.py`
**Data:** `results/attention_entropy/entropy_data.json`, `results/attention_entropy_d2_2/entropy_data.json`
**Figures:** `results/plots/attention_entropy_comparison.png`, `results/plots/dca_d2_2_attention_entropy.png`

### Method

Forward hooks were registered on the `attn_dropout` submodule inside each of Sortformer's 18 `TransformerEncoderBlock` layers (`transformer_encoder.layers.{i}.first_sub_layer.attn_dropout`). NeMo's `MultiHeadAttention.forward()` feeds the post-softmax attention matrix into `attn_dropout` as its first input argument before applying dropout, so the hook captures the clean attention weights at shape `(B, 8, T, T)` in float32, regardless of the model's bf16 inference dtype. No monkey-patching of model code was required.

Shannon entropy of each layer's attention was computed as:

```
H[b, h, q] = −∑_k A[b, h, q, k] · log(A[b, h, q, k] + ε)
```

averaged over batch, heads, and query positions. Maximum entropy for a uniform distribution over T=125 frames is ln(125) = 4.828 nats. Per-head entropy was also recorded to produce ±1σ bands.

### Windows analysed — dca_d1_1

Four 10 s windows from dca_d1_1:

| Window | Offset | GT content | Role |
|--------|--------|------------|------|
| Good (primary) | 55 s | Short D1-1/DAL209 burst; model correctly tracks | Correct tracking |
| Bad (primary) | 95 s | Dense 3-speaker exchange; model locks to one speaker | High confusion |
| Good (replication) | 60 s | Overlaps first speech onset at 63 s | Intermediate |
| Silence | 115 s | No speech (exchange ended at 112 s) | Near-uniform baseline |

### Results — dca_d1_1

| Window | Layer-17 H (nats) | σ across heads |
|--------|------------------|---------------|
| 55 s — correct tracking | 4.678 | 0.092 |
| 95 s — high confusion | **3.979** | **0.184** |
| 60 s — first speech onset | 4.121 | 0.169 |
| 115 s — silence | 4.802 | 0.033 |

**Primary gap (55 s − 95 s) at layer 17: 0.699 nats (14.5% of maximum entropy)**

The divergence between the correct-tracking and high-confusion windows grows monotonically from layer 1 (Δ = 0.25 nats) to layer 17 (Δ = 0.70 nats), with the sharpest widening in layers 14–17.

The per-head standard deviation at layer 17 is also notable: σ = 0.033 nats in silence (all 8 heads equally diffuse), rising to σ = 0.184 nats in the high-confusion window. Silence provides near-uniform attention (H ≈ 4.80), confirming that entropy variation is input-driven rather than a consequence of model initialisation.

### Windows analysed — dca_d2_2

Four 10 s windows from dca_d2_2, selected based on per-window CER from Phase 3.4:

| Window | Offset | CER | Role |
|--------|--------|-----|------|
| Primary low-CER | 2955 s | 0.002 | Lowest nonzero CER in range |
| Primary high-CER | 2935 s | 0.461 | Highest CER in range |
| Replication low-CER | 2940 s | 0.000 | Single-speaker window |
| Replication high-CER | 2950 s | 0.403 | Second-highest CER in range |

### Results — dca_d2_2

| Window | Layer-17 H (nats) | σ across heads |
|--------|------------------|---------------|
| 2955 s — low CER | 3.968 | 0.205 |
| 2935 s — high CER | **4.264** | 0.297 |
| 2940 s — low CER (replication) | 3.973 | 0.120 |
| 2950 s — high CER (replication) | **4.338** | 0.299 |

**Primary gap (low − high CER) at layer 17: −0.296 nats. Replication gap: −0.365 nats.**

The direction is **opposite** to dca_d1_1: here the high-CER windows have *higher* layer-17 entropy than the low-CER windows. This is consistent with the null correlation result from Phase 3.4 (r = +0.076, p = 0.789 across 15 windows).

### Interpretation

**Layer-17 attention entropy differs between windows with different CER, but the direction is not consistent across recordings and the magnitude does not predict per-window CER.** On dca_d1_1 the high-confusion window has lower entropy than the correct-tracking window (Δ = −0.699 nats). On dca_d2_2 the relationship reverses (high-CER windows have higher entropy, Δ = +0.296 to +0.365 nats). Across 15 dca_d2_2 windows, the Pearson correlation between H17 and CER is r = +0.076, p = 0.789 — not statistically significant.

The more straightforward explanation is that entropy at layer 17 reflects the complexity of the acoustic content in the window: windows with dense multi-speaker activity produce higher entropy than single-speaker windows because the attention must track more distinct frame patterns. High CER also occurs in dense multi-speaker windows because those are precisely the contexts where Sortformer's speaker assignment fails. The two are correlated with speaker count and transition density as a common driver, not with each other directly.

Specific observations that support this:

1. **On dca_d1_1**, the high-confusion window at 95 s has a dense controller/pilot/co-pilot exchange that the model reduces to one speaker label — a very different acoustic structure from the sparse 55 s window. The entropy difference (0.699 nats) likely reflects that difference in acoustic complexity, not attention collapse per se.

2. **On dca_d2_2**, the two lowest-entropy windows (H17 ≈ 3.90 at 2920s and 2925s) both have CER = 0.0 because they contain single-speaker stretches; the model correctly assigns one speaker and gets full credit. Low entropy + low CER is the expected outcome for simple single-speaker input.

3. **Head standard deviation increases in multi-speaker windows** (σ ≈ 0.18–0.30 in high-CER windows vs σ ≈ 0.09–0.12 in low-CER/silence). This reflects head specialisation under richer input, not collapse.

**Conclusion:** Attention entropy at layer 17 is a useful descriptive quantity that tracks acoustic complexity within a window. It is not a reliable predictor of per-window CER, and it does not provide causal evidence for a specific lock-up mechanism. The attention pattern differences observed between windows are consistent with the model behaving as designed — attending more broadly on complex input — rather than exhibiting a pathological collapse. Further investigation would require comparing windows with matched speaker count and transition density but different CER outcomes.

---

## Phase 3.3: Speaker Embedding Analysis — CEC 599 Deliverable 2c

**Date:** April 7, 2026
**Script:** `src/spkdiar/analysis/speaker_embeddings.py`
**Model:** TitaNet-Large (`nvidia/speakerverification_en_titanet_large`, 25.3M params, 192-dim embeddings)

### Method

For each speaker cue in the ATC0R STM ground truth (quality ≤ 3, duration ≥ 0.5 s), the corresponding audio segment is extracted and fed to TitaNet twice: once loaded at 16 kHz (native), and once loaded at 8 kHz then resampled to 16 kHz before the model sees it. The 8 kHz condition bandlimits the audio content to 4 kHz, simulating ATC VHF radio bandwidth (300–3400 Hz), while keeping the model's input sample rate unchanged. Intra-speaker cosine similarity (consistency) and inter-speaker cosine similarity (confusion) are computed across all pairwise combinations of cues. The separability margin is intra − inter.

Three recording conditions were analysed:

| Condition | Speakers | Cues | Duration |
|-----------|---------|------|---------|
| dca_d1_1 (first 350 s) | 4 | 19 | 350 s |
| dca_d1_1 (full) | 32 | 389 | ~88 min |
| log_id_1 (full) | 88 | 659 | ~120 min |

### Results

| Condition | Rate | Intra sim | Inter sim | **Margin** |
|-----------|------|-----------|-----------|------------|
| dca_d1_1 (350 s, 4 spk) | 16k | 0.771 | 0.233 | **0.538** |
| dca_d1_1 (350 s, 4 spk) | 8k | 0.784 | 0.298 | **0.486** |
| dca_d1_1 (full, 32 spk) | 16k | 0.641 | 0.322 | **0.318** |
| dca_d1_1 (full, 32 spk) | 8k | 0.642 | 0.367 | **0.276** |
| log_id_1 (full, 88 spk) | 16k | 0.621 | 0.191 | **0.429** |
| log_id_1 (full, 88 spk) | 8k | 0.628 | 0.215 | **0.413** |

### Interpretation

**TitaNet maintains a substantial separability margin across all conditions.** Even with 88 speakers on VHF radio audio, the 16 kHz margin is 0.43 and the 8 kHz margin is 0.41 — a difference of only 0.02 nats between rates. This is the central finding of this deliverable.

**The 8 kHz degradation is modest and consistent.** Across all three conditions, bandlimiting to 4 kHz raises mean inter-speaker similarity by 0.03–0.05, slightly compressing the margin. This is small relative to the absolute margin values (0.28–0.54), confirming that ATC VHF audio already lacks high-frequency speaker-identifying features (4–8 kHz formants, breathiness, fricatives) even in the nominally 16 kHz recordings. The embedding collapse is largely inherent to the domain, not an artifact of sample rate.

**The ECAPA-TDNN collapse is architecture-specific.** Pyannote 3.1 uses ECAPA-TDNN embeddings and collapsed 32–88 speakers to 1–6 clusters across all three recordings tested in Phase 2. TitaNet-Large on the same audio maintains a 0.28–0.54 margin across the same speaker counts. The difference is not domain (both models see the same VHF audio) but architecture: ECAPA-TDNN relies heavily on higher-frequency spectral envelope features that are absent in VHF radio, while TitaNet's deeper convolutional trunk and multi-scale context aggregation extract residual speaker identity from the 300–3400 Hz passband.

**dca_d1_1 full shows lower margin (0.318) than log_id_1 (0.429)** despite fewer speakers. This is because dca_d1_1 involves one DCA departure controller (D1-2, 169 cues) with heavy within-sector reuse: the same controller talks to many different pilots, and within-speaker variation in D1-2's embedding is higher than for the more homogeneous log_id_1 session. This inflates the inter-speaker similarity for dca_d1_1.

**Figures:** `results/plots/embedding_similarity_distributions.png` (3×2 panel showing similarity histograms for all conditions), `results/plots/embedding_tsne.png` (log_id_1 88-speaker t-SNE at 16k vs 8k), `results/plots/embedding_tsne_dca_d1_1_full.png` (32-speaker t-SNE). The t-SNE of log_id_1 shows visible per-speaker clustering at both rates, confirming that TitaNet can partially separate even the most crowded ATC sessions.

---

## Phase 3.4: Entropy vs CER Correlation — dca_d2_2 Null Result

**Date:** April 9, 2026

**Script:** `src/spkdiar/analysis/entropy_vs_cer.py`

**Setup:** dca_d2_2, windows with start in [2880, 2960)s. For each of the 15 usable windows (1 skipped — no reference speech): (a) load existing offline Sortformer pred RTTM, compute per-window CER vs GT RTTM clipped to window (pyannote, collar=0.25s, UEM-bounded); (b) load audio at 16 kHz, run Sortformer with `AttentionCapture` hooks to get layer-17 mean attention entropy.

**Results (15 windows):**

| Window (start) | H17 (nats) | CER | FA | MISS |
|---|---|---|---|---|
| 2880s | 4.649 | 0.000 | 0.000 | 0.000 |
| 2890s | 4.590 | 0.000 | 0.000 | 0.000 |
| 2895s | 4.303 | 0.238 | 0.026 | 0.000 |
| 2900s | 4.280 | 0.180 | 0.019 | 0.000 |
| 2905s | 4.392 | 0.232 | 0.028 | 0.000 |
| 2910s | 4.072 | 0.320 | 0.007 | 0.000 |
| 2915s | 3.995 | 0.383 | 0.124 | 0.000 |
| 2920s | 3.924 | 0.000 | 0.011 | 0.000 |
| 2925s | 3.900 | 0.000 | 0.021 | 0.000 |
| 2930s | 4.003 | 0.197 | 0.035 | 0.000 |
| 2935s | 4.264 | 0.461 | 0.044 | 0.000 |
| 2940s | 3.973 | 0.000 | 0.038 | 0.007 |
| 2945s | 4.265 | 0.279 | 0.173 | 0.000 |
| 2950s | 4.338 | 0.403 | 0.025 | 0.000 |
| 2955s | 3.968 | 0.002 | 0.000 | 0.035 |

**Pearson r = +0.076, p = 0.789, n = 15** — not statistically significant.

**Interpretation:** Layer-17 entropy does not predict per-window CER in this recording and offset range. The two lowest-entropy windows (H17 = 3.90, 3.92 at 2925s and 2920s) have CER = 0.0; the highest-CER windows (0.46 at 2935s, 0.40 at 2950s) have mid-to-high entropy (4.26, 4.34). The sign of the trend is also opposite to what an attention-collapse hypothesis would predict.

CER in this region is better explained by per-window speaker count and transition density. Single-speaker windows receive CER = 0 regardless of entropy; windows with multiple active speakers receive high CER because Sortformer's speaker assignment is already unreliable in this part of the recording. Entropy varies with the same driver (speaker count and acoustic complexity), not independently.

A full 18-layer entropy profile was subsequently generated for the highest-CER (2935s, CER=0.461) and lowest-nonzero-CER (2955s, CER=0.002) windows — see Phase 3.1 for the dca_d2_2 results and the consolidated interpretation.

**Outputs:** `results/entropy_cer/dca_d2_2_entropy_cer.json`, `results/plots/entropy_vs_cer_scatter.png`, `results/plots/dca_d2_2_attention_entropy.png`

---

## Upcoming experiments

- [ ] LS-EEND simulated checkpoint (1–8 speaker variety) — download and test
- [x] Attention entropy extraction from Sortformer layers (diagnostic evidence for lock-up) ← done
- [x] Speaker embedding analysis: TitaNet at 8/16 kHz on ATC — CEC 599 deliverable 2c ← done
- [x] Entropy vs CER correlation (null result — collapse is precondition, not trigger) ← done
- [ ] Waterfall plots comparing all systems visually on same windows
- [ ] Engineering design decisions note (syllabus deliverable 1e)
- [ ] Extend to all 16 recordings for final paper numbers

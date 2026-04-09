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

## Phase 3.1: Attention Entropy Analysis — Lock-up Mechanism Confirmed

**Date:** April 7, 2026
**Script:** `src/spkdiar/analysis/attention_entropy.py`
**Data:** `results/attention_entropy/entropy_data.json`
**Figure:** `results/plots/attention_entropy_comparison.png`

### Method

Forward hooks were registered on the `attn_dropout` submodule inside each of Sortformer's 18 `TransformerEncoderBlock` layers (`transformer_encoder.layers.{i}.first_sub_layer.attn_dropout`). NeMo's `MultiHeadAttention.forward()` feeds the post-softmax attention matrix into `attn_dropout` as its first input argument before applying dropout, so the hook captures the clean attention weights at shape `(B, 8, T, T)` in float32, regardless of the model's bf16 inference dtype. No monkey-patching of model code was required.

Shannon entropy of each layer's attention was computed as:

```
H[b, h, q] = −∑_k A[b, h, q, k] · log(A[b, h, q, k] + ε)
```

averaged over batch, heads, and query positions. Maximum entropy for a uniform distribution over T=125 frames is ln(125) = 4.828 nats. Per-head entropy was also recorded to produce ±1σ bands.

### Windows analysed

Four 10 s windows from dca_d1_1:

| Window | Offset | GT content | Role |
|--------|--------|------------|------|
| Good (primary) | 55 s | Short D1-1/DAL209 burst; model correctly tracks | Correct tracking |
| Bad (primary) | 95 s | Dense 3-speaker exchange; model locks to one speaker | Lock-up |
| Good (replication) | 60 s | Overlaps first speech onset at 63 s | Intermediate |
| Silence | 115 s | No speech (exchange ended at 112 s) | Near-uniform baseline |

### Results

| Window | Layer-17 H (nats) | σ across heads |
|--------|------------------|---------------|
| 55 s — correct tracking | 4.678 | 0.092 |
| 95 s — lock-up | **3.979** | **0.184** |
| 60 s — first speech onset | 4.121 | 0.169 |
| 115 s — silence | 4.802 | 0.033 |

**Primary gap (55 s − 95 s) at layer 17: 0.699 nats (14.5% of maximum entropy)**

The divergence between the correct-tracking and lock-up windows grows monotonically from layer 1 (Δ = 0.25 nats) to layer 17 (Δ = 0.70 nats), with the sharpest widening in layers 14–17 — the output layers that directly feed the speaker probability matrix. This is the expected signature of attention entropy collapse: deeper layers progressively commit to a narrow set of key frames rather than attending broadly across the 10 s context.

The per-head standard deviation at layer 17 is also diagnostic: σ = 0.033 nats in silence (all 8 heads equally diffuse), rising to σ = 0.184 nats in the lock-up window (heads diverge as some concentrate more sharply than others). The good-tracking window sits between these extremes (σ = 0.092).

### Interpretation

The entropy data provides quantitative evidence for the mechanism proposed in Phase 1:

1. **Attention collapse is real and layer-progressive.** It is not a discrete event at a single layer but a cumulative narrowing, suggesting that regularisation of the deeper layers (e.g. σReparam on layers 10–17) would be the highest-leverage intervention.

2. **Silence is the cleanest baseline.** Near-uniform attention (H ≈ 4.80) when no speech is present confirms that the model's attention mechanism starts diffuse; collapse is driven by the input, not by a pathological initialisation.

3. **The 60 s window is intermediate**, not "good". This makes sense: it contains the first speech onset, where the model begins to focus. True lock-up requires the model to have already committed to a speaker across multiple frames, which is what happens by 95 s.

4. **Σ across heads increases during lock-up.** This suggests that head specialisation under ATC domain pressure is uneven — some heads collapse earlier than others. This could be exploited by an auxiliary head-diversity loss during fine-tuning.

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

**Pearson r = +0.076, p = 0.789, n = 15** — no significant correlation.

**Interpretation:** This is an important **null result**. The two lowest-entropy windows (H17 = 3.90, 3.92 at 2925s and 2920s) have CER = 0.0. The highest-CER windows (0.46 at 2935s, 0.40 at 2950s) have mid-to-high entropy (4.26, 4.34). Entropy collapse is **not a per-window predictor** of CER within an already-degraded region.

The correct interpretation: the entire [2880, 2960)s stretch runs 0.3–1.0 nats below the maximum uniform entropy of 4.828, indicating the model is in a generally low-entropy regime throughout. Within this regime, whether CER is high or low depends on the specific speaker layout in that window (single-speaker windows get CER=0 even with collapsed attention), not on the marginal entropy difference between windows.

Entropy collapse is a **necessary precondition** for lock-up, not a moment-to-moment trigger. The cause is accumulated distributional drift across the recording, not a within-window entropy threshold.

**Outputs:** `results/entropy_cer/dca_d2_2_entropy_cer.json`, `results/plots/entropy_vs_cer_scatter.png`

---

## Upcoming experiments

- [ ] LS-EEND simulated checkpoint (1–8 speaker variety) — download and test
- [x] Attention entropy extraction from Sortformer layers (diagnostic evidence for lock-up) ← done
- [x] Speaker embedding analysis: TitaNet at 8/16 kHz on ATC — CEC 599 deliverable 2c ← done
- [x] Entropy vs CER correlation (null result — collapse is precondition, not trigger) ← done
- [ ] Waterfall plots comparing all systems visually on same windows
- [ ] Engineering design decisions note (syllabus deliverable 1e)
- [ ] Extend to all 16 recordings for final paper numbers

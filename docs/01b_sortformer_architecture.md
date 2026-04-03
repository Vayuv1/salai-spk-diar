# Signal Flow and Architecture of NVIDIA's Sortformer Models for Speaker Diarization

**CEC 599: Speaker Diarization — Deliverable 1b**
Shital Pandey | Spring 2026 | Instructor: Dr. Jianhua Liu

**Focus: Streaming Sortformer and the Arrival-Order Speaker Cache (AOSC)**

---

## 1. Overview and motivation

Sortformer (Park, Medennikov, Dhawan, et al., 2024) is an end-to-end speaker diarization model that resolves the speaker label permutation problem through temporal ordering rather than combinatorial search. The key idea: instead of evaluating all possible speaker-to-output-slot assignments (which grows factorially with speaker count), Sortformer assigns speakers to output slots in the order they first appear in the audio. The first speaker to talk gets slot 0, the second gets slot 1, and so on.

The Streaming Sortformer extension (Medennikov, Park, Wang, et al., Interspeech 2025) adapts this for real-time processing using an Arrival-Order Speaker Cache (AOSC) that maintains speaker identity across streaming chunks while preserving the arrival-time ordering property.

---

## 2. Offline Sortformer: complete signal flow

### 2.1 Stage 1 — Acoustic feature extraction

Raw 16 kHz mono audio is converted to 80-channel log-mel spectrograms:

- **Window**: 25 ms Hann window
- **Stride**: 10 ms hop
- **FFT size**: 512 points
- **Mel filters**: 80 channels
- **Normalization**: Per-feature mean/variance normalization
- **Dither**: 1e-5 (small random noise to prevent numerical issues with silence)

A 10-second input produces approximately 1,000 mel frames, each an 80-dimensional vector.

### 2.2 Stage 2 — Fast Conformer encoder (NEST pre-encoder)

The mel spectrogram enters an 18-layer Fast Conformer encoder. This module is pre-trained using NVIDIA's NEST (Neural Encoder for Speech Transformers) framework with BEST-RQ self-supervised learning on ~25,000 hours of speech. The encoder architecture:

**Subsampling (8× downsampling):**
Three depthwise separable convolutional layers progressively reduce temporal resolution:
- Conv1: stride 2, reducing 10 ms → 20 ms per frame
- Conv2: stride 2, reducing 20 ms → 40 ms per frame
- Conv3: stride 2, reducing 40 ms → 80 ms per frame

After subsampling, a 10-second input has approximately **125 frames** at 80 ms resolution.

**18 Fast Conformer blocks:**
Each block follows the "macaron" sandwich pattern (detailed in Deliverable 1d):

x̃ = x + ½·FFN(x) → x' = x̃ + MHSA(x̃) → x'' = x' + Conv(x') → y = LN(x'' + ½·FFN(x''))

Key parameters:
- Hidden dimension: d_model = 512
- Attention heads: 8 (head dimension = 64)
- Convolution kernel size: 9
- FFN inner dimension: 2048 (expansion factor 4)
- Dropout: 0.1
- Positional encoding: Relative sinusoidal (Shaw et al., 2018)

**Output**: 512-dimensional frame embeddings at 80 ms resolution. Total encoder parameters: ~115M.

### 2.3 Stage 3 — Sortformer Transformer module

The 512-dimensional encoder outputs are projected down to 192 dimensions through a linear layer, then processed by the Sortformer-specific Transformer:

**18 Transformer encoder layers:**
- Hidden size: 192
- Attention heads: 8 (head dimension: 24)
- FFN inner dimension: 768
- Activation: ReLU
- **Dropout: 0.5** on attention scores, attention output, and FFN output (training only; disabled at inference)

This is a standard Transformer encoder — it uses self-attention, not cross-attention. Every frame attends to every other frame within the processing window. The extremely high 0.5 dropout rate during training acts as a strong regularizer, preventing the model from relying on any single attention pattern. At inference, dropout is turned off, which means the model can develop sharper attention patterns than it ever saw during training — a factor potentially relevant to the lock-up pattern observed on ATC audio.

### 2.4 Stage 4 — Output head

Two feedforward layers map the 192-dimensional representations to 4 sigmoid outputs per frame:

FFN: 192 → 192 → 4, with sigmoid activation on the final layer

Output matrix: **P ∈ ℝ^{T×4}**, where each element p_{t,k} ∈ [0,1] represents the probability that speaker k is active at frame t. Sigmoid (not softmax) is used because multiple speakers can overlap.

**Total model size: ~123M parameters.**

### 2.5 Complete signal flow summary

```
Raw audio (16 kHz)
    ↓ 25ms window, 10ms stride, 80-channel mel filterbank
Log-mel spectrogram [~1000 × 80]
    ↓ 3 depthwise separable conv layers (8× downsampling)
Subsampled features [~125 × 512]
    ↓ 18 Fast Conformer blocks (d=512, 8 heads)
Encoder output [~125 × 512]
    ↓ Linear projection (512 → 192)
Projected features [~125 × 192]
    ↓ 18 Transformer encoder layers (d=192, 8 heads)
Contextual representations [~125 × 192]
    ↓ 2 FFN layers + sigmoid
Speaker probabilities [~125 × 4]
```

---

## 3. Sort Loss: resolving permutation through temporal ordering

### 3.1 The permutation problem

In multi-speaker diarization, the model must assign each detected speaker to a specific output slot. But the assignment is arbitrary — there is no inherent reason why "the controller" should be output slot 0 versus slot 2. Traditional EEND models handle this with Permutation Invariant Training (PIT), which evaluates all K! possible assignments and uses the one with the lowest loss:

L_PIT = min_{φ ∈ Perm(K)} (1/K) Σ_k L_BCE(y_{φ(k)}, q_k)

For K=4 speakers, this evaluates 24 permutations. For K=8, it would require 40,320 evaluations — computationally expensive and a hard scaling barrier.

### 3.2 Sort Loss definition

Sort Loss eliminates the combinatorial search by defining a deterministic ordering. Given ground truth speaker activities Y = [y₁, ..., y_K]^T and predicted activities P = [q₁, ..., q_K]^T:

**Step 1 — Define arrival time:**
Ψ(y_k) = min{t' | y_{k,t'} ≠ 0}

This finds the first frame where speaker k is active.

**Step 2 — Sort speakers by arrival:**
A sorting function η orders speakers such that Ψ(y_{η(1)}) ≤ Ψ(y_{η(2)}) ≤ ... ≤ Ψ(y_{η(K)})

**Step 3 — Compute loss with sorted assignment:**
L_Sort = (1/K) Σ_{k=1}^{K} L_BCE(y_{η(k)}, q_k)

The first speaker to appear is always compared against output slot 0, the second against slot 1, etc. No search over permutations is needed — the assignment is deterministic. Computational complexity drops from O(K!) to O(K log K).

### 3.3 Hybrid loss for best performance

The paper's best results use a hybrid of Sort Loss and PIT:

L_hybrid = 0.5 · L_Sort + 0.5 · L_PIL

where L_PIL is PIT loss. The hybrid provides two complementary training signals: Sort Loss enforces temporal ordering as a regularizer, while PIT ensures the model finds optimal assignments even when arrival-time ordering is ambiguous (e.g., speakers starting simultaneously).

### 3.4 Positional embeddings are essential

Sort Loss only works when the model can distinguish frame positions — it needs to know that frame 10 comes before frame 50 to establish arrival order. Standard self-attention is permutation-equivariant (it produces the same output regardless of input ordering). **Relative positional embeddings break this symmetry**, allowing the model to learn temporal relationships. The Sortformer paper shows that removing positional embeddings causes Sort Loss training to fail completely, while PIT-only training is unaffected.

### 3.5 Known limitation: speaker count overestimation

The paper notes that models trained with Sort Loss or hybrid loss tend to **overestimate the number of speakers** in longer recordings. PIT-trained models provide more accurate speaker counts. This is relevant for ATC audio, where the actual speaker count within any 10-second window is typically 2–4 but the model has 4 output slots that may all activate spuriously.

---

## 4. Streaming Sortformer: the AOSC mechanism

### 4.1 The streaming challenge

Offline Sortformer processes an entire recording (or a long segment) at once, allowing bidirectional self-attention across all frames. Streaming requires processing audio incrementally, in chunks, while maintaining consistent speaker-slot assignments across chunks. The fundamental challenge: how do you tell the model that "speaker A in chunk 3" is the same as "speaker A in chunk 1" when each chunk is processed independently?

### 4.2 Input construction for each streaming step

At each streaming step, the model receives a composite input assembled from four segments:

```
[Speaker Cache (M frames)] [FIFO Queue (F frames)] [Current Chunk (C frames)] [Right Context (R frames)]
```

- **Speaker Cache**: M = 188 frames (15.04 seconds at 80 ms/frame). Contains representative embeddings from previously identified speakers, organized by arrival time.
- **FIFO Queue**: Recent frames providing immediate temporal context. Updated every step by appending the current chunk and discarding oldest frames.
- **Current Chunk**: The new audio being processed. Size varies by latency target.
- **Right Context**: Future frames for look-ahead (only available in non-real-time or buffered modes).

The model processes this composite input through its standard encoder and Transformer layers, then extracts only the current chunk's output probabilities for the final diarization result.

### 4.3 AOSC: the 7-step speaker cache update

After processing each chunk, the speaker cache is updated through a carefully designed procedure:

**Step 1 — Compute speaker scores:**
For each frame i in the speaker cache, compute a score S_i for the most likely speaker assignment:

S_i = log(P_{i,k*}) + Σ_{j≠k*} log(1 − P_{i,j})

where k* = argmax_k P_{i,k}. This score reflects both confidence in the assigned speaker and distinctiveness from other speakers.

**Step 2 — Detect silence:**
Frames where no speaker probability exceeds 0.5 are classified as silence. The average embedding of all silence frames is computed for later use.

**Step 3 — Disable non-speech scores:**
For silence frames, all speaker scores are set to −∞ so they cannot be selected as speaker representatives.

**Step 4 — Apply recency bias:**
New frames from the current chunk receive a small score boost δ = 0.05. This ensures recently observed speaker characteristics are slightly preferred over older ones, allowing the cache to adapt to speaker acoustic changes over time.

**Step 5 — Guarantee minimum speaker representation:**
For each detected speaker, a minimum of K = 33 frames receive a score boost of Δ. This prevents any registered speaker from being completely dropped from the cache, even if they have been silent for a long time. This is a critical self-correction mechanism — a speaker cannot be "forgotten" once registered.

**Step 6 — Insert silence separators:**
A = 3 silence embedding frames are inserted between speaker segments in the cache. These act as boundary markers, helping the model distinguish where one speaker's representation ends and another's begins.

**Step 7 — Select top-M frames:**
From all candidate frames (previous cache + new chunk), select the M frames with highest scores, preserving their original temporal (arrival-time) order.

### 4.4 Training augmentation: random speaker permutation

During training, speakers within the cache are **randomly permuted** at each streaming step. This is a targeted augmentation: it prevents the model from learning fixed associations between speaker identity and cache position, forcing it to rely on acoustic content rather than positional shortcuts. This augmentation is essential — without it, the model degrades when speaker ordering differs between training and inference.

### 4.5 Latency configurations

| Config | Chunk (frames) | Right Ctx | FIFO | Cache | Latency | RTF |
|--------|----------------|-----------|------|-------|---------|-----|
| Ultra-low | 3 | 1 | 188 | 188 | 0.32s | 0.180 |
| Low | 6 | 7 | 188 | 188 | 1.04s | 0.093 |
| Medium | 124 | 1 | 124 | 188 | 10.0s | 0.005 |
| High | 340 | 40 | 40 | 188 | 30.4s | 0.002 |

All configurations use the same 188-frame speaker cache. The tradeoff is between latency and right-context look-ahead. Even at 0.32s latency, DER degradation is modest (CALLHOME-all: 11.50% vs 10.09% at 10s).

### 4.6 The streaming model configuration in our codebase

From the `DiarizationConfig` in `e2e_diarize_speech.py`, the streaming parameters are:

```python
spkcache_len: int = 188          # Speaker cache size (frames)
spkcache_update_period: int = 144 # How often to update cache
fifo_len: int = 188              # FIFO queue length
chunk_len: int = 6               # Current chunk size
chunk_left_context: int = 1      # Left context frames
chunk_right_context: int = 7     # Right context frames
```

These correspond to the "Low latency" (1.04s) configuration from the paper.

---

## 5. Benchmark results

### 5.1 Offline Sortformer

| Dataset | Collar | DER (no PP) | DER (with PP) |
|---------|--------|-------------|---------------|
| DIHARD III Eval (≤4 spk) | 0.0s | 16.28% | 14.76% |
| CALLHOME part2 (2 spk) | 0.25s | 6.49% | 5.85% |
| CALLHOME part2 (all spk) | 0.25s | — | ~12.6% |

### 5.2 Streaming Sortformer

| Dataset | Latency | DER (no PP) | DER (with PP) |
|---------|---------|-------------|---------------|
| CALLHOME-all | 10.0s | 11.10% | 10.09% |
| CALLHOME-all | 1.04s | 11.19% | 10.27% |
| CALLHOME-all | 0.32s | 12.36% | 11.50% |
| DIHARD III-all | 10.0s | 20.39% | 19.02% |
| DIHARD III (≤4 spk) | 10.0s | 15.66% | 14.79% |

Streaming at 10s latency **outperforms offline** on CALLHOME (10.09% vs ~12.6%) and matches on DIHARD III (14.79% vs 14.76% for ≤4 speakers). The paper attributes this to the offline model's degradation on recordings exceeding its 90-second training length, while streaming's fixed-window processing avoids this mismatch.

### 5.3 Our ATC0 results

Using `diar_sortformer_4spk-v1.nemo` (offline) on the ATC0 dataset:
- **DER: 22.04%** (FA=2.68%, MISS=2.83%, CER=16.53%)

The confusion error rate (CER) at 16.53% dominates, consistent with the lock-up pattern where speech is detected correctly but assigned to the wrong speaker.

---

## 6. Multi-speaker ASR integration

Sortformer's arrival-time ordering enables clean integration with ASR through sinusoidal speaker kernels:

γ_k = sin(2πkz/M), for k = 0, 1, ..., K-1

where z is the frame index and M is a modulation parameter. The speaker probability matrix P is multiplied element-wise with each kernel, then summed and added to the normalized ASR encoder states:

h'_t = LN(h_t) + Σ_k P_{t,k} · γ_k(t)

This allows a standard ASR decoder to distinguish which speaker is producing which tokens, enabling speaker-attributed transcription using standard cross-entropy loss — no PIT needed in the ASR branch.

---

## 7. Available models on HuggingFace

| Model | Mode | Speakers | Notes |
|-------|------|----------|-------|
| nvidia/diar_sortformer_4spk-v1 | Offline | ≤4 | Used in our experiments |
| nvidia/diar_streaming_sortformer_4spk-v2 | Streaming | ≤4 | With AOSC |
| nvidia/diar_streaming_sortformer_4spk-v2.1 | Streaming | ≤4 | Updated version |

---

## 8. Architectural implications for ATC audio

The Sortformer architecture has several properties that interact poorly with ATC communication characteristics:

1. **4-speaker hard cap**: ATC sessions routinely involve 10+ speakers within minutes. Any 10-second window may contain transmissions from speakers the model cannot represent.

2. **Sort Loss assumes temporal progression**: In meetings, speakers arrive and persist. In ATC, pilots appear once or twice then disappear. The "arrival order" concept breaks down when most speakers are transient.

3. **No explicit speaker representations**: Unlike attractor-based models, Sortformer has no interpretable speaker vectors. Speaker identity is distributed across attention patterns, making it impossible to inspect or correct speaker assignments.

4. **Inference without dropout**: The 0.5 dropout used during training prevents attention collapse, but at inference this safety net is removed. On out-of-distribution ATC audio, unconstrained attention can concentrate on a subset of frames, causing the observed lock-up.

5. **Streaming as partial mitigation**: The AOSC's guaranteed minimum speaker representation and chunked processing inherently limit the extent of any lock-up within a single chunk. Testing streaming Sortformer on ATC0 is a high-priority experiment.

---

## References

1. Park, T., Medennikov, N., Dhawan, K., et al. (2024). Sortformer: Seamless Integration of Speaker Diarization and ASR by Bridging Timestamps and Tokens. *arXiv:2409.06656*. Accepted at ICML 2025.
2. Medennikov, N., Park, T., Wang, H., et al. (2025). Streaming Sortformer: Speaker Cache-Based Online Speaker Diarization with Arrival-Time Ordering. *Interspeech 2025*. arXiv:2507.18446.
3. Rekesh, D., et al. (2023). Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition. *arXiv:2305.05084*.
4. NVIDIA NeMo Framework. Speaker Diarization Configuration Files. https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/configs.html

# The Architecture of Conformer and Fast Conformer

**CEC 599: Speaker Diarization — Deliverable 1d**
Shital Pandey | Spring 2026 | Instructor: Dr. Jianhua Liu

---

## 1. Why Conformer matters for speaker diarization

The Conformer (Gulati et al., Interspeech 2020) is the backbone encoder used by both Sortformer and many other speech processing systems. It serves as the acoustic feature extractor that converts raw mel spectrograms into frame-level representations carrying both speech content and speaker identity information. Understanding its architecture is essential because the quality and nature of these representations directly determines the performance of downstream diarization.

Fast Conformer (Rekesh et al., ICASSP 2023) is NVIDIA's optimized variant that achieves 2.8× speedup with no accuracy loss. Sortformer uses Fast Conformer as its NEST encoder module.

---

## 2. The core insight: combining convolution with self-attention

Speech has a dual nature — it contains both **local patterns** (phonemes, formant transitions, voice onset times) and **global dependencies** (speaking rate, intonation contours, speaker identity). Pure convolution captures local patterns well but struggles with long-range dependencies. Pure self-attention captures global dependencies but is computationally expensive and may miss fine-grained local patterns.

The Conformer combines both through a "macaron" architecture that interleaves convolution and self-attention within each block. The original paper demonstrated that this combination outperforms both Transformer-only and convolution-only architectures on speech recognition benchmarks.

---

## 3. Conformer block: the macaron sandwich

Each Conformer block applies four sub-modules in sequence, with a distinctive pattern of half-step residual connections:

### 3.1 First feed-forward module (half-step)

x̃ = x + ½ · FFN₁(x)

The FFN consists of: LayerNorm → Linear(d → 4d) → Swish activation → Dropout → Linear(4d → d) → Dropout

The ½ scaling factor comes from the Macaron-Net paper (Lu et al., 2019), which showed that placing two half-step FFN modules around the attention module works better than a single full-step FFN after it. The intuition: the first FFN pre-processes features for attention, while the second post-processes attention output.

### 3.2 Multi-head self-attention module

x' = x̃ + MHSA(x̃)

Standard multi-head self-attention with **relative positional encoding** (Shaw et al., 2018):

Attention(Q, K, V) = softmax((QK^T + Q·R^T) / √d_k) · V

where R is a learned relative position embedding matrix. Unlike absolute positional encodings that assign a fixed vector to each position, relative encodings represent the distance between positions. This makes the model robust to different input lengths and allows it to generalize to longer sequences than seen during training.

Key parameters in Sortformer's configuration:
- d_model = 512
- Number of heads: 8
- Head dimension: d_k = 64
- Relative positional encoding (required for Sort Loss to work)

### 3.3 Convolution module

x'' = x' + Conv(x')

The convolution module is the distinguishing feature of Conformer. Its internal structure:

LayerNorm → Pointwise Conv (d → 2d) → GLU activation → **Depthwise Conv** → BatchNorm → Swish → Pointwise Conv (d → d) → Dropout

Breaking down each component:

**Pointwise convolution (d → 2d)**: A 1×1 convolution that doubles the channels. This is essentially a linear projection applied to each frame independently.

**Gated Linear Unit (GLU)**: Splits the 2d channels into two halves (a, b), then computes a ⊙ σ(b). The sigmoid-gated half controls which features pass through, acting as a learned feature selector.

**Depthwise convolution**: This is the core local pattern extractor. Unlike standard convolution that mixes all channels, depthwise convolution applies a separate 1-D filter to each channel independently. With kernel size k, each output frame depends on the k neighboring frames in its channel only. This is computationally cheap (O(d·k·T) vs O(d²·k·T) for standard conv) while still capturing local temporal patterns.

- Original Conformer kernel size: 31 (covering ~310 ms at 10 ms resolution)
- Fast Conformer kernel size: 9 (covering ~720 ms at 80 ms resolution due to 8× downsampling)

**BatchNorm**: Applied instead of LayerNorm because convolution operates on local patterns where batch statistics are more appropriate than per-instance statistics.

**Swish activation**: f(x) = x · σ(x), a smooth non-linearity that allows small negative values to pass (unlike ReLU).

### 3.4 Second feed-forward module (half-step) + LayerNorm

y = LayerNorm(x'' + ½ · FFN₂(x''))

Same structure as FFN₁. The final LayerNorm stabilizes the output for the next block.

### 3.5 Complete block equation

y = LN(x + ½·FFN₁(x) + MHSA(x + ½·FFN₁(x)) + Conv(x + ½·FFN₁(x) + MHSA(x + ½·FFN₁(x))) + ½·FFN₂(...))

In practice, this is computed sequentially through the four residual sub-modules.

---

## 4. Fast Conformer modifications

Rekesh et al. (2023) introduced four changes that reduce computation by 2.8× while preserving or improving accuracy:

### 4.1 Aggressive 8× downsampling

**Original Conformer**: 2 convolutional layers with stride 2 each → **4× downsampling** (10 ms → 40 ms per frame)

**Fast Conformer**: 3 depthwise separable convolutional layers with stride 2 each → **8× downsampling** (10 ms → 80 ms per frame)

This halves the number of frames entering the Conformer blocks compared to standard Conformer. Since self-attention is O(T²) and convolution/FFN are O(T), halving T reduces attention cost by 4× and linear costs by 2×.

The depthwise separable convolutions in the subsampling layers (instead of standard convolutions) further reduce parameter count. A depthwise separable convolution factorizes a standard convolution into a depthwise convolution (one filter per channel) followed by a pointwise convolution (1×1 conv mixing channels), reducing parameters from k·C_in·C_out to k·C_in + C_in·C_out.

### 4.2 Reduced subsampling channels

The subsampling convolutions use 256 channels instead of 512, reducing early-stage computation. The full 512-dimensional representation is produced by a linear projection after subsampling.

### 4.3 Smaller convolution kernel

Kernel size reduced from 31 to 9. With 8× downsampling, each frame already represents 80 ms, so a kernel of 9 covers **720 ms** of audio — comparable to the original Conformer's 31-kernel coverage of ~310 ms at 40 ms resolution. The effective temporal receptive field is actually larger.

### 4.4 Computational savings

| Metric | Conformer (17 layers) | Fast Conformer (18 layers) | Reduction |
|--------|----------------------|---------------------------|-----------|
| Encoder GMACS | 143.2 | 48.7 | 2.9× |
| WER (LS test-other) | 5.19% | 4.99% | Better |
| Max audio length (A100) | ~15 min | ~675 min | 45× |

The extra layer (18 vs 17) compensates for reduced per-frame computation, and the overall result is both faster and more accurate.

### 4.5 Limited-context attention for long-form

For processing hour-long recordings, Fast Conformer introduces limited-context attention:

- Each frame attends only to frames within a local window of w frames
- A global token aggregates information across the entire sequence
- Complexity changes from O(T²) to O(T·w) — linear in T

In Sortformer's offline mode, full-context attention is used (att_context_size: [-1, -1]). In streaming mode, right context is limited (typically 7 frames = 560 ms with 50% probability during training).

---

## 5. How the encoder feeds diarization

### 5.1 Frame representations carry speaker information

Each output frame from the Fast Conformer encoder is a 512-dimensional vector that implicitly encodes:

- **Acoustic content**: What phonemes, words, or noise are present at this moment
- **Speaker characteristics**: Vocal tract resonance patterns, F0 range, speaking style
- **Temporal context**: Information about neighboring frames through convolution and attention
- **Positional information**: Where in the sequence this frame appears, via relative positional encoding

For diarization, the speaker characteristics component is the most critical. The encoder's ability to produce speaker-discriminative representations determines whether the downstream Transformer and output head can correctly assign speakers.

### 5.2 The NEST pre-training connection

In Sortformer, the Fast Conformer encoder is pre-trained using NVIDIA's NEST (Neural Encoder for Speech Transformers) framework. NEST uses **BEST-RQ** (BERT-based Speech pre-Training with Random-projection Quantizer) — a self-supervised method where:

1. Random segments of the input features are masked
2. A random projection quantizer creates discrete pseudo-labels
3. The encoder learns to predict the pseudo-labels of masked regions

This pre-training on ~25,000 hours of speech teaches the encoder general speech representations before diarization-specific fine-tuning. The resulting encoder captures robust acoustic patterns but has never seen VHF radio-distorted audio, which may contribute to poor representation quality on ATC data.

### 5.3 Impact on downstream diarization

If the encoder produces similar embeddings for different speakers (because their voices sound alike after VHF radio distortion), no amount of Transformer processing or Sort Loss training can reliably separate them. The encoder is the foundation — its representations define the upper bound of diarization accuracy.

For ATC audio specifically, the narrow bandwidth (300–3,400 Hz) eliminates formant structure above 3.4 kHz, reducing the discriminative information available. AM radio artifacts add non-speech energy patterns that the encoder may not handle well, given its training on wideband conversational audio.

---

## 6. Conformer variants in the diarization landscape

### 6.1 Usage across architectures

| System | Encoder | Layers | d_model | Downsampling |
|--------|---------|--------|---------|--------------|
| Sortformer | Fast Conformer | 18 | 512 | 8× |
| LS-EEND | Conformer (Retention) | 4 | 256 | 10× |
| pyannote 3.x | Modified Conformer | — | — | varies |
| NeMo MSDD | Fast Conformer | 18 | 512 | 8× |

### 6.2 Conformer for EEND (from Transformer to Conformer)

Jeoung et al. (ICASSP 2022) showed that replacing the Transformer encoder in SA-EEND with a Conformer encoder improved DER on CALLHOME from ~12% to ~10%, confirming that the local-global feature combination benefits diarization as well as ASR.

---

## 7. Practical implications for our work

### 7.1 The 80 ms resolution bottleneck

With 8× downsampling, speaker activities are predicted at 80 ms resolution. Each 80 ms frame aggregates information from a 160 ms window (due to overlapping convolution kernels in subsampling). For ATC transmissions lasting 1–3 seconds:

- 1-second utterance: ~12 frames
- 2-second utterance: ~25 frames
- 3-second utterance: ~37 frames

The model has relatively few frames per speaker turn to establish acoustic identity, particularly for brief pilot readbacks.

### 7.2 Depthwise separable convolution as a design principle

The shift from standard convolution to depthwise separable convolution throughout Fast Conformer reflects a broader trend in efficient neural architecture design. This factorization reduces parameters by a factor of k (kernel size) while maintaining the same receptive field. For our future work on model adaptation or fine-tuning, this efficiency enables training on smaller GPU budgets.

### 7.3 The convolution-attention balance for radio speech

Standard Conformer was designed for wideband speech where both local (formant) and global (prosody) features are informative. For narrowband VHF radio speech, the local features are degraded while global patterns (speaking rate, transmission timing) may be more robust. Adjusting the relative importance of convolution versus attention (e.g., through the kernel size or attention context window) could be beneficial for ATC-domain adaptation.

---

## References

1. Gulati, A., Qin, J., Chiu, C., et al. (2020). Conformer: Convolution-augmented Transformer for Speech Recognition. *Interspeech 2020*. arXiv:2005.08100.
2. Rekesh, D., Koluguri, N. R., Kriman, S., et al. (2023). Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition. *ICASSP 2023*. arXiv:2305.05084.
3. Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). Self-Attention with Relative Position Representations. *NAACL 2018*.
4. Lu, Y., Li, Z., He, D., Sun, Z., Dong, B., Qin, T., Wang, L., & Liu, T. (2019). Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View. *arXiv:1906.02762*.
5. NVIDIA NeMo Framework. Fast Conformer configuration: https://github.com/NVIDIA-NeMo/NeMo/blob/main/examples/asr/conf/fastconformer/

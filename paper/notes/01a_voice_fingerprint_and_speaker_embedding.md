# The Principle of Voice Fingerprint and Speaker Embedding

**CEC 599: Speaker Diarization — Deliverable 1a**
Shital Pandey | Spring 2026 | Instructor: Dr. Jianhua Liu

---

## 1. What makes a voice unique

Every human voice carries a distinct acoustic signature shaped by anatomy and habit. The vocal tract — the air column from the larynx through the mouth and nasal cavity — acts as a resonant filter whose shape determines the spectral envelope of produced speech. The length and cross-sectional profile of this tract vary from person to person, producing characteristic resonant frequencies called formants. The fundamental frequency (F0), determined by vocal fold mass and tension, further distinguishes speakers. Beyond anatomy, learned speaking patterns such as speaking rate, intonation contours, and habitual articulation contribute to what we informally call a "voice fingerprint."

For computational speaker recognition, the goal is to distill these distinguishing characteristics into a compact numerical representation — a fixed-dimensional vector — that captures speaker identity while discarding irrelevant variation like background noise, channel effects, and linguistic content. This vector is called a **speaker embedding**.

---

## 2. From raw audio to speaker-discriminative features

### 2.1 Acoustic feature extraction

The standard front-end for speaker embedding systems converts raw waveforms into spectral representations. The most common pipeline:

1. **Pre-emphasis filtering**: A first-order high-pass filter (coefficient ~0.97) boosts higher frequencies to compensate for the natural spectral tilt of speech.

2. **Framing and windowing**: The signal is divided into overlapping frames (typically 25 ms with 10 ms stride) and multiplied by a Hamming or Hann window to reduce spectral leakage.

3. **Short-time Fourier Transform (STFT)**: Each windowed frame is transformed to the frequency domain using a 512-point FFT.

4. **Mel filterbank**: The power spectrum is passed through 40–80 triangular filters spaced according to the mel scale, which approximates human auditory perception. The mel scale compresses higher frequencies where human discrimination is coarser.

5. **Log compression and optional cepstral transform**: Taking the logarithm of filterbank energies produces log-mel features. An optional Discrete Cosine Transform yields Mel-Frequency Cepstral Coefficients (MFCCs), though modern deep learning systems typically work with log-mel spectrograms directly.

The resulting feature matrix has shape [T × F], where T is the number of frames and F is the number of frequency bins (e.g., 80 for 80-channel log-mel).

### 2.2 The fundamental challenge: variable-length to fixed-length

Speech utterances vary in duration, but downstream tasks (comparison, clustering, classification) require fixed-dimensional representations. The core technical problem in speaker embedding is this: how do you compress a variable-length sequence of frame-level features into a single vector that retains speaker identity?

Three major architectures have solved this problem with increasing effectiveness.

---

## 3. X-vectors: the TDNN-based foundation

Snyder, Garcia-Romero, Sell, Povey, and Khudanpur (2018) introduced x-vectors at ICASSP 2018, establishing the deep learning paradigm for speaker embeddings that replaced the earlier i-vector approach.

### 3.1 Architecture

The x-vector system uses a Time-Delay Neural Network (TDNN), which is a 1-D convolutional network operating along the time axis:

**Frame-level layers (layers 1–5):**
- Layer 1: Input features → 512 dims, context {t-2, t-1, t, t+1, t+2}
- Layer 2: 512 → 512, context {t-2, t, t+2}
- Layer 3: 512 → 512, context {t-3, t, t+3}
- Layer 4: 512 → 512, full context at this layer
- Layer 5: 512 → 1500, full context

Each layer processes a dilated temporal context, gradually expanding the receptive field from 5 frames to the entire utterance.

**Statistics pooling layer:**
This is the key mechanism that converts variable-length frame sequences to fixed-length vectors. Given frame-level outputs h₁, h₂, ..., h_T from layer 5:

- Mean: μ = (1/T) Σ_t h_t
- Standard deviation: σ = √[(1/T) Σ_t (h_t − μ)²]
- Pooled output: [μ; σ] — a concatenation of mean and standard deviation

This produces a 3000-dimensional vector (1500-dim mean + 1500-dim std) regardless of input duration.

**Segment-level layers (layers 6–7):**
- Layer 6: 3000 → 512 (this is where the x-vector is extracted)
- Layer 7: 512 → 512
- Softmax output layer: 512 → N_speakers (training only)

### 3.2 Training

X-vectors are trained as a speaker classification task on a large speaker-labeled dataset (typically VoxCeleb1+2, containing ~7,000 speakers). The model learns to classify input segments into speaker identities. After training, the softmax layer is discarded, and the 512-dimensional activation at layer 6 is used as the speaker embedding.

Data augmentation is critical — the original x-vector recipe uses additive noise from the MUSAN corpus (music, speech, noise) and reverberation from simulated Room Impulse Responses (RIRs). This teaches the model to produce speaker-consistent embeddings across varying acoustic conditions.

### 3.3 Scoring and PLDA

Raw x-vectors are not directly compared with cosine similarity. Instead, a Probabilistic Linear Discriminant Analysis (PLDA) backend models the distribution of speaker embeddings:

- **Between-speaker variability**: captured by a low-rank matrix
- **Within-speaker variability**: captured by a diagonal or full covariance matrix

PLDA produces a log-likelihood ratio score for whether two embeddings come from the same speaker. Before PLDA scoring, embeddings are centered, whitened, and length-normalized.

### 3.4 Limitations

X-vectors suffer from several weaknesses: the TDNN's limited temporal context means long-range dependencies are poorly captured; the simple mean+std pooling treats all frames equally regardless of their informativeness (silence frames contribute as much as speech frames); and the architecture lacks mechanisms to emphasize the most discriminative frequency regions.

**Reference:** Snyder, D., Garcia-Romero, D., Sell, G., Povey, D., & Khudanpur, S. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition. *ICASSP 2018*.

---

## 4. ECAPA-TDNN: attention-driven speaker embeddings

Desplanques, Thienpondt, and Demuynck (2020) proposed ECAPA-TDNN at Interspeech 2020, achieving **0.87% Equal Error Rate (EER)** on VoxCeleb1 — a dramatic improvement over x-vectors. ECAPA-TDNN became the standard embedding extractor used by most modern diarization systems, including pyannote and DiariZen.

### 4.1 Three architectural innovations

**Squeeze-Excitation (SE) Res2Net blocks:**
Each residual block uses the Res2Net structure, which splits channels into groups and processes them hierarchically — each group receives the output of the previous group, creating multi-scale feature extraction within a single layer. The Squeeze-Excitation mechanism adds channel attention:

1. **Squeeze**: Global average pooling across time produces a channel descriptor c ∈ ℝ^C
2. **Excitation**: Two fully connected layers (C → C/r → C with ReLU and sigmoid) produce channel weights s ∈ ℝ^C
3. **Scale**: Each channel is multiplied by its corresponding weight: x_out = s ⊙ x

This allows the model to learn which frequency channels are most important for speaker discrimination — for instance, upweighting formant regions and downweighting noise-dominated bands.

**Multi-layer Feature Aggregation (MFA):**
Rather than using only the final layer's output for pooling, ECAPA-TDNN concatenates frame-level features from all SE-Res2Net blocks. This provides the pooling layer with both low-level acoustic details and high-level speaker characteristics.

**Channel- and Context-Dependent Attentive Statistics Pooling:**
This replaces x-vectors' simple mean+std pooling with an attention mechanism that weights frames differently based on their content:

1. Frame-level features h_t pass through a TDNN layer and tanh nonlinearity
2. Attention scores α_t = softmax(v^T · tanh(W · h_t + b)) weight each frame
3. Weighted mean: μ = Σ_t α_t · h_t
4. Weighted standard deviation: σ = √[Σ_t α_t · (h_t − μ)²]
5. Output: [μ; σ]

The attention is both **channel-dependent** (different channels can have different attention patterns) and **context-dependent** (attention scores consider temporal context through the TDNN layer). This means informative speech frames receive higher weights than silence or noise frames.

### 4.2 Training with AAM-Softmax loss

ECAPA-TDNN uses Additive Angular Margin Softmax (AAM-Softmax) loss instead of standard softmax:

L = -log(e^(s·cos(θ_y + m)) / (e^(s·cos(θ_y + m)) + Σ_{j≠y} e^(s·cos(θ_j))))

where θ_y is the angle between the embedding and the weight vector of the true class, m is the angular margin (typically 0.2), and s is the scale factor (typically 30). The margin penalty forces the model to produce more discriminative embeddings by requiring a larger angular separation between classes.

### 4.3 Output dimensions and usage

The standard ECAPA-TDNN produces **192-dimensional** embeddings. For speaker verification, pairs of embeddings are compared using cosine similarity (no PLDA needed). For diarization, embeddings extracted from short segments (1.5–3 seconds) are clustered using agglomerative hierarchical clustering or spectral clustering.

**Reference:** Desplanques, B., Thienpondt, J., & Demuynck, K. (2020). ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification. *Interspeech 2020*.

---

## 5. TitaNet: NVIDIA's depth-wise separable approach

Koluguri, Park, and Ginsburg (2022) introduced TitaNet at ICASSP 2022, part of NVIDIA's NeMo framework. TitaNet achieves **0.68% EER** on VoxCeleb1 and **1.73% DER** on AMI, surpassing ECAPA-TDNN while being available in multiple sizes.

### 5.1 Architecture

TitaNet uses 1-D depth-wise separable convolutions instead of standard TDNNs, reducing parameter count while maintaining representational capacity. The architecture has three mega-blocks:

- **Prolog block**: Single 1-D convolution processing input features
- **Body blocks (B₁, B₂, B₃)**: Each contains R residual sub-blocks with:
  - 1-D depth-wise separable convolution (depthwise + pointwise)
  - SE module with global context (not just local)
  - Batch normalization and ReLU
- **Epilog block**: Final convolution before pooling

The global context SE module differs from ECAPA-TDNN's SE by incorporating channel-wise statistics from the entire utterance, not just local context. This provides a stronger global speaker representation at every layer.

### 5.2 Model sizes

| Model | Parameters | Embedding Dim | EER (VoxCeleb1) |
|-------|-----------|---------------|-----------------|
| TitaNet-S | 6.3M | 192 | 1.10% |
| TitaNet-M | 13.4M | 192 | 0.84% |
| TitaNet-L | 25.3M | 192 | 0.68% |

All three produce 192-dimensional embeddings, making them drop-in replacements for ECAPA-TDNN. The NeMo framework uses TitaNet-L as the default speaker embedding extractor in its diarization pipelines.

**Reference:** Koluguri, N. R., Park, T., & Ginsburg, B. (2022). TitaNet: Neural Model for Speaker Representation with 1D Depth-wise Separable Convolutions and Global Context. *ICASSP 2022*.

---

## 6. Self-supervised embeddings: WavLM

Chen et al. (2022) presented WavLM in IEEE JSTSP, taking a fundamentally different approach. Instead of training on speaker labels, WavLM learns speech representations through self-supervised pre-training on 94,000 hours of unlabeled audio, then transfers to downstream tasks.

### 6.1 Pre-training objective

WavLM uses masked speech denoising: portions of the input waveform are masked, and overlapping speech from other utterances is mixed in as noise. The model must predict the masked content from the original clean speech, forcing it to learn both speech content and speaker characteristics while being robust to interference.

### 6.2 Architecture

WavLM uses the standard Transformer encoder architecture (similar to wav2vec 2.0 and HuBERT):
- CNN feature encoder: 7 convolutional layers producing 512-dim features at 20ms resolution
- Transformer encoder: 24 layers (Large model) with gated relative position bias
- Output: 1024-dimensional representations per frame

### 6.3 Speaker diarization performance

When fine-tuned for speaker verification, WavLM achieves **12.6% relative DER reduction** on CALLHOME compared to ECAPA-TDNN. Han, Landini, Rohdin, et al. (ICASSP 2025) integrated WavLM into the DiariZen pipeline, substantially outperforming previous baselines. The self-supervised features capture richer speaker characteristics than supervised embeddings, particularly for speakers with limited enrollment data.

**Reference:** Chen, S., et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing. *IEEE Journal of Selected Topics in Signal Processing*, 16(6), 1505–1518.

---

## 7. How speaker embeddings serve diarization

### 7.1 The clustering pipeline

In traditional (non-end-to-end) diarization, speaker embeddings are the central component:

1. **Voice Activity Detection (VAD)**: Identify speech regions using Silero VAD, MarbleNet, or pyannote's segmentation model
2. **Segmentation**: Divide speech regions into short, uniform segments (1.5–3 seconds) with overlap
3. **Embedding extraction**: Extract a speaker embedding for each segment using ECAPA-TDNN, TitaNet, or WavLM
4. **Scoring**: Compute pairwise similarity between all segment embeddings (cosine similarity or PLDA)
5. **Clustering**: Group segments by speaker using agglomerative clustering, spectral clustering, or VBx (Bayesian HMM)
6. **Re-segmentation**: Refine segment boundaries using Viterbi decoding or similar

The quality of speaker embeddings directly determines diarization accuracy. If two segments from different speakers produce similar embeddings, no amount of downstream processing can separate them.

### 7.2 End-to-end systems bypass explicit embeddings

Sortformer, LS-EEND, and other end-to-end diarization models do not extract explicit speaker embeddings. Instead, they learn internal representations that implicitly capture speaker identity within the model's hidden states. Sortformer's Fast Conformer encoder produces 512-dimensional frame representations that encode both speech content and speaker characteristics, but these are never extracted or compared as standalone embeddings.

This is a key architectural difference: clustering-based systems produce interpretable speaker vectors that can be analyzed, compared, and debugged, while end-to-end systems pack speaker identity into opaque hidden representations.

### 7.3 Relevance to ATC audio

For ATC audio, speaker embeddings face specific challenges:

- **VHF radio distortion** (300–3,400 Hz bandwidth, AM modulation) degrades the spectral characteristics that embeddings rely on
- **Short utterances** (1–3 seconds) provide limited acoustic material for robust embedding extraction
- **Many speakers per session** (10–100+ pilot contacts) exceed the capacity of most clustering methods
- **No enrollment data**: Pilots are encountered only once or a few times, with no prior speaker model

These challenges suggest that for ATC, text-based speaker identification leveraging callsigns and ICAO phraseology structure (as in BERTraffic by Zuluaga-Gomez et al., IEEE SLT 2022) may provide stronger speaker discrimination than acoustic embeddings alone.

---

## 8. Evaluation metrics for speaker embeddings

### 8.1 Speaker verification metrics

- **Equal Error Rate (EER)**: The operating point where False Acceptance Rate equals False Rejection Rate. Lower is better. State-of-the-art on VoxCeleb1: 0.68% (TitaNet-L).
- **minDCF**: Minimum Detection Cost Function, a weighted combination of miss and false alarm rates at a specific operating point. Used in NIST SRE evaluations.

### 8.2 Downstream diarization metrics

- **Diarization Error Rate (DER)**: DER = (missed speech + false alarm + speaker confusion) / total reference speech duration. The speaker confusion component directly reflects embedding quality.
- **Jaccard Error Rate (JER)**: Computes intersection-over-union per speaker then averages, giving equal weight to all speakers regardless of speaking time.

---

## Summary

Speaker embeddings have evolved from x-vectors' simple statistics pooling through ECAPA-TDNN's attention mechanisms to TitaNet's efficient depth-wise separable architecture. Self-supervised approaches like WavLM push the frontier further by learning from massive unlabeled data. For diarization, these embeddings serve as the acoustic foundation for clustering-based pipelines, while end-to-end systems like Sortformer learn equivalent representations implicitly. The choice of embedding architecture and its robustness to domain-specific acoustic distortions (such as VHF radio effects in ATC) directly determines downstream diarization accuracy.

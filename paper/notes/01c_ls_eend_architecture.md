# Signal Flow and Architecture of LS-EEND for Speaker Diarization

**CEC 599: Speaker Diarization — Deliverable 1c**
Shital Pandey | Spring 2026 | Instructor: Dr. Jianhua Liu

---

## 1. Context and motivation

LS-EEND (Long-Form Streaming End-to-End Neural Diarization) was proposed by Liang and Li (2024, arXiv:2410.06670, accepted IEEE/ACM TASLP 2025). It addresses two major limitations of prior EEND systems:

1. **Quadratic complexity**: Standard self-attention in EEND models scales as O(T²) with sequence length T, making processing of hour-long recordings impractical
2. **Fixed-length constraint**: Block-wise approaches (BW-EDA-EEND) require fixed block sizes and suffer from inter-block speaker inconsistency

LS-EEND solves both by replacing self-attention with the **Retention mechanism** (adapted from RetNet) in the encoder and using an **online self-attention-based attractor decoder** for speaker tracking. The result is a frame-in-frame-out streaming system with O(TD²) complexity — linear in sequence length.

The official implementation is at: https://github.com/Audio-WestlakeU/FS-EEND

---

## 2. Foundational EEND concepts (prerequisites)

### 2.1 EEND formulation

End-to-End Neural Diarization (Fujita et al., Interspeech 2019) reformulated speaker diarization as frame-level multi-label classification. Given T frames of audio features X = [x₁, ..., x_T], the model produces:

ŷ_{t,s} = σ(f(X)_{t,s})

where ŷ_{t,s} ∈ [0,1] is the probability that speaker s is active at frame t, and σ is the sigmoid function. Multiple speakers can be active simultaneously (unlike softmax, which enforces mutual exclusivity).

### 2.2 Permutation Invariant Training (PIT)

Because the assignment of speakers to output slots is arbitrary, EEND uses PIT loss:

L_PIT = min_{φ ∈ Perm(S)} Σ_t Σ_s L_BCE(ŷ_{t,φ(s)}, y_{t,s})

This evaluates all S! permutations and selects the one with lowest binary cross-entropy (BCE) loss. For S=2, only 2 permutations need checking. For larger S, the Hungarian algorithm reduces this to O(S³).

### 2.3 Encoder-Decoder Attractors (EDA-EEND)

Horiguchi et al. (Interspeech 2020) addressed EEND's fixed-speaker-count limitation by introducing attractor vectors. An LSTM encoder-decoder generates S attractor vectors a₁, ..., a_S from the encoder output. Each attractor represents one speaker. Speaker activity is computed as:

ŷ_{t,s} = σ(e_t · a_s)

where e_t is the frame embedding and · denotes dot product. An additional existence probability p_s = σ(Linear(a_s)) determines how many speakers are present — the model generates attractors until p_s drops below a threshold.

---

## 3. LS-EEND complete signal flow

### 3.1 Input features

Log-mel spectrograms with 23 features at 10 ms resolution (different from Sortformer's 80-channel features), downsampled by a factor of 10 to produce frame embeddings at **100 ms resolution**.

### 3.2 Architecture overview

```
Log-mel features [T × 23]
    ↓ Linear projection + 10× subsampling
Frame features [T/10 × D] where D = 256
    ↓ 4 Conformer-Retention blocks (causal encoder)
Encoded frames e₁, ..., e_T [T/10 × D]
    ↓ Online attractor decoder
Attractors a₁, ..., a_S [S × D]  (updated at every frame)
    ↓ Dot product + sigmoid
Speaker activities ŷ [T/10 × S]
```

### 3.3 The encoder: Conformer blocks with Retention

Each of the 4 encoder blocks follows the Conformer macaron pattern, but with **Retention replacing self-attention**:

x̃ = x + ½·FFN(x) → x' = x̃ + Retention(x̃) → x'' = x' + CausalConv(x') → y = LN(x'' + ½·FFN(x''))

The convolution module uses **causal convolution** (no future frames) for streaming compatibility.

---

## 4. The Retention mechanism in detail

### 4.1 Why Retention instead of self-attention?

Standard self-attention computes:

Attention(Q, K, V) = softmax(QK^T / √d) · V

For a sequence of T frames, QK^T is a T×T matrix — O(T²) computation and memory. For a 1-hour recording at 100 ms resolution, T = 36,000, making the attention matrix ~5 GB in float32. This is impractical.

Retention (from RetNet, Sun et al., 2023) provides an equivalent mechanism with two formulations:

### 4.2 Parallel form (for training)

During training, the entire sequence is available, and Retention can be computed in parallel:

R(X) = (QK^T ⊙ D) · V

where D is a **causal decay mask**:

D_{nm} = γ^{n-m}  if n ≥ m, else 0

The decay factor γ ∈ (0, 1) causes older frames to contribute exponentially less to the current frame's output. This is the key difference from self-attention: instead of a softmax-normalized attention matrix, Retention uses a geometrically decaying causal mask. No softmax means no attention entropy collapse.

### 4.3 Recurrent form (for streaming inference)

During streaming, each frame is processed one at a time using a recurrent formulation:

**State update:** S_n = γ · S_{n-1} + k_n^T · v_n

**Output:** o_n = q_n · S_n

where:
- S_n ∈ ℝ^{D×D} is a fixed-size recurrent state matrix
- q_n, k_n, v_n ∈ ℝ^D are the query, key, value for frame n
- γ is the decay factor

Each new frame requires exactly **O(D²) operations** regardless of position in the stream. The state matrix S_n is a compressed representation of all past frames, exponentially decayed. Total complexity for T frames: O(TD²) — linear in T.

### 4.4 Chunk-wise form (for balanced efficiency)

For practical efficiency, a hybrid chunk-wise form processes B frames at a time:

1. Within each chunk: use the parallel form
2. Across chunks: use the recurrent form to pass state

This balances training parallelism with streaming capability.

### 4.5 Critical adaptation: modified decay rate

The standard RetNet decay γ was designed for language modeling, where older tokens become less relevant fairly quickly. For speaker diarization, older frames may contain critical speaker identity information that must persist for seconds or minutes.

Liang and Li found that the standard γ caused **speaker embedding discontinuities** — visible in t-SNE visualizations as speaker clusters fragmenting over time. Their adapted γ values are larger (closer to 1.0), preserving speaker-relevant information over much longer periods. The exact values are learned per attention head during training.

This adaptation is one of the most important design decisions in LS-EEND. Too aggressive decay (small γ) causes the model to forget speakers during silences. Too conservative decay (γ close to 1.0) overwhelms the state with old information, reducing sensitivity to speaker changes.

---

## 5. The online attractor decoder

### 5.1 Why attractors matter

The attractor decoder is what makes LS-EEND fundamentally different from Sortformer. Sortformer uses fixed output slots with no explicit speaker representations — speaker identity is implicit in the model's internal states. LS-EEND generates **explicit speaker vectors (attractors)** that are continuously updated and can be inspected, compared, and analyzed.

### 5.2 Decoder architecture

The attractor decoder models two dimensions simultaneously:

**Temporal dimension (via Retention):** Each speaker's attractor evolves over time, accumulating acoustic evidence. The same Retention mechanism used in the encoder tracks how each attractor changes as new frames arrive.

**Speaker dimension (via self-attention):** At each time step, all S attractors attend to each other through standard self-attention. This cross-speaker interaction is critical — it allows attractors to **actively differentiate** from each other. If two attractors start drifting toward similar representations, the self-attention mechanism can push them apart.

### 5.3 Frame-by-frame attractor update

At each frame t:

1. **Encode** the new frame through the 4 Conformer-Retention encoder blocks, producing e_t
2. **Update** each attractor using the decoder's Retention mechanism, incorporating e_t
3. **Differentiate** attractors through cross-speaker self-attention
4. **Compute** speaker activities: ŷ_{t,s} = σ(a_{s,t}^T · ẽ_t), where ẽ_t is the L2-normalized frame embedding and a_{s,t} is the current attractor for speaker s

The L2 normalization ensures that the dot product represents cosine similarity, making the activity decision dependent on the direction of the vectors rather than their magnitude.

### 5.4 Speaker existence detection

Like EDA-EEND, LS-EEND determines the number of active speakers through an existence probability:

p_s = σ(Linear(a_s))

Attractors with p_s below a threshold are considered inactive. LS-EEND supports up to **8 speakers** (compared to Sortformer's hard cap of 4).

---

## 6. Training procedure

### 6.1 Progressive training curriculum

LS-EEND uses a multi-stage training curriculum that progressively increases difficulty:

1. **Stage 1**: Train on simulated mixtures with 1–2 speakers, short segments
2. **Stage 2**: Increase to 1–4 speakers, medium segments
3. **Stage 3**: Scale up to 1–8 speakers with longer segments
4. **Stage 4**: Fine-tune on real conversation data (CALLHOME, DIHARD)
5. **Stage 5**: Train on hour-long recordings using the chunk-wise Retention form

Each stage builds on the previous stage's weights. This curriculum prevents catastrophic forgetting — the model retains its ability to handle simple cases while learning to handle complex ones.

### 6.2 Simulation data

Training simulations use the LibriSpeech corpus combined with noise augmentation. Speaker mixtures are created by randomly combining single-speaker utterances with realistic overlap patterns, silence distributions, and speaker counts following statistics from real conversation datasets.

---

## 7. Benchmark results

### 7.1 Reported DER

| Dataset | Mode | DER |
|---------|------|-----|
| CALLHOME (2 spk) | Online | 9.61% |
| CALLHOME (all spk) | Online | 12.11% |
| DIHARD III (full) | Online | 19.61% |
| AMI (headset) | Online | 20.76% |
| Simulated (2 spk) | Online | 4.35% |
| Simulated (3 spk) | Online | 9.98% |
| Simulated (8 spk) | Online | 36.02% |

### 7.2 Comparison with Sortformer

| Metric | Sortformer (Offline) | Streaming Sortformer (10s) | LS-EEND (Online) |
|--------|---------------------|---------------------------|-------------------|
| CALLHOME-all | ~12.6% | 10.09% | 12.11% |
| DIHARD III | 14.76% (≤4spk) | 19.02% (all) | 19.61% (all) |
| Max speakers | 4 | 4 | 8 |
| Latency | Full recording | 10.0s | Per-frame |
| RTF | — | 0.005 | 0.028 |

LS-EEND is competitive with Sortformer while being a true per-frame streaming system with higher speaker capacity.

---

## 8. FS-EEND codebase structure

The official implementation at https://github.com/Audio-WestlakeU/FS-EEND contains:

- **FS-EEND**: Frame-wise Streaming EEND with non-autoregressive self-attention-based attractors (ICASSP 2024 — the predecessor to LS-EEND)
- **LS-EEND**: The long-form streaming extension (TASLP 2025)

Key source files relevant to our adaptation work:
- Model architecture definitions
- Retention mechanism implementation
- Attractor decoder with cross-speaker attention
- Training scripts with progressive curriculum
- Evaluation scripts with DER computation

The syllabus lists adapting this framework and updating its dependencies as Deliverable 2b.

---

## 9. Architectural comparison: why LS-EEND may resist lock-up

The lock-up pattern observed with Sortformer on ATC audio — where speaker assignment collapses to a single dominant speaker — has clear architectural explanations for why LS-EEND might handle it differently:

### 9.1 No attention entropy collapse risk

Sortformer's 18-layer Transformer uses softmax-based self-attention, which is theoretically susceptible to entropy collapse (attention weights concentrating on a few frames). LS-EEND's Retention mechanism uses exponential decay instead of softmax — there is no entropy to collapse. The decay mask is fixed and deterministic, not learned or data-dependent.

### 9.2 Explicit attractor self-correction

LS-EEND's cross-speaker self-attention in the decoder explicitly maintains attractor distinctiveness. If two attractors start converging (which would cause speaker confusion), the self-attention mechanism can detect this and push them apart. Sortformer has no equivalent mechanism — speaker slot separation is entirely implicit.

### 9.3 Per-frame updates prevent accumulation

LS-EEND updates attractors at every frame using the recurrent Retention form. If a speaker assignment error occurs at one frame, the next frame provides a fresh opportunity for correction. Sortformer's offline mode processes all frames simultaneously, meaning an error in the attention pattern affects all frames at once.

### 9.4 But LS-EEND has its own weaknesses

- **PIT for ≤8 speakers**: Still requires permutation search, though O(S³) with Hungarian algorithm
- **Retention decay over silence**: Extended silence periods (common in ATC) can cause attractor degradation
- **Smaller model capacity**: 4 Conformer blocks with 256 dimensions versus Sortformer's 18+18 layers with 512+192 dimensions
- **Less training data**: Primarily trained on simulated mixtures, not the 7,000+ hours used by Sortformer

---

## 10. Relevance to our ATC diarization work

Testing LS-EEND on our ATC0 dataset would directly address whether the lock-up pattern is Sortformer-specific or a general property of neural diarization on ATC audio. If LS-EEND does not exhibit lock-up:

- The cause is architectural (likely attention entropy collapse in Sortformer's Transformer)
- Solutions should focus on attention stabilization or switching to attractor-based approaches

If LS-EEND also exhibits lock-up:

- The cause is domain-related (ATC audio characteristics that confuse all neural diarization models)
- Solutions should focus on domain adaptation, ATC-specific training data, or hybrid acoustic-linguistic approaches

---

## References

1. Liang, D. & Li, X. (2024). LS-EEND: Long-Form Streaming End-to-End Neural Diarization with Online Attractor Extraction. *arXiv:2410.06670*. Accepted IEEE/ACM TASLP 2025.
2. Fujita, Y., Kanda, N., Horiguchi, S., Nagamatsu, K., & Watanabe, S. (2019). End-to-End Neural Speaker Diarization with Permutation-Free Objectives. *Interspeech 2019*.
3. Horiguchi, S., Fujita, Y., Watanabe, S., Xue, N., & Nagamatsu, K. (2020). End-to-End Speaker Diarization for an Unknown Number of Speakers with Encoder-Decoder Based Attractors. *Interspeech 2020*.
4. Sun, Y., Dong, L., Huang, S., et al. (2023). Retentive Network: A Successor to Transformer for Large Language Models. *arXiv:2307.08621*.
5. Han, Z., Lee, C., & Stolcke, A. (2021). BW-EDA-EEND: Streaming End-to-End Neural Speaker Diarization for a Variable Number of Speakers. *ICASSP 2021*.
6. FS-EEND GitHub Repository: https://github.com/Audio-WestlakeU/FS-EEND

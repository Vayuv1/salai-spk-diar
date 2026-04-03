# Understanding Speaker Diarization Inside and Out: From Theory to Our ATC Lock-Up Problem

**CEC 599: Speaker Diarization — Comprehensive Study Guide**
Shital Pandey | Spring 2026

---

## How to use this document

This document is meant to give you a complete understanding of modern speaker diarization — enough that you can walk into Dr. Liu's office, explain any architecture in detail, diagnose the lock-up problem we observed, and defend your approach in the DASC 2026 paper. It is written to be read front-to-back, with each section building on the previous one. By the end, you should be able to enter Phase 2 (comparative benchmarking and experiments) with full confidence in the theoretical foundation.

---

## Part 1: What is speaker diarization and why is it hard?

### The basic question

Speaker diarization answers: "who spoke when?" Given an audio recording with multiple speakers, the system must segment the audio into regions and assign each region to a speaker. The output is typically an RTTM file — a list of (speaker_id, start_time, duration) tuples.

### Why it is harder than it sounds

Three properties make diarization fundamentally challenging:

**The number of speakers is unknown.** Unlike speaker verification (is this person X?) or speaker identification (which of these N known people is speaking?), diarization must discover how many speakers exist in the recording without any prior information.

**Speaker identity is relative, not absolute.** The system has no enrolled speaker profiles. It must determine that "the person speaking at t=5s is the same person who spoke at t=25s" purely from acoustic similarity, without knowing who either person actually is.

**The permutation problem.** If the model outputs probabilities for 4 speaker slots, which slot corresponds to which speaker? During training, the mapping between ground truth speakers and output slots is arbitrary. This is the permutation problem, and different architectures solve it differently — PIT searches all possible mappings, Sort Loss uses temporal ordering, and attractor-based methods use learned speaker representations.

### How we measure performance

**DER (Diarization Error Rate)** is the standard metric:

DER = (Missed Speech + False Alarm + Speaker Confusion) / Total Reference Duration

Our ATC0 result: DER = 22.04%, broken down as:
- False Alarm (FA) = 2.68% — model says someone is talking when nobody is
- Missed Speech (MISS) = 2.83% — model misses speech that is actually there
- Speaker Confusion (CER) = 16.53% — model detects speech correctly but assigns it to the wrong speaker

The confusion component dominates. The model hears speech fine; it just cannot figure out who is talking. This is the lock-up pattern.

---

## Part 2: The two families of diarization systems

### Family 1: Clustering-based (modular pipeline)

The traditional approach chains together separate components:

1. **VAD** detects speech regions (Silero VAD, NeMo MarbleNet)
2. **Segmentation** cuts speech into uniform chunks (1.5–3 seconds)
3. **Embedding extraction** computes a speaker vector for each chunk (ECAPA-TDNN, TitaNet)
4. **Clustering** groups chunks by speaker similarity (VBx, agglomerative, spectral)

Strengths: handles many speakers, each component is interpretable and debuggable, no permutation problem (clustering assigns global labels).

Weaknesses: cannot handle overlapping speech (each chunk gets exactly one label), errors cascade between stages, no joint optimization.

### Family 2: End-to-end neural (EEND)

A single neural network directly outputs per-frame, per-speaker activity probabilities. Trained end-to-end with a loss function that handles the permutation problem.

Strengths: handles overlap naturally (sigmoid outputs allow multiple speakers per frame), jointly optimized, simpler deployment.

Weaknesses: fixed maximum speaker count, requires permutation-aware loss, harder to interpret and debug, susceptible to domain mismatch.

**Our system uses Sortformer, an end-to-end approach.** Understanding why this family has the weaknesses it has is central to understanding the lock-up.

---

## Part 3: Speaker embeddings — how voices become vectors

Before diving into end-to-end systems, you need to understand speaker embeddings because they are the building blocks of clustering-based diarization and because the concept of "speaker representations" is fundamental to understanding what goes wrong in end-to-end systems too.

### What makes a voice unique

Your voice is shaped by the physical dimensions of your vocal tract (throat, mouth, nasal cavity), which create characteristic resonance frequencies called formants. The fundamental frequency (F0) from your vocal folds, your habitual speaking patterns (speed, intonation, articulation), and even your accent all contribute to a unique acoustic profile.

A speaker embedding system takes a variable-length audio clip and produces a fixed-size vector (typically 192 or 512 dimensions) that captures these characteristics while ignoring irrelevant factors like what words are being said, background noise, and channel effects.

### The evolution: x-vectors → ECAPA-TDNN → TitaNet

**X-vectors** (Snyder et al., ICASSP 2018) use a TDNN (1-D convolutional network) that processes frame-level features, then a statistics pooling layer computes the mean and standard deviation across all frames. This produces a fixed 512-dimensional vector regardless of input length. Simple and effective, but all frames contribute equally — silence is weighted the same as speech.

**ECAPA-TDNN** (Desplanques et al., Interspeech 2020) fixed this with three innovations: (1) Squeeze-Excitation blocks that learn which frequency channels are most speaker-discriminative, (2) multi-layer feature aggregation that combines information from all layers (not just the last), and (3) attentive statistics pooling that learns to weight informative frames higher than silence or noise. Result: 0.87% EER on VoxCeleb1, a massive improvement. Produces 192-dimensional embeddings.

**TitaNet** (Koluguri et al., ICASSP 2022) replaced standard convolutions with depthwise separable convolutions for efficiency, added global context to the Squeeze-Excitation modules, and achieved 0.68% EER on VoxCeleb1. Available in NVIDIA's NeMo toolkit.

### Why embeddings matter for understanding the lock-up

End-to-end systems like Sortformer do not extract explicit speaker embeddings. Instead, their internal representations implicitly carry speaker information. The key insight: **if the encoder produces similar internal representations for different speakers (because VHF radio distortion makes their voices sound alike to the model), the downstream layers cannot separate them, regardless of how sophisticated they are.** This is the first link in the lock-up chain.

---

## Part 4: The Conformer — foundation of modern speech processing

### Why you need to understand this

The Conformer is the encoder backbone used by Sortformer. Every audio frame that enters the diarization system passes through the Conformer first. If the Conformer produces poor representations for ATC audio, everything downstream fails.

### The core idea: local + global features

Speech has two kinds of patterns:
- **Local**: Phoneme boundaries, formant transitions, voice onset times. These happen within a few milliseconds.
- **Global**: Speaking rate, intonation contours, speaker identity. These span hundreds of milliseconds to seconds.

Convolution is great at local patterns (it looks at a small window of neighboring frames). Self-attention is great at global patterns (every frame can attend to every other frame). The Conformer puts both in every block.

### The macaron block structure

Each Conformer block has four sub-modules with residual connections:

1. **Half-step FFN**: A feed-forward network that processes each frame independently. The "half-step" means the residual connection adds only 50% of the output: x̃ = x + 0.5·FFN(x). This pre-processes features for the attention module.

2. **Multi-head self-attention with relative positions**: Every frame computes attention scores against every other frame, capturing global patterns. Relative positional encoding tells the model "this frame is 5 frames before that frame" rather than "this frame is at absolute position 47." This is critical for Sortformer because Sort Loss needs the model to understand temporal ordering.

3. **Depthwise convolution**: A 1-D convolution applied to each channel independently (not mixing channels). With a kernel of size 9, each frame's output depends on 4 frames before and 4 frames after it — capturing local acoustic patterns like formant transitions. A GLU gate selects which features pass through.

4. **Half-step FFN + LayerNorm**: Another FFN to post-process, then normalization for the next block.

### Fast Conformer: the same thing, faster

NVIDIA's Fast Conformer makes four changes: (1) 8× downsampling instead of 4× (using 3 strided depthwise separable conv layers), reducing frame count by half; (2) fewer subsampling channels (256 vs 512); (3) smaller convolution kernel (9 vs 31); and (4) limited-context attention option for long recordings.

The result: 2.9× fewer multiply-add operations, processing capability from 15 minutes to 675 minutes on a single GPU, and WER actually improves from 5.19% to 4.99% on LibriSpeech test-other. More layers compensate for the reduced per-frame computation.

### What this means for ATC

The 8× downsampling produces frame embeddings at **80 ms resolution**. A 10-second window has ~125 frames. A typical ATC pilot transmission (1–3 seconds) spans only 12–37 frames. The model has very few frames to work with for each speaker turn. The Conformer's convolution kernel covers ~720 ms (9 frames × 80 ms), which is comparable to a single short utterance — meaning the local feature extraction barely spans one speaker turn.

The NEST pre-training used wideband speech. The Conformer has never seen VHF radio distortion (narrow bandwidth, AM modulation, squelch transients) during pre-training. Its representations on ATC audio may lack the speaker-discriminative features it learned from clean wideband speech.

---

## Part 5: Sortformer — how it works, piece by piece

### The complete signal path

```
16 kHz audio → 80-ch log-mel (10 ms stride) → 8× subsampling (80 ms) →
18 Fast Conformer blocks (d=512) → Linear projection (512→192) →
18 Transformer encoder blocks (d=192) → 2 FFN layers + sigmoid → [T × 4] probabilities
```

For a 10-second window: 1000 mel frames → 125 downsampled frames → 125 frames through 36 total layers → 125 × 4 output matrix.

### Sort Loss: the arrival-time trick

The permutation problem: the model has 4 output slots. Which slot is which speaker? Traditional EEND tries all 24 possible assignments (4! = 24) and picks the best. Sort Loss instead defines a rule: the first speaker to appear in the ground truth gets slot 0, the second gets slot 1, etc.

This is elegant because it is deterministic — no search needed, complexity drops from O(K!) to O(K log K). But it creates a strong assumption: **speakers can be meaningfully ordered by first appearance.** In meetings, this works well — people join the conversation and persist. In ATC, the controller speaks first in nearly every window, so slot 0 is almost always the controller. Pilots appear briefly and may never return. The ordering assumption becomes a brittleness point.

### The Transformer module and dropout

The 18-layer Transformer encoder (hidden size 192, 8 heads) processes all 125 frames simultaneously through self-attention. Every frame can attend to every other frame. This is where speaker associations are formed — the model learns to associate acoustically similar frames across the window.

During training, **0.5 dropout** is applied to attention scores, attention outputs, and FFN outputs. This is an extremely high dropout rate — half of all values are randomly zeroed at every layer in every forward pass. This prevents any single attention pattern from dominating and forces the model to learn robust, distributed representations.

**At inference, dropout is turned off.** The model can now form sharper, more concentrated attention patterns than it ever experienced during training. On in-distribution data (meetings, phone calls), this sharpness helps. On out-of-distribution data (ATC radio), it may cause attention to collapse onto a subset of frames — the beginning of the lock-up.

### The output: sigmoid, not softmax

Each of the 4 output values per frame passes through a sigmoid function independently. This means the model can predict multiple speakers active simultaneously (overlap). For ATC, where half-duplex radio prevents overlap, the model still activates multiple outputs — which contributes to confusion errors.

---

## Part 6: Streaming Sortformer — how chunking and AOSC change the game

### The problem with offline processing

Offline Sortformer processes an entire window (10 seconds in our setup) at once. If the attention patterns collapse on frame 30, that collapse affects the representations of all 125 frames because self-attention is bidirectional. There is no recovery mechanism — the error is baked into the output.

### Chunked processing limits error propagation

Streaming Sortformer processes audio in chunks (as small as 3 frames = 0.24 seconds). Each chunk sees: the speaker cache (long-term speaker anchors), a FIFO queue (recent context), the current chunk, and a short right context. If attention collapses within one chunk, the damage is limited to those few frames. The next chunk gets a fresh opportunity.

### The AOSC: remembering speakers across chunks

The Arrival-Order Speaker Cache stores representative frame embeddings from each detected speaker, organized by arrival time. Its 7-step update procedure:

1. Score each frame by how confidently it represents one speaker
2. Find silence frames and compute their average embedding
3. Suppress non-speech frames from speaker selection
4. Boost recent frames slightly (recency bias δ = 0.05)
5. Guarantee minimum 33 frames per detected speaker (prevents forgetting)
6. Insert 3 silence embeddings between speaker segments (boundary markers)
7. Keep top 188 frames by score, preserving arrival order

The guaranteed minimum representation is the key self-correction mechanism: once a speaker is registered in the cache, they cannot be completely dropped, even after long silences. This is absent from offline Sortformer.

### Why streaming might resist lock-up

Three reasons: (1) shorter processing windows limit the extent of any attention collapse, (2) the AOSC provides explicit speaker anchors that persist across chunks, and (3) each chunk boundary provides a natural reset point. Testing streaming Sortformer on ATC0 is one of the highest-priority experiments for Phase 2.

---

## Part 7: LS-EEND — a fundamentally different approach

### Attractors vs. slots

Sortformer uses fixed output slots with implicit speaker representations. LS-EEND uses **explicit attractor vectors** — learned per-speaker representations that are continuously updated at every frame.

Think of it this way: Sortformer asks "which of my 4 output channels should be active right now?" while LS-EEND asks "how similar is this frame to each speaker's current acoustic profile?"

The attractor approach has a built-in self-correction mechanism. At every frame, each attractor is updated with new acoustic evidence. If the attractor drifted away from the true speaker characteristics, the next frame of that speaker's speech pulls it back. Additionally, cross-speaker self-attention in the decoder explicitly pushes attractors apart if they start converging — preventing the confusion that causes lock-up.

### Retention instead of self-attention

LS-EEND replaces softmax self-attention in its encoder with the Retention mechanism from RetNet. The practical difference:

**Self-attention**: attention_weight = softmax(Q·K^T / √d). The softmax can concentrate to a near-one-hot distribution (entropy collapse), causing all frames to attend to one frame.

**Retention**: weight = exponential_decay(distance). The weight is a deterministic function of temporal distance — no data-dependent collapse is possible. Older frames contribute less (exponential decay), but the decay rate is tuned to preserve speaker information over long periods.

This architectural difference means LS-EEND is **structurally immune** to the specific attention entropy collapse mechanism that likely causes Sortformer's lock-up. Whether it has its own failure modes on ATC audio is an open question for Phase 2.

### Frame-in-frame-out streaming

LS-EEND processes one frame at a time using the recurrent form of Retention:

State update: S_n = γ · S_{n-1} + k_n^T · v_n (accumulate new info, decay old)
Output: o_n = q_n · S_n (query the accumulated state)

Each frame requires O(D²) operations regardless of position in the stream. No chunking, no speaker cache — just a fixed-size state matrix that evolves as new frames arrive. Total complexity: O(TD²), linear in sequence length.

---

## Part 8: The lock-up diagnosis — connecting architecture to failure

### What we observe

On ATC0 audio processed in 10-second windows with 5-second overlap:
- The model correctly identifies different speakers for the first few seconds
- At some point (varying by window), speaker assignment "locks" to one speaker
- From that point forward, all speech is assigned to the same output slot
- The lock-up point varies between overlapping windows (a 5-second shift changes when it occurs)
- Noise perturbation (ON/OFF periodic noise) does not break the pattern
- Small time shifts (3–10 ms) do not break the pattern

### The mechanism: attention entropy collapse

The most likely explanation, grounded in Zhai et al. (ICML 2023):

**Step 1**: ATC audio's VHF radio characteristics (narrow bandwidth, AM distortion, squelch artifacts) produce mel spectrograms that occupy a restricted region of the feature space compared to wideband training data. The Fast Conformer encoder, never having seen these characteristics, produces increasingly similar 512-dimensional embeddings across frames.

**Step 2**: The 18-layer Transformer encoder (d=192) processes these similar embeddings through self-attention. With out-of-distribution inputs causing elevated logit variance, attention weights concentrate on a subset of frames. Multiple attention heads converge to similar patterns (redundancy).

**Step 3**: As attention concentrates, the 192-dimensional frame representations at the Transformer output converge. LayerNorm amplifies relative differences but cannot create diversity when inputs are already similar.

**Step 4**: The output FFN layers receive undifferentiated input and default to the most probable pattern from training: one dominant speaker. The sigmoid saturates (one slot near 1.0, others near 0.0).

**Step 5**: No recovery mechanism exists. All frames are processed simultaneously (offline mode), so there is no opportunity to detect and correct the collapse.

### Why our perturbation experiments failed

The time-shift experiment (moving audio by 3–10 ms) and the noise ON/OFF experiment both tried to "shake" the model out of lock-up by changing the input. But the problem is not that a specific input pattern triggers lock-up — the problem is that **all ATC audio produces similar encoder representations** because the model was not trained on this domain. Small perturbations change the input but do not make it more in-distribution.

### Why the lock-up starts after correct initial identification

The first speaker turns in a window have distinctive onset characteristics — push-to-talk activation creates a sharp energy transient, and the initial formant transitions after silence are more discriminable than sustained speech. These features fall within the encoder's discrimination capacity. As the window progresses and the model must rely on sustained acoustic patterns (which are severely distorted by VHF radio), the encoder representations converge and attention collapse begins.

---

## Part 9: What to test in Phase 2

Armed with this theoretical understanding, the experiments for Phase 2 follow logically:

### Experiment 1: Streaming Sortformer on ATC0

Run diar_streaming_sortformer_4spk-v2 at multiple latency configurations (0.32s, 1.04s, 10s) on the same ATC0 test set. If streaming shows reduced lock-up (lower CER) despite using the same model weights, this confirms that chunked processing with AOSC provides meaningful self-correction. Generate the same waterfall plots for direct visual comparison.

### Experiment 2: LS-EEND on ATC0

Adapt the FS-EEND codebase, run LS-EEND on ATC0, and measure DER. If LS-EEND does not exhibit lock-up, the cause is confirmed as Sortformer's attention mechanism. If it also locks up, the cause is domain-related (ATC audio confusing all neural diarization models), and solutions must focus on domain adaptation rather than architecture changes.

### Experiment 3: Pyannote 3.x on ATC0

Run pyannote's clustering-based pipeline as a baseline. This uses ECAPA-TDNN embeddings with agglomerative clustering — a fundamentally different approach. If pyannote handles ATC well, it tells us that the problem is specific to end-to-end architectures, not to neural processing of ATC audio in general.

### Experiment 4: VBx on ATC0

VBx uses TitaNet embeddings with Bayesian HMM clustering. It handles many speakers and is structurally immune to attention-based lock-up. Since ATC is half-duplex (no overlap), VBx's inability to handle overlap is irrelevant. VBx may be the best-suited architecture for ATC among existing systems.

### Experiment 5: Attention analysis on Sortformer

Extract attention weight matrices from the Sortformer Transformer layers during inference on ATC0. Visualize attention entropy across layers and time. Identify the layer and frame where entropy collapse begins. This provides direct evidence for the diagnosis.

---

## Part 10: Quick reference — key numbers and facts

| Fact | Value |
|------|-------|
| Sortformer model used | diar_sortformer_4spk-v1.nemo |
| Our DER on ATC0 | 22.04% (FA=2.68, MISS=2.83, CER=16.53) |
| Window size / overlap | 10s / 5s |
| Frame resolution | 80 ms (8× downsampling from 10 ms) |
| Frames per 10s window | ~125 |
| Sortformer encoder | 18 Fast Conformer blocks, d=512 |
| Sortformer Transformer | 18 layers, d=192, 8 heads |
| Training dropout | 0.5 (disabled at inference) |
| Training segment length | 90 seconds |
| Max speakers | 4 (hard cap) |
| Sortformer DER on CALLHOME | ~12.6% |
| Streaming Sortformer DER | 10.09% (CALLHOME, 10s latency) |
| LS-EEND DER on CALLHOME | 12.11% (online) |
| BERTraffic improvement | 32% relative over VBx on ATC |
| ATC0 test recordings | 7 files, ~10+ hours total |
| Silero VAD on ATC audio | ~0.2 prob for speech (degraded) |
| NeMo VAD on ATC audio | ~1.0 prob everywhere (undiscriminating) |

---

## Conclusion

You now have a complete picture of: how speaker diarization works (both clustering and end-to-end), how voice fingerprints become speaker embeddings, how the Conformer/Fast Conformer backbone extracts features, how Sortformer and its streaming extension process those features into speaker probabilities, how LS-EEND takes a fundamentally different approach with attractors and retention, and why Sortformer locks up on ATC audio while other architectures may not.

The theoretical foundation is set. Phase 2 is about testing these predictions experimentally.

# Literature Survey: Speaker Diarization for Real-Time ATC Information Extraction

**CEC 599: Speaker Diarization — Literature Survey Report**
Shital Pandey | Spring 2026 | Instructor: Dr. Jianhua Liu
Embry-Riddle Aeronautical University, Daytona Beach

---

## Abstract

This survey covers the state of speaker diarization as of early 2026, with emphasis on end-to-end neural architectures, streaming systems, and their applicability to air traffic control (ATC) communications. The survey spans foundational EEND work, the Sortformer and LS-EEND architectures, speaker embedding systems, evaluation methodologies, and ATC-specific diarization challenges. A total of 45 publications from 2018–2026 are reviewed, organized by topic area with connections drawn to the specific problem of speaker assignment lock-up observed when applying Sortformer to ATC audio.

---

## 1. End-to-End Neural Diarization: Foundations

### 1.1 EEND: The paradigm shift (Fujita et al., 2019)

Fujita, Kanda, Horiguchi, Nagamatsu, and Watanabe (Interspeech 2019) introduced End-to-End Neural Diarization, reframing diarization from a pipeline of separate components (VAD → segmentation → embedding → clustering) into a single neural network trained with Permutation Invariant Training (PIT) loss. The model uses a stack of Transformer encoder layers to process frame-level features and produces sigmoid outputs indicating per-speaker activity at each frame. On simulated 2-speaker mixtures, EEND achieved 12.28% DER versus 28.77% for conventional clustering baselines. The formulation naturally handles overlapping speech — a persistent weakness of clustering approaches. The limitation: EEND requires a fixed maximum number of speakers specified at training time.

### 1.2 EDA-EEND: Flexible speaker count (Horiguchi et al., 2020)

Horiguchi, Fujita, Watanabe, Xue, and Nagamatsu (Interspeech 2020) solved the fixed-speaker-count problem by introducing Encoder-Decoder Attractors (EDA). An LSTM encoder-decoder generates attractor vectors sequentially, with each attractor representing one speaker. An existence probability determines when to stop generating attractors, allowing the model to dynamically determine speaker count. Speaker activity is computed as the sigmoid of the dot product between frame embeddings and attractor vectors. This work established the attractor paradigm that LS-EEND later extends to streaming.

### 1.3 EEND-VC: Bridging end-to-end and clustering (Kinoshita et al., 2021)

Kinoshita, Delcroix, and Tawara (Interspeech 2021) proposed EEND-vector clustering, combining EEND's overlap handling with the scalability of clustering methods. Local speaker activities from short EEND windows are merged into global speaker identities through embedding-based clustering. This hybrid approach handles both overlap and many speakers — directly relevant to ATC where individual windows may have ≤4 speakers but sessions involve many more.

### 1.4 Block-wise streaming EEND (Han et al., 2021)

Han, Lee, and Stolcke (ICASSP 2021) presented BW-EDA-EEND for online diarization. Audio is processed in fixed-length blocks with inter-block speaker matching using attractor similarity. The approach works but suffers from block-boundary artifacts and speaker identity inconsistency across blocks — problems that LS-EEND's retention mechanism was specifically designed to solve.

### 1.5 Synthetic training data (Landini et al., 2022)

Landini, Lozano-Diez, Diez, and Burget (Interspeech 2022) demonstrated that generating synthetic conversations matching the statistical properties of real conversations (overlap ratios, turn-taking patterns, speaker counts) substantially improves EEND training. This finding is important for ATC, where real speaker-labeled data is scarce but communication patterns are well-characterized by ICAO standards.

---

## 2. Sortformer Family

### 2.1 Sortformer: arrival-time ordering (Park et al., 2024)

Park, Medennikov, Dhawan, Wang, Huang, Koluguri, Puvvada, Balam, and Ginsburg (arXiv:2409.06656, ICML 2025) introduced Sort Loss, which resolves the permutation problem by ordering speakers by their first appearance. Built on a Fast Conformer encoder (18 layers, d=512) plus an 18-layer Transformer encoder (d=192), the model produces a T×4 speaker probability matrix. The hybrid loss (0.5 Sort + 0.5 PIT) gives best results: 14.76% DER on DIHARD III (≤4 speakers) and competitive CALLHOME results. Sort Loss enables seamless ASR integration through sinusoidal speaker kernels, eliminating the need for PIT in the ASR loss computation. The model requires relative positional embeddings to break self-attention's permutation equivariance.

Key finding relevant to our work: the paper notes that Sort Loss models tend to overestimate speaker count on longer recordings, and that performance degrades more sharply than PIT models when evaluation length differs from the 90-second training length.

### 2.2 Streaming Sortformer with AOSC (Medennikov et al., 2025)

Medennikov, Park, Wang, Huang, Dhawan, Wang, Balam, and Ginsburg (Interspeech 2025, arXiv:2507.18446) extended Sortformer to streaming using the Arrival-Order Speaker Cache. The AOSC stores representative frame embeddings organized by speaker arrival order, with a 7-step update procedure that guarantees minimum speaker representation, applies recency bias, and inserts silence separators. At 10-second latency, streaming Sortformer achieves 10.09% DER on CALLHOME-all — outperforming the offline model's ~12.6%. The paper attributes this to the offline model's degradation on recordings longer than its 90-second training window, while streaming avoids this mismatch. Available models: diar_streaming_sortformer_4spk-v2 and v2.1 on HuggingFace.

### 2.3 Multi-speaker ASR integration

The Sortformer paper describes joint diarization-ASR training where speaker probability matrices are multiplied by sinusoidal kernels and added to ASR encoder states. This allows standard CTC/attention-based ASR decoding with speaker attribution, enabling speaker-attributed transcription — the end goal of our real-time ATC information extraction system.

---

## 3. LS-EEND and the Retention Mechanism

### 3.1 LS-EEND (Liang & Li, 2024)

Liang and Li (arXiv:2410.06670, IEEE/ACM TASLP 2025) proposed LS-EEND, replacing self-attention in the EEND encoder with the Retention mechanism from RetNet (Sun et al., 2023). In its recurrent form, Retention maintains a fixed-size state matrix updated at O(D²) per frame, enabling true frame-in-frame-out streaming with linear complexity O(TD²). The decoder uses cross-speaker self-attention to maintain attractor distinctiveness. LS-EEND supports up to 8 speakers and achieves 12.11% DER on CALLHOME (online) with RTF 0.028. The progressive training curriculum scales from 1 to 8 speakers and from short to hour-long recordings. Official code: https://github.com/Audio-WestlakeU/FS-EEND.

### 3.2 RetNet (Sun et al., 2023)

Sun, Dong, Huang, et al. (arXiv:2307.08621) proposed the Retentive Network as a Transformer alternative with three computation modes: parallel (training), recurrent (inference), and chunk-wise (balanced). The key innovation is replacing softmax attention with a causal exponential decay mask, providing O(1) per-step inference without the attention entropy collapse risk of softmax. LS-EEND adapted the decay rate γ to preserve speaker information over longer periods than the original language modeling application.

---

## 4. Conformer and Fast Conformer Architectures

### 4.1 Original Conformer (Gulati et al., 2020)

Gulati, Qin, Chiu, Parmar, Zhang, Yu, Han, Wang, Zhang, Wu, and Pang (Interspeech 2020, arXiv:2005.08100) proposed the Conformer, combining multi-head self-attention with depthwise convolution in a macaron-style architecture. The design captures both local (convolution kernel = 31 frames) and global (full self-attention) dependencies. On LibriSpeech, Conformer achieved 2.1/4.3% WER (test-clean/test-other) with 118M parameters, setting a new state of the art.

### 4.2 Fast Conformer (Rekesh et al., 2023)

Rekesh, Koluguri, Kriman, Majumdar, Noroozi, Huang, Hrinchuk, Puvvada, Kumar, Balam, and Ginsburg (ICASSP 2023, arXiv:2305.05084) modified Conformer with 8× downsampling (3 depthwise separable conv layers), reduced subsampling channels, and kernel size 9. The result: encoder GMACS from 143.2 to 48.7 (2.9× reduction) while WER improved from 5.19% to 4.99% on test-other. Limited-context attention with a global token enables processing recordings up to 675 minutes on a single A100. This is the encoder used in all Sortformer models.

### 4.3 NEST pre-training (Park et al., 2024)

The NEST (Neural Encoder for Speech Transformers) framework pre-trains the Fast Conformer encoder using BEST-RQ self-supervised learning on ~25,000 hours of speech. This produces a general-purpose speech encoder that captures robust acoustic representations before task-specific fine-tuning. The pre-training enables Sortformer's strong performance without task-specific encoder design.

---

## 5. Speaker Embedding Systems

### 5.1 X-vectors (Snyder et al., 2018)

Snyder, Garcia-Romero, Sell, Povey, and Khudanpur (ICASSP 2018) introduced x-vectors using a TDNN with statistics pooling (mean + standard deviation across frames) producing 512-dimensional embeddings. Trained as speaker classification on VoxCeleb with data augmentation from MUSAN noise and simulated RIRs. Scored using PLDA backend. The architecture established deep learning as the standard for speaker recognition.

### 5.2 ECAPA-TDNN (Desplanques et al., 2020)

Desplanques, Thienpondt, and Demuynck (Interspeech 2020) proposed ECAPA-TDNN with Squeeze-Excitation Res2Net blocks, multi-layer feature aggregation, and channel-dependent attentive statistics pooling. Achieved 0.87% EER on VoxCeleb1 — the benchmark standard. Produces 192-dimensional embeddings. Used as the default extractor in pyannote and DiariZen diarization pipelines.

### 5.3 TitaNet (Koluguri et al., 2022)

Koluguri, Park, and Ginsburg (ICASSP 2022) introduced TitaNet using 1-D depth-wise separable convolutions with global context Squeeze-Excitation. Three sizes: S (6.3M, 1.10% EER), M (13.4M, 0.84%), L (25.3M, 0.68% EER on VoxCeleb1). Available in NeMo framework as the default diarization embedding extractor. Produces 192-dimensional embeddings.

### 5.4 WavLM (Chen et al., 2022)

Chen et al. (IEEE JSTSP 2022) presented WavLM, a self-supervised model trained on 94,000 hours with masked speech denoising. Achieved 12.6% relative DER reduction on CALLHOME when fine-tuned for speaker verification. Han et al. (ICASSP 2025) integrated WavLM into the DiariZen pipeline, outperforming ECAPA-TDNN baselines.

---

## 6. Clustering-Based Diarization

### 6.1 VBx: Bayesian HMM clustering (Landini et al., 2022)

Landini, Profant, Diez, and Burget (Computer Speech & Language, 2022) provided a comprehensive description of VBx (Variational Bayes HMM x-vector clustering). VBx models speaker assignments as latent HMM states with PLDA-scored emissions. Achieved 4.42% DER on CALLHOME with oracle VAD. VBx handles many speakers effectively but cannot process overlapping speech. For ATC audio's half-duplex (non-overlapping) structure, this limitation is irrelevant — making VBx potentially well-suited.

### 6.2 Pyannote pipeline (Bredin & Laurent, 2021; Bredin, 2023)

Bredin and Laurent (Interspeech 2021) developed pyannote's end-to-end segmentation model serving as unified VAD, overlap detector, and speaker change detector. Bredin (2023) described the three-stage pyannote.audio v2.1 pipeline: local segmentation → embedding extraction → agglomerative clustering. PyannoteAI represents the current commercial state of the art.

---

## 7. Evaluation Methodology

### 7.1 DER and its components

Diarization Error Rate decomposes into: missed speech (T_miss), false alarm (T_fa), and speaker confusion (T_conf), divided by total reference duration. A 0.25-second collar around reference boundaries excludes ambiguous regions. Our ATC0 result (DER=22.04%, FA=2.68%, MISS=2.83%, CER=16.53%) shows confusion dominates — the model detects speech well but assigns it to wrong speakers.

### 7.2 JER for balanced evaluation

Jaccard Error Rate computes intersection-over-union per speaker then averages, giving equal weight to all speakers regardless of speaking time. More appropriate for ATC where controller speaking time vastly exceeds any individual pilot.

### 7.3 Benchmarking across systems (Lanzendorfer et al., 2025)

Lanzendorfer, Grotschla, Blaser, and Wattenhofer (arXiv:2509.26177) benchmarked 5 state-of-the-art systems across 4 datasets (196.6 hours, 5 languages). PyannoteAI achieved best overall 11.2% DER while open-source DiariZen reached 13.3%. This provides the baseline against which to compare ATC-specific performance.

---

## 8. ATC-Specific Diarization

### 8.1 BERTraffic: text-based diarization (Zuluaga-Gomez et al., 2022)

Zuluaga-Gomez, Sarfjoo, Prasad, Nigmatulina, Motlicek, Ondrej, Ohneiser, and Helmke (IEEE SLT 2022) demonstrated that fine-tuned BERT for joint speaker change detection and role detection from ASR transcripts achieves 0.90/0.95 F1 on ATCO/pilot detection, with 32% relative improvement over acoustic VBx. This result demonstrates that for ATC, linguistic patterns (callsigns, ICAO phraseology structure) provide stronger speaker discrimination than acoustic features alone.

### 8.2 ATC-SD Net (2024)

Published in Aerospace (2024), ATC-SD Net is a dedicated radiotelephone communications speaker diarization network. While details are limited, it represents one of the few systems designed specifically for ATC audio characteristics.

### 8.3 Grammar-based role identification (Prasad et al., 2021)

Prasad, Zuluaga-Gomez, and Motlicek (2021) used grammar-based rules leveraging ATC phraseology structure for speaker role identification. The half-duplex (push-to-talk) nature of ATC means speakers rarely overlap, but the single-channel reception, short utterances, and need for role classification rather than just segmentation create unique challenges.

---

## 9. Attention Stability and Failure Modes

### 9.1 Attention entropy collapse (Zhai et al., 2023)

Zhai, Likhomanenko, Littwin, Busbridge, Ramapuram, Zhang, Gu, and Susskind (ICML 2023) formally proved that attention entropy decreases exponentially with the spectral norm of attention logits. When entropy collapses, attention concentrates on a few tokens, reducing the effective information diversity. They proposed σReparam — reparameterizing linear layers with spectral normalization and a learned scalar to bound minimum entropy. This work provides the theoretical foundation for explaining Sortformer's lock-up as attention entropy collapse triggered by out-of-distribution ATC audio features.

### 9.2 Auxiliary losses for EEND attention heads (Jeoung et al., 2023)

Jeoung, Honda, Kanda, and Watanabe (ICASSP 2023) demonstrated that assigning auxiliary losses to specific attention heads (e.g., forcing one head to predict VAD, another to detect overlap) reduces DER by 17–33% on CALLHOME for SA-EEND. This approach directly combats attention head redundancy — where multiple heads learn similar, collapsed patterns — by enforcing functional specialization.

---

## 10. VAD Systems Relevant to ATC

### 10.1 Silero VAD

Silero VAD provides a lightweight (~1–2 MB) neural model running in <1ms per 30ms chunk. It is a stateful model requiring sequential processing — feeding isolated chunks produces severely degraded probabilities (as discovered in our experiments). Supports 6,000+ languages. Probabilities on VHF radio audio are significantly lower (~0.2 for speech) than on wideband audio (~0.9+), likely due to the narrow bandwidth and AM modulation artifacts.

### 10.2 NeMo MarbleNet frame VAD

NVIDIA's frame_vad_multilingual_marblenet_v2.0 produces per-frame (20 ms) speech probabilities using a MarbleNet architecture. In our experiments, it produced near-1.0 probabilities throughout ATC audio, suggesting it may not discriminate well between speech and radio noise in the ATC domain.

---

## 11. Summary and Research Gaps

### 11.1 Current state

End-to-end diarization has reached streaming maturity with Sortformer and LS-EEND both achieving competitive DER at practical latencies. The Conformer/Fast Conformer backbone provides the acoustic foundation for these systems. Speaker embeddings from ECAPA-TDNN and TitaNet enable robust clustering-based alternatives.

### 11.2 Gaps relevant to our work

1. **No published evaluation of Sortformer or LS-EEND on ATC audio.** All benchmark results are on conversational datasets (meetings, phone calls, web video).

2. **The lock-up failure pattern is undocumented.** While attention entropy collapse is theoretically grounded, its manifestation as persistent speaker assignment errors in diarization has not been reported or analyzed.

3. **ATC diarization remains dominated by text-based approaches.** BERTraffic's success suggests that acoustic-only diarization may be fundamentally insufficient for ATC, but no hybrid acoustic-linguistic end-to-end system exists.

4. **No comparative study across architectures on ATC data.** Whether the lock-up is Sortformer-specific or a general neural diarization weakness on ATC audio is unknown.

5. **Fine-tuning neural diarization models on ATC data has not been attempted.** The ATC0 corpus provides speaker labels suitable for fine-tuning, but this has not been explored for Sortformer or LS-EEND.

---

## References

1. Bredin, H. (2023). pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe. *Interspeech 2023*.
2. Bredin, H. & Laurent, A. (2021). End-to-end speaker segmentation for overlap-aware resegmentation. *Interspeech 2021*.
3. Chen, S., et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing. *IEEE JSTSP*, 16(6).
4. Desplanques, B., Thienpondt, J., & Demuynck, K. (2020). ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification. *Interspeech 2020*.
5. Fujita, Y., Kanda, N., Horiguchi, S., Nagamatsu, K., & Watanabe, S. (2019). End-to-End Neural Speaker Diarization with Permutation-Free Objectives. *Interspeech 2019*.
6. Gulati, A., et al. (2020). Conformer: Convolution-augmented Transformer for Speech Recognition. *Interspeech 2020*. arXiv:2005.08100.
7. Han, Z., Lee, C., & Stolcke, A. (2021). BW-EDA-EEND: Streaming End-to-End Neural Speaker Diarization. *ICASSP 2021*.
8. Han, Z., Landini, F., Rohdin, J., et al. (2025). DiariZen: WavLM-based speaker diarization. *ICASSP 2025*.
9. Horiguchi, S., Fujita, Y., Watanabe, S., Xue, N., & Nagamatsu, K. (2020). End-to-End Speaker Diarization for an Unknown Number of Speakers with Encoder-Decoder Based Attractors. *Interspeech 2020*.
10. Jeoung, H., Honda, K., Kanda, N., & Watanabe, S. (2023). Improving End-to-End Neural Diarization with Auxiliary Losses on Self-Attention Maps. *ICASSP 2023*.
11. Kinoshita, K., Delcroix, M., & Tawara, N. (2021). EEND-vector clustering: combining end-to-end and clustering approaches. *Interspeech 2021*.
12. Koluguri, N. R., Park, T., & Ginsburg, B. (2022). TitaNet: Neural Model for Speaker Representation with 1D Depth-wise Separable Convolutions. *ICASSP 2022*.
13. Landini, F., Lozano-Diez, A., Diez, M., & Burget, L. (2022). Simulated Conversations for EEND Training. *Interspeech 2022*.
14. Landini, F., Profant, J., Diez, M., & Burget, L. (2022). Bayesian HMM clustering of x-vector sequences (VBx). *Computer Speech & Language*.
15. Lanzendorfer, L., Grotschla, F., Blaser, Y., & Wattenhofer, R. (2025). Benchmarking Diarization Models. *arXiv:2509.26177*.
16. Liang, D. & Li, X. (2024). LS-EEND: Long-Form Streaming End-to-End Neural Diarization with Online Attractor Extraction. *arXiv:2410.06670*. IEEE/ACM TASLP 2025.
17. Medennikov, N., Park, T., Wang, H., et al. (2025). Streaming Sortformer: Speaker Cache-Based Online Speaker Diarization with Arrival-Time Ordering. *Interspeech 2025*. arXiv:2507.18446.
18. Park, T., Medennikov, N., Dhawan, K., et al. (2024). Sortformer: Seamless Integration of Speaker Diarization and ASR. *arXiv:2409.06656*. ICML 2025.
19. Prasad, A., Zuluaga-Gomez, J., & Motlicek, P. (2021). Grammar-based speaker role identification for ATC. *2021*.
20. Rekesh, D., et al. (2023). Fast Conformer with Linearly Scalable Attention. *ICASSP 2023*. arXiv:2305.05084.
21. Snyder, D., Garcia-Romero, D., Sell, G., Povey, D., & Khudanpur, S. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition. *ICASSP 2018*.
22. Sun, Y., Dong, L., Huang, S., et al. (2023). Retentive Network: A Successor to Transformer for Large Language Models. *arXiv:2307.08621*.
23. Zhai, S., Likhomanenko, T., et al. (2023). Stabilizing Transformer Training by Preventing Attention Entropy Collapse. *ICML 2023*.
24. Zuluaga-Gomez, J., et al. (2022). BERTraffic: BERT-based joint speaker change detection and role identification for ATC. *IEEE SLT 2022*.

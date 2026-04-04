# CLAUDE.md — Project Context for Claude Code

## Project identity

This is `salai-spk-diar`, a speaker diarization research project for ATC (air traffic control) communications. It's part of Shital Pandey's PhD work at Embry-Riddle Aeronautical University (Daytona Beach), SaLAI Lab, under Dr. Jianhua Liu. The work serves three simultaneous goals: a CEC 599 independent study course (Speaker Diarization, Spring 2026), a DASC 2026 paper ("Enhancing Automatic ATC Information Extraction via Speaker Diarization"), and the broader Eclipse Aerospace collaboration on real-time ASR/NLP for pilot workload reduction.

## The core research problem

Sortformer-based speaker diarization exhibits a "lock-up" failure pattern on ATC radio audio: the model correctly identifies different speakers initially within a 10-second window, then collapses to assigning all subsequent speech to a single speaker. The confusion error rate (CER) dominates the DER. This project investigates why this happens and what to do about it.

## Diagnosis (from Phase 1 theory work)

The lock-up is most likely caused by **attention entropy collapse** triggered by domain mismatch between ATC VHF radio audio (narrow bandwidth 300-3400 Hz, AM modulation, squelch artifacts) and Sortformer's conversational training data. The 18-layer Transformer encoder uses softmax self-attention which can concentrate on a subset of frames when input representations are too similar (out-of-distribution). At inference, the 0.5 training dropout is disabled, allowing sharper attention patterns than the model ever saw during training.

## Environment

- Hardware: ASUS ROG NUC 970, RTX 4090 (24 GB), Ubuntu 24.04
- Python 3.12.2, CUDA 12.4, uv 0.8.16
- NeMo 2.7.2, PyTorch 2.6.0+cu124, pyannote.audio 3.4.0
- Project uses `uv` with `pyproject.toml` for dependency management
- Source code is in `src/spkdiar/` (installed as editable package)

## Dataset: ATC0R

Revised ATC0 dataset with corrected timestamps, casing, punctuation, quality levels L1-L4.

**STM format** (one line per cue):
```
<rec_id> <channel> <quality>><speaker>><listener> <start> <end> <transcript>
```
Example: `dca_d1_1 1 1>D1-1>DAL209 63.380 66.070 Delta Two Zero Nine, turn left heading two eight zero.`

- 16 recordings, ~22 hours total
- 21-90 speakers per recording
- STM files in `data/atc0r/stm/`, audio in `data/atc0r/audio/`
- Generated RTTM in `data/processed/rttm/`, manifests in `data/processed/manifests/`
- 16,912 windowed segments (10s window, 5s shift)

## Phase 2 experiment results so far

### Sortformer Offline (diar_sortformer_4spk-v1.nemo)
- **DER: 10.71%** (FA=0.81%, MISS=1.38%, CER=8.52%)
- Tested on dca_d1_1, dca_d1_3, log_id_1 (first 350s each, 213 windows)
- CER dominates — model detects speech well but misassigns speakers (lock-up pattern)
- Probability tensors saved in `results/sortformer_offline/prob_tensors/`

### Streaming Sortformer with AOSC (diar_streaming_sortformer_4spk-v2.nemo)
- 10s latency: DER=34.83% (FA=27.55%, MISS=0.46%, **CER=6.82%**)
- 1s latency: DER=69.49% (FA=61.24%, MISS=0.38%, CER=7.86%)
- **Key finding**: CER improves vs offline (6.82% vs 8.52%) — AOSC helps with speaker assignment
- FA explodes because streaming model's probability calibration doesn't match ATC's sparse speech
- Threshold 0.72 (instead of 0.5) reduces streaming DER from 31.1% to 21.4%

### Pyannote 3.1 (clustering-based)
- **DER: 32.49%** — massive under-clustering
- dca_d1_1: 32 speakers → detected 2
- dca_d1_3: 46 speakers → detected 1
- log_id_1: 88 speakers → detected 6
- VHF radio distortion makes speaker embeddings too similar for clustering to separate

### Summary
Sortformer offline is the best system on ATC audio by a wide margin. The lock-up pattern hurts it but pyannote's clustering fails much worse.

## Current task: FS-EEND / LS-EEND porting (Phase 2 continuation)

The FS-EEND codebase (`external/FS-EEND/`) targets Python 3.9, PyTorch 1.13, PyTorch Lightning 1.8. We are porting it to work with our current environment (PyTorch 2.6, Python 3.12). This is syllabus deliverable 2b ("adapting or modifying existing diarization frameworks").

The goal: extract the LS-EEND model architecture, load the pretrained CALLHOME checkpoint (`ch.ckpt` from Google Drive), run inference on our ATC0R data, and compare DER/CER with Sortformer. If LS-EEND does NOT exhibit the lock-up pattern, it confirms the cause is Sortformer's attention mechanism. If it DOES lock up, the cause is domain-related.

Key architectural differences from Sortformer:
- Uses Retention mechanism (not softmax self-attention) — immune to attention entropy collapse
- Has explicit attractor vectors (not implicit output slots)
- Cross-speaker self-attention in decoder pushes attractors apart
- Per-frame updates provide continuous self-correction
- Supports up to 8 speakers (vs Sortformer's 4)
- All audio must be 8 kHz

Pretrained checkpoints (Google Drive):
- Simulated 1-8 speaker: https://drive.google.com/file/d/1uWY8JvjHJJ-SvGiNS-6s3q10g4CY2ePt/view
- CALLHOME finetuned: https://drive.google.com/file/d/1W8nYAB6YoEKMM5KZX-apVADvHaYc2Fre/view
- DIHARD III: https://drive.google.com/file/d/115iaEG1OZwXa9tSyScXGtWeOk9JLfpER/view

## Remaining Phase 2 work after FS-EEND

- Waterfall plots comparing all systems visually on same windows
- Attention entropy extraction from Sortformer layers (diagnostic evidence)
- Speaker embedding analysis at 8 kHz vs 16 kHz (syllabus deliverable 2c)
- Engineering design decisions note (syllabus deliverable 1e)

## Phase 3 (after Phase 2)

Targeted experiments informed by theory: attention regularization (σReparam or auxiliary head losses), domain fine-tuning on ATC data, potential hybrid system combining Sortformer/LS-EEND with text-based speaker identification (BERTraffic approach).

## CEC 599 course deliverables status

- [x] 1a: Voice fingerprint and speaker embedding notes (docs/)
- [x] 1b: Sortformer architecture notes (docs/)
- [x] 1c: LS-EEND architecture notes (docs/)
- [x] 1d: Conformer architecture notes (docs/)
- [ ] 1e: Engineering design decisions (needs experimental results)
- [ ] 2a: Stitching/stabilization strategies
- [ ] 2b: FS-EEND framework adaptation ← CURRENT TASK
- [ ] 2c: Speaker embeddings at 8/16 kHz
- [ ] 2d: Fine-tuning diarization models

## Key dates

- April 17, 2026: PhD qualifying exam oral
- May 14, 2026: Pilot workload testing
- DASC 2026 paper deadline: TBD

## Important conventions

- Use `uv run` to execute all Python commands
- Source code in `src/spkdiar/`, installed as editable package
- ATC0R STM files are the single source of truth for ground truth (not HuggingFace)
- Quality level 4 cues are excluded from ground truth
- Speaker IDs in RTTM are prefixed with recording ID: `{rec_id}_{speaker}`
- Windowed manifest uniq_id format: `{rec_id}-{start_ms}-{duration_ms}`
- Frame resolution: 80ms (8× downsampling from 10ms mel frames)
- Results go in `results/`, probability tensors in `results/*/prob_tensors/`
- Large files (audio, models, .npy) are git-ignored
- Models are in `models/` directory (symlinked or downloaded)
- The pyannote runner needs `torch.load` monkey-patched for weights_only=False (PyTorch 2.6 issue)
- NeMo's `score_labels` returns a tuple in 2.7.2, use `metric[0]` not `abs(metric)`

## Shital's preferences

- Direct, dense responses without filler
- Theory before experiments — understand why before trying to fix
- Work should simultaneously serve course, paper, and research goals
- Dr. Liu expects solid understanding and results, not just running scripts
- No AI-sounding language in written deliverables
- HuggingFace handle: theVayu

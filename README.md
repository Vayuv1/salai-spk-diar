# salai-spk-diar

Speaker diarization research for ATC communications.

**SaLAI Lab** — Embry-Riddle Aeronautical University, Daytona Beach
PI: Dr. Jianhua Liu | PhD Student: Shital Pandey

## What this repo does

Investigates the "lock-up" failure pattern observed when applying Sortformer-based
speaker diarization to air traffic control (ATC) radio communications. Compares
multiple diarization architectures (Sortformer offline/streaming, LS-EEND, pyannote,
NeMo MSDD) on the ATC0R dataset to diagnose whether the failure is architecture-specific
or domain-general.

## Setup

```bash
# Clone
git clone git@github.com:<your-username>/salai-spk-diar.git
cd salai-spk-diar

# Create virtual environment and install all dependencies
uv sync

# Place audio files
# Copy or symlink your ATC0R audio files into data/atc0r/audio/
# Copy your ATC0R STM files into data/atc0r/stm/

# Prepare data (generates RTTM + NeMo manifests from STM files)
uv run python -m spkdiar.data.prep_all --stm-dir data/atc0r/stm --audio-dir data/atc0r/audio

# Run experiments
uv run python -m spkdiar.inference.run_sortformer --config configs/sortformer_offline.yaml
```

## Project structure

```
salai-spk-diar/
├── pyproject.toml          # Dependencies (managed by uv)
├── data/
│   ├── atc0r/
│   │   ├── audio/          # MP3/WAV files (git-ignored)
│   │   └── stm/            # ATC0R STM annotation files (tracked)
│   └── processed/          # Generated RTTM, manifests (git-ignored)
├── src/spkdiar/            # All source code
│   ├── data/               # Data prep: STM parsing, RTTM/manifest generation
│   ├── inference/          # Diarization runners for each system
│   ├── analysis/           # DER evaluation, plotting, attention analysis
│   └── utils/              # Shared helpers
├── configs/                # Experiment YAML configs
├── scripts/                # Top-level bash run scripts
├── results/                # Experiment outputs (git-ignored)
├── docs/                   # Course deliverables and notes
├── notebooks/              # Exploration notebooks
└── tests/                  # Unit tests
```

## Hardware

- ASUS ROG NUC 970, Ubuntu 24.04
- NVIDIA GeForce RTX 4090 (24 GB)
- CUDA 12.4 / Driver 555.42

## Related work

- DASC 2026 paper: "Enhancing Automatic ATC Information Extraction via Speaker Diarization"
- CEC 599: Speaker Diarization (Spring 2026, Dr. Jianhua Liu)
- Eclipse Aerospace Inc. collaboration

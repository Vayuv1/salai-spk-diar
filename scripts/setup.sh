#!/usr/bin/env bash
# ==============================================================================
# setup.sh — Bootstrap the salai-spk-diar project on the NUC
#
# Run this ONCE after cloning the repo:
#     chmod +x scripts/setup.sh && ./scripts/setup.sh
# ==============================================================================
set -euo pipefail

echo "============================================"
echo "  salai-spk-diar — Project Setup"
echo "============================================"
echo ""

# --- 1. Verify prerequisites ---
echo "[1/5] Checking prerequisites..."

command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
command -v git >/dev/null 2>&1 || { echo "ERROR: git not found."; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "WARNING: nvidia-smi not found. CUDA may not work."; }

echo "  uv:    $(uv --version)"
echo "  git:   $(git --version)"
echo "  python target: 3.12"
echo ""

# --- 2. Create virtual environment and install dependencies ---
echo "[2/5] Creating virtual environment and installing dependencies..."
echo "  This will take a few minutes (PyTorch + NeMo are large)..."
echo ""

uv sync

echo ""
echo "  Environment created at .venv/"
echo ""

# --- 3. Verify critical imports ---
echo "[3/5] Verifying critical imports..."

uv run python -c "
import torch
print(f'  torch:      {torch.__version__}')
print(f'  CUDA avail: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:        {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:       {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

uv run python -c "
try:
    import nemo
    print(f'  nemo:       {nemo.__version__}')
except ImportError:
    print('  nemo:       NOT INSTALLED (will retry)')
"

uv run python -c "
try:
    import pyannote.audio
    print(f'  pyannote:   {pyannote.audio.__version__}')
except ImportError:
    print('  pyannote:   NOT INSTALLED (may need manual install)')
"

echo ""

# --- 4. Create data directory structure ---
echo "[4/5] Setting up data directories..."

mkdir -p data/atc0r/audio
mkdir -p data/atc0r/stm
mkdir -p data/processed/rttm
mkdir -p data/processed/manifests
mkdir -p results

echo "  data/atc0r/audio/    — Place your MP3/WAV files here"
echo "  data/atc0r/stm/      — Place your ATC0R .stm files here"
echo "  data/processed/      — Generated RTTM and manifests go here"
echo "  results/             — Experiment outputs go here"
echo ""

# --- 5. Summary ---
echo "[5/5] Setup complete!"
echo ""
echo "============================================"
echo "  NEXT STEPS"
echo "============================================"
echo ""
echo "  1. Copy your ATC0R .stm files into:  data/atc0r/stm/"
echo "  2. Copy your audio files into:        data/atc0r/audio/"
echo "  3. Run data preparation:"
echo ""
echo "     uv run python -m spkdiar.data.prep_all \\"
echo "         --stm-dir data/atc0r/stm \\"
echo "         --audio-dir data/atc0r/audio"
echo ""
echo "  4. Run Sortformer inference (once prep is done):"
echo ""
echo "     uv run python -m spkdiar.inference.run_sortformer \\"
echo "         --manifest data/processed/manifests/windowed_10s_5s.jsonl \\"
echo "         --model-path <path-to-diar_sortformer_4spk-v1.nemo>"
echo ""
echo "============================================"

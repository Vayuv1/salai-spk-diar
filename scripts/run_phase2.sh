#!/usr/bin/env bash
# ==============================================================================
# run_phase2.sh — Run all Phase 2 comparative experiments
#
# Usage:
#     chmod +x scripts/run_phase2.sh
#     ./scripts/run_phase2.sh          # Run all experiments
#     ./scripts/run_phase2.sh quick    # Quick test (3 recordings, 350s each)
# ==============================================================================
set -euo pipefail

MODE="${1:-quick}"  # "quick" or "full"

WINDOWED_MANIFEST="data/processed/manifests/windowed_10s_5s.jsonl"
FULL_MANIFEST="data/processed/manifests/full_manifest.jsonl"
OFFLINE_MODEL="models/diar_sortformer_4spk-v1.nemo"
STREAMING_MODEL="models/diar_streaming_sortformer_4spk-v2.nemo"

# Quick mode: 3 diverse recordings, first 350s
QUICK_RECS="dca_d1_1,dca_d1_3,log_id_1"
QUICK_MAX_OFFSET="350"

echo "============================================"
echo "  Phase 2: Comparative Diarization"
echo "  Mode: ${MODE}"
echo "============================================"
echo ""

# --- Experiment 1: Sortformer Offline ---
echo ">>> [1/4] Sortformer Offline"
if [ "$MODE" = "quick" ]; then
    uv run python -m spkdiar.inference.run_sortformer \
        --manifest "$WINDOWED_MANIFEST" \
        --model-path "$OFFLINE_MODEL" \
        --out-dir results/sortformer_offline \
        --rec-ids "$QUICK_RECS" \
        --max-offset "$QUICK_MAX_OFFSET"
else
    uv run python -m spkdiar.inference.run_sortformer \
        --manifest "$WINDOWED_MANIFEST" \
        --model-path "$OFFLINE_MODEL" \
        --out-dir results/sortformer_offline
fi
echo ""

# --- Experiment 2: Streaming Sortformer (10s latency) ---
echo ">>> [2/4] Streaming Sortformer (10s latency)"
if [ -f "$STREAMING_MODEL" ]; then
    if [ "$MODE" = "quick" ]; then
        uv run python -m spkdiar.inference.run_streaming \
            --manifest "$WINDOWED_MANIFEST" \
            --model-path "$STREAMING_MODEL" \
            --out-dir results/sortformer_streaming_10s \
            --latency medium \
            --rec-ids "$QUICK_RECS" \
            --max-offset "$QUICK_MAX_OFFSET"
    else
        uv run python -m spkdiar.inference.run_streaming \
            --manifest "$WINDOWED_MANIFEST" \
            --model-path "$STREAMING_MODEL" \
            --out-dir results/sortformer_streaming_10s \
            --latency medium
    fi
else
    echo "  SKIPPED — streaming model not found at $STREAMING_MODEL"
    echo "  Download with: uv run python -c \""
    echo "    from nemo.collections.asr.models import SortformerEncLabelModel"
    echo "    m = SortformerEncLabelModel.from_pretrained('nvidia/diar_streaming_sortformer_4spk-v2')"
    echo "    m.save_to('$STREAMING_MODEL')\""
fi
echo ""

# --- Experiment 3: Streaming Sortformer (1s latency) ---
echo ">>> [3/4] Streaming Sortformer (1s latency)"
if [ -f "$STREAMING_MODEL" ]; then
    if [ "$MODE" = "quick" ]; then
        uv run python -m spkdiar.inference.run_streaming \
            --manifest "$WINDOWED_MANIFEST" \
            --model-path "$STREAMING_MODEL" \
            --out-dir results/sortformer_streaming_1s \
            --latency low \
            --rec-ids "$QUICK_RECS" \
            --max-offset "$QUICK_MAX_OFFSET"
    else
        uv run python -m spkdiar.inference.run_streaming \
            --manifest "$WINDOWED_MANIFEST" \
            --model-path "$STREAMING_MODEL" \
            --out-dir results/sortformer_streaming_1s \
            --latency low
    fi
else
    echo "  SKIPPED — streaming model not found"
fi
echo ""

# --- Experiment 4: Pyannote 3.1 ---
echo ">>> [4/4] Pyannote 3.1"
if [ -n "${HF_TOKEN:-}" ]; then
    if [ "$MODE" = "quick" ]; then
        uv run python -m spkdiar.inference.run_pyannote \
            --manifest "$FULL_MANIFEST" \
            --out-dir results/pyannote \
            --rec-ids "$QUICK_RECS" \
            --max-duration "$QUICK_MAX_OFFSET"
    else
        uv run python -m spkdiar.inference.run_pyannote \
            --manifest "$FULL_MANIFEST" \
            --out-dir results/pyannote
    fi
else
    echo "  SKIPPED — HF_TOKEN not set"
    echo "  Run: export HF_TOKEN=hf_xxxxx"
fi
echo ""

echo "============================================"
echo "  All experiments complete"
echo "  Results in: results/"
echo "============================================"

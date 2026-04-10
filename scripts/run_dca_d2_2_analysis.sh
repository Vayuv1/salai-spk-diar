#!/usr/bin/env bash
set -euo pipefail

REC="dca_d2_2"
MIN_OFF=2850
MAX_OFF=2970
T_START=2880
T_END=2960

#echo "=== Step 1a: Sortformer Offline on ${REC} ==="
#uv run python -m spkdiar.inference.run_sortformer \
#    --manifest data/processed/manifests/windowed_10s_5s.jsonl \
#    --model-path models/diar_sortformer_4spk-v1.nemo \
#    --out-dir results/sortformer_offline \
#    --rec-ids "$REC" \
#    --max-offset "$MAX_OFF"
#
#echo "=== Step 1b: Streaming Sortformer on ${REC} ==="
#uv run python -m spkdiar.inference.run_streaming \
#    --manifest data/processed/manifests/windowed_10s_5s.jsonl \
#    --model-path models/diar_streaming_sortformer_4spk-v2.nemo \
#    --out-dir results/sortformer_streaming_10s \
#    --latency medium \
#    --rec-ids "$REC" \
#    --max-offset "$MAX_OFF"
#
#echo "=== Step 1c: LS-EEND on ${REC} ==="
#uv run python -m spkdiar.inference.run_lseend \
#    --model-path models/lseend_callhome.ckpt \
#    --audio-dir data/atc0r/audio \
#    --rttm-dir data/processed/rttm \
#    --out-dir results/lseend \
#    --rec-ids "$REC" \
#    --max-duration 3000

echo "=== Step 1d: Pyannote on ${REC} ==="
uv run python -c "
import torch
_orig = torch.load
def _patch(*a, **kw):
    kw['weights_only'] = False
    return _orig(*a, **kw)
torch.load = _patch

import sys
sys.argv = ['', '--manifest', 'data/processed/manifests/full_manifest.jsonl',
            '--out-dir', 'results/pyannote',
            '--rec-ids', '${REC}',
            '--max-duration', '3000']
from spkdiar.inference.run_pyannote import main
main()
"

echo "=== Step 2: Waterfall plot ==="
uv run python -m spkdiar.analysis.plot_waterfall \
    --rec-id "$REC" \
    --offline-dir results/sortformer_offline/prob_tensors \
    --streaming-dir results/sortformer_streaming_10s/prob_tensors \
    --rttm-dir data/processed/rttm \
    --out-dir results/plots \
    --offset-min "$T_START" \
    --windows-per-grid 6 \
    --num-grids 2

echo "=== Step 3: Timeline plot ==="
uv run python -m spkdiar.analysis.plot_timeline \
    --rec-id "$REC" \
    --t-start "$T_START" --t-end "$T_END" \
    --out-dir results/plots

echo "=== Done ==="
echo "Check results/plots/ for output figures"
# Codex Reproducibility Audit Log

Started: 2026-04-19T16:14:56-04:00
Repository: `/home/pandeys2/Workspace/salai-spk-diar`
Objective: Revalidate the DASC 2026 ATC diarization paper from code and rerun evidence, not from existing paper text or stale artifacts.

## Ground Rules

- Existing paper text, plots, tables, logs, and result files are treated as candidate artifacts, not ground truth.
- Original artifacts are preserved. New regenerated outputs will use distinct filenames/directories where practical.
- Every material claim in the revised paper must map to literature, rerun outputs, or narrowly stated interpretation.

## 2026-04-19 16:14:56 - Audit start

- Confirmed working directory: `/home/pandeys2/Workspace/salai-spk-diar`
- Initial high-level directories observed:
  - `configs`
  - `data`
  - `docs`
  - `external`
  - `models`
  - `results`
  - `scripts`
  - `src`
  - `tests`

## 2026-04-19 16:15-16:16 - Initial file inventory

Paper-related files:

- `docs/main.tex`
- `docs/references.bib`
- `docs/Comparative_Evaluation_and_Domain_Adaptation_of_Neural_Speaker_Diarization_for_Air_Traffic_Control_Communications.pdf`
- `docs/phase2_experiment_log.md`

Top-level run scripts:

- `scripts/run_phase2.sh`
- `scripts/run_dca_d2_2_analysis.sh`
- `scripts/setup.sh`

Source modules of interest:

- Data prep:
  - `src/spkdiar/data/make_manifest.py`
  - `src/spkdiar/data/make_rttm.py`
  - `src/spkdiar/data/make_finetune_manifest.py`
  - `src/spkdiar/data/prep_all.py`
- Inference:
  - `src/spkdiar/inference/run_sortformer.py`
  - `src/spkdiar/inference/run_streaming.py`
  - `src/spkdiar/inference/run_pyannote.py`
  - `src/spkdiar/inference/run_lseend.py`
- Training:
  - `src/spkdiar/training/finetune_sortformer.py`
- Analysis / plotting:
  - `src/spkdiar/analysis/gen_fig1_der_comparison.py`
  - `src/spkdiar/analysis/gen_fig2_paired_waterfall_v2.py`
  - `src/spkdiar/analysis/gen_fig4_attention_entropy_v2.py`
  - `src/spkdiar/analysis/role_cer_analysis.py`
  - `src/spkdiar/analysis/plot_timeline.py`
  - `src/spkdiar/analysis/plot_waterfall.py`
  - `src/spkdiar/analysis/speaker_embeddings.py`
  - `src/spkdiar/analysis/gen_fig3_embeddings.py`

## 2026-04-19 16:16 - Git state captured

`git status --short` at audit start:

```text
 m external/FS-EEND
 M src/spkdiar/analysis/gen_fig1_der_comparison.py
 M src/spkdiar/analysis/gen_fig3_embeddings.py
?? docs/Comparative_Evaluation_and_Domain_Adaptation_of_Neural_Speaker_Diarization_for_Air_Traffic_Control_Communications.pdf
?? docs/main.tex
?? docs/references.bib
?? src/spkdiar/analysis/gen_fig2_lockup_comparison_v4.py
?? src/spkdiar/analysis/gen_fig2_lockup_comparison_v5.py
?? src/spkdiar/analysis/gen_fig2_paired_waterfall_v2.py
?? src/spkdiar/analysis/gen_fig2_rttm_comparison.py
?? src/spkdiar/analysis/gen_fig4b_entropy_dca_d2_2.py
```

Interpretation:

- The worktree is already dirty before Codex edits.
- Existing modified and untracked files will not be overwritten blindly.
- The paper source itself is currently untracked, so revisions must preserve traceability carefully.

## 2026-04-19 16:16-16:18 - Dataset split traced from code

Authoritative split source identified:

- `src/spkdiar/data/make_finetune_manifest.py`

Hard-coded held-out evaluation recordings in code:

- `dca_d1_1`
- `dca_d2_2`
- `dfw_a1_1`
- `log_id_1`

Training recordings implied by generated manifest:

- `dca_d1_2`
- `dca_d1_3`
- `dca_d1_4`
- `dca_d2_1`
- `dca_f1_1`
- `dca_f1_2`
- `dca_f2_1`
- `dca_f2_2`
- `dca_f2_3`
- `dfw_d1_1`
- `log_id_2`
- `log_id_3`

Manifest counts observed:

- `data/processed/manifests/finetune_train.jsonl`: 1453 windows
- `data/processed/manifests/finetune_eval.jsonl`: 456 windows
- `data/processed/manifests/windowed_10s_5s.jsonl`: 16912 windows
- `data/processed/manifests/full_manifest.jsonl`: 16 recordings

Per-recording fine-tuning validation-window counts:

- `dca_d1_1`: 119
- `dca_d2_2`: 92
- `dfw_a1_1`: 86
- `log_id_1`: 159

Correction added later in the audit:

- These four counts sum to 456 and correspond to the 90 s fine-tuning validation manifest, not to the held-out 10 s evaluation subset.
- The held-out 10 s evaluation subset was later confirmed directly from the rerun outputs to contain 3989 scored windows in total:
  - `dca_d1_1`: 1056
  - `dca_d2_2`: 717
  - `dfw_a1_1`: 781
  - `log_id_1`: 1435

## 2026-04-19 16:18-16:21 - Model and checkpoint inventory

Observed model files in `models/`:

- `models/diar_streaming_sortformer_4spk-v2.nemo`
- `models/lseend_callhome.ckpt`
- Fine-tuned Sortformer exports already present:
  - `models/diar_sortformer_4spk-v1-atc.nemo`
  - `models/diar_sortformer_4spk-v1-atc-750steps.nemo`
  - `models/diar_sortformer_4spk-v1-atc-1500steps.nemo`
  - `models/diar_sortformer_4spk-v1-atc-2ksteps.nemo`
  - `models/diar_sortformer_4spk-v1-atc-3ksteps.nemo`
  - `models/diar_sortformer_4spk-v1-atc-5ksteps.nemo`
  - `models/diar_sortformer_4spk-v1-atc-10ksteps.nemo`

Observed fine-tuning checkpoint history in `results/finetune/checkpoints/`:

- Interval checkpoints exist through at least `step=10000`
- The training script default is `--max-steps 1000`
- This discrepancy must be documented and resolved in the rerun narrative

Potential issue detected:

- The baseline pretrained offline `.nemo` referenced by the training and inference scripts is not yet confirmed present in `models/` from the initial `find` output. This needs direct verification before reruns.

## 2026-04-19 16:21-16:24 - Paper claims inspected

Current paper source: `docs/main.tex`

Key observations from the draft:

- The abstract and results sections contain precise numerical claims that may be stale.
- The paper currently claims:
  - A “comparative evaluation” across Sortformer offline/streaming, pyannote 3.1, and LS-EEND
  - A revised ATC dataset of 16 recordings
  - Fine-tuning on 12 recordings and evaluation on 4 held-out recordings
  - CER reductions and cross-facility generalization
  - Streaming AOSC CER reduction
  - Embedding-based explanation for pyannote collapse
  - Attention entropy interpretation that is already somewhat cautious but may still be stronger than evidence supports

Figures referenced by `docs/main.tex`:

- `fig2_paired_waterfall.pdf`
- `fig4_attention_entropy_v2.pdf`
- `fig3_embedding_similarity.pdf`
- `fig1_der_comparison.pdf`

Tables referenced by `docs/main.tex`:

- `tab:main_results`
- `tab:finetune_results`

## 2026-04-19 16:24-16:29 - Authoritative script candidates identified

Likely authoritative experiment path for the paper:

- Train/eval split generation:
  - `src/spkdiar/data/make_finetune_manifest.py`
- Pretrained offline Sortformer inference:
  - `src/spkdiar/inference/run_sortformer.py`
- Streaming Sortformer inference:
  - `src/spkdiar/inference/run_streaming.py`
- Pyannote baseline:
  - `src/spkdiar/inference/run_pyannote.py`
- LS-EEND baseline:
  - `src/spkdiar/inference/run_lseend.py`
- Fine-tuning:
  - `src/spkdiar/training/finetune_sortformer.py`
- DER comparison figure:
  - `src/spkdiar/analysis/gen_fig1_der_comparison.py`
- Lock-up visual / paired waterfall:
  - `src/spkdiar/analysis/gen_fig2_paired_waterfall_v2.py`
- Attention entropy figure:
  - `src/spkdiar/analysis/gen_fig4_attention_entropy_v2.py`
- Per-role CER analysis:
  - `src/spkdiar/analysis/role_cer_analysis.py`

Important caveat:

- `gen_fig1_der_comparison.py` currently hard-codes numbers directly in the script. This is not sufficient evidence by itself. The figure must be regenerated from rerun metrics or from a machine-generated result file derived from reruns.

## 2026-04-19 16:29-16:31 - Environment verification

Commands run:

- `date -Iseconds`
- `python --version`
- `./.venv/bin/python --version`
- `.venv` import checks for `nemo`, `pyannote.audio`, `pyannote.metrics`, `lightning`, `librosa`
- `.venv` Torch/CUDA check

Observed environment:

- Python 3.12.2 in both system and project venv
- Torch `2.6.0+cu124`
- Major packages import successfully inside `.venv`

Critical runtime issue:

- `torch.cuda.is_available()` returned `False` inside the sandboxed process
- Warnings included:
  - `CUDA initialization: Unexpected error from cudaGetDeviceCount()`
  - `Can't initialize NVML`

Interpretation:

- GPU access is likely blocked by the execution sandbox rather than by missing packages
- GPU-backed NeMo inference / fine-tuning may require escalated execution

## Open items before reruns

- Verify whether `models/diar_sortformer_4spk-v1.nemo` exists or whether the pretrained path in code is stale.
- Inspect `scripts/run_phase2.sh` and any other orchestration script for the exact commands used historically.
- Determine whether pyannote reruns are possible without external network/auth at audit time.
- Build a machine-readable result path for regenerated metrics so figures and paper text can cite rerun outputs directly.

## 2026-04-19 16:32-16:40 - Additional audit findings

### Pretrained model path

- `models/diar_sortformer_4spk-v1.nemo` does exist, but as a symlink:
  - `/home/pandeys2/Workspace/salai/speaker_diarization/runtime/pretrained_models/diar_sortformer_4spk-v1.nemo`
- This means pretrained reruns depend on an external file outside the workspace root.
- Because unrestricted execution is available, the symlink should still be usable for reruns.

### Historical orchestration scripts

- `scripts/run_phase2.sh` is the top-level comparative run script for quick/full modes.
- It calls:
  - `spkdiar.inference.run_sortformer`
  - `spkdiar.inference.run_streaming`
  - `spkdiar.inference.run_pyannote`
- It does not include fine-tuning or a full-recording metric table export path.
- `scripts/run_dca_d2_2_analysis.sh` is a focused analysis helper for the `dca_d2_2` example region and figure generation; it is not the authoritative comparative evaluation pipeline.

### GPU accessibility

Sandboxed Python could not see CUDA, but unrestricted execution can:

- `nvidia-smi` reports: `NVIDIA GeForce RTX 4090, 24564 MiB`
- Under unrestricted execution, `.venv` Torch reports:
  - `torch 2.6.0+cu124`
  - `cuda_available True`
  - `device_count 1`
  - `device0 NVIDIA GeForce RTX 4090`
  - `bf16 True`

Conclusion:

- Reruns requiring GPU should be launched with unrestricted execution.

### Fine-tuning run history in repo

`results/finetune/logs/finetune/` contains multiple training runs:

- `version_1`: 1000 steps
- `version_2`: 10000 steps
- `version_3`: 5000 steps
- `version_4`: 3000 steps
- `version_5`: 2000 steps
- `version_6`: 1500 steps
- `version_7`: 750 steps

Important discrepancy:

- The repo contains many later fine-tuning runs beyond 1000 steps.
- The paper currently centers 1000-step fine-tuning.
- The rerun requested by the user will therefore explicitly regenerate the 1000-step condition, even though later checkpoints exist.

### Pyannote rerun risk

- `HF_TOKEN` is not set in the current environment.
- No obvious `pyannote` model cache was found under `~/.cache/huggingface`.
- Existing file `results/pyannote/summary.json` only reports one-recording output:
  - `overall_der = 0.5475925739094111`
  - `n_recordings = 1`

Implication:

- Pyannote rerun may be blocked by missing authentication and/or model cache.
- This will need an explicit runtime check during execution.

## 2026-04-19 16:40-16:49 - Evaluation methodology audit

Problem discovered:

- No reusable script currently exports the paper’s full-recording DER/FA/MISS/CER table for Sortformer.
- `src/spkdiar/analysis/gen_fig1_der_comparison.py` hard-codes the paper numbers.
- The paper numbers therefore are not yet traceable to a machine-generated table artifact.

Tested hypotheses against existing RTTM artifacts:

1. **Global full-file stitched evaluation** of center-cropped per-window RTTMs
   - Did **not** match the paper numbers.
   - Produced much higher DER/CER because speaker slot identities are independent across windows.

2. **Window-wise evaluation aggregated across all 10 s windows** for each recording
   - **Did** reproduce the pretrained paper table approximately:
     - `dca_d1_1`: DER 0.23599, FA 0.03957, MISS 0.02078, CER 0.17564
     - `dca_d2_2`: DER 0.23142, FA 0.04201, MISS 0.01065, CER 0.17876
     - `dfw_a1_1`: DER 0.47225, FA 0.29389, MISS 0.01236, CER 0.16600
     - `log_id_1`: DER 0.44959, FA 0.31868, MISS 0.02785, CER 0.10307
   - These align closely with the rounded percentages in the current paper and `docs/phase2_experiment_log.md`.

Interpretation:

- The paper’s “full recordings” wording is misleading if the actual metric is the weighted aggregation of independent per-window evaluations across the full recording span.
- The revised paper should describe this evaluation protocol precisely.

### Critical discrepancy already confirmed

Using the same window-wise aggregation on existing `results/sortformer_finetuned/pred_rttm`:

- `dca_d1_1`: DER 0.14466, FA 0.02409, MISS 0.02538, CER 0.09519
- `dca_d2_2`: DER 0.12985, FA 0.02521, MISS 0.02312, CER 0.08152
- `dfw_a1_1`: DER 0.21812, FA 0.12710, MISS 0.02567, CER 0.06534
- `log_id_1`: DER 0.12870, FA 0.04420, MISS 0.02937, CER 0.05513

This does **not** match the current paper on `dca_d2_2`:

- Paper claims: DER 10.92%, FA 1.56%, MISS 1.61%, CER 3.75%
- Existing artifact evaluation gives approximately: DER 12.99%, FA 2.52%, MISS 2.31%, CER 8.15%

Conclusion:

- At least one fine-tuned artifact set currently in `results/sortformer_finetuned` is inconsistent with the paper’s reported 1000-step result.
- This is consistent with the user’s warning about a known inconsistency in the DER comparison section.
- Fresh reruns and a machine-generated metrics table are required before paper revision.

## 2026-04-19 18:07 - Post-crash reconstruction of completed reruns and paper revision

The IDE crashed after the heavy reruns had already completed. The entries below reconstruct the completed work from surviving command history, generated artifacts, and output files. Only artifacts that remain on disk are treated as evidence.

### Pretrained offline Sortformer rerun completed

Command used:

```text
uv run python -m spkdiar.inference.run_sortformer \
  --manifest data/processed/manifests/windowed_10s_5s.jsonl \
  --model-path models/diar_sortformer_4spk-v1.nemo \
  --out-dir results/repro/sortformer_pretrained_eval4_rerun \
  --rec-ids dca_d1_1,dca_d2_2,dfw_a1_1,log_id_1
```

Authoritative output:

- `results/repro/sortformer_pretrained_eval4_rerun/eval_metrics.json`

Per-recording metrics:

- `dca_d1_1`: DER 20.8720, FA 2.6582, MISS 1.4164, CER 16.7974
- `dca_d2_2`: DER 20.2195, FA 2.4094, MISS 0.6669, CER 17.1432
- `dfw_a1_1`: DER 47.3849, FA 30.9608, MISS 0.4995, CER 15.9246
- `log_id_1`: DER 44.7313, FA 33.1415, MISS 1.5326, CER 10.0572
- Overall: DER 36.6048, FA 21.1529, MISS 1.0415, CER 14.4104

Interpretation:

- These rerun values confirm that the repository’s held-out Sortformer baseline is recoverable.
- They also confirm that the paper’s metric source is the exported window-level NeMo evaluation path, not a stitched full-file RTTM evaluation.

### 1000-step fine-tuning rerun completed

Command used:

```text
uv run python -m spkdiar.training.finetune_sortformer \
  --model-path models/diar_sortformer_4spk-v1.nemo \
  --train-manifest data/processed/manifests/finetune_train.jsonl \
  --eval-manifest data/processed/manifests/finetune_eval.jsonl \
  --out-dir results/repro/finetune_1k_rerun \
  --max-steps 1000 \
  --lr 1e-5 \
  --batch-size 4 \
  --warmup-steps 100 \
  --val-interval 200 \
  --ckpt-interval 200 \
  --seed 42 \
  --export-model models/diar_sortformer_4spk-v1-atc-1ksteps-rerun.nemo
```

Key outputs:

- Exported model: `models/diar_sortformer_4spk-v1-atc-1ksteps-rerun.nemo`
- Checkpoints: `results/repro/finetune_1k_rerun/checkpoints/`
- Training log: `results/repro/finetune_1k_rerun/logs/finetune/version_0/metrics.csv`

Final validation row from `metrics.csv`:

- `val_loss = 0.07861328125`
- `val_ats_loss = 0.10498046875`
- `val_pil_loss = 0.052490234375`
- `val_f1_acc = 0.8676561117`
- `val_f1_acc_ats = 0.7942429781`
- `val_precision = 0.8646892905`
- `val_recall = 0.8731861115`

Important note:

- The codebase contains many later fine-tuning runs, but this rerun intentionally regenerated the requested 1000-step condition with an explicit seed and deterministic trainer settings.

### Fine-tuned offline Sortformer rerun completed

Command used:

```text
uv run python -m spkdiar.inference.run_sortformer \
  --manifest data/processed/manifests/windowed_10s_5s.jsonl \
  --model-path models/diar_sortformer_4spk-v1-atc-1ksteps-rerun.nemo \
  --out-dir results/repro/sortformer_finetuned_eval4_rerun \
  --rec-ids dca_d1_1,dca_d2_2,dfw_a1_1,log_id_1
```

Authoritative output:

- `results/repro/sortformer_finetuned_eval4_rerun/eval_metrics.json`

Per-recording metrics:

- `dca_d1_1`: DER 12.6187, FA 1.5982, MISS 2.0033, CER 9.0172
- `dca_d2_2`: DER 10.7288, FA 1.5582, MISS 1.6131, CER 7.5574
- `dfw_a1_1`: DER 22.0556, FA 13.8389, MISS 1.9581, CER 6.2586
- `log_id_1`: DER 11.7359, FA 4.4719, MISS 1.8187, CER 5.4454
- Overall: DER 14.8961, FA 6.2744, MISS 1.8614, CER 6.7604

Relative reductions vs pretrained offline:

- CER reduction by recording: 46.3%, 55.9%, 60.7%, 45.9%
- DER reduction by recording: 39.5%, 46.9%, 53.5%, 73.8%
- Overall CER reduction: 53.1%
- Overall DER reduction: 59.3%

Critical discrepancy confirmed:

- The fresh rerun does **not** support the old paper claim that `dca_d2_2` fine-tuned CER was 3.75%.
- The rerun-backed value is 7.5574%, with DER 10.7288%.

### Streaming Sortformer rerun completed

Command used:

```text
uv run python -m spkdiar.inference.run_streaming \
  --manifest data/processed/manifests/windowed_10s_5s.jsonl \
  --model-path models/diar_streaming_sortformer_4spk-v2.nemo \
  --out-dir results/repro/sortformer_streaming10_eval4_rerun \
  --latency medium \
  --rec-ids dca_d1_1,dca_d2_2,dfw_a1_1,log_id_1
```

Authoritative output:

- `results/repro/sortformer_streaming10_eval4_rerun/eval_metrics.json`

Per-recording metrics:

- `dca_d1_1`: DER 43.3353, FA 29.0442, MISS 0.5518, CER 13.7394
- `dca_d2_2`: DER 43.1277, FA 32.5407, MISS 0.3196, CER 10.2674
- `dfw_a1_1`: DER 12.4377, FA 7.8780, MISS 0.4017, CER 4.1580
- `log_id_1`: DER 15.4757, FA 8.8545, MISS 0.5672, CER 6.0540
- Overall: DER 24.8135, FA 16.6360, MISS 0.4701, CER 7.7074

Interpretation:

- Streaming reduces CER on all four held-out recordings relative to pretrained offline.
- The gain is uneven: the DCA pair suffers a large FA increase, while the DFW and LOG recordings improve strongly in both DER and CER.
- Overall CER reduction vs pretrained offline is 46.5%; overall DER reduction is 32.2%.

### Role-conditioned CER analysis rerun completed

Command used:

```text
./.venv/bin/python -m spkdiar.analysis.role_cer_analysis
```

Authoritative output:

- `results/paper_figures/role_cer_table.json`

Relative CER reductions:

- Controller CER reduction: 44.8%, 55.1%, 66.2%, 43.4%
- Pilot CER reduction: 38.6%, 43.3%, 49.3%, 39.6%

Interpretation:

- Both roles improve after fine-tuning.
- The controller gains are consistently larger in this held-out set.

### Figure regeneration completed

Commands used:

```text
./.venv/bin/python -m spkdiar.analysis.gen_fig1_der_comparison
./.venv/bin/python -m spkdiar.analysis.gen_fig2_paired_waterfall_v2 \
  --pretrained-prob-dir results/repro/sortformer_pretrained_eval4_rerun/prob_tensors \
  --finetuned-prob-dir results/repro/sortformer_finetuned_eval4_rerun/prob_tensors \
  --pretrained-rttm-dir results/repro/sortformer_pretrained_eval4_rerun/pred_rttm \
  --finetuned-rttm-dir results/repro/sortformer_finetuned_eval4_rerun/pred_rttm
./.venv/bin/python -m spkdiar.analysis.gen_fig3_embeddings
```

Outputs:

- `results/paper_figures/fig1_der_comparison.pdf`
- `results/paper_figures/fig2_paired_waterfall.pdf`
- `results/paper_figures/fig3_embedding_similarity.pdf`

Additional evidence files already present for embedding claims:

- `results/speaker_embeddings/dca_d1_1_full/similarity_stats.json`
- `results/speaker_embeddings/log_id_1_full/similarity_stats.json`

Embedding margins confirmed:

- `dca_d1_1` 16 kHz margin: 0.3182
- `dca_d1_1` 8 kHz margin: 0.2756
- `log_id_1` 16 kHz margin: 0.4293
- `log_id_1` 8 kHz margin: 0.4130

Interpretation:

- Speaker-discriminative information survives the ATC channel, although it is degraded.
- The paper should phrase this cautiously and avoid overextending the result to any specific clustering pipeline.

### Manifest-capacity caveat quantified

Fine-tuning manifest statistics gathered during this session:

- `finetune_train.jsonl`: 1453 windows total; 197 windows (13.56%) contain at least 5 labeled speakers
- `finetune_eval.jsonl`: 456 windows total; 86 windows (18.86%) contain at least 5 labeled speakers
- Held-out evaluation subset of `windowed_10s_5s.jsonl`: 3989 windows total; 0 windows contain at least 5 labeled speakers

Conclusion:

- The four-speaker ceiling is a real training-window limitation.
- It does **not** affect the held-out 10 s evaluation windows used for the main paper numbers.

### Paper scope reduced and revised

The original draft overclaimed relative to the surviving rerun evidence. The paper was rewritten to keep only claims backed by the rerun artifacts above.

Key revisions made:

- Title narrowed from a broad comparative cross-architecture framing to a Sortformer-focused, rerun-backed framing
- Abstract updated to the corrected pretrained, streaming, and fine-tuned results
- Results section rewritten around the three rerun-backed Sortformer configurations
- Unsupported quantitative pyannote and LS-EEND claims removed from the main paper text
- Attention-entropy section removed from the main paper because it was not necessary to support the revised claims
- Explicit evaluation-protocol caveat added: reported DER is a window-level aggregate, not stitched full-file DER
- Explicit four-speaker-capacity caveat added for the 90 s fine-tuning manifests
- Figure paths normalized via `\graphicspath{{../results/paper_figures/}}`

Supporting note added:

- `docs/claim_to_evidence_traceability.md`

### Static paper sanity checks completed

Checks performed:

- Citation-key check between `docs/main.tex` and `docs/references.bib`
  - Result: no missing bibliography keys
- Toolchain availability check
  - `latexmk`: not installed
  - `pdflatex`: not installed
  - `bibtex`: not installed

Implication:

- I could not run a full PDF compile check in this environment.
- The manuscript revision was therefore validated statically, not by actual TeX compilation.

## 2026-04-19 21:00-21:40 - Must-have analyses completed and paper upgraded

### Benchmark paper reviewed for quality target

Reference document reviewed:

- `docs/paper-asr-dasc-26-data-sel-aug-seg.pdf`

Derived internal planning artifacts:

- `docs/paper_quality_benchmark_notes.md`
- `docs/paper_revision_outline.md`

Key standard adopted from the reference paper:

- explicit scope boundaries
- explicit result questions
- setup detail sufficient to defend each metric
- figures that answer specific questions rather than decorate the paper
- clear limitations and reproducibility framing

### Window-difficulty analysis completed

Artifacts:

- `results/paper_figures/window_difficulty_summary.json`
- `results/paper_figures/fig_window_difficulty.pdf`

Key findings:

- Fine-tuning materially reduces CER in hard multi-speaker windows:
  - 2-speaker windows: 14.96 -> 6.95
  - 3-speaker windows: 26.42 -> 12.55
- Fine-tuning materially reduces CER in rapid-turn windows:
  - 3 speaker changes: 22.21 -> 9.94
- Default streaming's main weakness is elevated FA during active-speech windows, not silent-window hallucination.

### Speaker-capacity analysis completed

Artifacts:

- `results/paper_figures/speaker_capacity_histogram.json`
- `results/paper_figures/fig_speaker_capacity.pdf`

Confirmed counts:

- `finetune_train.jsonl`: 1453 windows total; 197 windows (13.56%) with at least 5 speakers
- `finetune_eval.jsonl`: 456 windows total; 86 windows (18.86%) with at least 5 speakers
- Held-out subset of `windowed_10s_5s.jsonl`: 3989 windows total; 0 windows with at least 5 speakers

### Checkpoint-response sweep completed

Artifacts:

- `results/repro/checkpoint_sweep_1k/checkpoint_sweep_summary.csv`
- `results/paper_figures/fig_checkpoint_sweep.pdf`

Held-out metrics by checkpoint:

- 200: DER 18.5151, FA 7.9159, MISS 1.8552, CER 8.7440
- 400: DER 14.7944, FA 5.7771, MISS 1.8485, CER 7.1687
- 600: DER 16.2289, FA 7.2917, MISS 1.6461, CER 7.2911
- 800: DER 14.8284, FA 6.2324, MISS 1.8200, CER 6.7761
- 1000: DER 14.8961, FA 6.2744, MISS 1.8614, CER 6.7604

Interpretation:

- checkpoint behavior is non-monotonic
- best DER occurs at step 400
- best CER occurs at step 1000

### Streaming latency frontier completed

Artifacts:

- `results/repro/streaming_latency_sweep_eval4/latency_sweep_summary.csv`
- `results/paper_figures/fig_streaming_frontier.pdf`

Held-out metrics:

- medium / 10.0 s: DER 24.8135, FA 16.6360, MISS 0.4701, CER 7.7074
- low / 1.04 s: DER 55.0535, FA 46.1583, MISS 0.5371, CER 8.3582
- ultra-low / 0.32 s: DER 70.3255, FA 61.0534, MISS 0.6558, CER 8.6163

Interpretation:

- lowering latency sharply increases DER
- the increase is dominated by FA, not CER

### Streaming threshold-transfer analysis completed

Important tooling note:

- The first threshold sweep attempted during this session was invalid because threshold overrides were not taking effect while bypass postprocessing remained enabled.
- The issue was fixed in:
  - `src/spkdiar/inference/run_streaming.py`
  - `src/spkdiar/inference/run_sortformer.py`
  - `src/spkdiar/analysis/reevaluate_prob_tensors.py`
  - `src/spkdiar/analysis/sweep_streaming_thresholds.py`
- The corrected sweep reran from scratch and replaced the invalid result path in the paper workflow.

Calibration artifacts:

- Calibration manifest:
  - `results/repro/manifests/windowed_10s_5s_calibration_excl_eval4_nonoverlap_30min.jsonl`
- Calibration inference:
  - `results/repro/streaming_medium_calibration_nonoverlap_30min/eval_metrics.json`
- Corrected threshold sweep:
  - `results/repro/streaming_medium_calibration_threshold_sweep_nonoverlap_30min_postproc/threshold_sweep_summary.csv`
  - `results/repro/streaming_medium_calibration_threshold_sweep_nonoverlap_30min_postproc/threshold_sweep_summary.json`
- Held-out transfer result:
  - `results/repro/sortformer_streaming10_eval4_calibrated_0p75/eval_metrics.json`
- Figure:
  - `results/paper_figures/fig_streaming_threshold_transfer.pdf`

Calibration sweep result:

- Threshold 0.75 selected by minimum validation DER
- Validation metrics at threshold 0.75:
  - DER 25.5726
  - FA 7.1636
  - MISS 6.8993
  - CER 11.5096

Held-out transfer result at threshold 0.75:

- DER 17.4370
- FA 5.4550
- MISS 5.3645
- CER 6.6176

Comparison to default held-out streaming point:

- DER: 24.8135 -> 17.4370
- FA: 16.6360 -> 5.4550
- MISS: 0.4701 -> 5.3645
- CER: 7.7074 -> 6.6176

Interpretation:

- a large fraction of the default streaming FA penalty is calibration-sensitive
- calibrated streaming reaches CER comparable to fine-tuned offline, but not the same DER because MISS rises materially

### Paper source rewritten around the completed analyses

Updated manuscript:

- `docs/main.tex`

Major revisions relative to the earlier narrowed draft:

- added explicit scope boundaries and roadmap in the introduction
- expanded setup with manifest counts, held-out window counts, calibration-split construction, and evaluation mechanics
- added a speaker-capacity figure in the setup section
- added a per-recording system comparison table
- added checkpoint-response, latency-frontier, and threshold-transfer sections
- added difficulty-stratified analysis section
- retained qualitative, role-conditioned, and embedding analyses under a clearer mechanistic-analysis framing
- strengthened discussion and reproducibility framing

Remaining environment limitation:

- `latexmk`, `pdflatex`, and `bibtex` remain unavailable in this environment
- the revised manuscript was checked statically but not compiled to PDF during this audit

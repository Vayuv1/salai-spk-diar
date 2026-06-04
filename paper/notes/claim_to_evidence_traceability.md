# Claim-to-Evidence Traceability Note

Updated: 2026-04-19
Repository: `/home/pandeys2/Workspace/salai-spk-diar`

## Artifact Provenance

- `docs/main.tex`
  - Final paper draft revised to use only rerun-backed Sortformer claims and the completed must-have analyses.
- `results/repro/sortformer_pretrained_eval4_rerun/eval_metrics.json`
  - Produced by `src/spkdiar/inference/run_sortformer.py`
  - Pretrained offline held-out metrics used in the main system-comparison tables.
- `results/repro/sortformer_streaming10_eval4_rerun/eval_metrics.json`
  - Produced by `src/spkdiar/inference/run_streaming.py`
  - Default streaming 10.0 s latency held-out metrics used in the main system-comparison tables and latency discussion.
- `results/repro/sortformer_streaming10_eval4_calibrated_0p75/eval_metrics.json`
  - Produced by `src/spkdiar/analysis/reevaluate_prob_tensors.py`
  - Held-out streaming metrics after validation-selected threshold transfer.
- `results/repro/sortformer_finetuned_eval4_rerun/eval_metrics.json`
  - Produced by `src/spkdiar/inference/run_sortformer.py` using the rerun 1000-step fine-tuned checkpoint
  - Fine-tuned held-out metrics used in the main system-comparison tables and fine-tuning discussion.
- `results/repro/finetune_1k_rerun/logs/finetune/version_0/metrics.csv`
  - Produced by `src/spkdiar/training/finetune_sortformer.py`
  - Confirms the 1000-step rerun completed and records training/validation history.
- `models/diar_sortformer_4spk-v1-atc-1ksteps-rerun.nemo`
  - Exported by `src/spkdiar/training/finetune_sortformer.py`
  - Checkpoint used for the fine-tuned held-out inference rerun.
- `results/repro/checkpoint_sweep_1k/checkpoint_sweep_summary.csv`
  - Produced by `src/spkdiar/analysis/sweep_sortformer_checkpoints.py`
  - Input for the checkpoint-response plot and checkpoint-selection discussion.
- `results/repro/streaming_latency_sweep_eval4/latency_sweep_summary.csv`
  - Produced by `src/spkdiar/analysis/sweep_streaming_latencies.py`
  - Input for the streaming latency-frontier plot and latency discussion.
- `results/repro/streaming_medium_calibration_threshold_sweep_nonoverlap_30min_postproc/threshold_sweep_summary.csv`
  - Produced by `src/spkdiar/analysis/sweep_streaming_thresholds.py`
  - Input for threshold selection and the threshold-transfer plot.
- `results/paper_figures/fig_checkpoint_sweep.pdf`
  - Produced by `src/spkdiar/analysis/gen_fig_checkpoint_sweep.py`
  - Visualizes held-out checkpoint-response behavior.
- `results/paper_figures/fig_streaming_frontier.pdf`
  - Produced by `src/spkdiar/analysis/gen_fig_streaming_frontier.py`
  - Visualizes the held-out streaming latency frontier.
- `results/paper_figures/fig_streaming_threshold_transfer.pdf`
  - Produced by `src/spkdiar/analysis/gen_fig_streaming_threshold_transfer.py`
  - Visualizes the validation sweep and held-out threshold transfer.
- `results/paper_figures/window_difficulty_summary.json`
  - Produced by `src/spkdiar/analysis/window_difficulty_analysis.py`
  - Backs the difficulty-stratified window analysis.
- `results/paper_figures/fig_window_difficulty.pdf`
  - Produced by `src/spkdiar/analysis/window_difficulty_analysis.py`
  - Visualizes the difficulty-stratified analysis.
- `results/paper_figures/speaker_capacity_histogram.json`
  - Produced by `src/spkdiar/analysis/gen_fig_speaker_capacity.py`
  - Backs the speaker-capacity caveat.
- `results/paper_figures/fig_speaker_capacity.pdf`
  - Produced by `src/spkdiar/analysis/gen_fig_speaker_capacity.py`
  - Visualizes the speaker-count distribution across manifests.
- `results/paper_figures/fig1_der_comparison.pdf`
  - Produced by `src/spkdiar/analysis/gen_fig1_der_comparison.py`
  - Input metrics: pretrained and fine-tuned `eval_metrics.json` files above.
- `results/paper_figures/fig2_paired_waterfall.pdf`
  - Produced by `src/spkdiar/analysis/gen_fig2_paired_waterfall_v2.py`
  - Input artifacts: pretrained and fine-tuned RTTM/probability tensors from the rerun directories above.
- `results/paper_figures/role_cer_table.json`
  - Produced by `src/spkdiar/analysis/role_cer_analysis.py`
  - Used for the controller/pilot CER reduction paragraph.
- `results/paper_figures/fig3_embedding_similarity.pdf`
  - Produced by `src/spkdiar/analysis/gen_fig3_embeddings.py`
  - The numeric embedding-margin claims are backed by:
    - `results/speaker_embeddings/dca_d1_1_full/similarity_stats.json`
    - `results/speaker_embeddings/log_id_1_full/similarity_stats.json`

## Major Claims and Evidence

- Claim: the pretrained offline Sortformer reaches 36.6% DER and 14.4% CER overall on the four held-out recordings.
  - Evidence: `results/repro/sortformer_pretrained_eval4_rerun/eval_metrics.json`

- Claim: the default streaming Sortformer at 10.0 s latency reaches 24.8% DER and 7.7% CER overall.
  - Evidence: `results/repro/sortformer_streaming10_eval4_rerun/eval_metrics.json`

- Claim: default streaming lowers CER on all four held-out recordings but shows a DCA-specific FA tradeoff.
  - Evidence:
    - Pretrained per-recording CER/FA in `results/repro/sortformer_pretrained_eval4_rerun/eval_metrics.json`
    - Streaming per-recording CER/FA in `results/repro/sortformer_streaming10_eval4_rerun/eval_metrics.json`

- Claim: streaming calibration selected on held-in data reduces held-out streaming DER from 24.8% to 17.4% and CER from 7.7% to 6.6%.
  - Evidence:
    - Validation sweep in `results/repro/streaming_medium_calibration_threshold_sweep_nonoverlap_30min_postproc/threshold_sweep_summary.csv`
    - Held-out transfer result in `results/repro/sortformer_streaming10_eval4_calibrated_0p75/eval_metrics.json`

- Claim: the validation-selected threshold is 0.75 under the minimum-DER rule.
  - Evidence:
    - `results/repro/streaming_medium_calibration_threshold_sweep_nonoverlap_30min_postproc/threshold_sweep_summary.csv`
    - `results/paper_figures/streaming_threshold_transfer_summary.json`

- Claim: the 1000-step fine-tuned offline model is the strongest overall DER configuration at 14.9% DER and 6.8% CER.
  - Evidence:
    - `results/repro/finetune_1k_rerun/logs/finetune/version_0/metrics.csv`
    - `results/repro/sortformer_finetuned_eval4_rerun/eval_metrics.json`

- Claim: checkpoint behavior is non-monotonic, with best DER at step 400 and best CER at step 1000.
  - Evidence:
    - `results/repro/checkpoint_sweep_1k/checkpoint_sweep_summary.csv`
    - `results/paper_figures/fig_checkpoint_sweep.pdf`

- Claim: lowering streaming latency from 10.0 s to 1.04 s and 0.32 s sharply increases DER, primarily through FA rather than CER.
  - Evidence:
    - `results/repro/streaming_latency_sweep_eval4/latency_sweep_summary.csv`
    - `results/paper_figures/fig_streaming_frontier.pdf`

- Claim: fine-tuning reduces CER by 45.9--60.7% across the four held-out recordings relative to the pretrained offline baseline.
  - Evidence:
    - Pretrained and fine-tuned per-recording CER values in the two `eval_metrics.json` files above

- Claim: fine-tuning reduces controller CER by 43.4--66.2% and pilot CER by 38.6--49.3%.
  - Evidence: `results/paper_figures/role_cer_table.json`

- Claim: fine-tuning materially reduces confusion in hard multi-speaker and rapid-turn windows.
  - Evidence:
    - `results/paper_figures/window_difficulty_summary.json`
    - `results/paper_figures/fig_window_difficulty.pdf`

- Claim: default streaming's main weakness is elevated FA during active-speech windows rather than silent-window hallucination.
  - Evidence:
    - `results/paper_figures/window_difficulty_summary.json`
    - `results/paper_figures/fig_window_difficulty.pdf`

- Claim: the paired waterfall example shows a pretrained slot-locking pattern that is alleviated after fine-tuning.
  - Evidence:
    - `results/paper_figures/fig2_paired_waterfall.pdf`
    - Source probabilities and RTTMs in:
      - `results/repro/sortformer_pretrained_eval4_rerun/`
      - `results/repro/sortformer_finetuned_eval4_rerun/`

- Claim: cue-level TitaNet-Large embeddings remain separable on ATC audio.
  - Evidence:
    - `results/speaker_embeddings/dca_d1_1_full/similarity_stats.json`
    - `results/speaker_embeddings/log_id_1_full/similarity_stats.json`
    - `results/paper_figures/fig3_embedding_similarity.pdf`

- Claim: the paper reports window-level aggregate DER rather than stitched full-file DER.
  - Evidence:
    - Exported NeMo metric files in the rerun directories above
    - Evaluation-path inspection documented in `docs/codex_reproducibility_audit_log.md`

- Claim: the four-speaker ceiling affects some 90 s fine-tuning windows but not the held-out 10 s evaluation windows.
  - Evidence:
    - `results/paper_figures/speaker_capacity_histogram.json`
    - `results/paper_figures/fig_speaker_capacity.pdf`
    - Fine-tuning manifests:
      - `data/processed/manifests/finetune_train.jsonl`
      - `data/processed/manifests/finetune_eval.jsonl`
    - Held-out evaluation manifest:
      - `data/processed/manifests/windowed_10s_5s.jsonl`

## Scripts-to-Paper Map

- Main system-comparison tables
  - `src/spkdiar/inference/run_sortformer.py`
  - `src/spkdiar/inference/run_streaming.py`
  - `src/spkdiar/analysis/reevaluate_prob_tensors.py`

- Checkpoint-response figure and checkpoint-selection paragraph
  - `src/spkdiar/analysis/sweep_sortformer_checkpoints.py`
  - `src/spkdiar/analysis/gen_fig_checkpoint_sweep.py`

- Streaming latency-frontier figure and latency paragraph
  - `src/spkdiar/analysis/sweep_streaming_latencies.py`
  - `src/spkdiar/analysis/gen_fig_streaming_frontier.py`

- Streaming threshold-transfer figure and calibration paragraph
  - `src/spkdiar/analysis/sweep_streaming_thresholds.py`
  - `src/spkdiar/analysis/reevaluate_prob_tensors.py`
  - `src/spkdiar/analysis/gen_fig_streaming_threshold_transfer.py`

- Speaker-capacity figure
  - `src/spkdiar/analysis/gen_fig_speaker_capacity.py`

- Difficulty-stratified analysis
  - `src/spkdiar/analysis/window_difficulty_analysis.py`

- Figure 1 DER decomposition
  - `src/spkdiar/analysis/gen_fig1_der_comparison.py`

- Figure 2 qualitative slot-tracking example
  - `src/spkdiar/analysis/gen_fig2_paired_waterfall_v2.py`

- Figure 3 embedding similarity
  - `src/spkdiar/analysis/gen_fig3_embeddings.py`

- Role-conditioned CER paragraph
  - `src/spkdiar/analysis/role_cer_analysis.py`

## Scope Note

The revised paper intentionally excludes broad quantitative claims about pyannote and LS-EEND because this pass did not retain fresh, protocol-matched rerun artifacts for those systems at the same evidentiary standard as the Sortformer family.

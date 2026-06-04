# Paper Revision Outline

Updated: 2026-04-19
Target paper: `docs/main.tex`
Quality bar: `docs/paper-asr-dasc-26-data-sel-aug-seg.pdf`

This outline converts the benchmark note into an executable revision plan for
the ATC diarization paper. It is not a writing template. It is a section-level
map of what the final draft must explain, what question each result answers,
and which generated artifact backs each claim.

## Required Framing Changes

- State scope explicitly in the introduction.
  This paper is a reproducible evaluation and lightweight adaptation study of
  the Sortformer family on ATC0R under one repository-defined protocol.

- State non-claims explicitly in the introduction.
  The paper does not claim stitched full-file DER, universal cross-architecture
  superiority, or a solved streaming operating point without calibration.

- Separate contributions from findings.
  Contributions are the reproducible protocol, the rerun-backed comparison, and
  the mechanistic analyses. Per-facility differences and checkpoint behavior are
  findings, not primary novelty claims.

## Section Plan

## 1. Introduction

Questions to answer:

- Why is ATC diarization hard enough that standard benchmark intuition fails?
- Why is Sortformer the focus of this paper?
- What exactly is being evaluated?
- What are the paper's concrete contributions?

Required content:

- ATC-specific failure modes:
  narrow-band VHF, sparse half-duplex exchanges, many speakers per recording,
  facility-to-facility acoustic variation.
- Explicit scope statement:
  one protocol, three Sortformer configurations, four held-out recordings, and
  machine-traceable reruns only.
- Explicit contributions list:
  protocol, system comparison, fine-tuning gains, checkpoint/latency/calibration
  analyses, and difficulty/capacity analyses.
- Short roadmap paragraph:
  setup, main results, mechanistic analyses, limitations.

## 2. Related Work

Questions to answer:

- What parts of the diarization literature matter here?
- What gap does this paper fill without overstating novelty?

Required content:

- EEND / Sortformer / streaming Sortformer context.
- Narrow ATC-specific diarization context.
- A short closing paragraph explaining that this work is narrower than a broad
  model benchmark and is centered on protocol-verified reruns.

## 3. Experimental Setup

Questions to answer:

- What data was used?
- What models were run?
- How were fine-tuning and evaluation carried out?
- What caveats are built into the setup?

Required subsections:

- Dataset and split
- Model variants
- Fine-tuning configuration
- Windowed evaluation protocol
- Speaker-capacity caveat

Required details:

- 16 recordings across three facilities and about 22 hours total.
- Exact held-out recordings:
  `dca_d1_1`, `dca_d2_2`, `dfw_a1_1`, `log_id_1`.
- Training recordings:
  `dca_d1_2`, `dca_d1_3`, `dca_d1_4`, `dca_d2_1`, `dca_f1_1`, `dca_f1_2`,
  `dca_f2_1`, `dca_f2_2`, `dca_f2_3`, `dfw_d1_1`, `log_id_2`, `log_id_3`.
- Manifest sizes:
  `windowed_10s_5s.jsonl`, `finetune_train.jsonl`, `finetune_eval.jsonl`.
- Exact inference settings for:
  pretrained offline, streaming latency presets, fine-tuned offline.
- Fine-tuning details:
  frozen encoder, trainable decoder/output layers, optimizer, lr, warmup, batch
  size, precision, checkpoint cadence.
- Evaluation protocol:
  10 s windows with 5 s shift, 0.25 s collar, overlap retained, aggregated
  window-level metrics rather than stitched full-file DER.
- Capacity figure support:
  held-out 10 s windows never exceed 4 speakers; 90 s train/val windows do.

Artifacts:

- `results/paper_figures/fig_speaker_capacity.pdf`
- `results/paper_figures/speaker_capacity_histogram.json`

## 4. Main Results

This section should be broken into result questions, not a single prose block.

### 4.1 Which Sortformer variant is strongest on the held-out protocol?

Required artifacts:

- Overall metrics table
- Per-recording table

Required evidence:

- `results/repro/sortformer_pretrained_eval4_rerun/eval_metrics.json`
- `results/repro/sortformer_streaming10_eval4_rerun/eval_metrics.json`
- `results/repro/sortformer_finetuned_eval4_rerun/eval_metrics.json`

Required points:

- Pretrained offline is the reference baseline.
- Streaming lowers CER on all held-out recordings but can worsen DER through FA.
- Fine-tuned offline is the strongest overall system under this protocol.

### 4.2 Which checkpoint is actually best?

Required artifact:

- `results/paper_figures/fig_checkpoint_sweep.pdf`

Required evidence:

- `results/repro/checkpoint_sweep_1k/checkpoint_sweep_summary.csv`

Required points:

- Improvement is not monotonic with step count.
- Best DER occurs at step 400.
- Best CER occurs at step 1000.
- Therefore the paper should describe 1000 steps as the chosen adaptation
  condition and report the checkpoint-response tradeoff explicitly, rather than
  implying the final checkpoint is automatically optimal.

### 4.3 What does streaming latency buy, and what does it cost?

Required artifact:

- `results/paper_figures/fig_streaming_frontier.pdf`

Required evidence:

- `results/repro/streaming_latency_sweep_eval4/latency_sweep_summary.csv`

Required points:

- CER changes only modestly from medium to low/ultra-low latency.
- DER degrades sharply at lower latency because FA rises sharply.
- The streaming tradeoff is therefore calibration- and operating-point-sensitive,
  not simply a confusion-only story.

### 4.4 How much of the streaming error is calibration versus intrinsic?

Required artifacts:

- `results/paper_figures/fig_streaming_threshold_transfer.pdf`
- `results/paper_figures/streaming_threshold_transfer_summary.json`

Required evidence:

- Corrected validation sweep output
- Held-out transfer reevaluation using the selected threshold

Required points:

- Threshold selection must be done on held-in data only.
- The selected threshold is transferred unchanged to the held-out recordings.
- This section must state whether the held-out streaming FA penalty narrows
  materially after calibration or whether the latency tradeoff remains dominant.

## 5. Difficulty And Mechanistic Analysis

### 5.1 Which windows are still hard after fine-tuning?

Required artifact:

- `results/paper_figures/fig_window_difficulty.pdf`

Required evidence:

- `results/paper_figures/window_difficulty_summary.json`

Required points:

- Fine-tuning materially reduces CER in 2-speaker and 3-speaker windows.
- Fine-tuning helps most on windows with several speaker changes.
- Streaming's main pathology is elevated FA during active-speech windows, not
  silent-window hallucination.

### 5.2 What do the qualitative traces show?

Required artifact:

- `results/paper_figures/fig2_paired_waterfall.pdf`

Required points:

- Use this as a mechanistic illustration of slot lock-up and recovery.
- Keep the text explicitly qualitative and tied back to the CER reductions.

### 5.3 Are controller and pilot gains comparable?

Required evidence:

- `results/paper_figures/role_cer_table.json`

Required points:

- Report controller and pilot CER reductions separately.
- Avoid causal overclaiming; describe the role asymmetry as an observation.

### 5.4 Does ATC audio still preserve speaker identity information?

Required artifact:

- `results/paper_figures/fig3_embedding_similarity.pdf`

Required evidence:

- `results/speaker_embeddings/dca_d1_1_full/similarity_stats.json`
- `results/speaker_embeddings/log_id_1_full/similarity_stats.json`

Required points:

- Embedding separability survives VHF degradation to a meaningful degree.
- State clearly that this does not by itself prove clustering success.

## 6. Discussion

Questions to answer:

- What is robust across reruns?
- What is facility-dependent?
- What limitations remain?
- What future work is actually justified by the evidence?

Required content:

- Robust findings:
  fine-tuning helps substantially; confusion is a core baseline failure mode.
- Facility dependence:
  streaming FA differs strongly by site/recording.
- Limitations:
  four held-out recordings, windowed aggregation, 4-speaker output ceiling,
  training/eval mismatch in 90 s windows.
- Future work tied to evidence:
  larger-capacity diarizers, protocol-matched non-Sortformer baselines, richer
  streaming calibration, and possibly text-assisted or role-aware diarization.

## 7. Reproducibility Paragraph

Required content:

- State that all metric claims are backed by emitted JSON/CSV outputs.
- Name the result directories and scripts that produced the tables and figures.
- Mention the threshold-sweep bug and its fix so the calibration story is
  auditable rather than silently corrected.

## Figures And Tables To Include

- Overall metrics table
- Per-recording metrics table
- `fig1_der_comparison.pdf`
- `fig_checkpoint_sweep.pdf`
- `fig_streaming_frontier.pdf`
- `fig_streaming_threshold_transfer.pdf`
- `fig_window_difficulty.pdf`
- `fig_speaker_capacity.pdf`
- `fig2_paired_waterfall.pdf`
- `fig3_embedding_similarity.pdf`

## Claim Discipline Rules

- Do not report a number unless it comes from a specific emitted artifact.
- Do not treat checkpoint 1000 as automatically optimal.
- Do not treat streaming improvements as uniform across facilities.
- Do not conflate calibration gains with intrinsic model gains.
- Do not let qualitative figures carry quantitative claims by themselves.

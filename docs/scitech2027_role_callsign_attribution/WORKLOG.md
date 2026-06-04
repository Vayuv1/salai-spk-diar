# SciTech 2027 Work Log

## Scope
- Objective: prepare an AIAA SciTech 2027 extended abstract on role and callsign attribution for flight-deck avionics support in ATC voice operations.
- Constraint: keep all DASC 2026 paper materials read-only and isolated from this work.
- Working directory for all new or revised SciTech files: `docs/scitech2027_role_callsign_attribution/`.

## Git status before this revision pass
- Checked with `git status --short --branch` on 2026-05-21.
- The repository was already dirty before this pass.
- Pre-existing tracked modifications outside the SciTech directory included:
  - `external/FS-EEND`
  - `src/spkdiar/analysis/gen_fig1_der_comparison.py`
  - `src/spkdiar/analysis/gen_fig3_embeddings.py`
  - `src/spkdiar/analysis/ieee_style.py`
  - `src/spkdiar/analysis/role_cer_analysis.py`
  - `src/spkdiar/inference/run_pyannote.py`
  - `src/spkdiar/inference/run_sortformer.py`
  - `src/spkdiar/inference/run_streaming.py`
  - `src/spkdiar/training/finetune_sortformer.py`
- Those DASC-related or unrelated modifications were left untouched.

## Read-only evidence inspected for this pass
- DASC paper and bibliography:
  - `docs/main.tex`
  - `docs/references.bib`
  - `docs/dasc2026_sortformer_atc_submission_ready.pdf`
- Companion ASR paper:
  - `docs/paper-asr-dasc-26-data-sel-aug-seg.pdf`
- Result artifacts used for verified numerical claims:
  - `results/repro/sortformer_pretrained_eval4_rerun/eval_metrics.json`
  - `results/repro/sortformer_streaming10_eval4_rerun/eval_metrics.json`
  - `results/repro/sortformer_streaming10_eval4_calibrated_0p75/eval_metrics.json`
  - `results/repro/sortformer_finetuned_eval4_rerun/eval_metrics.json`
  - `results/paper_figures/role_cer_table.json`
- Annotation / alignment infrastructure:
  - `src/spkdiar/data/stm_parser.py`
  - `src/spkdiar/data/make_rttm.py`
  - `data/processed/manifests/windowed_10s_5s.jsonl`

## What changed in this revision pass
- Reframed the abstract as a Digital Avionics submission following Dr. Liu's comments.
- Made the add-on flight-deck avionics application the primary story instead of ATC facility monitoring.
- Integrated IASMS and ATM support into the abstract body, figure wording, and contribution framing without making them read like pasted subtopic keywords.
- Removed forward-looking aircraft-automation language and related sensitive terminology from the public-facing draft.
- Updated the SciTech-only framework figure so the output stage now reads as avionics-support outputs rather than generic monitoring outputs.
- Rewrote the SciTech abstract substantially so it reads as a public conference submission rather than an internal note.
- Replaced the awkward author block with a cleaner professional title/affiliation block.
- Reframed the story around the operational monitoring problem:
  - transcript-only ATC monitoring is insufficient,
  - role and callsign attribution are the missing intermediate layer,
  - completed diarization work is preliminary foundation rather than the main contribution.
- Removed internal/project-management language from the paper body.
- Kept only verified numerical claims from completed diarization and role-conditioned analyses.
- Avoided introducing completed callsign or readback metrics that are not supported by repository artifacts.
- Added one new SciTech-only pipeline figure.
- Added two tables:
  - verified preliminary foundation
  - planned full-paper evaluation protocol
- Tightened the manual reference list and kept only public references in the abstract.
- Simplified the public-facing ASR discussion so the abstract no longer depends on exact ASR WER values for persuasion; the completed numerical claims in the paper are now limited to verified diarization and role-conditioned results.
- Rebuilt the SciTech-only pipeline figure as a simpler academic block diagram with shorter labels and a cleaner output stage.
- Standardized table formatting with consistent caption styling, controlled float placement, and improved row spacing.
- Removed the unpublished companion-ASR result statement from the paper body and replaced it with a public-citation-based description of the planned ASR branch.
- Tightened table wording so both tables read as concise conference tables rather than over-compressed notes.
- Rebalanced Table 1 and Table 2 column widths so the descriptive columns use the available page width more effectively and the rendered tables no longer look visually right-heavy.
- Replaced one role-attribution sentence that could be read as inference-time use of annotation format with wording that correctly refers to annotation-derived training labels.
- Strengthened the callsign-branch sentence from `assumes` to `will use`, while keeping callsign evaluation framed as planned full-paper work.
- Rebuilt the SciTech-only figure and recompiled the shareable PDF after the avionics reframing.
- Visually rechecked the title page, figure page, table page, and references page in the final PDF.
- Performed a final framing pass so the abstract reads as a Digital Avionics submission built around a flight-deck add-on capability, with IASMS and ATM support treated as natural secondary uses rather than repeated subtopic labels.

## New SciTech-only files added or regenerated
- `gen_pipeline_figure.py`
- `fig_role_callsign_framework.pdf`
- `fig_role_callsign_framework.png`
- `extended_abstract_submission.tex`
- `extended_abstract_submission.pdf`
- `extended_abstract_submission_ready.pdf`
- Updated:
  - `extended_abstract_draft.md`
  - `CLAIMS_LEDGER.md`
  - `WORKLOG.md`

## Design choices
- No DASC figure, table, script, result artifact, or build file was modified or regenerated.
- The new figure was created inside the SciTech directory so the abstract can evolve independently of the DASC package.
- This abstract is now framed for the SciTech submission category `Digital Avionics`.
- Intended SciTech subtopic: `Avionics technologies for safe and efficient vehicle operation in national airspace`.
- The current public-facing abstract does not cite the unpublished companion ASR paper. Instead, it treats the ASR branch as companion project evidence in the text and records the exact supporting path in the claims ledger.
- Exact ASR WER numbers were intentionally omitted from the revised abstract body, even though some can be verified from the companion PDF, because the user requested a clean, honest abstract without unsupported or unnecessary numerical ASR emphasis.

## Claims still requiring confirmation for a future full paper
- Any completed callsign-extraction metric
- Any completed readback-pairing metric
- Any completed joint role--callsign attribution metric
- Any completed downstream miscommunication-detection metric
- Those items would require either:
  - imported domain-tuned Whisper outputs from the external ASR repository, or
  - new evaluation artifacts generated in this SciTech workspace

## Build notes
- The shareable PDF is built from `extended_abstract_submission.tex`.
- The figure is generated by `gen_pipeline_figure.py` inside this SciTech directory.
- The PDF was rebuilt after the rewrite and then checked visually page by page.
- The final formatting pass included direct inspection of the figure asset and fresh page renders from the compiled PDF to verify figure readability, table ordering, caption consistency, and reference placement.
- The current shareable PDF is `extended_abstract_submission_ready.pdf`.

## DASC integrity note
- DASC paper files, DASC figures, DASC result artifacts, DASC scripts, and DASC build files were verified untouched in this pass.

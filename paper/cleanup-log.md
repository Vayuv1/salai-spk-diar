# DASC 2026 Workspace Cleanup Log

Date: 2026-06-04

## Safety Snapshot

- Main repo snapshot commit pushed before cleanup: `e3328c2` (`Snapshot before workspace cleanup`).
- User-approved exception: `external/FS-EEND` was preserved locally and not pushed. Its local git state was kept intact and not deleted or reset.

## Initial Inventory

### Source Files

- `docs/main.tex`
- `docs/main_anonymous.tex`
- `docs/main_identified.tex`
- `docs/references.bib`

### Figures / Assets

- Active paper figure set: `results/paper_figures/` (`55` files)
- Review screenshot batches: `docs/docs/page_check*/page-*.png` (`80` files total)

### Notes / Docs

- `docs/00-abstract-submission.md`
- `docs/01a_voice_fingerprint_and_speaker_embedding.md`
- `docs/01b_sortformer_architecture.md`
- `docs/01c_ls_eend_architecture.md`
- `docs/01d_conformer_architecture.md`
- `docs/02_literature_survey_report.md`
- `docs/03_complete_explanation_guide.md`
- `docs/abstract_review_concerns_addressed.md`
- `docs/checkpoint_selection_resolution.md`
- `docs/citation_audit_report.md`
- `docs/citation_strengthening_note_final.md`
- `docs/claim_to_evidence_traceability.md`
- `docs/codex-task.docx`
- `docs/codex_reproducibility_audit_log.md`
- `docs/corrected_reference_list.md`
- `docs/dasc_paper_revisions.txt`
- `docs/dasc_submission_checklist.md`
- `docs/editorial_memo_final_revision.md`
- `docs/feedback.pdf`
- `docs/figure7_disposition.md`
- `docs/figure_table_audit_final.md`
- `docs/final_proofreading_log.md`
- `docs/final_refinement_memo.md`
- `docs/paper_quality_benchmark_notes.md`
- `docs/paper_revision_outline.md`
- `docs/phase2_experiment_log.md`
- `docs/proofreading_report.md`
- `docs/reference_verification.md`
- `docs/reviewer_check_final.md`
- `docs/terminology_abbreviation_audit_final.md`

### Compiled Paper PDFs

- `docs/Comparative_Evaluation_and_Domain_Adaptation_of_Neural_Speaker_Diarization_for_Air_Traffic_Control_Communications.pdf`
- `docs/dasc2026_sortformer_atc_submission_ready.pdf`
- `docs/dasc2026_sortformer_atc_submission_ready_refresh.pdf`
- `docs/main.pdf`
- `docs/main_anonymous.pdf`
- `docs/main_identified.pdf`
- `docs/paper-asr-dasc-26-data-sel-aug-seg.pdf`

### Generated Build Artifacts

- `docs/main.aux`
- `docs/main.bbl`
- `docs/main.blg`
- `docs/main.fdb_latexmk`
- `docs/main.fls`
- `docs/main.log`
- `docs/main_anonymous.aux`
- `docs/main_anonymous.bbl`
- `docs/main_anonymous.blg`
- `docs/main_anonymous.fdb_latexmk`
- `docs/main_anonymous.fls`
- `docs/main_anonymous.log`
- `docs/main_identified.aux`
- `docs/main_identified.bbl`
- `docs/main_identified.blg`
- `docs/main_identified.fdb_latexmk`
- `docs/main_identified.fls`
- `docs/main_identified.log`
- `main.aux`
- `main.fdb_latexmk`
- `main.fls`
- `main.log`

### Kept But Flagged For Review

- `docs/results/paper_figures/*`
  Reason: `4` files that appear to be a partial duplicate figure copy, not the active build path.
- `docs/scitech2027_role_callsign_attribution/**/*`
  Reason: `37` unrelated SciTech files intentionally left untouched.

## Files Moved

### Source

- `docs/main.tex -> paper/src/main.tex`
- `docs/main_anonymous.tex -> paper/src/main_anonymous.tex`
- `docs/main_identified.tex -> paper/src/main_identified.tex`
- `docs/references.bib -> paper/src/references.bib`

### Compiled PDFs

- `docs/Comparative_Evaluation_and_Domain_Adaptation_of_Neural_Speaker_Diarization_for_Air_Traffic_Control_Communications.pdf -> paper/output/Comparative_Evaluation_and_Domain_Adaptation_of_Neural_Speaker_Diarization_for_Air_Traffic_Control_Communications.pdf`
- `docs/dasc2026_sortformer_atc_submission_ready.pdf -> paper/output/dasc2026_sortformer_atc_submission_ready.pdf`
- `docs/dasc2026_sortformer_atc_submission_ready_refresh.pdf -> paper/output/dasc2026_sortformer_atc_submission_ready_refresh.pdf`
- `docs/main.pdf -> paper/output/main.pdf`
- `docs/main_anonymous.pdf -> paper/output/main_anonymous.pdf`
- `docs/main_identified.pdf -> paper/output/main_identified.pdf`
- `docs/paper-asr-dasc-26-data-sel-aug-seg.pdf -> paper/output/paper-asr-dasc-26-data-sel-aug-seg.pdf`

### Notes / Docs

- `docs/00-abstract-submission.md -> paper/notes/00-abstract-submission.md`
- `docs/01a_voice_fingerprint_and_speaker_embedding.md -> paper/notes/01a_voice_fingerprint_and_speaker_embedding.md`
- `docs/01b_sortformer_architecture.md -> paper/notes/01b_sortformer_architecture.md`
- `docs/01c_ls_eend_architecture.md -> paper/notes/01c_ls_eend_architecture.md`
- `docs/01d_conformer_architecture.md -> paper/notes/01d_conformer_architecture.md`
- `docs/02_literature_survey_report.md -> paper/notes/02_literature_survey_report.md`
- `docs/03_complete_explanation_guide.md -> paper/notes/03_complete_explanation_guide.md`
- `docs/abstract_review_concerns_addressed.md -> paper/notes/abstract_review_concerns_addressed.md`
- `docs/checkpoint_selection_resolution.md -> paper/notes/checkpoint_selection_resolution.md`
- `docs/citation_audit_report.md -> paper/notes/citation_audit_report.md`
- `docs/citation_strengthening_note_final.md -> paper/notes/citation_strengthening_note_final.md`
- `docs/claim_to_evidence_traceability.md -> paper/notes/claim_to_evidence_traceability.md`
- `docs/codex-task.docx -> paper/notes/codex-task.docx`
- `docs/codex_reproducibility_audit_log.md -> paper/notes/codex_reproducibility_audit_log.md`
- `docs/corrected_reference_list.md -> paper/notes/corrected_reference_list.md`
- `docs/dasc_paper_revisions.txt -> paper/notes/dasc_paper_revisions.txt`
- `docs/dasc_submission_checklist.md -> paper/notes/dasc_submission_checklist.md`
- `docs/editorial_memo_final_revision.md -> paper/notes/editorial_memo_final_revision.md`
- `docs/feedback.pdf -> paper/notes/feedback.pdf`
- `docs/figure7_disposition.md -> paper/notes/figure7_disposition.md`
- `docs/figure_table_audit_final.md -> paper/notes/figure_table_audit_final.md`
- `docs/final_proofreading_log.md -> paper/notes/final_proofreading_log.md`
- `docs/final_refinement_memo.md -> paper/notes/final_refinement_memo.md`
- `docs/paper_quality_benchmark_notes.md -> paper/notes/paper_quality_benchmark_notes.md`
- `docs/paper_revision_outline.md -> paper/notes/paper_revision_outline.md`
- `docs/phase2_experiment_log.md -> paper/notes/phase2_experiment_log.md`
- `docs/proofreading_report.md -> paper/notes/proofreading_report.md`
- `docs/reference_verification.md -> paper/notes/reference_verification.md`
- `docs/reviewer_check_final.md -> paper/notes/reviewer_check_final.md`
- `docs/terminology_abbreviation_audit_final.md -> paper/notes/terminology_abbreviation_audit_final.md`

### Review Screenshots

- `docs/docs/page_check/page-*.png -> paper/review/page_checks/page_check/page-*.png` (`8` files)
- `docs/docs/page_check2/page-*.png -> paper/review/page_checks/page_check2/page-*.png` (`8` files)
- `docs/docs/page_check3/page-*.png -> paper/review/page_checks/page_check3/page-*.png` (`8` files)
- `docs/docs/page_check4/page-*.png -> paper/review/page_checks/page_check4/page-*.png` (`8` files)
- `docs/docs/page_check5/page-*.png -> paper/review/page_checks/page_check5/page-*.png` (`8` files)
- `docs/docs/page_check6/page-*.png -> paper/review/page_checks/page_check6/page-*.png` (`8` files)
- `docs/docs/page_check7/page-*.png -> paper/review/page_checks/page_check7/page-*.png` (`8` files)
- `docs/docs/page_check8/page-*.png -> paper/review/page_checks/page_check8/page-*.png` (`8` files)
- `docs/docs/page_check9/page-*.png -> paper/review/page_checks/page_check9/page-*.png` (`8` files)
- `docs/docs/page_check10/page-*.png -> paper/review/page_checks/page_check10/page-*.png` (`8` files)

## Files Copied

- `results/paper_figures/* -> paper/figures/*` (`55` files copied so the active paper workspace is self-contained)
- `paper/{src,figures,output} -> submission-frozen/{src,figures,output}`
- `submission-frozen/{src,figures,output} -> camera-ready/{src,figures,output}`

## Source Update

- Updated `paper/src/main.tex`
  - `\graphicspath{{../results/paper_figures/}} -> \graphicspath{{../figures/}}`

## Files Deleted

All deleted files were regenerable LaTeX build artifacts and were removed only after a successful rebuild.

### Deleted From `docs/`

- `docs/main.aux`
- `docs/main.bbl`
- `docs/main.blg`
- `docs/main.fdb_latexmk`
- `docs/main.fls`
- `docs/main.log`
- `docs/main_anonymous.aux`
- `docs/main_anonymous.bbl`
- `docs/main_anonymous.blg`
- `docs/main_anonymous.fdb_latexmk`
- `docs/main_anonymous.fls`
- `docs/main_anonymous.log`
- `docs/main_identified.aux`
- `docs/main_identified.bbl`
- `docs/main_identified.blg`
- `docs/main_identified.fdb_latexmk`
- `docs/main_identified.fls`
- `docs/main_identified.log`

### Deleted From Repo Root

- `main.aux`
- `main.fdb_latexmk`
- `main.fls`
- `main.log`

### Deleted From Active Workspace Output

- `paper/output/main.aux`
- `paper/output/main.bbl`
- `paper/output/main.blg`
- `paper/output/main.fdb_latexmk`
- `paper/output/main.fls`
- `paper/output/main.log`
- `paper/output/main_anonymous.aux`
- `paper/output/main_anonymous.bbl`
- `paper/output/main_anonymous.blg`
- `paper/output/main_anonymous.fdb_latexmk`
- `paper/output/main_anonymous.fls`
- `paper/output/main_anonymous.log`
- `paper/output/main_identified.aux`
- `paper/output/main_identified.bbl`
- `paper/output/main_identified.blg`
- `paper/output/main_identified.fdb_latexmk`
- `paper/output/main_identified.fls`
- `paper/output/main_identified.log`

### Deleted From `submission-frozen/output`

- `submission-frozen/output/main_anonymous.aux`
- `submission-frozen/output/main_anonymous.bbl`
- `submission-frozen/output/main_anonymous.blg`
- `submission-frozen/output/main_anonymous.fdb_latexmk`
- `submission-frozen/output/main_anonymous.fls`
- `submission-frozen/output/main_anonymous.log`
- `submission-frozen/output/main_identified.aux`
- `submission-frozen/output/main_identified.bbl`
- `submission-frozen/output/main_identified.blg`
- `submission-frozen/output/main_identified.fdb_latexmk`
- `submission-frozen/output/main_identified.fls`
- `submission-frozen/output/main_identified.log`

### Deleted From `camera-ready/output`

- `camera-ready/output/main_anonymous.aux`
- `camera-ready/output/main_anonymous.bbl`
- `camera-ready/output/main_anonymous.blg`
- `camera-ready/output/main_anonymous.fdb_latexmk`
- `camera-ready/output/main_anonymous.fls`
- `camera-ready/output/main_anonymous.log`
- `camera-ready/output/main_identified.aux`
- `camera-ready/output/main_identified.bbl`
- `camera-ready/output/main_identified.blg`
- `camera-ready/output/main_identified.fdb_latexmk`
- `camera-ready/output/main_identified.fls`
- `camera-ready/output/main_identified.log`

## Validation

- Rebuilt `paper/src/main.tex` into `paper/output/main.pdf`
- Rebuilt `paper/src/main_anonymous.tex` into `paper/output/main_anonymous.pdf`
- Rebuilt `paper/src/main_identified.tex` into `paper/output/main_identified.pdf`
- Rebuilt `submission-frozen/src/main_anonymous.tex` into `submission-frozen/output/main_anonymous.pdf`
- Rebuilt `submission-frozen/src/main_identified.tex` into `submission-frozen/output/main_identified.pdf`
- Rebuilt `camera-ready/src/main_anonymous.tex` into `camera-ready/output/main_anonymous.pdf`
- Rebuilt `camera-ready/src/main_identified.tex` into `camera-ready/output/main_identified.pdf`

## Final Folder Structure

```text
paper/
  src/
    main.tex
    main_anonymous.tex
    main_identified.tex
    references.bib
  figures/                   55 files
  notes/                     30 files
  output/
    Comparative_Evaluation_and_Domain_Adaptation_of_Neural_Speaker_Diarization_for_Air_Traffic_Control_Communications.pdf
    dasc2026_sortformer_atc_submission_ready.pdf
    dasc2026_sortformer_atc_submission_ready_refresh.pdf
    main.pdf
    main_anonymous.pdf
    main_identified.pdf
    paper-asr-dasc-26-data-sel-aug-seg.pdf
  review/
    page_checks/             80 files
  cleanup-log.md

submission-frozen/
  src/
  figures/
  output/

camera-ready/
  src/
  figures/
  output/

docs/
  results/paper_figures/     flagged, untouched duplicate partial copy
  scitech2027_role_callsign_attribution/  flagged, untouched SciTech subtree
```

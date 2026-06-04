# Final Refinement Memo (Advisor-Ready Pass)

Updated: 2026-04-20
Manuscript: `docs/main.tex`
PDF: `docs/dasc2026_sortformer_atc_submission_ready.pdf`

This pass made targeted, non-structural refinements to the current draft. It intentionally avoided wholesale rewrites and preserved the current section flow and tone.

## Targeted Changes Applied

- Title tightened to explicitly signal offline/streaming comparison and calibration without broadening scope.
- Checkpoint-selection logic made explicit in the checkpoint-sensitivity subsection so the “best DER at step 400 vs. best CER at step 1000” point is defensible while keeping the main reported fine-tuned model unchanged.
- ATC0 Revised annotation→RTTM preparation clarified with one concise sentence describing how STM segments and speaker IDs map into RTTM regions.
- Data-partition table clarified so “Size” is explicitly a window count (`# Windows`) and the caption states this directly.
- Removed “values are percentages” style phrasing and replaced it with direct `%` usage (with table captions/headers indicating units where appropriate).
- Figure 7 (cue-level embedding analysis) caption and surrounding prose tightened to clearly mark it as auxiliary evidence and to avoid any implied dependence on the diarization pipeline.
- Discussion limitations updated with a single brief sentence acknowledging that operational deployment in safety-critical avionics would require broader validation and certification-oriented evidence.

## Author Block

- Not modified in `docs/main.tex`. The draft remains compatible with the author block you already inserted in your working version.

# Editorial Memo: Final Manuscript Revision

Updated: 2026-04-20
Target manuscript: `docs/main.tex`
Submission PDF: `docs/dasc2026_sortformer_atc_submission_ready.pdf`

## Purpose of This Pass

This pass converted the Sortformer paper from a technically correct but internally framed draft into a standalone conference manuscript written for external review. The benchmark for quality was the DASC ASR paper in `docs/paper-asr-dasc-26-data-sel-aug-seg.pdf`, but the Sortformer paper was not rewritten to imitate that paper's wording or rhetorical cadence.

## Major Editorial Changes

- Rewrote the title, abstract, introduction, discussion, and conclusion so that the paper now reads as a finished ATC diarization study rather than a project summary or reproducibility note.
- Tightened scope control. The paper now states explicitly that it studies one model family under a matched offline/streaming ATC protocol rather than claiming a broad benchmark over all diarization approaches.
- Strengthened terminology hygiene. Abbreviations and specialized terms are expanded or defined when first introduced, including `ATC`, `VHF`, `DER`, `CER`, `FA`, `MISS`, `EEND`, `LS-EEND`, `NEST`, `AOSC`, `LDC`, `STM`, `RTTM`, `AdamW`, `bf16`, `speaker slots`, `collar scoring`, `overlap retained`, and `threshold calibration`.
- Reworked the experimental setup so it is audit-friendly. The final version now explains the ATC0 Revised corpus, what “revised” means, the three disjoint data partitions, why the calibration subset is non-overlapping, how 10 s and 90 s windows are used, what is frozen during fine-tuning, which hyperparameters matter, and how segment-level scoring is computed.
- Reorganized the results into a clearer scientific argument: overall comparison, checkpoint sensitivity, latency frontier, threshold calibration and transfer, difficulty-stratified analysis, and auxiliary evidence.
- Removed the weak cherry-picked qualitative window figure from the main paper. The manuscript now relies on aggregate evidence rather than a selected local example to support the fine-tuning claim.
- Revised figure and table captions so they explain what the reader should conclude rather than merely restating the title of the visual.
- Rewrote the discussion to interpret the practical meaning of the findings for ATC diarization, including why offline and streaming systems differ, why calibration matters, where adaptation helps, and what the remaining limitations are.
- Softened the AI disclosure so it no longer reads like a tooling log while preserving IEEE-style transparency.

## Net Result

The manuscript now presents itself as a finished DASC conference paper with explicit scope, reviewer-facing protocol detail, stronger transitions, clearer interpretation of results, and a more mature technical tone.

# Final Submit-Readiness Proofreading Log

Date: 2026-04-26

## Scope of this pass

- reread `docs/main.tex` from top to bottom
- rechecked Dr. Liu's annotated feedback in `docs/feedback.pdf`
- rebuilt the two schematic figures from scratch
- recompiled the manuscript with `latexmk`
- rendered the compiled PDF to page images
- visually inspected every page of the compiled PDF
- iterated until no visible figure overlap, caption mismatch, reference-style inconsistency, or structural layout issue remained

## Text edits made

1. Normalized figure references in the body to `Fig.` style throughout.
2. Reworked the introduction transition around the research questions so that the protocol summary, results preview, and contribution list read as one continuous argument.
3. Kept Dr. Liu's preferred contribution framing:
   - `We endeavor to answer these questions with the following work:`
4. Added citation support directly to the operational-importance paragraph in the introduction.
5. Expanded `STM` at first use as `Segment Time Marked (STM)`.
6. Replaced `speaker stream` with `speaker channel` for clearer wording.
7. Kept dataset wording consistent:
   - `author-revised data from the ATC0 dataset`
   - `three ATC facilities`
   - `the revised data`
8. Kept the partition description explicit that fine-tuning train, fine-tuning val., and calibration all come from the same 12 non-evaluation recordings with different windowing/sampling policies.
9. Clarified `difficulty-stratified` in the analysis section as grouping by local conversational complexity within each 10 s window.
10. Shortened a few repeated or overly long sentences without changing claims or results.

## Figures regenerated or materially revised

### Fig. 1
- File: `results/paper_figures/fig_protocol_overview.pdf`
- Script: `src/spkdiar/analysis/gen_fig_protocol_overview.py`
- Recreated from scratch.
- Final design:
  - panel (a): 10 s analysis windows on separate lanes with a clean 5 s shift indicator
  - panel (b): segment-level scoring schematic with external labels for collar and overlap retained
- Removed overlapping title/tick-label conflict from earlier drafts.
- Removed dense in-figure explanatory prose and moved explanation to the caption.

### Fig. 2
- File: `results/paper_figures/fig_model_adaptation_overview.pdf`
- Script: `src/spkdiar/analysis/gen_fig_model_adaptation_overview.py`
- Recreated from scratch.
- Final design:
  - panel (a): shared offline/streaming Sortformer backbone
  - panel (b): frozen-encoder ATC adaptation path
- Removed hatch-filled label regions and replaced them with plain light boxes.
- Kept the AOSC branch explicit and labeled `streaming only`.

### Fig. 8 / DER decomposition figure
- File: `results/paper_figures/fig1_der_comparison.pdf`
- Script: `src/spkdiar/analysis/gen_fig1_der_comparison.py`
- Retained with the previously improved component colors and hatching.
- Verified visually in the compiled PDF that the three DER components are distinguishable in both color and grayscale print.

## Citation / bibliography status

- No fabricated references were added.
- No unsupported new claims were introduced.
- Existing citations retained for:
  - ATC0 dataset
  - Sortformer
  - Streaming Sortformer
  - LS-EEND
  - pyannote
  - Powerset loss / pyannote method support
  - TitaNet
  - BERTraffic
  - DIHARD / CALLHOME
- Bibliography compiled successfully under IEEE numbered style.

## Visual inspection results

Checked directly in the compiled PDF:
- page 1: title, abstract, introduction opening, research-question block
- page 2: Fig. 1 placement and readability
- page 3: Fig. 2 placement and readability
- page 4: setup-to-results transition, Table I, Table II
- page 5: checkpoint / latency / threshold figures and captions
- page 6: difficulty, DER decomposition, and embedding figures
- page 7: discussion, conclusion, references

Verified:
- no text overlaps in Fig. 1 or Fig. 2
- no labels placed on top of hatched or patterned regions in the new schematics
- no figure-reference order issues
- no caption/label mismatches
- no nearly empty trailing page
- final PDF ends cleanly in 7 pages

## Compilation status

- Command used:
  - `latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex`
- Result:
  - success
- Final output:
  - `docs/main.pdf`
  - `docs/dasc2026_sortformer_atc_submission_ready.pdf`
- Final PDF properties:
  - 7 pages
  - US Letter
  - 292,247 bytes

## Remaining notes

- The local DASC submission checklist confirms anonymous placeholders for the submission PDF, so the anonymous author block was retained.
- The standard IEEE reminder about final camera-ready column balancing / font checks remains expected, but there are no current visible layout defects or unresolved compilation problems in the verified PDF.

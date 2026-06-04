# DASC Submission Checklist

Updated: 2026-04-20
Target paper: `docs/dasc2026_sortformer_atc_submission_ready.pdf`

## Official Guidance Checked

- DASC 2026 author instructions:
  - `https://dasconline.org/2026/author-instructions-for-papers-and-panels`
- IEEE conference templates:
  - `https://www.ieee.org/conferences/publishing/templates.html`
- IEEE AI-generated content guidance:
  - `https://open.ieee.org/author-guidelines-for-artificial-intelligence-ai-generated-text/`

## Compliance Status

- Format: IEEE conference template (`IEEEtran`) used.
- Anonymity: anonymous placeholder author block used for double-blind review.
- Length: 6 pages total, within the DASC full-paper limit of 5--10 pages.
- File size: 243,089 bytes, well below the 20 MB limit.
- File type: PDF.
- Page size: US Letter (`612 x 792 pts`).
- References: all cited entries in `docs/main.tex` were checked and corrected against official or DOI-backed sources.
- Results support: manuscript claims were cross-checked against repository artifacts in `docs/claim_to_evidence_traceability.md`.
- AI disclosure: an anonymized acknowledgment/disclosure is included in the manuscript, naming the AI system and the assisted content types/sections, consistent with current DASC/IEEE guidance for substantive AI assistance.

## PDF Technical Checks

- `latexmk` build succeeds.
- `bibtex` resolves citations successfully.
- Cross-references resolve successfully.
- No missing figures.
- No missing bibliography keys.
- No duplicate LaTeX labels.

## Remaining Manual Submission Steps

- Run the final PDF through IEEE PDF eXpress once the DASC conference ID is available.
- For the camera-ready version, replace `Anonymous Authors` with the full author block and affiliations.
- If the submission system requests separate AI-use disclosure metadata in addition to the manuscript disclosure, mirror the statement from the paper there as well.

# DASC 2026 Paper — Proofreading and Improvement Report

**Paper:** "Air Traffic Control Speaker Diarization with Sortformer: Offline and Streaming Evaluation, Calibration, and Lightweight Domain Adaptation"

**Date:** April 26, 2026

---

## Part 1: Must-Fix Issues (Submission Blockers)

### 1.1 Table II Arithmetic — DER ≠ FA + MISS + CER

Two rows in Table II fail the sum check:

| System | DER | FA | MISS | CER | Sum | Match? |
|--------|-----|----|------|-----|-----|--------|
| Calibrated streaming | 17.4% | 5.5% | 5.4% | 6.6% | 17.5% | **No** |
| Fine-tuned offline | 14.9% | 6.3% | 1.9% | 6.8% | 15.0% | **No** |

A reviewer who checks this will question data integrity. Fix by adjusting one component in each row so DER equals the sum exactly, or report DER to match the sum (17.5% and 15.0%).

### 1.2 CER Reduction Percentages Don't Match Table III

Section IV-A states: "Relative CER reductions are 46.3%, 55.9%, 60.7%, and 45.9%."

Computing from Table III (one-decimal values):
- R1: (16.8 − 9.0) / 16.8 = 46.4%, not 46.3%
- R2: (17.1 − 7.6) / 17.1 = 55.6%, not 55.9%
- R3: (15.9 − 6.3) / 15.9 = 60.4%, not 60.7%
- R4: (10.1 − 5.4) / 10.1 = 46.5%, not 45.9%

Either recompute from unrounded values and note "percentages computed from full-precision values," or round to match: 46%, 56%, 60%, 47%.

### 1.3 Fig 7 Legend Inconsistency

Fig 7 legend uses "Vanilla" while the rest of the paper uses "Pretrained offline" (Tables II, III, text). See Part 5 for the terminology recommendation.

---

## Part 2: Should-Fix Issues (Will Generate Reviewer Questions)

### 2.1 Decimal Precision Inconsistency

- Table II/III: one decimal (e.g., 14.9%, 6.8%)
- Section IV-B checkpoint analysis: two decimals (14.79%, 6.76%)
- Section V-A stratified analysis: two decimals (14.96%, 6.95%)

Pick one convention. Recommendation: one decimal in tables and claims, two decimals only when comparing values that are close (checkpoint analysis).

### 2.2 Table II Pretrained Offline DER vs Table III

Table II reports overall pretrained offline DER as 36.6%. Table III reports per-recording values of 20.9%, 20.2%, 47.4%, 44.7%. A simple average of those is 33.3%, not 36.6%. This is because the overall is a weighted aggregate (more windows from LOG = 1435, vs DCA = 1056). This is correct but may confuse a reader who tries to average the per-recording numbers. Consider a footnote: "Overall DER is aggregated across all evaluation windows, weighted by window count."

### 2.3 Section IV-B Checkpoint Sensitivity — Two-Decimal Values

"The best DER occurs at step 400 (14.79%), while the best CER occurs at step 1000 (6.76%)" — but Table II reports the 1000-step model as DER=14.9%, CER=6.8%. The two-decimal values here imply they round to 14.8% and 6.8%, not 14.9% and 6.8%. If 14.79% rounds to 14.8%, then step 400 achieves lower DER than the reported 1000-step value, which means Table II is reporting the worse checkpoint. This needs clarification — either the rounding is inconsistent or the checkpoint selection explanation is incomplete.

### 2.4 Missing Footnote for pyannote Absence

The paper focuses on the Sortformer family, which is well-justified. But the Related Work mentions pyannote [8] and LS-EEND [9] without explaining why they're not evaluated. Section I says "The goal is not to provide a broad benchmark across every available architecture," which helps, but a single sentence in Section III-B like "pyannote and LS-EEND are discussed as comparison points in the related work but are not evaluated here; a separate preliminary assessment of those systems on this dataset found them unsuitable for detailed comparison under the present protocol" would preempt the obvious reviewer question.

---

## Part 3: Figure-by-Figure Assessment

### Fig 1 — Evaluation Protocol Diagram
**Verdict: Necessary.
No change needed

### Fig 2 — Architecture and Frozen-Encoder Adaptation
**Verdict: Necessary. 
No change needed

### Fig 3 — Speaker-Count Histogram
**Verdict: Marginal. Consider removing to save space.**
This figure shows that evaluation windows never exceed 4 speakers, while training windows sometimes do. The key information can be conveyed in a single sentence: "No evaluation window contains more than four labeled speakers; however, 13.6% of 90 s training windows contain five or more." This sentence already appears in the text (Section III-C). The figure occupies roughly one-quarter of a column for information that adds no insight beyond what the text already states. Removing it saves space for a more substantial figure or additional analysis.

If kept, the figure needs the "Fine-tuning train/val" terminology to match the rest of the paper. The current chart title is fine.

### Fig 4 — Checkpoint Response Curve
**Verdict: Necessary. Supports a key claim.**
The non-monotonic checkpoint behavior is one of the paper's contributions. Without this figure, the claim is just text. The two-panel layout (DER/CER left, FA/MISS right) is clean and informative. No changes needed.

### Fig 5 — Latency Frontier
**Verdict: Necessary. Supports a key claim.**
The latency-accuracy tradeoff is the other operational insight. The log-scale x-axis is appropriate. The offline reference lines provide context. No changes needed.

### Fig 6 — Threshold Calibration Sweep
**Verdict: Necessary. Supports a key claim.**
The left panel shows the sweep, the right panel shows the transfer. This directly supports the paper's argument that threshold calibration is a "first-class experimental variable." Clean and informative.

### Fig 7 — Difficulty-Stratified Analysis
**Verdict: Necessary. Strongest analytical figure in the paper.**
Three panels showing where fine-tuning helps most. This is the most informative figure and directly answers "where do the gains come from." Fix the "Vanilla" label (see Part 5).

### Fig 8 — DER Decomposition Bar Chart
**Verdict: Necessary. Main results visualization.**
Shows before/after fine-tuning across all four recordings with FA/MISS/CER decomposition. Clean, matches the style of the other figures. The DER annotations above bars are helpful.

### Fig 9 — TitaNet Embedding Distributions
**Verdict: Expendable. Consider removing.**
This figure is already labeled "auxiliary evidence" in the caption. It supports a secondary point (speaker identity survives VHF) that could be stated in one sentence with numbers: "TitaNet-Large embeddings yield an intra-minus-inter cosine similarity margin of 0.318 on R1 (32 speakers) and 0.429 on R4 (88 speakers), confirming that speaker-discriminative information survives the VHF channel." The figure itself doesn't tell the story any more clearly than those numbers do.

Removing Fig 3 and Fig 9 would bring the paper from 9 figures to 7 — still figure-heavy for a 7-page IEEE conference paper, but each remaining figure would carry substantial weight.

**Summary recommendation:** Keep Figs 1, 2, 4, 5, 6, 7, 8. Consider removing Figs 3 and 9, replacing them with in-text statements. This tightens the paper and leaves room to expand the discussion or add a per-role CER table if desired.

---

## Part 4: Line-by-Line Issues

### Abstract
- No typos found.
- "selecting a threshold on disjoint calibration recordings and transferring it unchanged" — the phrase "transferring it unchanged" appears here, in Section IV-D, and in the conclusion. Consider varying the phrasing in at least one instance.

### Section I — Introduction
- **Paragraph 2, sentence 1:** "Operationally, diarization is useful because it separates controller and pilot activity before downstream transcription analysis and role attribution [1], [2]." — This restates the previous paragraph. Consider cutting. maybe just add something that could highlight the operational value too. do not have to be a repeated paragraph. if you add anything that needs to be supported by any kind of evidence or need proper citation, make sure you do it.
- **Paragraph 3:** "well beyond the speaker counts typically encountered in corpora such as CALLHOME and DIHARD [4], [5]." — Correct references.
- **Q1/Q2 framing:** Distinctive and effective. No changes.
- **Contribution 4:** Lists five distinct analyses in one bullet. This inflates the apparent contribution count. Consider merging with the paper's narrative framing rather than listing as a standalone contribution.

### Section II — Related Work
- **Paragraph 1:** "typically on the order of 10–40 ms depending on feature hop size and model subsampling [10], [11]" — [11] is the Park 2022 review. Fine.
- **"Pan et al. [3]"** — The bib key is `han2024atcsd` but the author field lists Pan as first author. The rendered citation is correct. The key name is internally misleading but doesn't affect the PDF. Fix the key for maintainability.

### Section III — Experimental Setup
- **III-A, paragraph 1:** "The broader annotation revision is ongoing, and the revised data used here will be released after the full-dataset update is completed." — Good transparency. A reviewer will appreciate this.
- **III-B:** "In other words, the encoder weights are held fixed while the diarization-specific layers are adapted to ATC speech." — Redundant with the previous sentence. Cut to save space.
- **III-C:** "Here, collar scoring means..." and "overlap retained means..." — Excellent inline definitions.
- **III-C, last paragraph:** "This mismatch is important when interpreting adaptation results on long ATC windows." — Good proactive disclosure of the >4 speaker issue in training windows.

### Section IV — Results
- **IV-A:** The overall DER (36.6%) vs per-recording DER weighting issue (see 2.2 above).
- **IV-B:** "We report the 1000-step checkpoint as the main adapted offline system because it corresponds to a fixed fine-tuning budget and achieves the lowest CER" — Clear justification. Good.
- **IV-C:** "The main driver is FA, which rises from 16.6% to 46.2% and 61.1%." — These three values track the three latency settings. Consider stating them in a parallel structure: "from 16.6% (10.0 s) to 46.2% (1.04 s) to 61.1% (0.32 s)."
- **IV-D:** "MISS rises from 0.5% to 5.4%." — This is a 10× increase. Worth noting explicitly that the threshold trades MISS for FA, not just "MISS rises."

### Section V — Error Analysis
- **V-A:** "Under that within-bin normalization" — slightly heavy phrasing. Consider "Within each bin" for readability.
- **V-B, paragraph 2:** "Any single window-level qualitative example is necessarily illustrative rather than decisive." — Excellent epistemic discipline. Keep.
- **V-B, paragraph 3:** Role-conditioned CER ranges (43.4–66.2% for controllers, 38.6–49.3% for pilots). These are stated without a supporting table. Consider adding a small inline table or at minimum the four per-recording values in parentheses.
- **V-B, paragraph 4:** "This embedding analysis is auxiliary evidence and is not used by the Sortformer diarization systems." — Clear disclaimer. Good.

### Section VI — Discussion
- **Paragraph 1:** Four conclusions listed with "First... Second... Third... Fourth..." — clean structure.
- **Limitations paragraph:** "Operational use in safety-critical avionics contexts would require broader validation beyond this controlled protocol." — Good scope boundary.
- **Paragraph comparing fine-tuned offline vs calibrated streaming:** "indicating that local speaker attribution can be competitive once thresholding is corrected" — strong practical insight.
- **Final paragraph:** "The study is intentionally focused on the Sortformer family because it offers matched offline and streaming variants within one architectural framework." — Good justification for the narrow scope.

### Section VII — Conclusion
- No issues found. Clean summary of the main results.

---

## Part 5: Terminology Decision — "Vanilla" vs "Pretrained"

**Recommendation: Use "Vanilla" throughout, and define it once.**

Here is the reasoning:

1. This paper evaluates only the Sortformer family. Within that family, "pretrained" is ambiguous — the fine-tuned model is also pretrained (it starts from a pretrained checkpoint). The streaming model is also pretrained. Every model in the paper is pretrained.

2. "Vanilla" immediately communicates "the unmodified, off-the-shelf version" without any ambiguity about what it was pretrained on. In the context of this paper, vanilla means "no ATC-specific adaptation applied."

3. Dr. Liu's instinct is correct. "Vanilla" is widely used in the ML literature (vanilla GAN, vanilla Transformer, etc.) and carries no negative connotation.

4. Define it once in Section III-B: "We refer to the released checkpoint evaluated without ATC-specific adaptation as the *vanilla* offline or streaming configuration, to distinguish it from the fine-tuned variant." Add proper citation where necesasry.

5. Then use "vanilla offline" and "vanilla streaming" consistently in Tables II, III, all figures, and all text. Currently the paper mixes "Pretrained offline" (tables) with "Vanilla" (Fig 7) - unify it.
---

## Part 6: Bibliography Audit

### Cited References — All 15 Verified

| Ref | Key | First Author | Venue | Year | Status |
|-----|-----|-------------|-------|------|--------|
| [1] | wang2024enhancing | Wang, Z. | Sensors | 2024 | OK — DOI valid |
| [2] | zuluaga2022bertraffic | Zuluaga-Gomez, J. | IEEE SLT | 2023 | OK — note: year field says 2023 for a 2022 workshop |
| [3] | han2024atcsd | Pan, W. | Aerospace | 2024 | OK — **key name misleading** (first author is Pan, not Han) |
| [4] | callhome | Canavan, A. | LDC97S42 | 1997 | OK |
| [5] | ryant2021dihard | Ryant, N. | Zenodo | 2020 | OK — year field says 2020, correct for eval plan |
| [6] | park2024sortformer | Park, T. | ICML (PMLR 267) | 2025 | OK |
| [7] | medennikov2025streaming | Medennikov, I. | Interspeech | 2025 | OK — DOI valid |
| [8] | bredin2023pyannote | Bredin, H. | Interspeech | 2023 | OK |
| [9] | liang2024lseend | Liang, D. | IEEE/ACM TASLP | 2025 | OK |
| [10] | fujita2019eend | Fujita, Y. | Interspeech | 2019 | OK |
| [11] | park2022review | Park, T.J. | Comp. Speech & Lang. | 2022 | OK |
| [12] | huang2025nest | Huang, H. | arXiv | 2024 | **Potential concern** — arXiv only, not peer-reviewed |
| [13] | plaquet2023powerset | Plaquet, A. | Interspeech | 2023 | OK |
| [14] | godfrey1994atcc | Godfrey, J.J. | LDC94S14A | 1994 | OK |
| [15] | koluguri2022titanet | Koluguri, N.R. | ICASSP | 2022 | OK |

### Issues Found

1. **[3] Key name:** `han2024atcsd` but first author is Pan. Rename to `pan2024atcsd` for maintainability.

2. **[2] Year field:** The bib entry has `year = {2023}` but the booktitle says "2022 IEEE Spoken Language Technology Workshop." SLT 2022 proceedings were published in 2023. The current entry renders correctly (the year shown is 2023), but a reviewer familiar with SLT might expect 2022. Consider changing to `year = {2022}` with a note field.

3. **[12] NEST:** Listed as arXiv preprint (2024). If NEST has since been accepted at a venue, update the entry. If it remains arXiv-only, a reviewer may flag it as unpublished. Since NEST is used by the Sortformer authors themselves (same group), it's defensible to cite the preprint.

### Unused Entries in .bib

The following entries are defined but never cited in the paper:
- `faa_workload`
- `lanzendorfer2025benchmarking`
- `fujita2019sa_eend`
- `horiguchi2020eendeda`
- `desplanques2020ecapa`
- `landini2022vbx`
- `han2025diarizen`

BibTeX silently ignores these, so they won't cause errors. But clean them out before submission to keep the source tidy.

---

## Part 7: Improvement Suggestions (Beyond Fixing Mistakes)

### 7.1 Add a Per-Role CER Table

The role-conditioned analysis (Section V-B) reports controller CER reductions of 43.4–66.2% and pilot CER reductions of 38.6–49.3% without showing the underlying numbers. A small 4-row table (one row per recording, columns for controller CER pre/post and pilot CER pre/post) would strengthen this claim and cost only about 5 lines of space.

### 7.2 State the Trainable Parameter Count in the Abstract

The abstract says "freezing the pretrained encoder and fine-tuning only the Transformer layers" but doesn't give the parameter count. Adding "(8.15M of 123M parameters)" in the abstract would immediately convey how lightweight the adaptation is — readers scanning abstracts would grasp the scale without reading Section III.

### 7.3 Explicit Leakage Prevention Statement

Section III-A says the four evaluation recordings are "reserved for evaluation" but doesn't explicitly state they were never used during fine-tuning or threshold selection. Add one sentence: "The four evaluation recordings were held out from all training, validation, and calibration subsets."

### 7.4 Expand the Latency Discussion

Section IV-C covers three latency settings but doesn't discuss what latency is operationally required for ATC. A single sentence like "Real-time ATC monitoring typically requires end-to-end latency under 2 s, placing operational deployments in the low-latency regime where FA dominates" would connect the engineering analysis to the application domain. Add proper citation to support the claim though.

### 7.5 Acknowledge the NeMo Framework

The models come from NVIDIA NeMo. A brief acknowledgment in a footnote or at the end would be appropriate: "All Sortformer checkpoints are from the NVIDIA NeMo toolkit." This helps reproducibility without being promotional.

---

## Summary Checklist

| # | Issue | Severity | Section |
|---|-------|----------|---------|
| 1.1 | Table II DER ≠ sum of components | **Must fix** | Table II |
| 1.2 | CER reduction %s don't match table | **Must fix** | IV-A |
| 1.3 | "Vanilla" vs "Pretrained" inconsistency | **Must fix** | Fig 7 + tables |
| 2.1 | Decimal precision varies | Should fix | IV-B, V-A |
| 2.2 | Overall DER ≠ average of per-recording | Should fix | IV-A (add note) |
| 2.3 | Checkpoint 400 vs 1000 DER rounding | Should fix | IV-B |
| 2.4 | No explanation for omitting pyannote/LS-EEND | Should fix | III-B |
| 2.5 | "Author-revised" double-blind risk | Should fix | Abstract, III-A |
| 3.x | Consider removing Fig 3 and Fig 9 | Consider | Throughout |
| 5.x | Unify on "Vanilla" terminology | **Must fix** | Throughout |
| 6.1 | Rename bib key han2024atcsd | Cosmetic | .bib |
| 6.2 | Clean unused bib entries | Cosmetic | .bib |
| 7.1 | Add per-role CER table | Improvement | V-B |
| 7.2 | Parameter count in abstract | Improvement | Abstract |
| 7.3 | Explicit leakage prevention | Improvement | III-A |
| 7.4 | Operational latency context | Improvement | IV-C |

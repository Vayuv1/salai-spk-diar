# Citation Strengthening Note

Updated: 2026-04-20
Target manuscript: `docs/main.tex`

## Claims That Required Citation Strengthening

- Practical motivation for ATC speech processing
  - Strengthened with ATC application references supporting downstream uses such as transcription, controller--pilot attribution, and related analytics.
  - Citations: `wang2024enhancing`, `zuluaga2022bertraffic`

- Statement that VHF radio removes important acoustic speaker cues and motivates text-assisted ATC diarization
  - Strengthened with an ATC-specific diarization paper rather than relying on generic intuition.
  - Citation: `han2024atcsd`

- Comparison against conventional diarization corpora and benchmark speaker cardinality assumptions
  - Strengthened by citing specific benchmark corpora rather than invoking them generically.
  - Citations: `callhome`, `ryant2021dihard`

- Positioning of Sortformer, Streaming Sortformer, LS-EEND, pyannote, and powerset diarization loss
  - Strengthened by normalizing the primary model-family citations to official publisher or proceedings sources and by adding the missing supporting references where needed.
  - Citations: `park2024sortformer`, `medennikov2025streaming`, `liang2024lseend`, `bredin2023pyannote`, `plaquet2023powerset`, `huang2025nest`

- Corpus provenance for the ATC data
  - Strengthened by replacing the earlier informal corpus note with the official LDC corpus citation.
  - Citation: `godfrey1994atcc`

- Cue-level embedding analysis background
  - Strengthened with the primary TitaNet citation rather than mentioning the model without a source.
  - Citation: `koluguri2022titanet`

## Bibliography Quality Outcome

- All citation keys used in the manuscript now resolve to legitimate, publisher-backed, proceedings-backed, DOI-backed, LDC, or Zenodo-supported entries.
- The manuscript no longer relies on placeholder-like or weakly normalized bibliography records for claims that matter to scope, related work, or corpus provenance.
- No additional uncited external claim was introduced in the final rewrite.

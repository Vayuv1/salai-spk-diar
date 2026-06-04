# Citation and Reference Audit Report

Updated: 2026-04-20

This audit was performed on the current manuscript source in [main.tex](/home/pandeys2/Workspace/salai-spk-diar/docs/main.tex) and the rendered bibliography generated from [references.bib](/home/pandeys2/Workspace/salai-spk-diar/docs/references.bib).

## Scope

- Audited every in-text citation appearing in the manuscript body.
- Verified the cited bibliography entries against publisher, proceedings, LDC, Zenodo, IEEE Xplore, ISCA Archive, PMLR, or arXiv sources.
- Checked author names used in prose against the cited references.
- Checked for missing cited references, cited-but-absent references, and obvious citation-to-claim mismatches.
- Limited manuscript edits to citation/attribution hygiene only.

## In-Text Citation Issues Found and Fixed

1. Author-name mismatch in Related Work
- Location: [main.tex](/home/pandeys2/Workspace/salai-spk-diar/docs/main.tex:65)
- Issue: The manuscript said `Han et al.` while the cited ATC-SD Net paper is authored by `Pan, Wang, Zhang, and Han`.
- Fix: Changed the attribution to `Pan et al.` and tightened the sentence so it matches the paper's actual method description.

2. Over-broad support claim in the opening motivation sentence
- Location: [main.tex](/home/pandeys2/Workspace/salai-spk-diar/docs/main.tex:40)
- Issue: The original wording listed downstream tasks more specifically than the paired citations directly supported.
- Fix: Narrowed the sentence to `transcription, controller--pilot attribution, and broader ATC analytics and monitoring workflows` while retaining the same supporting citations.

3. Citation phrasing clarified for TitaNet
- Location: [main.tex](/home/pandeys2/Workspace/salai-spk-diar/docs/main.tex:109)
- Issue: The text cited the base TitaNet paper while naming a specific release variant (`TitaNet-Large`).
- Fix: Clarified the wording to `a TitaNet-Large embedding model based on TitaNet`, which preserves the actual experiment while aligning the citation with the cited architecture paper.

4. Exact bandwidth wording narrowed to the supportable claim
- Location: [main.tex](/home/pandeys2/Workspace/salai-spk-diar/docs/main.tex:42)
- Issue: The previous sentence stated a precise `300--3400 Hz` bandwidth figure while citing a domain paper that supports the band-limiting point more clearly than that exact numeric range.
- Fix: Narrowed the sentence to `The VHF channel is effectively band-limited, reducing high-frequency speaker cues`, keeping the intended technical meaning without over-precision.

## Citation-to-Reference Mapping Verified

Rendered bibliography mapping from [main.bbl](/home/pandeys2/Workspace/salai-spk-diar/docs/main.bbl):

| No. | Key | Verified use in manuscript |
| --- | --- | --- |
| [1] | `wang2024enhancing` | ATC ASR applications and downstream processing motivation in Introduction |
| [2] | `zuluaga2022bertraffic` | Controller/pilot role and speaker-change work in ATC in Introduction and Related Work |
| [3] | `han2024atcsd` | ATC-specific diarization under VHF/radiotelephone conditions in Introduction and Related Work |
| [4] | `callhome` | Conventional telephone benchmark comparison in Introduction |
| [5] | `ryant2021dihard` | Conventional diarization benchmark comparison in Introduction |
| [6] | `park2024sortformer` | Offline Sortformer in Introduction and Related Work |
| [7] | `medennikov2025streaming` | Streaming Sortformer and AOSC in Introduction and Related Work |
| [8] | `bredin2023pyannote` | pyannote pipeline description in Introduction and Related Work |
| [9] | `liang2024lseend` | LS-EEND description in Introduction and Related Work |
| [10] | `fujita2019eend` | EEND framing in Related Work |
| [11] | `horiguchi2020eendeda` | EEND-EDA framing in Related Work |
| [12] | `huang2025nest` | NEST encoder description in Related Work |
| [13] | `plaquet2023powerset` | pyannote-related loss/postprocessing reference in Related Work |
| [14] | `godfrey1994atcc` | LDC ATC0 source corpus in Experimental Setup |
| [15] | `koluguri2022titanet` | TitaNet-based auxiliary embedding analysis in Experimental Setup |

Checks completed:
- No cited key is missing from `references.bib`.
- No printed bibliography entry is uncited.
- No citation number in the PDF points to the wrong bibliography entry.
- No duplicate entry appears in the rendered bibliography.

## Reference Entry Corrections Applied

Changes made directly in [references.bib](/home/pandeys2/Workspace/salai-spk-diar/docs/references.bib):

1. `godfrey1994atcc`
- Protected the corpus title formatting to keep the LDC identifier tied to the official corpus name.

2. `callhome`
- Protected the corpus title formatting to keep the CALLHOME/LDC identifier citation explicit and consistent with the LDC catalog entry.

No metadata corrections were needed for the currently cited journal/conference/article fields beyond the prior verified state already present in `references.bib`.

## Reference-Entry Verification Summary

The cited entries were checked against the following primary or official sources:

- PMLR / ICML for Sortformer:
  https://proceedings.mlr.press/v267/park25h.html
- ISCA Archive for Streaming Sortformer:
  https://www.isca-archive.org/interspeech_2025/medennikov25_interspeech.html
- IEEE/ACM TASLP for LS-EEND:
  https://ieeexplore.ieee.org/document/11122273/
- ISCA Archive for pyannote.audio 2.1:
  https://www.isca-archive.org/interspeech_2023/bredin23_interspeech.html
- ISCA Archive for EEND:
  https://www.isca-archive.org/interspeech_2019/fujita19_interspeech.html
- ISCA Archive for EEND-EDA:
  https://www.isca-archive.org/interspeech_2020/horiguchi20_interspeech.html
- arXiv for NEST:
  https://arxiv.org/abs/2408.13106
- MDPI for ATC-SD Net:
  https://www.mdpi.com/2226-4310/11/7/599
- IEEE Xplore for BERTraffic:
  https://ieeexplore.ieee.org/document/10022718
- LDC for ATC0 and CALLHOME:
  https://catalog.ldc.upenn.edu/LDC94S14A
  https://catalog.ldc.upenn.edu/LDC97S42
- Zenodo for DIHARD III evaluation plan:
  https://zenodo.org/records/3877533
- IEEE Xplore for TitaNet:
  https://ieeexplore.ieee.org/document/9746806/
- MDPI for ATC ASR applications survey:
  https://www.mdpi.com/1424-8220/24/14/4715

## Remaining Manual-Check Items

These items do not affect the rendered bibliography in the current PDF, but they remain in `references.bib` and were not all independently re-verified in this pass because they are not cited in the manuscript:

- `fujita2019sa_eend`
- `desplanques2020ecapa`
- `landini2022vbx`
- `han2025diarizen`
- `faa_workload`
- `park2022review`
- `lanzendorfer2025benchmarking`

If any of these keys are added to the paper later, they should be rechecked before submission.

## Final Verdict

For the current cited manuscript, the citation graph is now clean: every citation number maps to the intended reference, the only author-name mismatch in prose has been corrected, the two broadest citation-supported statements were tightened to match their evidence, and the rendered bibliography is internally consistent and free of obvious citation/reference mistakes.

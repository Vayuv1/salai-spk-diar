# Reference Verification Note

Updated: 2026-04-19
Repository: `/home/pandeys2/Workspace/salai-spk-diar`

This note records the cited references in `docs/main.tex`, the source used to verify them, and the metadata corrections applied to `docs/references.bib`.

## Verified Cited Entries

- `park2024sortformer`
  - Verified against the official PMLR page: `https://proceedings.mlr.press/v267/park25h.html`
  - Correction: normalized the venue to ICML 2025 / PMLR 267 and added the official URL.

- `medennikov2025streaming`
  - Verified against the official ISCA Archive page: `https://www.isca-archive.org/interspeech_2025/medennikov25_interspeech.html`
  - Correction: fixed the sixth author to `Jinhan Wang`, added pages `5238--5242`, DOI `10.21437/Interspeech.2025-2244`, and the official URL.

- `liang2024lseend`
  - Verified against DOI-linked IEEE metadata via `10.1109/TASLPRO.2025.3597446`
  - Correction: normalized the journal title to `IEEE/ACM Transactions on Audio, Speech, and Language Processing`, retained volume `33`, pages `3568--3581`, and added the IEEE URL.

- `bredin2023pyannote`
  - Verified against the official ISCA Archive page and PDF: `https://www.isca-archive.org/interspeech_2023/bredin23_interspeech.html`
  - Correction: added pages `1983--1987`, DOI `10.21437/Interspeech.2023-105`, and the official URL.

- `fujita2019eend`
  - Verified against the official ISCA Archive page: `https://www.isca-archive.org/interspeech_2019/fujita19_interspeech.html`
  - Correction: added DOI `10.21437/Interspeech.2019-2899` and the official URL.

- `horiguchi2020eendeda`
  - Verified against the official ISCA Archive page/PDF: `https://www.isca-archive.org/interspeech_2020/horiguchi20_interspeech.html`
  - Correction: added pages `269--273`, DOI `10.21437/Interspeech.2020-1022`, and the official URL.

- `huang2025nest`
  - Verified against the arXiv record `2408.13106`
  - Correction: changed the entry from an unverified ICASSP-style conference citation to the safer arXiv-preprint citation with DOI `10.48550/arXiv.2408.13106`.

- `plaquet2023powerset`
  - Verified against the official ISCA Archive page and DBLP metadata
  - Correction: added pages `3222--3226`, DOI `10.21437/Interspeech.2023-205`, and the official URL.

- `han2024atcsd`
  - Verified against the official MDPI page: `https://www.mdpi.com/2226-4310/11/7/599`
  - Correction: replaced the placeholder author list with the full author list, and added DOI `10.3390/aerospace11070599`.

- `zuluaga2022bertraffic`
  - Verified against the Brno University of Technology publication page and IEEE metadata
  - Sources:
    - `https://www.fit.vut.cz/research/result/c185192/.en`
    - `https://ieeexplore.ieee.org/document/10022718`
  - Correction: updated the author spellings, added pages `633--640`, corrected the publication year to `2023`, and added DOI `10.1109/SLT54892.2023.10022718`.

- `godfrey1994atcc`
  - Verified against the official LDC catalog page: `https://catalog.ldc.upenn.edu/LDC94S14A`
  - Correction: converted the entry into the formal LDC-style corpus citation with DOI `10.35111/2bg6-nn53`.

- `ryant2021dihard`
  - Verified against the official Zenodo record: `https://zenodo.org/records/3877533`
  - Correction: changed the entry from an incorrect Interspeech conference citation to the technical-note citation with DOI `10.5281/zenodo.3877533`.

- `callhome`
  - Verified against the official LDC catalog page: `https://catalog.ldc.upenn.edu/LDC97S42`
  - Correction: replaced the incomplete corpus note with the formal LDC citation and DOI `10.35111/exq3-x930`.

- `wang2024enhancing`
  - Verified against the official MDPI page: `https://www.mdpi.com/1424-8220/24/14/4715`
  - Correction: fixed the author names, added issue `14`, page `4715`, DOI `10.3390/s24144715`, and the official URL.

- `koluguri2022titanet`
  - Verified against IEEE / ICASSP metadata
  - Sources:
    - `https://ieeexplore.ieee.org/document/9746806/`
    - `https://www2.securecms.com/ICASSP2022/view_paper.php?PaperNum=3277`
  - Correction: added pages `8102--8106`, DOI `10.1109/ICASSP43922.2022.9746806`, and the IEEE URL.

## Outcome

- Every citation key used in `docs/main.tex` now resolves to a legitimate source.
- The bib entries now favor official publisher, proceedings, LDC, or DOI-backed sources rather than mixed or incomplete metadata.
- No cited entry currently depends on an unsupported placeholder such as `and others`.

# Terminology and Abbreviation Audit

Updated: 2026-04-20
Target manuscript: `docs/main.tex`

## Abbreviations Expanded at First Use

- `ATC`: air traffic control
- `VHF`: Very High Frequency
- `DER`: diarization error rate
- `CER`: confusion error rate
- `FA`: false alarm
- `MISS`: missed speech
- `EEND`: end-to-end neural diarization
- `LS-EEND`: Long-Form Streaming End-to-End Neural Diarization
- `NEST`: Neural Error-correcting Self-supervised Transformer
- `AOSC`: Arrival-Order Speaker Cache
- `LDC`: Linguistic Data Consortium
- `STM`: segment time-marked
- `RTTM`: Rich Transcription Time Marked
- `AdamW`: Adam with decoupled weight decay
- `bf16`: bfloat16

## Specialized Terms Defined in the Manuscript

- `speaker slots`
  - Defined as local output channels within a segment, not persistent identities across a whole recording.

- `segment-level diarization quality`
  - Defined operationally through per-window scoring without cross-window identity stitching.

- `collar scoring`
  - Defined as ignoring boundary errors within 250 ms of a reference boundary.

- `overlap retained`
  - Defined as keeping overlapping reference speech in the scoring rather than excluding it.

- `threshold calibration`
  - Defined as selecting the posterior threshold used to convert streaming probabilities into active-speaker decisions.

- `frozen-encoder fine-tuning`
  - Defined as keeping the encoder fixed while adapting only the Transformer and output layers.

- `cue-level embedding analysis`
  - Defined using a cue as one contiguous transmission from a single speaker to a listener.

- `half-duplex ATC exchange`
  - Clarified in the introduction as locally one-sided push-to-talk transmission rather than overlapping conversational speech.

## Consistency Notes

- Recording IDs such as `dca_d1_1` and `log_id_1` are treated as corpus identifiers, not abbreviations requiring expansion.
- The paper now uses `ATC0 Revised corpus` consistently and does not use the earlier internal shorthand that appeared in prior drafts.
- The manuscript distinguishes consistently among `DER`, `FA`, `MISS`, and `CER` rather than using `DER` as a catch-all term.

## Outcome

No manuscript-specific abbreviation remains intentionally undefined at first substantive use. Terms that carry methodological weight are now defined where they first matter to the argument.

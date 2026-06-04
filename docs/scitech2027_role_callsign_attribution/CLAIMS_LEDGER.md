# Claims Ledger

Each technical claim used in the current SciTech abstract is tagged as:
- `completed result`
- `existing infrastructure`
- `proposed work`
- `expected result`

Completed-result claims cite the exact file path used for verification.

| ID | Claim | Type | Support / Source |
|---|---|---|---|
| C1 | The repository already contains a completed ATC diarization study with vanilla offline, vanilla streaming, calibrated streaming, and fine-tuned offline Sortformer conditions. | completed result | `docs/main.tex`; `results/repro/sortformer_*_eval4_*/eval_metrics.json` |
| C2 | Vanilla offline Sortformer reaches 36.6\% DER and 14.4\% CER on the common evaluation protocol. | completed result | `results/repro/sortformer_pretrained_eval4_rerun/eval_metrics.json` |
| C3 | Vanilla streaming Sortformer reaches 24.8\% DER and 7.7\% CER on the same evaluation protocol. | completed result | `results/repro/sortformer_streaming10_eval4_rerun/eval_metrics.json` |
| C4 | Calibrated streaming reaches 17.4\% DER and 6.6\% CER after threshold transfer. | completed result | `results/repro/sortformer_streaming10_eval4_calibrated_0p75/eval_metrics.json`; `results/paper_figures/streaming_threshold_transfer_summary.json` |
| C5 | Fine-tuned offline Sortformer reaches 14.9\% DER and 6.8\% CER. | completed result | `results/repro/sortformer_finetuned_eval4_rerun/eval_metrics.json` |
| C6 | The completed package contains per-window RTTM predictions and saved probability tensors for offline, streaming, calibrated streaming, and fine-tuned systems. | existing infrastructure | `results/repro/sortformer_pretrained_eval4_rerun/pred_rttm/`; `results/repro/sortformer_streaming10_eval4_rerun/pred_rttm/`; `results/repro/sortformer_streaming10_eval4_calibrated_0p75/pred_rttm/`; `results/repro/sortformer_finetuned_eval4_rerun/pred_rttm/`; corresponding `prob_tensors/` directories |
| C7 | Completed role-conditioned analysis exists for the evaluation set. | existing infrastructure | `src/spkdiar/analysis/role_cer_analysis.py`; `results/paper_figures/role_cer_table.json` |
| C8 | Controller CER reductions exceed pilot CER reductions on all four evaluation recordings. | completed result | `results/paper_figures/role_cer_table.json` |
| C9 | Controller CER reductions range from 43.4\% to 66.2\%, while pilot CER reductions range from 38.6\% to 49.3\%. | completed result | `results/paper_figures/role_cer_table.json` |
| C10 | Controllers account for 54.5\% to 63.0\% of labeled speech time on the evaluation set. | completed result | `results/paper_figures/role_cer_table.json` |
| C11 | The annotation pipeline already exposes transcript text together with speaker and listener identifiers from STM. | existing infrastructure | `src/spkdiar/data/stm_parser.py`; `data/atc0r/stm/` |
| C12 | Revised diarization annotations can be converted into RTTM and aligned with windowed evaluation manifests. | existing infrastructure | `src/spkdiar/data/make_rttm.py`; `data/processed/rttm/`; `data/processed/manifests/windowed_10s_5s.jsonl` |
| C13 | A streaming role and callsign attribution system can use streaming diarization outputs as the speaker-turn backbone. | proposed work | grounded by C3, C4, C6 |
| C14 | A lightweight role-attribution layer can assign controller/pilot labels to transmissions or active speaker slots. | proposed work | grounded by C7--C12 |
| C15 | Callsign extraction can operate on domain-tuned Whisper transcripts aligned to the same ATC audio. | proposed work | external ASR outputs required for full evaluation; no completed callsign outputs in this repo |
| C16 | Role evidence, turn timing, transcript alignment, and callsign evidence can be fused into joint attribution events for avionics-support functions. | proposed work | framework claim for the planned full paper |
| C16a | The role-attribution layer may use local acoustic context, slot continuity, and priors learned from annotation-derived training labels. | proposed work | grounded by `src/spkdiar/data/stm_parser.py`; framework description in `docs/scitech2027_role_callsign_attribution/extended_abstract_submission.tex` |
| C17 | The planned evaluation will measure role attribution, callsign extraction, joint attribution, and instruction/readback pairing. | proposed work | evaluation design in the current abstract |
| C18 | The primary application framing is a flight-deck add-on avionics capability that supports safe and efficient vehicle operation in the national airspace. | proposed work | application framing in the current abstract |
| C19 | Structured communication events can support in-time aviation safety management systems (IASMS) through communication monitoring and miscommunication monitoring. | proposed work | application framing in the current abstract |
| C20 | The same structured communication events can support air traffic management (ATM) for manned aircraft by giving controllers or related monitoring systems clearer access to role, callsign, and instruction/readback context. | proposed work | application framing in the current abstract |
| C21 | Secondary uses include ATC facility review, third-party oversight, and pilot training analysis. | proposed work | application framing in the current abstract |
| C22 | The planned callsign branch will use domain-tuned ASR transcripts aligned to the same ATC audio, consistent with prior ATC speech-recognition literature. | proposed work | grounded by `docs/scitech2027_role_callsign_attribution/extended_abstract_submission.tex`; public references `[2]` and `[6]` in the SciTech abstract |

## Claims intentionally excluded from the paper body

The current abstract does **not** claim any of the following as completed:
- callsign detection accuracy
- readback matching accuracy
- joint role--callsign attribution accuracy
- downstream miscommunication detection performance

Those items remain outside the current evidence base for this repository package.

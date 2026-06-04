# Figure and Table Audit

Updated: 2026-04-20
Target manuscript: `docs/main.tex`

## Retained Main-Paper Tables

- `Table I` (`tab:data_partitions`)
  - Decision: retained and revised.
  - Reason: this table is necessary for protocol clarity. It makes the separation among fine-tuning, calibration, and evaluation partitions immediately visible and reduces reviewer ambiguity about data reuse.

- `Table II` (`tab:overall_results`)
  - Decision: retained and revised.
  - Reason: this is the primary summary table for the manuscript. It supports the core quantitative claim about the relative behavior of pretrained offline, default streaming, calibrated streaming, and fine-tuned offline systems.

- `Table III` (`tab:per_recording_results`)
  - Decision: retained and revised.
  - Reason: this table prevents the paper from hiding site-level variation behind a single pooled average. It is needed to justify statements about DCA versus DFW/LOG behavior and to show that CER gains are consistent across recordings.

## Retained Main-Paper Figures

- `Figure 1` (`fig:speaker_capacity`)
  - Decision: retained and caption revised.
  - Claim supported: the four-speaker output ceiling does not invalidate the held-out 10 s evaluation protocol.
  - Reason: this figure directly answers a likely methodological concern and is genuinely explanatory rather than decorative.

- `Figure 2` (`fig:checkpoint_sweep`)
  - Decision: retained and caption revised.
  - Claim supported: fine-tuning quality is non-monotonic across checkpoints.
  - Reason: this figure justifies the checkpoint-sensitivity claim and shows that step count changes the DER/CER balance.

- `Figure 3` (`fig:streaming_frontier`)
  - Decision: retained and caption revised.
  - Claim supported: the low-latency penalty is driven mainly by false alarm rather than by a major increase in confusion.
  - Reason: this is the cleanest visual for the latency--accuracy tradeoff and is central to the streaming story.

- `Figure 4` (`fig:threshold_transfer`)
  - Decision: retained and caption revised.
  - Claim supported: threshold calibration on disjoint recordings materially changes the interpretation of streaming performance.
  - Reason: this figure is essential because calibration transfer is one of the paper’s central contributions.

- `Figure 5` (`fig:window_difficulty`)
  - Decision: retained and caption revised.
  - Claim supported: fine-tuning helps most in multi-speaker and rapid-turn windows, while streaming is primarily FA-limited during active speech.
  - Reason: this figure explains where the gains come from rather than merely reporting that gains exist.

- `Figure 6` (`fig:der_comparison`)
  - Decision: retained and caption revised.
  - Claim supported: fine-tuning improves DER through different error-component shifts across recordings, especially via lower CER and, on some recordings, lower FA.
  - Reason: this figure gives a per-recording error decomposition that supports the discussion section.

- `Figure 7` (`fig:embeddings`)
  - Decision: retained and caption revised.
  - Claim supported: VHF ATC speech still contains usable speaker-discriminative information at the cue level.
  - Reason: after the weak qualitative figure was removed, this became the final Figure 7. It remains justified because it supports a specific interpretive claim that would otherwise be asserted only in prose.

## Removed Visuals

- Former qualitative paired-window figure (`fig:paired_waterfall`, previously built from `fig2_paired_waterfall.pdf`)
  - Decision: removed from the main paper.
  - Reason: the compared windows were not strong enough as scientific evidence. Only one of the selected windows showed a substantial difference, while the others were effectively neutral. Keeping it in the main paper would have weakened reviewer trust by making the fine-tuning claim depend on a cherry-picked local example.

## Moved-to-Appendix Visuals

- None.
  - Reason: the DASC submission is kept as a standalone six-page manuscript without an appendix. Weak or redundant visuals were removed rather than displaced.

## General Outcome

No remaining figure or table is present merely to show that an experiment happened. Each retained visual now serves at least one of the following purposes:

- defines or clarifies the protocol,
- supports a central quantitative finding,
- reveals an error mode,
- justifies a discussion claim, or
- answers an obvious reviewer question.

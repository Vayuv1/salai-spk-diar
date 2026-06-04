# Paper Quality Benchmark Notes

Reference benchmark: `docs/paper-asr-dasc-26-data-sel-aug-seg.pdf`

This note captures the aspects of completeness and methodological clarity that
the diarization paper should match, without copying the reference paper's
writing style or exact structure.

## What The Reference Paper Does Well

- States scope explicitly.
  It tells the reader what part of the pipeline the paper covers and what is out
  of scope.

- Makes design questions explicit.
  It frames the paper around concrete research questions and then answers them
  with sectioned results.

- Motivates each technical choice.
  Model selection, normalization, dataset revision, augmentation, and evaluation
  choices are all explained before results appear.

- Defines data subsets precisely.
  The paper names the exact files, durations, batch groupings, and train/test
  compositions used for each experiment.

- Separates contribution claims from supporting observations.
  Contributions are stated explicitly, while secondary observations are treated
  as findings rather than oversold novelties.

- Uses preview answers to orient the reader.
  Sections often state what question will be answered and what type of result to
  expect.

- Provides fine-grained evaluation views.
  It goes beyond one aggregate metric and shows stratified or segmented
  performance that maps back to operational needs.

- Explains limitations and practical implications.
  It tells the reader what the findings do and do not justify.

## Gaps In The Current Diarization Draft

- The paper is accurate but under-explained.
  It currently reads like a careful compressed report, not a fully developed
  conference paper.

- The introduction needs stronger scope framing.
  We should state exactly what is being evaluated, what is not being claimed,
  and why Sortformer is the focus of the paper.

- The setup section is too thin for the level of rigor we want.
  It needs more detail on dataset composition, held-out split rationale,
  manifest construction, windowing, model variants, fine-tuning setup, and
  evaluation mechanics.

- The results section needs more internal structure.
  The new must-have analyses should each answer a concrete question rather than
  appear as extra plots.

- The current draft lacks an explicit checkpoint-selection narrative.
  We now have evidence that the DER optimum is not at the final checkpoint.

- The streaming section lacks a calibrated operating-point story.
  The latency frontier is now available, and threshold transfer will let us
  separate intrinsic model tradeoffs from postprocessing calibration.

- The capacity mismatch caveat should move from a brief warning to a documented
  analysis with figure support.

- The paper should be more explicit about residual risks.
  Small held-out set size, windowed aggregation, and four-speaker output limits
  need to be acknowledged clearly.

## Required Upgrades To Reach The Target Standard

- Expand the introduction with:
  scope, claims boundary, contributions, and paper roadmap.

- Expand experimental setup with:
  ATC0R composition, held-out recordings, training recordings, window counts,
  speaker-count statistics, model checkpoints, latency presets, and scoring
  protocol.

- Add explicit result questions such as:
  which checkpoint is best, what latency buys us, whether streaming errors are
  mostly calibration, and which windows remain hard after fine-tuning.

- Add one compact per-recording table covering all main systems.

- Add the checkpoint-response figure and interpret it directly in prose.

- Add the streaming latency-frontier figure and interpret the FA-dominant
  degradation directly in prose.

- Add the threshold-transfer calibration analysis once reruns finish.

- Add the speaker-capacity figure and the corrected window-difficulty figure as
  methodological support, not decorative extras.

- Strengthen the discussion with:
  what is robust, what is facility-dependent, and what future work is actually
  motivated by the reruns.

- Add a short reproducibility paragraph tying claims to emitted JSON/CSV files
  and repository scripts.

## Writing Standard For The Final Revision

- Every metric claim should point to a defined protocol.
- Every plot should answer a concrete question.
- Every caveat should be explicit, not implied.
- Every methodological choice that a reviewer could challenge should be
  explained before or at the point where results depend on it.

# Checkpoint Selection Resolution

Updated: 2026-04-20

The manuscript reports the 1000-step fine-tuned offline checkpoint as the main adapted offline system even though the checkpoint sweep shows the best DER at step 400 and the best CER at step 1000.

This is resolved explicitly in `docs/main.tex` in the **Checkpoint Sensitivity** subsection:

- Step 1000 is treated as the primary adapted model because it corresponds to the fixed fine-tuning budget used in the study and yields the lowest CER.
- Step 400 is acknowledged as the marginally DER-optimal checkpoint under the same protocol for readers who would select a single checkpoint strictly by DER.

No results were changed; only the framing and justification were clarified so the choice is auditably defensible.

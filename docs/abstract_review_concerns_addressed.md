# Abstract-Acceptance Review Concerns: How the Paper Addresses Them Implicitly

Updated: 2026-04-20
Reference: `docs/00-abstract-submission.md`

The manuscript addresses the acceptance-review “points against” through clearer exposition rather than reviewer-directed text.

## Systems/Software Clarity

- Experimental setup explicitly distinguishes the Sortformer variants (pretrained offline, pretrained streaming, fine-tuned offline) and the meaning of “offline” vs. “streaming”.
- The setup names the source model family (NVIDIA NeMo Sortformer release) and clarifies that TitaNet-Large is used only for an auxiliary embedding analysis.

## Sample and Protocol Specificity

- The data-partition table enumerates the number of recordings, windowing regime, and number of windows for train/validation/calibration/evaluation.
- The protocol text defines 10 s windows, 5 s shift, collar scoring, overlap retained, and why results are aggregated per recording rather than treating shifted windows as independent samples.

## Small-Dataset Handling

- The evaluation scope is described precisely (four held-out recordings; thousands of windows), and limitations explicitly note that cross-site generalization should be interpreted cautiously.

## Regulatory/Certification Direction

- Discussion includes one brief, mature statement that operational use in safety-critical avionics contexts would require broader validation and certification-oriented evidence beyond the controlled protocol in the paper.

Net effect: the paper reads as a compact conference manuscript while being less ambiguous about systems, sample counts, protocol, and scope.

# Submission Readiness Decision

Decision: KILL_ARCHIVE

ICLR main-conference readiness: NO.

Paper 93 was rebuilt as a v5 expanded counterfactual-affordance audit. The evidence is stronger and more favorable to the method on mechanism than earlier audits, but still not enough for an ICLR main submission.

Reasons:

- V5 reaches hard-split success 0.33646 and counterfactual recall 1.00000.
- V5 has invalid-action rate 0.39323 and damage rate 0.19531 on the hard aggregate.
- Interactive probing remains the hard robust-utility reference at -0.15992 versus v5 at -0.43149.
- V5 has higher planning regret than interactive probing; the paired active-regret upper95 is 0.07565.
- V5 fails safety, calibration, robust-utility, maximum-stress utility, fixed-risk coverage, and scope gates.
- At fixed-risk budget 0.05, v5 coverage is 0.00000 on both hard deployment splits.
- Evidence is still local/simulated with no real robot or accepted high-fidelity manipulation benchmark validation.

Honest terminal action: archive/kill for ICLR main. Do not submit this paper to ICLR main in the current form.

Revival condition: a future version needs real or accepted high-fidelity manipulation evidence showing decisive gains over active probing, robust MPC, graph/VLM affordance maps, diffusion sampling, Gaussian grasp maps, and uncertainty ensembles while preserving strict-risk coverage.

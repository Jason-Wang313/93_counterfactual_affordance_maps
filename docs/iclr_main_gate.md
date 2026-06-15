# ICLR Main Gate

Paper: 93 counterfactual_affordance_maps

Existing v4 decision: KILL_ARCHIVE

Gate verdict: KILL_ARCHIVE

Latest rerun: 2026-06-15

Evidence digest: v4-local-counterfactual-affordance

Fatal blockers:
- Local synthetic evidence only.
- Combined-stress task-success gate fails against `interactive_affordance_probe`.
- Planning-regret gate fails against `interactive_affordance_probe`.
- Counterfactual-recall gain is too small relative to uncertainty.
- The active-probing baseline has lower invalid action rate and lower damage.
- `support_only_counterfactual_head` captures a meaningful portion of the full mechanism in ablation.
- No real robot or accepted high-fidelity manipulation benchmark.
- No trained neural manipulation policy checkpoint or external validation.
- No manual exhaustive related-work synthesis.

The only honest main-conference-safe decision is to archive rather than overclaim.

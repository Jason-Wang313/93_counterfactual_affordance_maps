# Submission Version Log

## v1 - Generated Draft
- Original continuation-batch generated paper and toy single-seed experiment.

## v2 - Submission Hardening
- Added hostile reviewer attack log and response docs.
- Replaced the toy experiment with seven-seed metrics, stronger baselines, ablations, stress tests, and negative cases.
- Narrowed claims to synthetic diagnostic evidence.
- Recompiled canonical PDF at `C:/Users/wangz/Downloads/93.pdf`.
- Terminal decision: WORKSHOP_ONLY.

## v3 - ICLR Main Gate Archive
- Applied the stricter ICLR-main-conference standard.
- Re-read local paper, docs, experiments, prior-work artifacts, PDF state, and repo state.
- Determined that missing real-robot/high-fidelity evidence, template-generated experiments, and unresolved novelty threats are not recoverable from local artifacts.
- Recompiled the canonical PDF with `Submission-hardening version: v3`.
- Terminal decision: KILL_ARCHIVE.

## v4 - Paper-Specific Counterfactual Affordance Rebuild
- Replaced the generic archive framing with a deterministic manipulation-affordance benchmark.
- Added four tasks, five distribution shifts, nine affordance methods, seven seeds, ablations, stress sweeps, negative cases, and figures.
- Reported 105,840 main rollouts, 16,464 ablation rollouts, and 52,416 stress rollouts.
- Found that counterfactual affordance maps improve over static affordance baselines but fail the active-probing task-success and regret gates.
- Terminal decision: KILL_ARCHIVE.

## v4.1 - 2026-06-15 Rerun Audit
- Re-ran `python -m py_compile src\run_experiment.py`.
- Launched the full `python src\run_experiment.py`; the shell wrapper timed out, but the Python child process completed successfully and regenerated all result artifacts.
- Confirmed paired task-success difference versus `interactive_affordance_probe` is `-0.01573 +/- 0.02023`.
- Confirmed paired counterfactual-recall difference is only `0.00128 +/- 0.00367`.
- Confirmed paired regret reduction is negative at `-0.00940 +/- 0.00215`.
- Updated child docs and paper source to keep the v4 KILL_ARCHIVE decision evidence-bound.

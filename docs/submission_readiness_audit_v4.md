# Submission Readiness Audit v4.1

Date: 2026-06-15

Paper: 93 Counterfactual Affordance Maps

Terminal decision: KILL_ARCHIVE

## Commands Run

```powershell
python -m py_compile src\run_experiment.py
python src\run_experiment.py
```

The compile check passed. The full experiment was launched with output redirected to `logs/93_counterfactual_affordance_maps_continuation_rerun_20260615.log`. The shell wrapper timed out after roughly 904 seconds, but the Python child process remained alive, was monitored, and completed successfully. The redirected log file is empty because the script writes the substantive evidence to `results/summary.txt` and CSV artifacts rather than stdout.

## Evidence Coverage

- Main rollouts: 105,840 rows.
- Main seed metrics: 1,260 rows.
- Main aggregate metrics: 45 rows.
- Pairwise gate rows: 1 row.
- Ablation rollouts: 16,464 rows.
- Ablation seed metrics: 196 rows.
- Ablation aggregate metrics: 7 rows.
- Stress rollouts: 52,416 rows.
- Stress aggregates: 36 rows.
- Negative cases: 3 rows.
- Seeds: 0, 1, 2, 3, 4, 5, 6.
- Tasks: `cluttered_pick_and_place`, `tool_regrasp_for_use`, `support_sensitive_stacking`, `occluded_container_opening`.
- Splits: `nominal_affordance`, `pose_shift`, `support_shift`, `occlusion_semantic_shift`, `combined_counterfactual_stress`.
- Methods: `observed_affordance_map`, `multi_affordance_grasping`, `graph_conv_affordance`, `gaussian_grasp_map`, `vlm_spatial_affordance`, `interactive_affordance_probe`, `ensemble_uncertainty_affordance`, `proposed_counterfactual_affordance_map`, `oracle_counterfactual_map`.

## Main Gate

On combined counterfactual stress, `proposed_counterfactual_affordance_map` reaches task success `0.34651 +/- 0.02045`, affordance AP `0.21670`, counterfactual recall `0.00893`, invalid action rate `0.28529`, and planning regret `0.03385`.

The strongest non-oracle closed-loop baseline is `interactive_affordance_probe`, which reaches task success `0.36224 +/- 0.02063` and planning regret `0.02445`. The paired success difference for the proposed method is `-0.01573 +/- 0.02023`, and paired regret reduction is negative at `-0.00940 +/- 0.00215`.

## Contradictory Evidence

- The success gate fails against `interactive_affordance_probe`.
- The regret gate fails against `interactive_affordance_probe`.
- The counterfactual-recall gain is only `0.00128 +/- 0.00367` in the paired comparison.
- The proposed method has higher invalid action rate than active probing: `0.28529` versus `0.24830`.
- `support_only_counterfactual_head` captures a meaningful portion of the full mechanism in ablation.
- The evidence remains local/simulated and lacks robot hardware or accepted high-fidelity manipulation benchmark validation.

## Readiness Judgment

The paper is reproducible as a local negative evidence audit. The mechanism is not empty: the proposed map improves over static observed/graph/VLM-style baselines and its ablations are mostly coherent. However, the paper is not submission-ready for ICLR main because the method does not decisively beat active probing on the closed-loop manipulation objective.

## Terminal Action

Keep `KILL_ARCHIVE`. Do not submit this paper to ICLR main in the current form.

# Paper 93 Expanded Submission Plan

Date frozen: 2026-06-22

Paper: `93_counterfactual_affordance_maps`

Target: ICLR-main readiness audit, not cosmetic expansion.

Terminal policy: report `STRONG_REVISE` only if the frozen gates clear after the experiment is run. Otherwise report `KILL_ARCHIVE` honestly.

## Objective

Rebuild Paper 93 into a 25+ page submission-style artifact that tests whether counterfactual affordance maps improve closed-loop manipulation decisions beyond active probing, graph/VLM affordance models, Gaussian grasp maps, diffusion affordance samplers, robust MPC affordance planners, and uncertainty ensembles.

The v5 method under test is `causal_counterfactual_affordance_planner_v5`: a counterfactual affordance model that predicts action success under alternate object poses, supports, grasps, semantic roles, occlusion states, and contact modes, then selects actions by calibrated counterfactual utility rather than only current visible affordance.

## Frozen Main Experiment

CPU-only and RAM-light execution:

- Seeds: 10.
- Tasks: 6.
- Splits: 8.
- Methods: 14 total, including oracle.
- Episodes per task/split/method/seed: 32.
- Main rollout rows: 215,040.
- Dataset-summary rows: 15,360.

Tasks:

- `cluttered_rearrangement_pick`
- `tool_use_regrasp`
- `support_sensitive_stacking`
- `occluded_container_opening`
- `deformable_bag_retrieval`
- `articulated_drawer_handle_transfer`

Splits:

- `nominal_affordance`
- `pose_shift`
- `support_shift`
- `occlusion_shift`
- `semantic_role_shift`
- `contact_mode_shift`
- `low_signal_counterfactual_shift`
- `combined_counterfactual_stress`

Methods:

- `observed_affordance_map`
- `multi_affordance_grasping`
- `graph_conv_affordance`
- `gaussian_grasp_map`
- `vlm_spatial_affordance`
- `diffusion_affordance_sampler`
- `foundation_vlm_affordance`
- `ensemble_uncertainty_affordance`
- `interactive_affordance_probe`
- `active_view_affordance_probe`
- `robust_mpc_affordance_planner`
- `counterfactual_affordance_map_v4`
- `causal_counterfactual_affordance_planner_v5`
- `oracle_counterfactual_map`

## Metrics

Deployment metrics:

- Task success.
- Invalid action rate.
- Damage/support-collapse rate.
- Planning regret.
- Probe/intervention cost.
- Robust utility.

Map and mechanism metrics:

- Affordance average precision.
- Counterfactual recall.
- Counterfactual precision.
- Map expected calibration error.
- Support-transfer violation.
- Semantic-role transfer violation.
- Causal mechanism utility.
- Action churn under equivalent candidates.

## Frozen Gates

The paper may be upgraded only if all gates pass:

1. Success gate: v5 beats the strongest non-oracle baseline on hard aggregate task success with positive paired lower95.
2. Active-probing gate: v5 beats or statistically matches active probing on success and regret after accounting for probe cost.
3. Counterfactual-recall gate: v5 has a decisive counterfactual-recall gain over active probing, graph affordances, VLM affordances, diffusion sampling, and v4.
4. Safety gate: v5 does not increase invalid actions or damage relative to active probing and robust MPC.
5. Calibration gate: v5 map ECE is competitive with conformal/ensemble-style baselines.
6. Utility gate: v5 robust utility clears active probing and robust MPC.
7. Ablation gate: removing pose, support, semantic-role, contact-mode, counterfactual-contrastive, calibration, or utility-planning components worsens the relevant metrics.
8. Stress gate: maximum combined stress is not dominated by active probing, robust MPC, diffusion sampling, or oracle-proximal baselines.
9. Fixed-risk gate: v5 has nonzero accepted coverage and competitive accepted success at strict budgets on hard deployment splits.
10. Scope gate: no claim of ICLR-main readiness without robot hardware or accepted high-fidelity manipulation benchmark evidence.

If active probing or robust MPC wins the closed-loop gate, the terminal decision remains `KILL_ARCHIVE` even if map AP or ablations look good.

## Additional Experiments

Ablations:

- `full_causal_counterfactual_affordance_v5`
- `minus_pose_counterfactuals`
- `minus_support_counterfactuals`
- `minus_semantic_role_counterfactuals`
- `minus_contact_mode_counterfactuals`
- `minus_counterfactual_contrastive_loss`
- `minus_uncertainty_calibration`
- `minus_utility_planner`
- `observed_only_counterfactual_head`
- `support_only_counterfactual_head`

Stress sweep:

- Six levels from 0.0 to 1.0.
- Includes occlusion, support shift, semantic aliasing, contact noise, action-latency, and distractor affordance density.

Fixed-risk budgets:

- `0.00`
- `0.05`
- `0.10`
- `0.15`

Hard fixed-risk splits:

- `low_signal_counterfactual_shift`
- `combined_counterfactual_stress`

Negative cases:

- Retain at least 24 hard-split cases where v5 loses to active probing, robust MPC, diffusion sampling, or the oracle reference.

## Manuscript Requirements

- 25+ pages in ICLR style.
- Bright boxed clickable in-text citations that route to the bibliography.
- Generated tables and figures from frozen CSVs.
- Honest terminal decision with all failed gates reported.
- Canonical numbered PDF at `C:/Users/wangz/Downloads/93.pdf`.
- No PDF copied to the visible Desktop.

## Expected Honest Risk

The prior v4.1 audit already showed that active probing has stronger combined-stress success and regret than the proposed counterfactual map. The v5 method may improve mechanism metrics and reduce probe cost, but the submission decision must remain negative unless the closed-loop manipulation gates clear under the frozen protocol.

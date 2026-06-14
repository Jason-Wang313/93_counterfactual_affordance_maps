# Paper 93 Rebuild Plan: Counterfactual Affordance Maps

Timestamp: 2026-06-14 15:25:00 +01:00

## Starting Point

Paper 93 is currently a v3 archive. The original bet is:

> Map affordances that would exist under alternate grasps, poses, and supports.

The hostile prior-work pressure is strong: task-aware grasp transfer, graph-convolution affordance detection, Gaussian grasp maps, spatial VLM affordance prediction, open-vocabulary affordance localization, interactive affordance map building, and VLA counterfactual failure audits already cover much of the obvious space. A rebuild cannot claim novelty from "affordance map plus more predictions." It must show that explicitly representing unrealized alternatives changes action choice.

## Rebuilt Claim Under Test

The strongest defensible claim is:

> A counterfactual affordance map is useful only if it predicts how affordances change under alternate grasps, poses, supports, and occlusions, and if those predictions improve closed-loop manipulation decisions beyond observed-only affordance maps, graph/VLM affordance baselines, active probing, and uncertainty ensembles.

This is a local evidence audit, not hardware validation.

## Benchmark Design

I will replace the template success-rate generator with a deterministic manipulation-affordance benchmark. Each episode samples an object/task scene with hidden support relations, occlusions, contact stability, semantic task constraints, and alternate candidate actions. The simulator will score both map prediction quality and the outcome of choosing actions from those maps.

Tasks:

1. `cluttered_pick_and_place`
2. `tool_regrasp_for_use`
3. `support_sensitive_stacking`
4. `occluded_container_opening`

Splits:

1. `nominal_affordance`
2. `pose_shift`
3. `support_shift`
4. `occlusion_semantic_shift`
5. `combined_counterfactual_stress`

## Methods To Compare

Strong baselines:

1. `observed_affordance_map`
2. `multi_affordance_grasping`
3. `graph_conv_affordance`
4. `gaussian_grasp_map`
5. `vlm_spatial_affordance`
6. `interactive_affordance_probe`
7. `ensemble_uncertainty_affordance`
8. `proposed_counterfactual_affordance_map`
9. `oracle_counterfactual_map`

## Metrics

Map metrics:

1. Affordance AP.
2. Counterfactual recall.
3. Calibration error.
4. Support-relation error.
5. Semantic affordance violation.

Closed-loop metrics:

1. Task success.
2. Invalid action rate.
3. Object damage/support collapse.
4. Planning regret to oracle.
5. Probe/intervention cost.
6. Recovery success after a wrong first action.

Statistics:

1. Seven deterministic seeds.
2. Per-task and per-split means with 95 percent confidence intervals.
3. Paired seed/task comparison against the strongest non-oracle baseline.
4. Explicit terminal decision in `results/summary.txt`.

## Ablations

The full method must beat stripped variants:

1. `full_counterfactual_affordance`
2. `minus_pose_counterfactuals`
3. `minus_support_counterfactuals`
4. `minus_semantic_counterfactuals`
5. `minus_uncertainty_calibration`
6. `observed_only_counterfactual_head`
7. `support_only_counterfactual_head`

If stripped variants match or beat the full method on task success or counterfactual recall without a clear tradeoff, the mechanism is not supported.

## Stress Tests

Stress axes:

1. Pose extrapolation.
2. Hidden support-relation shift.
3. Visual occlusion.
4. Semantic task ambiguity.
5. Contact/noise perturbation.
6. Combined maximum stress.

The stress sweep must show whether counterfactual maps help when observed affordance and graph/VLM baselines become overconfident.

## Paper Rewrite Requirements

After experiments:

1. Rewrite `paper/main.tex` as either a strong-revise evidence report or a negative evidence audit.
2. Replace template claims with measured claims only.
3. Include tables for combined stress, ablations, and failure cases.
4. Include figures for map quality, closed-loop task outcomes, calibration/regret, ablations, and stress curves.
5. Update README, child status, claims, final audit, and submission-readiness docs.
6. Build only `C:/Users/wangz/Downloads/93.pdf`; do not copy anything to Desktop.
7. Commit and push to `https://github.com/Jason-Wang313/93_counterfactual_affordance_maps`.

## Terminal Gate

Mark `STRONG_REVISE` only if all of the following are true:

1. `proposed_counterfactual_affordance_map` beats the strongest non-oracle baseline on combined-stress task success and counterfactual recall.
2. It also reduces invalid actions or planning regret without excessive probe/intervention cost.
3. Core ablations degrade in expected directions.
4. Maximum-stress curves do not reverse in favor of graph, VLM, active probing, or ensemble uncertainty baselines.
5. The paper honestly states the evidence is local/simulated and not robot hardware validation.

Otherwise mark `KILL_ARCHIVE`. A counterfactual affordance map that is matched by observed-only, graph/VLM, active-probe, or uncertainty baselines is not ICLR-main ready.

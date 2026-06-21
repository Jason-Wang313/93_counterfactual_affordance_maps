# Claims

- Mechanism claim tested: counterfactual affordance maps are useful only if they improve closed-loop decisions beyond observed-only maps, graph affordances, VLM/foundation affordances, diffusion sampling, uncertainty ensembles, active probing, robust MPC, and the v4 counterfactual map.
- Expanded evidence claim: v5 adds 10 seeds, 6 tasks, 8 splits, 14 methods, 10 ablations, 604,800 stress-sweep rows, fixed-risk deployment, paired seed tests, and 24 negative cases.
- Supported local finding: `causal_counterfactual_affordance_planner_v5` improves over v4 and static affordance baselines, and the ablation audit shows a real mechanism.
- Supported local finding: v5 reaches hard-split success 0.33646 and counterfactual recall 1.00000.
- Unsupported main claim: v5 is not ICLR-main-ready because it fails active-probe regret, safety, calibration, robust utility, stress dominance, fixed-risk coverage, and scope gates.
- Unsupported claim explicitly avoided: no claim of real-robot deployment, accepted high-fidelity manipulation benchmark validation, or SOTA affordance learning.

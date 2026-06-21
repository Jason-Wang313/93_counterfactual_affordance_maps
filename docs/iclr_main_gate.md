# ICLR Main Gate

Paper: 93 counterfactual_affordance_maps

Latest decision: KILL_ARCHIVE

Latest audit: v5 expanded audit, 2026-06-22

Evidence digest: v5-expanded-counterfactual-affordance

Gate outcomes:

- success_gate=True
- active_probe_gate=False
- recall_gate=True
- safety_gate=False
- calibration_gate=False
- utility_gate=False
- ablation_gate=True
- stress_gate=False
- fixed_risk_gate=False
- scope_gate=False

Fatal blockers:

- V5 buys hard-split success/recall with too much invalid action, damage, and regret.
- Interactive probing remains the hard robust-utility reference.
- Robust MPC remains a safer closed-loop deployment comparator.
- V5 fixed-risk coverage is zero at budget 0.05 on both hard splits.
- Calibration is worse than the best non-oracle calibration reference.
- Maximum-stress utility is dominated by active probing.
- Evidence remains local/simulated with no real robot or accepted high-fidelity manipulation benchmark.

The only honest main-conference-safe decision is to archive rather than overclaim.

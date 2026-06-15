# Paper 93 Terminal Audit

Date: 2026-06-15

Paper: `93_counterfactual_affordance_maps`

Decision: `KILL_ARCHIVE`

## Reproduction

- `python -m py_compile src\run_experiment.py`: passed.
- `python src\run_experiment.py`: launched with log redirection; shell wrapper timed out after roughly 904 seconds, but the Python child process continued and completed successfully.
- Rerun log path: `logs/93_counterfactual_affordance_maps_continuation_rerun_20260615.log`.
- Note: the rerun log file is empty because the script writes substantive evidence to `results/summary.txt` and CSV files rather than stdout.
- PDF target: `C:/Users/wangz/Downloads/93.pdf`.
- Visible Desktop copy: not allowed.

## Evidence Files

- `results/rollouts.csv`: 105,840 rows.
- `results/raw_seed_metrics.csv`: 1,260 rows.
- `results/metrics.csv`: 45 rows.
- `results/pairwise_stats.csv`: 1 row.
- `results/ablation_rollouts.csv`: 16,464 rows.
- `results/ablation_seed_metrics.csv`: 196 rows.
- `results/ablation_metrics.csv`: 7 rows.
- `results/stress_sweep_raw.csv`: 52,416 rows.
- `results/stress_sweep.csv`: 36 rows.
- `results/negative_cases.csv`: 3 rows.

## Key Results

Combined counterfactual stress:

- `proposed_counterfactual_affordance_map`: success `0.34651 +/- 0.02045`, AP `0.21670`, counterfactual recall `0.00893`, invalid action `0.28529`, regret `0.03385`, cost `0.02000`.
- `interactive_affordance_probe`: success `0.36224 +/- 0.02063`, AP `0.22665`, counterfactual recall `0.00765`, invalid action `0.24830`, regret `0.02445`, cost `0.18000`.
- `ensemble_uncertainty_affordance`: success `0.34056`, AP `0.20365`, counterfactual recall `0.00808`, regret `0.04651`.
- Paired success difference versus `interactive_affordance_probe`: `-0.01573 +/- 0.02023`.
- Paired counterfactual-recall difference: `0.00128 +/- 0.00367`.
- Paired regret reduction: `-0.00940 +/- 0.00215`.

Ablation:

- Full counterfactual affordance: success `0.37032`, counterfactual recall `0.01105`, invalid action `0.27721`, regret `0.03104`.
- `minus_pose_counterfactuals`: success `0.34056`, counterfactual recall `0.00808`, invalid action `0.27083`, regret `0.03926`.
- `support_only_counterfactual_head`: success `0.34141`, counterfactual recall `0.00553`, invalid action `0.28529`, regret `0.05004`.

Maximum stress:

- `proposed_counterfactual_affordance_map`: success `0.34341`, counterfactual recall `0.00824`, invalid action `0.29121`, regret `0.03292`.
- `interactive_affordance_probe`: success `0.33654`, counterfactual recall `0.00962`, invalid action `0.25687`, regret `0.02601`.
- `oracle_counterfactual_map`: success `0.37500`, counterfactual recall `0.01236`, invalid action `0.25137`, regret `0.01138`.

## Terminal Reason

The rerun verifies that the counterfactual map is locally useful but not ICLR-main ready. It improves over static affordance maps and has a coherent ablation signal, but it loses the practical combined-stress closed-loop gate to active probing on task success, invalid actions, and regret. The evidence is also local simulated evidence rather than robot or accepted high-fidelity manipulation validation. The only honest terminal decision is `KILL_ARCHIVE`.

## PDF Verification

- Build command: two-pass `pdflatex -interaction=nonstopmode -halt-on-error main.tex`.
- Canonical PDF: `C:/Users/wangz/Downloads/93.pdf`.
- PDF SHA256: `1C55928FFE37197B57CFF8F49A2127C1E9700DB0759EA2820DA994CAEB1E9478`.
- PDF size: 478,968 bytes.
- LaTeX log scan: no document warnings/errors requiring action after the second pass.
- Desktop copy: absent.

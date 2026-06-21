# 93 Counterfactual Affordance Maps

Submission-hardening version: v5 expanded ICLR-main audit

Terminal decision: KILL_ARCHIVE for ICLR main conference.

This repository contains a deterministic, CPU-only, RAM-light manipulation-affordance evidence audit for the claim that robots should map affordances under alternate grasps, poses, supports, semantic roles, occlusion states, and contact modes. The v5 rebuild expands the evidence to 10 seeds, 6 task families, 8 distribution splits, 14 methods, 10 ablations, a six-level stress sweep crossed with all splits, fixed-risk deployment budgets, paired seed tests, and 24 negative cases.

The method under test is `causal_counterfactual_affordance_planner_v5`. It improves over v4 and several static affordance baselines, but it is not submission-ready because active probing and robust MPC remain stronger on the hostile deployment criteria.

## Key Result

Hard aggregate over low-signal and combined counterfactual stress:

- `causal_counterfactual_affordance_planner_v5`: success 0.33646, counterfactual recall 1.00000, invalid action 0.39323, damage 0.19531, robust utility -0.43149.
- `interactive_affordance_probe`: success 0.29557, counterfactual recall 0.85359, invalid action 0.20521, damage 0.04427, robust utility -0.15992.
- `robust_mpc_affordance_planner`: success 0.25833, invalid action 0.23594, damage 0.05859, robust utility -0.22369.
- v5 beats active probing on hard success with paired lower95 0.01474, but fails regret, safety, calibration, robust-utility, stress, fixed-risk, and scope gates.
- Fixed-risk budget 0.05 coverage for v5 is 0.00000 on both hard splits.

Terminal gate vector: success true, active-probe false, recall true, safety false, calibration false, utility false, ablation true, stress false, fixed-risk false, scope false.

## Artifacts

- Canonical PDF: `C:/Users/wangz/Downloads/93.pdf`
- PDF pages: 25
- PDF SHA-256: `5222B202BEC63CA70F040A2B621EBB65775F4B601D819B68C122773538A4CD60`
- GitHub-safe raw stress artifact: `results/stress_sweep_raw.csv.gz`
- Public repo: https://github.com/Jason-Wang313/93_counterfactual_affordance_maps
- No PDF is copied to the visible Desktop.

## Reproduce Evidence

```powershell
python -m py_compile src\run_experiment.py
python src\run_experiment.py
```

## Rebuild Manuscript

```powershell
python scripts\generate_manuscript.py
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
Copy-Item .\main.pdf C:\Users\wangz\Downloads\93.pdf -Force
cd ..
python scripts\validate_submission_artifacts.py
```

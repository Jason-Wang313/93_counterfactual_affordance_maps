# Submission Readiness Audit v5

Date: 2026-06-22

Paper: 93 Counterfactual Affordance Maps

Terminal decision: KILL_ARCHIVE

## Commands Run

```powershell
python -m py_compile src\run_experiment.py
python src\run_experiment.py
python -m py_compile scripts\generate_manuscript.py scripts\validate_submission_artifacts.py
python scripts\generate_manuscript.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
python scripts\validate_submission_artifacts.py
```

## Evidence Coverage

- Main rollouts: 215,040 rows.
- Dataset summary rows: 15,360 rows.
- Main seed metrics: 1,120 rows.
- Main aggregate metrics: 1,568 rows.
- Main pairwise rows: 1,344 rows.
- Hard aggregate seed rows: 140 rows.
- Hard aggregate metrics: 196 rows.
- Hard aggregate pairwise rows: 168 rows.
- Ablation rollouts: 76,800 rows.
- Ablation metrics: 140 rows.
- Stress raw rows: 604,800 rows.
- GitHub packaging: raw stress rows are validated locally from `results/stress_sweep_raw.csv`; the public repo stores `results/stress_sweep_raw.csv.gz` because the uncompressed CSV exceeds GitHub's 100 MB file limit.
- Stress metrics: 1,176 rows.
- Fixed-risk raw rows: 69,120 rows.
- Fixed-risk metrics: 288 rows.
- Fixed-risk pairwise rows: 240 rows.
- Negative cases: 24 rows.

## Main Gate

On the hard aggregate, `causal_counterfactual_affordance_planner_v5` reaches success `0.33646`, counterfactual recall `1.00000`, invalid action `0.39323`, damage `0.19531`, regret `0.35070`, map ECE `0.38475`, and robust utility `-0.43149`.

The strongest hard success challenger is `interactive_affordance_probe`, which reaches success `0.29557`, invalid action `0.20521`, damage `0.04427`, regret `0.27682`, and robust utility `-0.15992`.

The v5 paired success lower95 versus interactive probing is `0.01474`, so the success gate passes. The active-probe regret upper95 is `0.07565`, the robust-utility lower95 is `-0.29889`, and the fixed-risk coverage at budget 0.05 is `0.00000` on both hard splits, so deployment readiness fails.

## Contradictory Evidence

- Active-probe gate fails.
- Safety gate fails.
- Calibration gate fails.
- Robust-utility gate fails.
- Stress gate fails.
- Fixed-risk gate fails.
- Scope gate fails.

## PDF Audit

- Canonical PDF: `C:/Users/wangz/Downloads/93.pdf`
- Pages: 25
- SHA-256: `5222B202BEC63CA70F040A2B621EBB65775F4B601D819B68C122773538A4CD60`
- Bright boxed citation links: enabled through `hyperref` citation borders.
- Desktop PDF leak: none.

## Readiness Judgment

The v5 mechanism is real and stronger than v4, but the paper is not ICLR-main-ready. A submission would be vulnerable to straightforward active-probing, robust-MPC, calibration, fixed-risk, and scope objections.

## Terminal Action

Keep `KILL_ARCHIVE`. Do not submit this paper to ICLR main in the current form.

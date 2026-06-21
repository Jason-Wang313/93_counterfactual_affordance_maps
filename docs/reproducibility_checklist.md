# Reproducibility Checklist

## What Reproduces

- [x] `python -m py_compile src/run_experiment.py`
- [x] `python src/run_experiment.py`
- [x] `results/rollouts.csv`
- [x] `results/dataset_summary.csv`
- [x] `results/raw_seed_metrics.csv`
- [x] `results/metrics.csv`
- [x] `results/pairwise_stats.csv`
- [x] `results/hard_aggregate_metrics.csv`
- [x] `results/hard_aggregate_pairwise_stats.csv`
- [x] `results/ablation_metrics.csv`
- [x] `results/stress_sweep.csv`
- [x] `results/fixed_risk_metrics.csv`
- [x] `results/negative_cases.csv`
- [x] `results/stress_sweep_raw.csv.gz` for GitHub-safe storage of the oversized raw stress CSV.
- [x] v5 figures under `figures/`
- [x] `python scripts/generate_manuscript.py`
- [x] `paper/main.tex`
- [x] `paper/references.bib`
- [x] Canonical PDF: `C:/Users/wangz/Downloads/93.pdf`
- [x] `python scripts/validate_submission_artifacts.py`

## What Does Not Reproduce

- [ ] Real robot results.
- [ ] Accepted high-fidelity benchmark runs.
- [ ] Hardware control stack.
- [ ] Independent external reproduction.
- [ ] Claims of ICLR-main readiness.

This is reproducible as a strong local negative evidence audit, not as an ICLR-main robotics system paper.

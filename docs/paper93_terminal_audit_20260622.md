# Paper 93 Terminal Audit - 2026-06-22

Terminal recommendation: KILL_ARCHIVE

ICLR main ready: no

Final artifact:

- `C:/Users/wangz/Downloads/93.pdf`
- 25 pages
- SHA-256 `5222B202BEC63CA70F040A2B621EBB65775F4B601D819B68C122773538A4CD60`

Validated commands:

- `python -m py_compile src\run_experiment.py`
- `python src\run_experiment.py`
- `python -m py_compile scripts\generate_manuscript.py scripts\validate_submission_artifacts.py`
- `python scripts\generate_manuscript.py`
- `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`
- `python scripts\validate_submission_artifacts.py`
- `pdftoppm` render and sampled visual QA
- GitHub push packaging with `results/stress_sweep_raw.csv.gz` for the oversized raw stress CSV.

Result summary:

- Hard success winner: `causal_counterfactual_affordance_planner_v5`
- Robust-utility winner among hard references: `interactive_affordance_probe`
- Best calibration reference: `observed_affordance_map`
- Maximum-stress utility reference: `interactive_affordance_probe`
- Fixed-risk v5 coverage at budget 0.05: zero on both hard splits

Gate vector:

- success: pass
- active probe: fail
- recall: pass
- safety: fail
- calibration: fail
- utility: fail
- ablation: pass
- stress: fail
- fixed risk: fail
- scope: fail

Conclusion: archive this evidence package. Future revival requires external manipulation validation and a safer deployed frontier, not just higher counterfactual recall.

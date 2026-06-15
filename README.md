# 93 Counterfactual Affordance Maps

Submission-hardening version: v4.1 rerun audit

Terminal decision: KILL_ARCHIVE for ICLR main conference.

This repository now contains a deterministic manipulation-affordance evidence audit for the claim that robots should map affordances under alternate grasps, poses, supports, and semantic roles. The rebuilt benchmark includes four tasks, five shifts, seven seeds, nine affordance methods, seven ablations, and a stress sweep.

The 2026-06-15 continuation rerun reproduced the same terminal decision: the proposed counterfactual map is useful locally, but active probing remains the stronger closed-loop baseline.

## Key Result

On combined counterfactual stress:

- Proposed counterfactual map: task success 0.347, AP 0.217, counterfactual recall 0.0089, regret 0.034.
- Interactive affordance probe: task success 0.362, AP 0.227, counterfactual recall 0.0077, regret 0.024.
- Graph-conv affordance: task success 0.323, AP 0.184, counterfactual recall 0.0034.
- Paired task-success difference vs strongest non-oracle baseline: -0.0157 +/- 0.0202.

The proposed method improves over static observed/graph/VLM affordance baselines, but it does not beat active probing decisively and lacks robot hardware or accepted high-fidelity validation.

## Reproduce Evidence

```powershell
python src\run_experiment.py
```

## Rebuild PDF

```powershell
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Canonical local PDF: `C:/Users/wangz/Downloads/93.pdf`

No PDF should be copied to the visible Desktop.

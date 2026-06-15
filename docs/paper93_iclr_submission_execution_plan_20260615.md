# Paper 93 ICLR-Main Submission Execution Plan

Date: 2026-06-15
Paper: `93_counterfactual_affordance_maps`
Repository: `https://github.com/Jason-Wang313/93_counterfactual_affordance_maps`

## Goal

Rebuild and audit Paper 93 under an ICLR-main submission standard, while refusing to upgrade the paper unless the evidence proves that counterfactual affordance maps improve closed-loop manipulation decisions beyond active probing, graph/VLM affordance models, Gaussian grasp maps, and uncertainty ensembles.

## Current Starting State

- The repository is on `main` at commit `a2a521d0299e364e5d28070730ac6759b70b68ac`.
- The existing v4 audit is terminally negative: `KILL_ARCHIVE`.
- Existing evidence suggests the proposed map improves over static observed/graph/VLM-style affordance baselines.
- Existing evidence also suggests it does not decisively beat `interactive_affordance_probe` on combined-stress task success, regret, or counterfactual recall.
- `C:/Users/wangz/Downloads/93.pdf` exists.
- `C:/Users/wangz/Desktop/93.pdf` does not exist and must not be created.

## Execution Steps

1. Re-run `python -m py_compile src/run_experiment.py`.
2. Re-run `python src/run_experiment.py` and save the full console transcript to the batch log directory.
3. Verify that the rerun regenerates main rollouts, seed metrics, aggregate metrics, paired statistics, ablations, stress sweeps, negative cases, and figures.
4. Independently audit the CSV files with pandas rather than trusting the prose summary.
5. Compare `proposed_counterfactual_affordance_map` against the strongest non-oracle baselines on:
   - combined-stress task success;
   - affordance AP;
   - counterfactual recall;
   - invalid action rate;
   - object damage/support collapse;
   - planning regret;
   - probe/intervention cost;
   - maximum-stress robustness.
6. Check that the full counterfactual mechanism beats stripped ablations in the dimensions needed to support the thesis.
7. Update the paper and documentation with measured claims only.
8. Rebuild `paper/main.pdf` with `pdflatex` and copy the final artifact only to `C:/Users/wangz/Downloads/93.pdf`.
9. Scan the LaTeX log for real warnings/errors and fix recoverable typesetting problems.
10. Update the root batch ledgers after the child repo is correct.
11. Commit, push, and verify the public GitHub repository.
12. Confirm the child git tree is clean, `origin/main` matches local `HEAD`, the numbered PDF exists in Downloads, and no Desktop PDF exists.

## Submission-Readiness Gates

Paper 93 may be marked `STRONG_REVISE` only if all gates pass:

1. The proposed counterfactual affordance map beats the strongest non-oracle baseline on combined-stress task success.
2. It beats or clearly matches active probing on planning regret without hiding behind lower probe cost.
3. It has a decisive counterfactual-recall improvement with uncertainty small enough to matter.
4. It reduces invalid actions or damage relative to graph/VLM/static affordance baselines and remains competitive with active probing.
5. Core ablations degrade in the expected directions.
6. Maximum-stress curves do not reverse in favor of active probing, graph affordances, VLM affordances, Gaussian grasp maps, or uncertainty ensembles.
7. The paper clearly labels the evidence as local simulated evidence and does not imply robot hardware validation.

If the proposed method loses the active-probing closed-loop gate, the terminal decision remains `KILL_ARCHIVE` even if map metrics or ablations look internally coherent.

## Expected Honest Outcome

The prior v4 evidence already suggests a likely `KILL_ARCHIVE` decision because active probing has better combined-stress task success and lower regret. The continuation rerun must verify that outcome from regenerated evidence rather than preserving it by inertia.

## Deliverables

- Updated rerun log in `C:/Users/wangz/robotics_massive_pool_paper_factory/logs/`.
- Updated Paper 93 result CSVs and figures if regenerated content changes.
- Updated child documentation and paper source with the rerun audit.
- Final numbered PDF at `C:/Users/wangz/Downloads/93.pdf` only.
- Updated root ledgers through Paper 93.
- Public GitHub repo pushed and verified clean.

# Experiment Rigor Checklist

## v5 Expanded Synthetic Rigor

- [x] Frozen plan before execution.
- [x] CPU-only, RAM-light deterministic runner.
- [x] 10 seeds.
- [x] 6 task families.
- [x] 8 distribution splits.
- [x] 14 methods including active probing, robust MPC, diffusion, VLM/foundation, uncertainty, v4, v5, and oracle.
- [x] 215,040 main rollout rows.
- [x] Paired seed confidence intervals.
- [x] Hard aggregate over low-signal and combined stress.
- [x] 10 ablations with 76,800 rollout rows.
- [x] Six-level stress sweep with 604,800 raw rows.
- [x] Fixed-risk deployment budgets 0.00, 0.05, 0.10, and 0.15.
- [x] 24 negative cases.
- [x] 25-page ICLR-style PDF with bright boxed clickable citations.
- [x] Artifact validator for row counts, PDF page count, links, terminal tokens, and no Desktop leak.

## ICLR Main Bar

- [ ] Real-robot validation.
- [ ] Accepted high-fidelity manipulation benchmark.
- [ ] Hardware timing and failure recovery study.
- [ ] Independent reproduction.
- [ ] External learned checkpoint baselines.
- [x] Strong local baselines and stress tests.
- [x] Honest terminal decision after frozen protocol.

Decision: v5 still fails ICLR main empirical-rigor and deployment gates; archive.

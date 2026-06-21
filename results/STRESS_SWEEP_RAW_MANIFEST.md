# Stress Sweep Raw Artifact

`results/stress_sweep_raw.csv` is generated locally by:

```powershell
python src\run_experiment.py
```

The uncompressed CSV has 604,800 data rows and is approximately 249 MiB, which exceeds GitHub's hard 100 MB file limit. The public repository therefore stores:

- `results/stress_sweep_raw.csv.gz`

The local workspace keeps the uncompressed CSV for validation. The validator accepts the local CSV when present and otherwise reads the compressed `.csv.gz` artifact.

Do not copy this file to the Desktop.

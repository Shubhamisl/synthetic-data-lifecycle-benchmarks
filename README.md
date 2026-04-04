# Synthetic Data Lifecycle

This repository contains a synthetic data generation research project and a cross-domain benchmarking extension under `benchmarks/`.

## Included

- core project source under `evaluation/`, `main.py`, and `config.py`
- benchmark pipeline scripts under `benchmarks/`
- benchmark tests under `tests/benchmarks/`
- design and implementation notes under `docs/superpowers/`
- lightweight benchmark outputs such as summary CSVs, interpretation notes, and publication-ready plots

## Excluded From Version Control

The repository intentionally excludes large or reproducible generated assets, including:

- raw and processed dataset CSVs
- trained model binaries
- large synthetic CSV outputs
- transient logs and cache folders

## Benchmark Entry Points

Run the benchmark stages independently:

```powershell
python -m benchmarks.download_datasets
python -m benchmarks.train_benchmark_models
python -m benchmarks.evaluate_benchmarks
python -m benchmarks.visualize_benchmarks
```

The cache-aware orchestrator is available at:

```powershell
python -m benchmarks.run_benchmarks
```

## Direction 3 In Colab

For the `dp_triangle/` pipeline, prefer the Colab-specific dependency set instead of the full base `requirements.txt`.
The base requirements include `tensorflow-privacy`, which is not needed for Direction 3 and can fail on modern Colab images.

Recommended Colab flow:

```python
!git clone https://github.com/Shubhamisl/synthetic-data-lifecycle-benchmarks.git
%cd synthetic-data-lifecycle-benchmarks
!pip install -q -r requirements_colab_direction3.txt
!python -m data.loader
!python -m dp_triangle.run_direction3 --dry-run
```

Before real training, remove the dry-run artifacts:

```python
from pathlib import Path

results = Path("results")
for path in results.glob("dp_synthetic_*.csv"):
    path.unlink(missing_ok=True)

for name in [
    "dp_triangle_dashboard.csv",
    "dp_subgroup_fairness.csv",
    "dp_post_hoc_debiasing.csv",
    "figure4_epsilon_tradeoff_curve.png",
    "figure5_pff_radar_chart.png",
    "figure6_intersectional_fairness.png",
    "figure7_post_hoc_recovery.png",
    "dp_direction3_dry_run.marker",
]:
    (results / name).unlink(missing_ok=True)
```

Then run the full experiment:

```python
!python -m dp_triangle.run_direction3
```

## Benchmark Summaries

Key saved outputs:

- `benchmarks/results/cross_domain_summary.csv`
- `benchmarks/results/mean_rank_table.csv`
- `benchmarks/results/benchmark_run_notes.md`
- `benchmarks/results/benchmark_research_summary.md`

Plots are saved under `benchmarks/plots/`.

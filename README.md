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

## Benchmark Summaries

Key saved outputs:

- `benchmarks/results/cross_domain_summary.csv`
- `benchmarks/results/mean_rank_table.csv`
- `benchmarks/results/benchmark_run_notes.md`
- `benchmarks/results/benchmark_research_summary.md`

Plots are saved under `benchmarks/plots/`.

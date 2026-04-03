# Cross-Domain Benchmark Research Summary

Date: 2026-04-03

## Executive Summary

Across four domains, neither generator dominated every metric. CTGAN was generally stronger on distributional fidelity, achieving the better mean JS rank (1.25 vs. 1.75), while TVAE was stronger on privacy and fairness, with the better mean MIA rank (1.25 vs. 1.75), the better mean demographic-parity rank (1.33 vs. 1.67), and the better overall mean rank (1.46 vs. 1.54). Utility was effectively tied at the aggregate level, with both models obtaining the same mean TSTR rank of 1.50.

Performance was highly dataset-dependent. On Bank Marketing, CTGAN came closest to the real-data baseline, reaching 88.51% TSTR against an 89.27% real baseline, a gap of only 0.76 percentage points. On Covertype, both generators underperformed the real baseline substantially, but TVAE preserved downstream utility more successfully than CTGAN, reaching 70.40% TSTR versus 50.04% for CTGAN. On Diabetes, TVAE provided the best balance of utility, privacy, and JS divergence, while CTGAN produced the lower demographic-parity difference. On Adult, CTGAN slightly improved utility and JS divergence, but TVAE showed lower privacy risk and lower demographic-parity difference.

## Compact Result Table

| Dataset | Domain | Real TSTR Baseline (%) | Best Synthetic TSTR (%) | Gap to Baseline (pp) | Lowest JS | Lowest MIA | Lowest DP |
|---|---|---:|---:|---:|---:|---:|---:|
| adult | Socioeconomic | 85.68 | 81.67 (CTGAN) | 4.01 | 0.0228 (CTGAN) | 0.3790 (TVAE) | 0.1810 (TVAE) |
| bank | Marketing / Finance | 89.27 | 88.51 (CTGAN) | 0.76 | 0.0118 (CTGAN) | 0.3389 (CTGAN) | 0.0132 (TVAE) |
| covertype | Environmental / Ecological | 88.25 | 70.40 (TVAE) | 17.85 | 0.0049 (CTGAN) | 0.4525 (TVAE) | N/A |
| diabetes | Healthcare | 74.03 | 66.88 (TVAE) | 7.14 | 0.0370 (TVAE) | 0.4557 (TVAE) | 0.0905 (CTGAN) |

## Mean-Rank Interpretation

| Model | Mean TSTR Rank | Mean JS Rank | Mean MIA Rank | Mean DP Rank | Overall Mean Rank |
|---|---:|---:|---:|---:|---:|
| CTGAN | 1.50 | 1.25 | 1.75 | 1.67 | 1.54 |
| TVAE | 1.50 | 1.75 | 1.25 | 1.33 | 1.46 |

This ranking pattern suggests a practical trade-off rather than a single universally best generator. If the benchmark objective prioritizes statistical fidelity, CTGAN is the stronger candidate. If the objective prioritizes lower privacy leakage and lower group disparity while keeping competitive utility, TVAE is the better default choice across domains.

## Manuscript-Ready Paragraph

We extended the synthetic data lifecycle benchmark from the original Adult baseline to four domains spanning socioeconomic, marketing, ecological, and healthcare data. Across datasets, CTGAN and TVAE showed complementary strengths rather than consistent dominance by a single model. CTGAN achieved the better mean rank for distributional fidelity, with a lower mean JS-divergence rank (1.25) than TVAE (1.75), and it produced the strongest near-baseline utility result on Bank Marketing, where synthetic-data TSTR reached 88.51% compared with an 89.27% real-data baseline. In contrast, TVAE achieved the better overall mean rank (1.46 vs. 1.54), driven by lower average membership-inference advantage and lower demographic-parity disparity. Utility outcomes were dataset-sensitive: TVAE clearly outperformed CTGAN on Covertype and Diabetes, whereas CTGAN was slightly stronger on Adult and substantially stronger on Bank. These findings indicate that cross-domain synthetic data benchmarking should report fidelity, utility, privacy, and fairness jointly, because model choice depends strongly on the target deployment setting rather than on a single headline metric.

## Reproducibility Note

Two implementation details are important for correct interpretation of this run:

- Bank categorical missing values were normalized to `"unknown"` before splitting and training.
- Covertype CTGAN used only the true categorical or indicator columns as discrete features because the literal all-columns-discrete configuration exceeded available memory.

These choices are documented in `benchmark_run_notes.md`.

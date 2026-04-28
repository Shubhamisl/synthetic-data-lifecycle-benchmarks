# Cross-Domain Benchmark Run Notes

Date: 2026-04-03

## Purpose

This note records the key methodological choices and runtime findings for the cross-domain benchmark run generated under `benchmarks/`.

## Final Outputs

Primary result tables:

- `cross_domain_summary.csv`
- `mean_rank_table.csv`

Primary plots:

- `plot1_tstr_heatmap.png`
- `plot2_cross_domain_dashboard.png`
- `plot3_mean_rank.png`
- `plot4_privacy_utility_all_domains.png`

Synthetic datasets saved:

- `adult_ctgan.csv`
- `adult_tvae.csv`
- `bank_ctgan.csv`
- `bank_tvae.csv`
- `covertype_ctgan.csv`
- `covertype_tvae.csv`
- `diabetes_ctgan.csv`
- `diabetes_tvae.csv`

## Important Methodology Notes

### 1. Bank missing-category normalization

The UCI Bank Marketing source retrieved through `ucimlrepo` contained missing values in categorical fields including:

- `job`
- `education`
- `contact`
- `poutcome`

These were normalized to the explicit category `"unknown"` during preprocessing before the train/test split. This was necessary because leaving the missing values in place caused downstream synthetic generation failures and would have introduced inconsistent handling across models.

### 2. Covertype CTGAN discrete-column adjustment

The original benchmark instruction requested passing all Covertype columns as discrete/categorical to CTGAN. In practice, this caused CTGAN to expand the transformed training matrix to an in-memory representation too large for the environment:

- approximate failure shape: `(46480, 13468)`
- approximate allocation request: `4.66 GiB`

With explicit user approval, the CTGAN configuration for Covertype was adjusted to treat only the true categorical or indicator columns as discrete:

- `Wilderness_Area1..4`
- `Soil_Type1..40`
- `Cover_Type`

Continuous terrain measurements such as elevation, aspect, slope, and distance variables remained continuous.

This change preserved the benchmark run while keeping the dataset semantics more faithful than forcing all columns into a categorical one-hot representation.

## Validation Notes

The final synthetic CSVs were reloaded from disk and checked to confirm:

- expected row counts
- schema consistency
- zero `NaN` values
- valid target values
- Covertype target class integrity

The benchmark test suite also passed after the final implementation state.

## Interpretation Reminder

Results for Covertype should be interpreted with the discrete-column adjustment above in mind. Comparisons across datasets remain valid within this benchmark run, but the Covertype CTGAN configuration is not identical to the literal all-columns-discrete instruction because that configuration was not executable in the available memory budget.

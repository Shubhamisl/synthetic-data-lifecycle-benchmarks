# Cross-Domain Benchmarking Design

Date: 2026-04-03
Project: `synthetic_data_lifecycle`
Scope: Add a new `benchmarks/` package for cross-domain synthetic data benchmarking without modifying or retraining the protected Adult artifacts in the existing lifecycle pipeline.

## Goal

Extend the current single-dataset synthetic data lifecycle into a cross-domain benchmarking workflow covering four datasets:

- `adult` as the baseline dataset using already-produced artifacts
- `bank` from UCI Bank Marketing
- `covertype` from UCI Covertype using a stratified sample
- `diabetes` from the Pima Indians Diabetes CSV source

The new work must live entirely inside a new `benchmarks/` folder and be runnable independently of the parent lifecycle stages.

## Skills Used

- `brainstorming` to define the design before any implementation
- `writing-plans` as the next workflow step after the user reviews this spec
- `test-driven-development` for implementing the new benchmark scripts with validation-first behavior
- `verification-before-completion` before claiming the benchmark pipeline is complete

## Constraints

- Do not modify files in the protected existing artifact set:
  - `data/adult_train.csv`
  - `data/adult_test.csv`
  - `results/ctgan_synthetic.csv`
  - `results/tvae_synthetic.csv`
  - `results/final_evaluation_table.csv`
  - `models/ctgan_model.pkl`
  - `models/tvae_model.pkl`
- Do not retrain Adult models.
- All new code and outputs belong under `benchmarks/`.
- Each benchmark script must be independently runnable with `python -m benchmarks.<module>`.
- Corrupted, inconsistent, or miscalculated data must be detected aggressively.
- If a dataset fails after fallback handling or validation, log the failure and stop the run immediately rather than producing partial benchmark outputs.

## Proposed Architecture

Create a new package at `benchmarks/` with these primary scripts:

- `download_datasets.py`
- `train_benchmark_models.py`
- `evaluate_benchmarks.py`
- `visualize_benchmarks.py`
- `run_benchmarks.py`

Also add a small internal helper module:

- `common.py`

And the required data folders:

- `datasets/`
- `synthetic/`
- `results/`
- `plots/`

This keeps the public interface exactly aligned with the requested five scripts while centralizing correctness-sensitive logic in one place.

## Common Module Responsibilities

`benchmarks/common.py` will hold shared configuration and safety logic:

- dataset registry metadata for `adult`, `bank`, `covertype`, and `diabetes`
- domain and fidelity labels for reporting
- path helpers for all benchmark artifacts
- dataset-specific target column, sensitive attribute, and valid class sets
- shared validation helpers for real and synthetic tabular data
- categorical detection and consistent label encoding
- safe summary-table formatting
- failure logging and warning logging
- subprocess helpers for the master runner

The registry prevents metric scripts, plotting scripts, and training scripts from drifting apart in how they interpret each dataset.

## Dataset Design

### Adult

- Source: copy from existing processed files
- Train/test:
  - copy `data/adult_train.csv` to `benchmarks/datasets/adult_train.csv`
  - copy `data/adult_test.csv` to `benchmarks/datasets/adult_test.csv`
- Synthetic artifacts:
  - copy `results/ctgan_synthetic.csv` to `benchmarks/synthetic/adult_ctgan.csv`
  - copy `results/tvae_synthetic.csv` to `benchmarks/synthetic/adult_tvae.csv`
- Evaluation baseline:
  - copy `results/final_evaluation_table.csv` to `benchmarks/results/adult_evaluation.csv`
- Target column: `income`
- Sensitive attribute: `sex`
- Domain: `Socioeconomic`
- Fidelity level: `Baseline`

### Bank

- Source: `ucimlrepo.fetch_ucirepo(id=222)` with fallback to a direct UCI CSV download if needed
- Preprocessing:
  - rename `y` to `target`
  - encode `yes -> 1`, `no -> 0`
  - create `age_group` as `young` for `age <= 40` and `older` for `age > 40`
  - drop original `age`
  - drop `duration` if present because it leaks outcome information
- Split:
  - 80/20 train/test
  - stratified on `target`
- Sensitive attribute: `age_group`
- Domain: `Marketing / Finance`
- Fidelity level: `Level 2`

### Covertype

- Source: `ucimlrepo.fetch_ucirepo(id=31)` with fallback to a direct UCI download if needed
- Preprocessing:
  - keep target `Cover_Type` as the target column
  - first take a stratified 10 percent sample from the full dataset
  - then split that sample into 80/20 train/test stratified on `Cover_Type`
  - if the 10 percent sample is too large to process reliably, reduce to a stratified 5 percent sample and log that downgrade
- Sensitive attribute: `None`
- Domain: `Environmental / Ecological`
- Fidelity level: `Level 1`
- Special training rule:
  - pass all columns as discrete/categorical to CTGAN because integer-coded columns must not be treated as continuous during synthesis

### Diabetes

- Source URL:
  - `https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv`
- Preprocessing:
  - assign the provided nine-column schema
  - replace zeros in `glucose`, `blood_pressure`, `skin_thickness`, `insulin`, and `bmi` with each column median
  - create `age_group` as `young` for `age <= 35` and `older` for `age > 35`
  - drop original `age`
- Split:
  - 80/20 train/test
  - stratified on `target`
- Sensitive attribute: `age_group`
- Domain: `Healthcare`
- Fidelity level: `Level 2`

## Download And Preprocessing Script

`benchmarks/download_datasets.py` will:

1. Ensure benchmark directories exist.
2. Process the four datasets one by one.
3. Save each dataset as:
   - `benchmarks/datasets/{name}_train.csv`
   - `benchmarks/datasets/{name}_test.csv`
4. Print for each dataset:
   - train/test shapes
   - class balance
   - column names and inferred column types
5. Print a final summary table including:
   - dataset name
   - total samples
   - number of features
   - target class count
   - sensitive attribute

### Download-Time Validation

Before saving and again after reloading from disk, validate:

- dataframe is not empty
- target column exists
- no duplicate column names
- no all-NaN columns
- train/test columns match exactly
- stratification retained all target classes in both splits
- sensitive attribute column exists when required
- diabetes zero-missing replacements were applied successfully
- covertype sampled target classes remain valid

If `ucimlrepo` fails, the script will try the direct-download fallback and log the reason for fallback.

## Training Script

`benchmarks/train_benchmark_models.py` will:

- skip Adult model training entirely
- copy Adult synthetic CSVs into the benchmark synthetic folder
- for `bank`, `covertype`, and `diabetes`:
  - load the benchmark training split
  - train `CTGAN(epochs=300, batch_size=500, verbose=True)`
  - train `TVAESynthesizer(..., epochs=300)`
  - generate exactly 10,000 synthetic rows per model
  - save:
    - `benchmarks/synthetic/{name}_ctgan.csv`
    - `benchmarks/synthetic/{name}_tvae.csv`

### Training Metadata Handling

- TVAE metadata will be detected from the real training dataframe and corrected for categorical columns where needed.
- For `covertype`, CTGAN discrete columns will include every column in the training dataframe.
- Synthetic target values for `covertype` will be validated to ensure only integer classes `1..7` appear.

### Synthetic Validation

After each synthetic dataset is generated, validate:

- shape is `(10000, N_features)`
- no NaN values exist
- columns match the training schema exactly and in order
- target column contains only valid class values
- no column is constant
  - constant columns are warnings, not hard failures
- for `covertype`, target values are integers in `[1, 7]`

Validation results will be printed immediately after generation. Hard failures are logged and stop the script.

## Evaluation Script

`benchmarks/evaluate_benchmarks.py` will compute four metrics for every dataset and model combination.

### Shared Encoding Rule

For any dataframe used in machine learning metrics:

- detect categorical columns via:
  - `df.select_dtypes(include=['object', 'category']).columns.tolist()`
- fit label encoders consistently across aligned real and synthetic columns before model training

### Metric 1: JS Divergence

- numeric columns only
- 50 bins per column
- use epsilon `1e-10`
- compute:
  - `M = 0.5 * (P + Q)`
  - `JS = 0.5 * KL(P || M) + 0.5 * KL(Q || M)`
- average across all numerical columns

The implementation will guard against zero-width histogram ranges and malformed numeric arrays.

### Metric 2: TSTR Accuracy

- use `RandomForestClassifier(n_estimators=100, random_state=42)`
- binary datasets:
  - `adult`
  - `bank`
  - `diabetes`
- multi-class dataset:
  - `covertype`
- compute:
  - `TSTR_Real_Baseline`: train on real train, test on real test
  - `TSTR_Accuracy`: train on synthetic, test on real test
- report percentages

### Metric 3: Membership Inference Attack Advantage

For each synthetic dataset:

- combine real train rows with label `1`
- combine synthetic rows with label `0`
- train a `RandomForestClassifier(n_estimators=100, random_state=42)`
- compute attack accuracy on a hold-out split
- calculate:
  - `MIA_Advantage = attack_accuracy - 0.5`

### Metric 4: Demographic Parity Difference

For datasets with a sensitive attribute:

- compute the absolute difference in positive target rate between the two groups
- use:
  - `sex` for Adult
  - `age_group` for Bank
  - `age_group` for Diabetes

For `covertype`, write `NaN` in the CSV and display `N/A` in printed summaries.

### Evaluation Outputs

Save:

- per-dataset results:
  - `benchmarks/results/{name}_evaluation.csv`
- cross-domain summary:
  - `benchmarks/results/cross_domain_summary.csv`
- mean rank table:
  - `benchmarks/results/mean_rank_table.csv`

Per-dataset result columns:

- `Model`
- `JS_Divergence`
- `TSTR_Accuracy`
- `MIA_Advantage`
- `Demographic_Parity`
- `TSTR_Real_Baseline`

Cross-domain summary adds:

- `Dataset`
- `Domain`
- `Fidelity_Level`

### Mean Rank Design

Rank models across datasets by metric:

- TSTR:
  - descending accuracy, higher is better
- JS:
  - ascending, lower is better
- MIA:
  - ascending, lower is better
- DP:
  - ascending, lower is better

Compute:

- `Mean_TSTR_Rank`
- `Mean_JS_Rank`
- `Mean_MIA_Rank`
- `Mean_DP_Rank`
- `Overall_Mean_Rank`

Datasets with `NaN` demographic parity values for a metric input, such as `covertype`, will be excluded only from that metric’s rank aggregation.

## Visualization Script

`benchmarks/visualize_benchmarks.py` will load the saved CSV outputs and generate four publication-ready PNG figures using:

- `plt.style.use('seaborn-v0_8-whitegrid')`
- `dpi=300`
- colors:
  - real baseline: `#9E9E9E`
  - CTGAN: `#FF9800`
  - TVAE: `#4CAF50`

### Plot 1

- cross-domain TSTR heatmap
- rows: datasets
- columns: Real Baseline, CTGAN, TVAE
- annotate each cell with the value and baseline gap note
- save to:
  - `benchmarks/plots/plot1_tstr_heatmap.png`

### Plot 2

- four-subplot grouped bar dashboard for:
  - JS divergence
  - TSTR accuracy
  - MIA advantage
  - demographic parity
- dataset groups side by side
- annotate bar values
- save to:
  - `benchmarks/plots/plot2_cross_domain_dashboard.png`

### Plot 3

- grouped horizontal bar chart of metric mean ranks by model
- save to:
  - `benchmarks/plots/plot3_mean_rank.png`

### Plot 4

- privacy-utility scatter with eight points
- x:
  - TSTR accuracy
- y:
  - MIA advantage
- marker by dataset
- color by model
- dashed baseline lines by dataset
- shaded ideal region `X > 83` and `Y < 0.15`
- save to:
  - `benchmarks/plots/plot4_privacy_utility_all_domains.png`

Before plotting, validate that required CSVs exist, columns are present, and all expected dataset-model combinations are available.

## Master Runner Design

`benchmarks/run_benchmarks.py` will provide cache-aware orchestration but will not be run until the four explicit scripts succeed manually.

It will:

1. check all benchmark dataset train/test CSVs
2. check all non-Adult synthetic CSVs
3. check evaluation outputs
4. check all four plots
5. run missing stages only
6. print the final summary table and mean rank table
7. print total elapsed time

Skipping is allowed only when the complete expected artifact set for a stage exists.

## Logging And Failure Policy

All scripts will log to:

- `benchmarks/results/benchmark_failures.log`

Each log entry should include:

- timestamp
- stage
- dataset
- severity
- message

Failure policy:

- warnings:
  - log and continue only for non-fatal issues such as constant synthetic columns
- hard failures:
  - log and stop immediately for invalid schema, invalid target classes, failed downloads after fallback, missing required files, or inconsistent evaluation inputs

This satisfies the requirement to mark failures clearly while preventing partial benchmark outputs from being treated as trustworthy.

## Reproducibility

Use deterministic seeds where supported:

- `random_state=42` for sampling and train/test splits
- `random_state=42` for downstream Random Forest models

Persist outputs with stable names so reruns and cache checks remain predictable.

## Verification Strategy

Every stage will verify inputs before use and outputs after writing:

- reload saved CSVs from disk for post-write validation
- compare schemas between train, test, and synthetic outputs
- confirm target class sets match dataset expectations
- confirm sensitive attribute availability where required
- verify required tables are complete before plotting
- validate rank inputs before saving summary tables

This design favors simple scripts like the existing lifecycle, but with stronger correctness controls around corruption, schema drift, and metric miscalculation.

## Implementation Order

After the user reviews and approves this spec:

1. invoke `writing-plans`
2. create the benchmark package structure
3. implement shared helpers and dataset download/preprocessing
4. implement training and synthetic validation
5. implement evaluation and ranking
6. implement plotting
7. run the four required commands in order
8. share outputs and generated plots

## Environment Note

This workspace does not currently appear to be a Git repository, so the design document cannot be committed from the current environment unless repository metadata becomes available later.

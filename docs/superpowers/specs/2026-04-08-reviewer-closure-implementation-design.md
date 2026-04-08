# Reviewer-Closure Implementation Design

## Goal

Address the remaining implementation-oriented reviewer concerns without turning the project into a second research cycle. The design should satisfy the current paper needs while remaining expandable if later reviewers request additional model families, broader iterative lifecycle experiments, or deeper scalability analysis.

## Scope Summary

### Build Now

1. Add one new model family to the non-DP cross-domain benchmark: `TABDDPM`.
2. Add one explicit iterative lifecycle experiment on `Adult`.
3. Add lightweight compute and scalability reporting.
4. Add reproducibility export generation.
5. Integrate all new work into the Drive-backed checkpoint-safe Colab notebook.

### Defer But Leave Hooks For

1. Additional benchmark model families, especially other tabular diffusion variants.
2. Iterative lifecycle runs on `Bank` and `Diabetes`.
3. Deeper compute profiling such as peak memory, per-epoch timing, and scaling curves.
4. Additional privacy and fairness diagnostics.
5. Differential privacy support for newly added benchmark models.

## Why This Scope

This plan is stronger than a manuscript-only response because it adds:

- a named new model family requested by reviewers,
- a concrete lifecycle iteration experiment instead of only a conceptual lifecycle description,
- reproducibility reporting beyond prose,
- a practical compute and scalability layer.

At the same time, it avoids high-risk scope expansion such as implementing DP triangle support for diffusion models or rerunning iterative lifecycle studies across every dataset.

## Existing Assets To Reuse

### Reusable Benchmark Infrastructure

- `benchmarks/train_benchmark_models.py`
- `benchmarks/run_benchmarks.py`
- `benchmarks/evaluate_benchmarks.py`
- `benchmarks/visualize_benchmarks.py`
- `benchmarks/download_datasets.py`
- existing benchmark datasets under `benchmarks/datasets/`

### Reusable Triangle Infrastructure

- `dp_triangle/common.py`
- `dp_triangle/train_dp_variants.py`
- `dp_triangle/evaluate_triangle.py`
- `dp_triangle/visualize_triangle.py`
- `dp_triangle/run_direction3.py`

### Reusable Artifacts

- cross-domain benchmark outputs in `benchmarks/results/`
- preserved DP outputs in `colab_artifacts/`
- curated paper-facing figures in `paper_assets/`
- checkpoint-safe notebook in `notebooks/direction3_multidataset_colab.ipynb`

## Architecture Principles

### 1. Registry-Based Model Expansion

The benchmark pipeline should use a model registry rather than hardcoded CTGAN/TVAE branches. `TABDDPM` is the immediate new model, but the structure must allow additional benchmark model families later.

### 2. Experiment Modes As First-Class Concepts

The project should distinguish between:

- cross-domain benchmark mode,
- iterative lifecycle mode,
- DP triangle mode.

These modes should share data loading and export conventions while keeping their execution logic separate.

### 3. Reporting As Exportable Artifacts

Reproducibility and compute summaries should be generated as explicit outputs, not buried inside notebook text or logs. The same pattern should be reusable for future appendix artifacts.

### 4. Checkpoint-Safe Notebook Stages

Every notebook stage should:

- run independently,
- write outputs into Drive-backed storage,
- export a zip archive after completion.

This avoids the data-loss problem previously encountered in Colab.

## Work Package 1: Model Registry And TABDDPM Integration

### Objective

Add `TABDDPM` to the non-DP cross-domain benchmark as a third benchmark model alongside CTGAN and TVAE.

### Build-Now Design

Create a benchmark model registry that maps:

- model id,
- training function,
- synthetic output naming,
- required dependencies,
- display label for plots and tables.

Recommended new module:

- `benchmarks/benchmark_models.py`

This module should centralize model registration and keep model-specific logic out of the benchmark runner.

### Expected Code Touches

- `benchmarks/train_benchmark_models.py`
- `benchmarks/run_benchmarks.py`
- `benchmarks/evaluate_benchmarks.py`
- `benchmarks/visualize_benchmarks.py`
- new `benchmarks/benchmark_models.py`

### Expected Outputs

- synthetic CSVs for TABDDPM on all four datasets,
- updated benchmark evaluation CSVs,
- updated `cross_domain_summary.csv`,
- updated `mean_rank_table.csv`,
- updated benchmark figures and report artifacts.

### Future Hook

Any future model family should only require:

1. one adapter,
2. one registry entry,
3. no evaluator rewrite.

## Work Package 2: Adult Iterative Lifecycle Experiment

### Objective

Create an explicit lifecycle-in-action experiment rather than relying on a static description of the lifecycle.

### Build-Now Design

Implement a two-round experiment on `Adult`:

- Round 1: baseline comparison across CTGAN, TVAE, and TABDDPM.
- Decision step: choose a model based on a defined lifecycle objective, for example balanced utility, privacy, and fairness.
- Round 2: run a second experiment using the chosen path, likely through the existing Adult DP triangle machinery or a reduced privacy-aware rerun.

The output must make the lifecycle decision visible. The experiment should not merely run two unrelated benchmarks.

Recommended new module:

- `benchmarks/run_iterative_lifecycle.py`

### Expected Outputs

- round-1 comparison artifact,
- decision summary artifact,
- round-2 result artifact,
- one lifecycle iteration summary table or figure suitable for the paper appendix or main text.

### Reuse

- Adult benchmark outputs,
- Adult DP triangle outputs,
- existing evaluation metrics.

### Future Hook

The orchestration should be dataset-parameterized so `Bank` and `Diabetes` can be added later without redesigning the workflow.

## Work Package 3: Compute And Scalability Reporting

### Objective

Respond to reviewer concern about computational complexity and scalability with a practical, evidence-backed reporting layer.

### Build-Now Design

Generate a lightweight compute summary for each benchmark model and dataset including:

- dataset row count,
- feature count,
- training duration,
- artifact size,
- optional device information,
- optional peak GPU memory if available.

This is a practical resource-awareness summary, not a full asymptotic complexity paper.

Recommended new module:

- `benchmarks/export_compute_summary.py`

### Expected Outputs

- `compute_summary.csv`
- one compact figure or markdown summary
- appendix-ready material for the paper

### Reuse

- existing logs,
- saved model files,
- dataset metadata,
- notebook runtime environment.

### Future Hook

Later work can extend this module to include memory traces, dataset scaling sweeps, or per-epoch timing without changing the current reporting contract.

## Work Package 4: Reproducibility Export Layer

### Objective

Make reproducibility visible through generated artifacts rather than relying on narrative description.

### Build-Now Design

Export structured summaries for:

- dataset manifest,
- model manifest,
- benchmark run manifest,
- hyperparameter summary,
- environment/runtime summary,
- result-file inventory.

Recommended new module:

- `benchmarks/export_reproducibility.py`

### Expected Outputs

- `reproducibility_manifest.csv`
- `hyperparameter_summary.csv`
- `environment_summary.md`
- `artifact_inventory.csv`

### Reuse

- saved logs,
- benchmark outputs,
- notebook environment,
- existing file tree.

### Future Hook

This export layer should later accommodate seeds, exact config snapshots, git commit hashes, and package lock captures.

## Work Package 5: Notebook Integration

### Objective

Integrate all new implementation work into the same Drive-backed checkpoint notebook pattern already adopted for Direction 3.

### Build-Now Notebook Structure

1. Environment and Drive setup
2. Repository clone/update
3. Dependency installation
4. GPU check
5. Dataset preparation
6. Cross-domain benchmark: CTGAN
7. Cross-domain benchmark: TVAE
8. Cross-domain benchmark: TABDDPM
9. Benchmark evaluation and figures
10. Adult iterative lifecycle experiment
11. Direction 3 DP triangle runs
12. Compute and scalability export
13. Reproducibility export
14. Final combined archive

### Notebook Design Rules

- every stage must save artifacts under Drive-backed paths,
- every major stage must create a stage-specific zip archive,
- reruns should prefer refresh of evaluation outputs over unnecessary retraining where possible,
- cells should be independently restartable.

### Future Hook

New models or experiment modes should appear as additional checkpoint stages rather than notebook rewrites.

## Output Layout

### Build Now

Recommended new output roots:

- `benchmarks/results/iterative_lifecycle/adult/`
- `benchmarks/results/compute/`
- `benchmarks/results/reproducibility/`

The existing benchmark outputs remain under:

- `benchmarks/results/`

The existing DP triangle outputs remain under:

- `results/`
- `benchmarks/results/dp_triangle/<dataset>/`

### Future Hook

This layout should support:

- additional iterative datasets,
- more benchmark models,
- more reporting artifacts,
- future appendix exports.

## What Is New Versus What Reuses Existing Results

### Mostly Reuse

- dataset preparation,
- benchmark evaluation,
- benchmark visualization,
- DP triangle pipeline,
- artifact archive pattern,
- cross-domain benchmark baseline results,
- existing paper-facing figure organization.

### New But Built On Existing Framework

- `TABDDPM` benchmark integration,
- benchmark model registry,
- compute summary export,
- reproducibility export,
- notebook stage additions.

### Closest To New Experimental Work

- explicit iterative lifecycle orchestration on Adult.

This is the only work package that is conceptually new rather than mainly a framework extension.

## Risks

### Low Risk

- reproducibility export layer,
- compute summary export layer,
- notebook checkpoint integration.

### Medium Risk

- TABDDPM integration,
- updating benchmark plots and reports for a third model.

### Highest Risk

- iterative lifecycle experiment orchestration,
- only because it must encode a meaningful decision loop rather than just chain two scripts.

## Success Criteria

The implementation is complete when:

1. `TABDDPM` appears in cross-domain benchmark outputs, figures, and summary tables.
2. Adult iterative lifecycle outputs exist and make the decision flow explicit.
3. Compute/scalability artifacts are generated automatically.
4. Reproducibility artifacts are generated automatically.
5. The notebook supports all stages with Drive persistence and per-stage archives.
6. The design remains extensible to more models, more datasets, and deeper analysis without structural rework.

## Out Of Scope For This Round

The following are intentionally deferred:

- DP triangle support for TABDDPM or other new benchmark models,
- iterative lifecycle experiments across all datasets,
- full formal complexity benchmarking,
- multiple additional model families beyond one new tabular diffusion-style baseline.

These are valid future extensions, but they should not block the current reviewer-closure implementation cycle.

from __future__ import annotations

import pandas as pd

from benchmarks.visualize_benchmarks import _configure_stdout_utf8, required_plot_paths, validate_plot_inputs


def test_required_plot_paths_lists_all_four_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmarks.visualize_benchmarks.PLOTS_DIR", tmp_path)

    paths = required_plot_paths()

    assert len(paths) == 4
    assert all(path.suffix == ".png" for path in paths)


def test_validate_plot_inputs_requires_expected_columns():
    summary_df = pd.DataFrame({"Dataset": ["adult"], "Model": ["CTGAN"]})
    rank_df = pd.DataFrame({"Model": ["CTGAN"]})

    try:
        validate_plot_inputs(summary_df, rank_df)
    except ValueError as exc:
        assert "missing columns" in str(exc)
    else:
        raise AssertionError("validate_plot_inputs should reject incomplete inputs")


def test_validate_plot_inputs_accepts_three_model_summary():
    summary_rows = []
    for dataset in ["adult", "bank", "covertype", "diabetes"]:
        for model in ["CTGAN", "TVAE", "TABDDPM"]:
            summary_rows.append(
                {
                    "Dataset": dataset,
                    "Model": model,
                    "JS_Divergence": 0.1,
                    "TSTR_Accuracy": 80.0,
                    "MIA_Advantage": 0.2,
                    "Demographic_Parity": 0.3,
                    "TSTR_Real_Baseline": 82.0,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    rank_df = pd.DataFrame(
        {
            "Model": ["CTGAN", "TVAE", "TABDDPM"],
            "Mean_TSTR_Rank": [2.0, 3.0, 1.0],
            "Mean_JS_Rank": [2.0, 3.0, 1.0],
            "Mean_MIA_Rank": [2.0, 3.0, 1.0],
            "Mean_DP_Rank": [2.0, 3.0, 1.0],
            "Overall_Mean_Rank": [2.0, 3.0, 1.0],
        }
    )

    validate_plot_inputs(summary_df, rank_df)


def test_configure_stdout_utf8_reconfigures_when_supported(monkeypatch):
    calls: list[dict[str, str]] = []

    class DummyStdout:
        def reconfigure(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr("sys.stdout", DummyStdout())

    _configure_stdout_utf8()

    assert calls == [{"encoding": "utf-8"}]

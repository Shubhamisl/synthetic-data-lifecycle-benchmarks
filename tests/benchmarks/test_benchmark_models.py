from __future__ import annotations

import builtins
import importlib

from benchmarks.benchmark_models import (
    BENCHMARK_MODELS,
    get_available_benchmark_model_ids,
    get_benchmark_model_display_names,
    get_benchmark_model_ids,
    get_benchmark_models,
    get_trainable_benchmark_model_ids,
)
from benchmarks.common import PROJECT_ROOT


def test_benchmark_models_registry_exposes_expected_default_models():
    assert tuple(BENCHMARK_MODELS) == ("ctgan", "tvae", "tabddpm")
    assert get_benchmark_model_ids() == ("ctgan", "tvae", "tabddpm")
    assert get_benchmark_model_display_names() == ("CTGAN", "TVAE", "TABDDPM")
    assert tuple(spec.model_id for spec in get_benchmark_models()) == ("ctgan", "tvae", "tabddpm")
    assert get_trainable_benchmark_model_ids()[:2] == ("ctgan", "tvae")
    assert get_available_benchmark_model_ids()[:2] == ("ctgan", "tvae")
    assert BENCHMARK_MODELS["ctgan"].adult_source_path == PROJECT_ROOT / "results" / "ctgan_synthetic.csv"
    assert BENCHMARK_MODELS["tvae"].adult_source_path == PROJECT_ROOT / "results" / "tvae_synthetic.csv"
    assert BENCHMARK_MODELS["tabddpm"].adult_source_path is None


def test_tabddpm_trainable_status_tracks_synthcity_availability(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("synthcity"):
            raise ImportError("synthcity is unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    reloaded = importlib.reload(importlib.import_module("benchmarks.benchmark_models"))

    assert tuple(reloaded.BENCHMARK_MODELS) == ("ctgan", "tvae", "tabddpm")
    assert reloaded.BENCHMARK_MODELS["tabddpm"].trainable is False
    assert reloaded.get_available_benchmark_model_ids() == ("ctgan", "tvae")
    assert reloaded.get_trainable_benchmark_model_ids() == ("ctgan", "tvae")

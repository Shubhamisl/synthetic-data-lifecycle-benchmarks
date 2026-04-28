from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .common import PROJECT_ROOT


@dataclass(frozen=True)
class BenchmarkModelSpec:
    model_id: str
    display_name: str
    available: bool = True
    trainable: bool = True
    adult_source_path: Path | None = None


def _synthcity_ddpm_available() -> bool:
    try:
        from synthcity.plugins.generic.plugin_ddpm import TabDDPMPlugin
    except Exception:
        return False

    return TabDDPMPlugin is not None


def _tabddpm_backend_available() -> bool:
    return _synthcity_ddpm_available()


TABDDPM_BACKEND_AVAILABLE = _tabddpm_backend_available()


BENCHMARK_MODELS: dict[str, BenchmarkModelSpec] = {
    "ctgan": BenchmarkModelSpec(
        model_id="ctgan",
        display_name="CTGAN",
        available=True,
        trainable=True,
        adult_source_path=PROJECT_ROOT / "results" / "ctgan_synthetic.csv",
    ),
    "tvae": BenchmarkModelSpec(
        model_id="tvae",
        display_name="TVAE",
        available=True,
        trainable=True,
        adult_source_path=PROJECT_ROOT / "results" / "tvae_synthetic.csv",
    ),
    "tabddpm": BenchmarkModelSpec(
        model_id="tabddpm",
        display_name="TABDDPM",
        available=TABDDPM_BACKEND_AVAILABLE,
        trainable=TABDDPM_BACKEND_AVAILABLE,
    ),
}


def get_benchmark_models() -> tuple[BenchmarkModelSpec, ...]:
    return tuple(BENCHMARK_MODELS.values())


def get_benchmark_model_ids() -> tuple[str, ...]:
    return tuple(BENCHMARK_MODELS)


def get_benchmark_model_display_names() -> tuple[str, ...]:
    return tuple(spec.display_name for spec in get_benchmark_models())


def get_benchmark_model_spec(model_id: str) -> BenchmarkModelSpec:
    return BENCHMARK_MODELS[model_id]


def get_available_benchmark_model_ids() -> tuple[str, ...]:
    return tuple(model_id for model_id, spec in BENCHMARK_MODELS.items() if spec.available)


def get_trainable_benchmark_model_ids() -> tuple[str, ...]:
    return tuple(model_id for model_id, spec in BENCHMARK_MODELS.items() if spec.trainable)

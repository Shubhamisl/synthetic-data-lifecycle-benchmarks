"""Benchmark test package."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_ROOT = PROJECT_ROOT / "benchmarks"
project_root_str = str(PROJECT_ROOT)
benchmark_root_str = str(BENCHMARKS_ROOT)

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

if benchmark_root_str not in __path__:
    __path__.append(benchmark_root_str)

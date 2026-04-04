"""Tests for Direction 3 package."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DP_TRIANGLE_ROOT = PROJECT_ROOT / "dp_triangle"
project_root_str = str(PROJECT_ROOT)
dp_triangle_root_str = str(DP_TRIANGLE_ROOT)

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

if dp_triangle_root_str not in __path__:
    __path__.append(dp_triangle_root_str)

"""
Shared helpers for metric definitions.

To add a new metric:
1. Create a new file in this package (e.g. my_metric.py)
2. Define a function that returns a GEval (or any BaseMetric)
3. Re-export it from __init__.py
"""

from __future__ import annotations

from settings.config import GEVAL_THRESHOLD, JUDGE_MODEL
from models.models import LANGUAGE_LABELS

__all__ = ["GEVAL_THRESHOLD", "JUDGE_MODEL", "LANGUAGE_LABELS"]

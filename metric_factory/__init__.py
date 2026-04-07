"""
metric_factory — pluggable metrics for OAN evals.

To add a new metric:
  1. Create  tests/evals/metric_factory/my_metric.py
  2. Define  def my_metric(**kwargs) -> GEval: ...
  3. Import + re-export it here
  4. Wire it into get_metrics_for_case() (valid / decline path)
"""

from __future__ import annotations

from deepeval.metrics import GEval


from metric_factory.response_validity_metric import response_validity_metric
from metric_factory.response_quality_metric import response_quality_metric
from metric_factory.language_quality import language_quality_metric

__all__ = [
    "response_validity_metric",
    "response_quality_metric",
    "language_quality_metric",
    "get_metrics_for_case",
]


def get_metrics_for_case(
    *,
    is_decline: bool,
    lang_code: str,
    judge_model: str | None = None,
) -> list[GEval]:
    """Return the correct metric list for a test case.

    * Decline cases  → [ResponseValidity]
    * Valid cases    → [ResponseValidity, ResponseQuality, LanguageQuality]
    """
    if is_decline:
        return [response_validity_metric(judge_model=judge_model)]

    return [
        response_validity_metric(judge_model=judge_model),
        response_quality_metric(judge_model=judge_model),
        language_quality_metric(lang_code, judge_model=judge_model),
    ]

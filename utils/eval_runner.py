"""
eval_runner.py — programmatic evaluation runner (no pytest).

This is used by the FastAPI endpoint to execute an eval run and emit
JSON/PDF reports on demand.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from settings.config import DATASET_PATH, MH_DATASET_PATH
from utils.dataset_loader import build_test_cases
from utils.evaluator import evaluate_case
from utils.execution import fetch_all_outputs, fetch_all_mh_outputs
from utils.report import ReportBuilder, save_report, save_report_pdf


def _build_file_prefix(run_meta: dict[str, Any] | None, default: str = "eval_report") -> str:
    """Build a descriptive file prefix from run metadata.

    Examples:
        None                           -> "eval_report"
        {name: "nightly", version: ""} -> "eval_report_nightly"
        {name: "qa", version: "v2.1"} -> "eval_report_qa_v2.1"
    """
    parts = [default]
    if run_meta:
        name = run_meta.get("name", "").strip()
        version = run_meta.get("version", "").strip()
        if name:
            # sanitise for filesystem safety
            parts.append(name.replace(" ", "_").replace("/", "-")[:40])
        if version:
            parts.append(version.replace(" ", "_").replace("/", "-")[:20])
    return "_".join(parts)


class EvalRunError(RuntimeError):
    pass


def run_eval(
    *,
    base_url: str,
    judge_model: str | None = None,
    api_key: str | None = None,
    service_api_key: str | None = None,
    dataset_path: str | None = None,
    max_workers: int | None = None,
    output_dir: str | Path = "reports",
    run_meta: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Run all test cases and return (report_dict, report_json_path)."""
    if api_key:
        # Provide the key for Deepeval / OpenAI (if required by the judge model).
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ.setdefault("DEEPEVAL_API_KEY", api_key)

    cases = build_test_cases(dataset_path or DATASET_PATH)
    outputs = fetch_all_outputs(
        cases,
        base_url=base_url,
        api_key=service_api_key,
        max_workers=max_workers,
    )

    builder = ReportBuilder()

    for tc in cases:
        actual_output = outputs.get(tc.name)
        if not actual_output:
            builder.add_api_error(tc)
            continue
        result = evaluate_case(tc, actual_output, judge_model=judge_model)
        builder.add_from_eval(tc, result, actual_output)

    report = builder.build(
        judge_model=judge_model,
        run_meta=run_meta,
        meta_extra={
            "base_url": base_url,
            "dataset_path": dataset_path or DATASET_PATH,
        },
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = _build_file_prefix(run_meta, "eval_report")
    out_dir = Path(output_dir)
    out_path = out_dir / f"{prefix}_{ts}.json"
    saved = save_report(report, out_path)

    return report, saved


def run_mh_eval(
    *,
    base_url: str,
    token: str,
    judge_model: str | None = None,
    api_key: str | None = None,
    dataset_path: str | None = None,
    max_workers: int | None = None,
    output_dir: str | Path = "reports",
    run_meta: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Run MH eval: fetch outputs via Mh_OANEvalClient, evaluate, return report."""
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ.setdefault("DEEPEVAL_API_KEY", api_key)

    cases = build_test_cases(dataset_path or MH_DATASET_PATH)
    outputs = fetch_all_mh_outputs(
        cases,
        base_url=base_url,
        token=token,
        max_workers=max_workers,
    )

    builder = ReportBuilder()

    for tc in cases:
        actual_output = outputs.get(tc.name)
        if not actual_output:
            builder.add_api_error(tc)
            continue
        result = evaluate_case(tc, actual_output, judge_model=judge_model)
        builder.add_from_eval(tc, result, actual_output)

    report = builder.build(
        judge_model=judge_model,
        run_meta=run_meta,
        meta_extra={
            "base_url": base_url,
            "dataset_path": dataset_path or MH_DATASET_PATH,
            "variant": "mh",
        },
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = _build_file_prefix(run_meta, "mh_eval_report")
    out_dir = Path(output_dir)
    out_path = out_dir / f"{prefix}_{ts}.json"
    saved = save_report(report, out_path)

    return report, saved


def render_pdf(report: dict[str, Any], output_dir: str | Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_meta = report.get("meta", {}).get("run_meta")
    prefix = _build_file_prefix(run_meta, "eval_report")
    out_dir = Path(output_dir)
    out_path = out_dir / f"{prefix}_{ts}.pdf"
    return save_report_pdf(report, out_path)

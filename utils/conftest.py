"""
conftest.py — pytest session hooks for OAN evaluation reports.

After the test session finishes, the collected CaseReport objects are
serialised to  reports/eval_report_<YYYYMMDD_HHMMSS>.json.

That JSON can be opened directly, fed into any PDF renderer (e.g. the
companion  report_to_pdf.py  utility), or imported into a spreadsheet.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

# Ensure local modules are importable whether pytest is launched from this
# directory or from its parent workspace.
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.report import report_builder, save_report


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:  # noqa: ARG001
    """Save the evaluation report once ALL tests have finished."""
    if not report_builder._cases:
        return  # nothing was collected (e.g. dry-run or wrong mark filter)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path("reports") / f"eval_report_{ts}.json"

    report = report_builder.build()
    saved = save_report(report, out_path)

    meta = report["meta"]
    total   = meta["total_cases"]
    passed  = meta["passed_cases"]
    failed  = meta["failed_cases"]
    errors  = meta["error_cases"]
    pct     = meta["overall_pass_rate"] * 100

    sep = "=" * 64
    print(
        f"\n\n{sep}\n"
        f"  EVAL REPORT  →  {saved}\n"
        f"{sep}\n"
        f"  Total : {total}   Passed : {passed}   "
        f"Failed : {failed}   Errors : {errors}\n"
        f"  Overall pass rate : {pct:.1f}%\n"
        f"{sep}\n"
    )

    # Surface the worst-offending categories for quick visibility
    top_fail = report["summary"]["top_failing_categories"]
    if top_fail:
        print("  Top failing categories:")
        for item in top_fail[:5]:
            if item["failed"] + item["error"] == 0:
                break
            print(
                f"    {item['category']:<35}  "
                f"failed={item['failed'] + item['error']}/{item['total']}  "
                f"({item['failure_rate'] * 100:.0f}%)"
            )
        print()

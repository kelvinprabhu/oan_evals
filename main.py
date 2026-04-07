from __future__ import annotations

import os
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException,Depends
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from auth.auth import verify_api_key
from utils.eval_runner import render_pdf, run_eval, run_mh_eval

app = FastAPI(title="OAN Eval API", version="1.0")


class RunMetadata(BaseModel):
    """Optional metadata to tag and identify a test run."""
    name: str = Field("", description="Run name (e.g. 'nightly', 'pre-release')")
    version: str = Field("", description="Version tag (e.g. 'v2.1.0')")
    note: str = Field("", description="Free-form note about this run")


class EvalRequest(BaseModel):
    base_url: str = Field(..., description="OAN API base URL (e.g. http://host:8000)")
    output_format: Literal["json", "pdf"] = Field("json", description="Output format")
    run_meta: Optional[RunMetadata] = Field(None, description="Optional run metadata for tagging")


class MHEvalRequest(BaseModel):
    token: str = Field(..., description="MH client auth token")
    url: str = Field(..., description="MH client base URL (e.g. http://host:8000)")
    output_format: Literal["json", "pdf"] = Field("json", description="Output format")
    run_meta: Optional[RunMetadata] = Field(None, description="Optional run metadata for tagging")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/run-eval",
    responses={
        400: {"description": "Bad request"},
        500: {"description": "Evaluation failed"},
    },
    dependencies=[Depends(verify_api_key)]
)
def run_eval_endpoint(req: EvalRequest):
    judge_model = os.environ.get("JUDGE_MODEL")
    api_key = os.environ.get("OPENAI_API_KEY")
    run_meta = req.run_meta.model_dump() if req.run_meta else None

    if not judge_model or not api_key:
        raise HTTPException(status_code=500, detail="JUDGE_MODEL and OPENAI_API_KEY must be set in environment")

    try:
        report, json_path = run_eval(
            base_url=req.base_url,
            judge_model=judge_model,
            api_key=api_key,
            run_meta=run_meta,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))

    if req.output_format == "json":
        return JSONResponse({
            "json_path": str(json_path),
            "report": report,
        })

    try:
        pdf_path = render_pdf(report, "reports")
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=pdf_path.name,
        headers={"X-Report-Json": str(json_path)},
    )


@app.post(
    "/run-eval-mh",
    responses={
        400: {"description": "Bad request"},
        500: {"description": "MH evaluation failed"},
    },
    dependencies=[Depends(verify_api_key)]
)
def run_eval_mh_endpoint(req: MHEvalRequest):
    judge_model = os.environ.get("JUDGE_MODEL")
    api_key = os.environ.get("OPENAI_API_KEY")
    run_meta = req.run_meta.model_dump() if req.run_meta else None

    if not judge_model or not api_key:
        raise HTTPException(status_code=500, detail="JUDGE_MODEL and OPENAI_API_KEY must be set in environment")

    try:
        report, json_path = run_mh_eval(
            base_url=req.url,
            token=req.token,
            judge_model=judge_model,
            api_key=api_key,
            run_meta=run_meta,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))

    if req.output_format == "json":
        return JSONResponse({
            "json_path": str(json_path),
            "report": report,
        })

    try:
        pdf_path = render_pdf(report, "reports")
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=pdf_path.name,
        headers={"X-Report-Json": str(json_path)},
    )
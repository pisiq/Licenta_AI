from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

# Ensure project root is importable when launched with `uvicorn api.main:app`.
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.schemas import (
    GenerationRequest,
    GenerationResponse,
    PipelinePdfOptions,
    PipelineResponse,
    ScoringJsonRequest,
    ScoringResponse,
)
from api.services import generate_review, model_version, score_from_json_source, score_from_pdf_bytes


app = FastAPI(
    title="Licenta Models API",
    description="HTTP API for scoring and review-generation models used by the .NET client.",
    version="1.0.0",
)


@app.get("/v1/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_version": model_version(),
        "cwd": os.getcwd(),
    }


@app.post("/v1/scoring/predict", response_model=ScoringResponse)
def scoring_predict(request: ScoringJsonRequest) -> ScoringResponse:
    try:
        scored, timings = score_from_json_source(
            parsed_json=request.parsed_json,
            json_path=request.json_path,
        )
        return ScoringResponse(
            request_id=str(uuid.uuid4()),
            model_version=model_version(),
            timings_ms=timings,
            data=scored,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}") from exc


@app.post("/v1/scoring/predict-pdf", response_model=ScoringResponse)
async def scoring_predict_pdf(file: UploadFile = File(...)) -> ScoringResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted.")

    try:
        scored, timings = score_from_pdf_bytes(
            pdf_bytes=await file.read(),
            filename=file.filename,
        )
        return ScoringResponse(
            request_id=str(uuid.uuid4()),
            model_version=model_version(),
            timings_ms=timings,
            data=scored,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF scoring failed: {exc}") from exc


@app.post("/v1/review/generate", response_model=GenerationResponse)
def review_generate(request: GenerationRequest) -> GenerationResponse:
    try:
        review_text, timings = generate_review(
            title=request.title,
            abstract=request.abstract,
            body=request.body,
            scores=request.scores,
            max_new_tokens=request.max_new_tokens,
            num_beams=request.num_beams,
        )
        return GenerationResponse(
            request_id=str(uuid.uuid4()),
            model_version=model_version(),
            timings_ms=timings,
            review_text=review_text,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Review generation failed: {exc}") from exc


@app.post("/v1/pipeline/predict-pdf", response_model=PipelineResponse)
async def pipeline_predict_pdf(
    file: UploadFile = File(...),
    include_review: bool = Form(default=True),
    num_beams: int = Form(default=4),
    max_new_tokens: int = Form(default=512),
) -> PipelineResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted.")

    try:
        options = PipelinePdfOptions(
            include_review=include_review,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        scored, timings = score_from_pdf_bytes(
            pdf_bytes=await file.read(),
            filename=file.filename,
        )

        review_text = ""
        if options.include_review:
            review_text, review_timings = generate_review(
                title=scored["title"],
                abstract=scored["abstract"],
                body=scored["body"],
                scores=scored["scores"],
                max_new_tokens=options.max_new_tokens,
                num_beams=options.num_beams,
            )
            timings["review"] = review_timings["total"]
            timings["total"] += review_timings["total"]

        return PipelineResponse(
            request_id=str(uuid.uuid4()),
            model_version=model_version(),
            timings_ms=timings,
            data=scored,
            review_text=review_text,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc


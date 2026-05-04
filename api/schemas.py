from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator


class ScoringJsonRequest(BaseModel):
    """Request body for scoring from parsed JSON content or local JSON path."""

    parsed_json: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Parsed paper JSON payload (metadata + sections).",
    )
    json_path: Optional[str] = Field(
        default=None,
        description="Local filesystem path to an existing parsed JSON file.",
    )

    @model_validator(mode="after")
    def validate_input_source(self) -> "ScoringJsonRequest":
        if not self.parsed_json and not self.json_path:
            raise ValueError("Provide either parsed_json or json_path.")
        return self


class GenerationRequest(BaseModel):
    """Request body for review text generation from already prepared fields."""

    title: str = ""
    abstract: str = ""
    body: str = ""
    scores: Dict[str, float]
    num_beams: int = Field(default=4, ge=1, le=12)
    max_new_tokens: int = Field(default=512, ge=64, le=2048)


class PipelinePdfOptions(BaseModel):
    """Optional generation settings for full PDF -> score + review pipeline."""

    include_review: bool = True
    num_beams: int = Field(default=4, ge=1, le=12)
    max_new_tokens: int = Field(default=512, ge=64, le=2048)


class ScoringResult(BaseModel):
    title: str = ""
    abstract: str = ""
    body: str = ""
    scores: Dict[str, float]


class ApiResponse(BaseModel):
    request_id: str
    model_version: str
    timings_ms: Dict[str, int]


class ScoringResponse(ApiResponse):
    data: ScoringResult


class GenerationResponse(ApiResponse):
    review_text: str


class PipelineResponse(ApiResponse):
    data: ScoringResult
    review_text: str = ""

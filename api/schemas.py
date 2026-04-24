"""Pydantic schemas for the malicious-URL detection API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response returned by ``GET /health``."""

    status:        Literal["ok"] = "ok"
    model_loaded:  bool
    model_name:    str | None = None
    n_features:    int


class URLPredictRequest(BaseModel):
    """Payload for ``POST /predict``.

    ``url`` is kept as a plain ``str`` (not ``HttpUrl``) because malicious
    URLs can legitimately be malformed and we still want to classify them.
    """

    url: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Raw URL to classify.",
        examples=["http://example.com/login?id=42"],
    )


class URLPredictResponse(BaseModel):
    """Response returned by ``POST /predict``."""

    url:           str
    label:         Literal["benign", "malicious"]
    probability:   float = Field(
        ..., ge=0.0, le=1.0,
        description="Probability the URL belongs to the 'malicious' class.",
    )
    model_name:    str = Field(
        ..., description="Identifier of the checkpoint used for inference.",
    )
    features_used: int = Field(
        ..., description="Number of features extracted (expected: 12).",
    )

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
    mode: Literal["binary", "multiclass"] = Field(
        "binary",
        description="Prediction mode: binary (benign/malicious) or multiclass.",
    )


class URLPredictResponse(BaseModel):
    """Response returned by ``POST /predict``."""

    url:           str
    mode:          Literal["binary", "multiclass"]
    label:         str
    probability:   float = Field(
        ..., ge=0.0, le=1.0,
        description="Probability of the predicted label.",
    )
    model_name:    str = Field(
        ..., description="Identifier of the checkpoint used for inference.",
    )
    features_used: int = Field(
        ..., description="Number of features extracted (expected: 12).",
    )
    probabilities: dict[str, float] | None = Field(
        None,
        description="Optional probability distribution for all supported classes.",
    )

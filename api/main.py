"""FastAPI application for malicious URL detection serving."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from api.schemas import HealthResponse, URLPredictRequest, URLPredictResponse
from api.utils import SELECTED_FEATURES, extract_features


_state: dict[str, Any] = {
    "model": None,
    "scaler": None,
    "imputer": None,
    "model_name": None,
}


def _to_probability(model: Any, x: np.ndarray) -> float:
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(x)[0, 1])
        return min(max(proba, 0.0), 1.0)

    if hasattr(model, "decision_function"):
        score = float(np.ravel(model.decision_function(x))[0])
        return float(1.0 / (1.0 + np.exp(-score)))

    pred = float(np.ravel(model.predict(x))[0])
    return float(1.0 if pred >= 0.5 else 0.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    root = Path(__file__).resolve().parent.parent
    ckpt_path = root / "checkpoints" / "best_model.joblib"

    if ckpt_path.exists():
        bundle = joblib.load(ckpt_path)
        _state["model"] = bundle.get("model")
        _state["scaler"] = bundle.get("scaler")
        _state["imputer"] = bundle.get("imputer")
        _state["model_name"] = bundle.get("model_name", "unknown")
        print(f"[startup] loaded checkpoint: {ckpt_path}")
    else:
        print(f"[startup] checkpoint not found: {ckpt_path}")

    yield


tags_metadata = [
    {"name": "System", "description": "Service status and operational endpoints."},
    {"name": "Inference", "description": "Malicious URL classification endpoints."},
]

app = FastAPI(
    title="Malicious URL Detection API",
    version="0.2.0",
    description=(
        "Predicts whether a URL is benign or malicious "
        "(phishing / malware / defacement / spam)."
    ),
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/app")


@app.get("/app", include_in_schema=False)
async def frontend() -> FileResponse:
    app_file = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
    if not app_file.exists():
        raise HTTPException(status_code=404, detail="Frontend file not found.")
    return FileResponse(app_file)


@app.get(
    "/health",
    tags=["System"],
    summary="Health check",
    response_model=HealthResponse,
)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_state["model"] is not None,
        model_name=_state["model_name"],
        n_features=len(SELECTED_FEATURES),
    )


@app.post(
    "/predict",
    tags=["Inference"],
    summary="Classify a single URL",
    response_model=URLPredictResponse,
)
async def predict(payload: URLPredictRequest) -> URLPredictResponse:
    if _state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train and export checkpoints first.",
        )

    x = extract_features(payload.url).reshape(1, -1)
    if _state["imputer"] is not None:
        x = _state["imputer"].transform(x)
    if _state["scaler"] is not None:
        x = _state["scaler"].transform(x)

    proba = _to_probability(_state["model"], x)
    label = "malicious" if proba >= 0.5 else "benign"

    return URLPredictResponse(
        url=payload.url,
        label=label,
        probability=proba,
        model_name=str(_state["model_name"] or "unknown"),
        features_used=len(SELECTED_FEATURES),
    )

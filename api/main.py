"""FastAPI application for malicious URL detection serving."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.schemas import (
    HealthResponse,
    URLPredictRequest,
    URLPredictResponse,
)
from api.utils import SELECTED_FEATURES, extract_features


# Process-wide model registry. Populated by the lifespan hook on startup.
_state: dict[str, Any] = {
    "model":        None,
    "scaler":       None,
    "imputer":      None,
    "model_name":   None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained checkpoint and preprocessing objects on startup.

    The winning artefact from notebook 03 (RandomForest / XGBoost_PSO /
    Improved_DNN — whichever topped the ``Accuracy ↓, FPR ↑`` ranking) is
    expected under ``checkpoints/best_model.joblib`` together with the
    fitted ``SimpleImputer`` and ``StandardScaler``.
    """
    # TODO: replace the stub below with actual joblib.load calls once the
    # training CLI (src/train.py) persists a checkpoint.
    print("[startup] model checkpoint loading is not implemented yet")
    yield
    # Nothing to clean up for sklearn / keras in-process models.


tags_metadata = [
    {"name": "System",    "description": "Service status and operational endpoints."},
    {"name": "Inference", "description": "Malicious URL classification endpoints."},
]

app = FastAPI(
    title="Malicious URL Detection API",
    version="0.1.0",
    description=(
        "Predicts whether a URL is benign or malicious "
        "(phishing / malware / defacement / spam). Classifier trained on "
        "12 lexical features selected from the ISCX-URL2016 dataset."
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
    """Redirect root path to the Swagger UI."""
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    tags=["System"],
    summary="Health check",
    response_model=HealthResponse,
)
async def health() -> HealthResponse:
    """Service availability probe used by the Docker HEALTHCHECK."""
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
    """Classify one URL as benign or malicious."""
    if _state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet — see README Roadmap §14.",
        )

    # Reference implementation (enabled once the checkpoint exists):
    #   x = extract_features(payload.url).reshape(1, -1)
    #   x = _state["imputer"].transform(x)
    #   x = _state["scaler"].transform(x)
    #   proba = float(_state["model"].predict_proba(x)[0, 1])
    #   label = "malicious" if proba >= 0.5 else "benign"
    #   return URLPredictResponse(
    #       url=payload.url, label=label, probability=proba,
    #       model_name=_state["model_name"],
    #       features_used=len(SELECTED_FEATURES),
    #   )
    raise HTTPException(
        status_code=501,
        detail="Prediction pipeline not wired up yet.",
    )

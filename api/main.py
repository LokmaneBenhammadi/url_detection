"""FastAPI application for malicious URL detection serving."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from api.schemas import HealthResponse, URLPredictRequest, URLPredictResponse
from api.utils import SELECTED_FEATURES, extract_features
from src.features import is_legitimate_domain

_state: dict[str, Any] = {
    'models': {},
    'model_names': {},
    'scaler': None,
    'imputer': None,
    'selected_features': [],
    'label_encoder': None,
    'classes': [],
}


def _to_probability(model: Any, x: np.ndarray) -> float:
    if hasattr(model, 'predict_proba'):
        proba = float(model.predict_proba(x)[0, 1])
        return min(max(proba, 0.0), 1.0)

    if hasattr(model, 'decision_function'):
        score = float(np.ravel(model.decision_function(x))[0])
        return float(1.0 / (1.0 + np.exp(-score)))

    pred = float(np.ravel(model.predict(x))[0])
    return float(1.0 if pred >= 0.5 else 0.0)


def _extract_domain(url: str) -> str | None:
    """Extract domain from a URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.hostname or parsed.netloc
        return domain.lower() if domain else None
    except Exception:
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    root = Path(__file__).resolve().parent.parent
    ckpt_path = root / 'checkpoints' / 'best_model.joblib'

    if ckpt_path.exists():
        bundle = joblib.load(ckpt_path)
        _state['models'] = bundle.get('models', {})
        _state['model_names'] = bundle.get('model_names', {})
        _state['imputer'] = bundle.get('imputer')
        _state['scaler'] = bundle.get('scaler')
        _state['selected_features'] = bundle.get('selected_features', SELECTED_FEATURES)
        _state['label_encoder'] = bundle.get('label_encoder')
        _state['classes'] = bundle.get('classes', [])
        print(f'[startup] loaded checkpoint: {ckpt_path}')
    else:
        print(f'[startup] checkpoint not found: {ckpt_path}')

    yield


tags_metadata = [
    {'name': 'System', 'description': 'Service status and operational endpoints.'},
    {'name': 'Inference', 'description': 'Malicious URL classification endpoints.'},
]

app = FastAPI(
    title='Malicious URL Detection API',
    version='0.2.0',
    description=(
        'Predicts whether a URL is benign or malicious ' 
        '(phishing / malware / defacement / spam).'
    ),
    openapi_tags=tags_metadata,
    docs_url='/docs',
    redoc_url='/redoc',
    openapi_url='/openapi.json',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/', include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url='/app')


@app.get('/app', include_in_schema=False)
async def frontend() -> FileResponse:
    app_file = Path(__file__).resolve().parent.parent / 'frontend' / 'index.html'
    if not app_file.exists():
        raise HTTPException(status_code=404, detail='Frontend file not found.')
    return FileResponse(app_file)


@app.get(
    '/health',
    tags=['System'],
    summary='Health check',
    response_model=HealthResponse,
)
async def health() -> HealthResponse:
    loaded = bool(_state['models'])
    model_name = _state['model_names'].get('binary') if loaded else None
    return HealthResponse(
        status='ok',
        model_loaded=loaded,
        model_name=model_name,
        n_features=len(SELECTED_FEATURES),
    )


@app.post(
    '/predict',
    tags=['Inference'],
    summary='Classify a single URL',
    response_model=URLPredictResponse,
)
async def predict(payload: URLPredictRequest) -> URLPredictResponse:
    if not _state['models']:
        raise HTTPException(
            status_code=503,
            detail='Model not loaded. Train and export checkpoints first.',
        )

    if payload.mode not in _state['models']:
        raise HTTPException(
            status_code=400,
            detail=f'Model mode {payload.mode} is not available.',
        )

    # Check domain reputation first
    domain = _extract_domain(payload.url)
    is_legit = is_legitimate_domain(domain) if domain else False

    x = extract_features(payload.url).reshape(1, -1)
    if _state['imputer'] is not None:
        x = _state['imputer'].transform(x)
    if _state['scaler'] is not None:
        x = _state['scaler'].transform(x)

    model = _state['models'][payload.mode]
    probabilities: dict[str, float] | None = None

    if payload.mode == 'binary':
        proba = _to_probability(model, x)
        
        # Override for legitimate domains
        if is_legit:
            label = 'benign'
            proba = 0.01  # Very low malicious probability
        else:
            label = 'malicious' if proba >= 0.8 else 'benign'
        
        probabilities = {
            'benign': float(1.0 - proba),
            'malicious': float(proba),
        }
    else:
        if not hasattr(model, 'predict_proba'):
            raise HTTPException(
                status_code=500,
                detail='Multiclass model does not support probability output.',
            )
        raw_probs = model.predict_proba(x)[0]
        best_idx = int(np.argmax(raw_probs))
        label = (
            _state['label_encoder'].inverse_transform([best_idx])[0]
            if _state['label_encoder'] is not None
            else str(_state['classes'][best_idx])
        )
        proba = float(raw_probs[best_idx])
        
        # Override for legitimate domains in multiclass
        if is_legit:
            label = 'benign'
            proba = 0.99  # Very high benign probability
            # Adjust probabilities to reflect benign
            raw_probs_copy = raw_probs.copy()
            raw_probs_copy[:] = 0.001  # Small equal probability for others
            benign_idx = list(_state['label_encoder'].classes_).index('benign') if 'benign' in _state['label_encoder'].classes_ else 0
            raw_probs_copy[benign_idx] = 0.996
            raw_probs = raw_probs_copy
        
        probabilities = {
            str(cls): float(p) for cls, p in zip(_state['classes'], raw_probs)
        }

    return URLPredictResponse(
        url=payload.url,
        mode=payload.mode,
        label=label,
        probability=proba,
        model_name=str(_state['model_names'].get(payload.mode, 'unknown')),
        features_used=len(SELECTED_FEATURES),
        probabilities=probabilities,
    )

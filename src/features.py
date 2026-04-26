from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

SELECTED_FEATURES: list[str] = [
    'Entropy_Domain',
    'argDomanRatio',
    'NumberRate_FileName',
    'CharacterContinuityRate',
    'argPathRatio',
    'ArgUrlRatio',
    'pathurlRatio',
    'domainUrlRatio',
    'domainlength',
    'NumberRate_AfterPath',
    'NumberofDotsinURL',
    'domain_token_count',
]

TARGET_CANDIDATES: list[str] = [
    'class', 'Class', 'Type', 'Label', 'label', 'URL_Type_obf_Type'
]

DATA_CANDIDATES: list[str] = [
    'All_clean.csv',
    'All.csv',
]

_SENTINEL = -1.0
_EPS = 1e-9

# List of known-legitimate domains (top Alexa/popular sites)
# These domains are unlikely to be malicious despite URL features
LEGITIMATE_DOMAINS = {
    'google.com', 'youtube.com', 'facebook.com', 'wikipedia.org',
    'amazon.com', 'twitter.com', 'instagram.com', 'linkedin.com',
    'reddit.com', 'github.com', 'stackoverflow.com', 'medium.com',
    'apple.com', 'microsoft.com', 'github.io', 'wordpress.com',
    'mozilla.org', 'w3.org', 'w3schools.com', 'npmjs.com',
    'docker.com', 'kubernetes.io', 'tensorflow.org', 'pytorch.org',
    'pixabay.com', 'unsplash.com', 'pexels.com', 'freepik.com',
    'bbc.com', 'cnn.com', 'reuters.com', 'bbc.co.uk',
    'gov.uk', 'gov.us', 'edu', 'org', 'net',  # Common suffixes
}


def _safe_div(num: float, den: float) -> float:
    return float(num) / (float(den) + _EPS)


def is_legitimate_domain(domain: str) -> bool:
    """Check if a domain is in the whitelist of known-legitimate sites."""
    if not domain:
        return False
    domain_lower = domain.lower().strip()
    # Remove www prefix for matching
    if domain_lower.startswith('www.'):
        domain_lower = domain_lower[4:]
    
    # Exact match or suffix match
    if domain_lower in LEGITIMATE_DOMAINS:
        return True
    # Check if it's a subdomain of a legitimate domain
    for legit in LEGITIMATE_DOMAINS:
        if domain_lower.endswith('.' + legit):
            return True
        if domain_lower == legit:
            return True
    return False


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts: dict[str, int] = {}
    for char in text:
        counts[char] = counts.get(char, 0) + 1
    length = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


def _normalized_entropy(text: str) -> float:
    if not text or len(text) < 2:
        return 0.0
    return _shannon_entropy(text) / math.log2(len(text))


def _digit_rate(text: str) -> float:
    if not text:
        return _SENTINEL
    return _safe_div(sum(1 for c in text if c.isdigit()), len(text))


def _longest_contiguous_run(text: str) -> int:
    if not text:
        return 0
    best = 1
    run = 1
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            run += 1
            if run > best:
                best = run
        else:
            run = 1
    return best


def _domain_token_count(domain: str) -> int:
    if not domain:
        return 0
    tokens = [t for t in re.split(r'[^a-zA-Z0-9]+', domain) if t]
    return len(tokens)


def _find_data_path(data_dir: Path, candidates: Iterable[str] | None = None) -> Path:
    candidates = list(candidates or DATA_CANDIDATES)
    for candidate in candidates:
        path = data_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No dataset CSV found in {data_dir}. Expected one of: {candidates}"
    )


def _find_target_column(df: pd.DataFrame) -> str:
    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"No target column found. Tried: {TARGET_CANDIDATES}. "
        f"Available columns sample: {list(df.columns[:20])}"
    )


def load_dataset(data_dir: Path | str) -> pd.DataFrame:
    data_dir = Path(data_dir)
    path = _find_data_path(data_dir)
    df = pd.read_csv(path, low_memory=False)
    if 'URL_Type_obf_Type' in df.columns and 'class' not in df.columns:
        # keep the original dataset target naming if only the raw target exists
        pass
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in SELECTED_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing selected features in dataset: {missing}")
    X = df[SELECTED_FEATURES].copy()
    return X


def build_targets(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    target_col = _find_target_column(df)
    y_raw = df[target_col].astype(str).copy()
    y_binary = np.where(y_raw == 'benign', 0, 1).astype(int)
    label_encoder = LabelEncoder()
    y_multi = label_encoder.fit_transform(y_raw)
    return y_binary, y_multi, label_encoder


def extract_features(url: str) -> np.ndarray:
    raw = (url or '').strip()
    normalized = raw if '://' in raw else f'http://{raw}'
    parsed = urlparse(normalized)

    domain = (parsed.hostname or parsed.netloc or '').lower()
    if domain.startswith('.'):
        domain = domain.lstrip('.')
    path = parsed.path or ''
    query = parsed.query or ''
    fragment = parsed.fragment or ''
    after_path = query + fragment
    file_name = ''
    if path and path != '/':
        file_name = path.rsplit('/', 1)[-1]

    url_for_rates = normalized
    if not url_for_rates:
        url_for_rates = raw

    features = {
        'Entropy_Domain': _normalized_entropy(domain) if domain else _SENTINEL,
        'argDomanRatio': _safe_div(len(query), len(domain)) if domain else _SENTINEL,
        'NumberRate_FileName': _digit_rate(file_name),
        'CharacterContinuityRate': _safe_div(
            _longest_contiguous_run(url_for_rates),
            len(url_for_rates),
        ) if url_for_rates else _SENTINEL,
        'argPathRatio': (
            _safe_div(len(query), len(path)) if path and path != '/' else _SENTINEL
        ),
        'ArgUrlRatio': _safe_div(len(query), len(url_for_rates)) if url_for_rates else _SENTINEL,
        'pathurlRatio': _safe_div(len(path), len(url_for_rates)) if url_for_rates else _SENTINEL,
        'domainUrlRatio': _safe_div(len(domain), len(url_for_rates)) if url_for_rates else _SENTINEL,
        'domainlength': float(len(domain)),
        'NumberRate_AfterPath': _digit_rate(after_path),
        'NumberofDotsinURL': float(url_for_rates.count('.')),
        'domain_token_count': float(_domain_token_count(domain)),
    }

    return np.array([features[name] for name in SELECTED_FEATURES], dtype=np.float64)


def binary_label_encoder() -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.fit(['benign', 'malicious'])
    return encoder

"""URL feature-extraction helpers for the malicious-URL detection API."""

from __future__ import annotations

import math
import re
from urllib.parse import urlparse

import numpy as np


# The 12 features selected by notebook 02 (MutualInfo, K=12).
# Order is load-bearing: it must match the column order the training-time
# StandardScaler and SimpleImputer were fitted on.
SELECTED_FEATURES: tuple[str, ...] = (
    "Entropy_Domain",
    "argDomanRatio",
    "NumberRate_FileName",
    "CharacterContinuityRate",
    "argPathRatio",
    "ArgUrlRatio",
    "pathurlRatio",
    "domainUrlRatio",
    "domainlength",
    "NumberRate_AfterPath",
    "NumberofDotsinURL",
    "domain_token_count",
)


_EPS = 1e-9


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den + _EPS)


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


def _digit_rate(text: str) -> float:
    if not text:
        return 0.0
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
    tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", domain) if t]
    return len(tokens)


def extract_features(url: str) -> np.ndarray:
    """Compute a 12-feature lexical vector for one URL.

    Returns a 1-D float64 array ordered exactly as SELECTED_FEATURES.
    """
    raw = (url or "").strip()
    normalized = raw if "://" in raw else f"http://{raw}"
    parsed = urlparse(normalized)

    domain = (parsed.hostname or parsed.netloc or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""
    after_path = query + fragment
    file_name = path.rsplit("/", 1)[-1] if path else ""

    url_for_rates = normalized

    features = {
        "Entropy_Domain": _shannon_entropy(domain),
        "argDomanRatio": _safe_div(len(query), len(domain)),
        "NumberRate_FileName": _digit_rate(file_name),
        "CharacterContinuityRate": _safe_div(
            _longest_contiguous_run(url_for_rates),
            len(url_for_rates),
        ),
        "argPathRatio": _safe_div(len(query), len(path)),
        "ArgUrlRatio": _safe_div(len(query), len(url_for_rates)),
        "pathurlRatio": _safe_div(len(path), len(url_for_rates)),
        "domainUrlRatio": _safe_div(len(domain), len(url_for_rates)),
        "domainlength": float(len(domain)),
        "NumberRate_AfterPath": _digit_rate(after_path),
        "NumberofDotsinURL": float(url_for_rates.count(".")),
        "domain_token_count": float(_domain_token_count(domain)),
    }

    return np.array([features[name] for name in SELECTED_FEATURES], dtype=np.float64)

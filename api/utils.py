"""URL feature-extraction helpers for the malicious-URL detection API."""

from __future__ import annotations

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


def extract_features(url: str) -> np.ndarray:
    """Compute the 12 lexical features for a single URL.

    Returns a 1-D float64 array of shape ``(12,)`` ordered according to
    :data:`SELECTED_FEATURES`.

    The ISCX-URL2016 dataset ships pre-extracted features only — the
    original extractor is a Weka/Java pipeline. A faithful Python port is
    pending (see README Roadmap §14).
    """
    raise NotImplementedError(
        "URL feature extraction is not implemented yet — port the "
        "ISCX-URL2016 extractor logic to Python first."
    )

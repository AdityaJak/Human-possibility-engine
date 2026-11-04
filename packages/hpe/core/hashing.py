"""Stable hashing of JSON-like data for manifests, preregistration and replay checks."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np


def _default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if hasattr(o, "model_dump"):
        return o.model_dump(mode="json")
    if isinstance(o, set | frozenset):
        return sorted(o)
    raise TypeError(f"not JSON-serialisable: {type(o)!r}")


def _normalise_floats(o: Any) -> Any:
    # Round floats to 12 significant digits so hashes are robust to repr noise but still
    # catch any real divergence. NaN/inf are encoded as strings.
    if isinstance(o, float):
        if math.isnan(o) or math.isinf(o):
            return str(o)
        return float(f"{o:.12g}")
    if isinstance(o, dict):
        return {str(k): _normalise_floats(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_normalise_floats(v) for v in o]
    return o


def stable_json(obj: Any) -> str:
    """Canonical JSON: sorted keys, no whitespace, normalised floats."""
    raw = json.loads(json.dumps(obj, default=_default))
    return json.dumps(_normalise_floats(raw), sort_keys=True, separators=(",", ":"))


def stable_hash(obj: Any, length: int = 16) -> str:
    """Hex SHA-256 prefix of ``stable_json(obj)``."""
    return hashlib.sha256(stable_json(obj).encode("utf-8")).hexdigest()[:length]

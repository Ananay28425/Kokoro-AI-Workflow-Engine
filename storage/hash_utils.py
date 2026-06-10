# Provides deterministic hashing helpers for cache keys and stable IDs.
from __future__ import annotations

import hashlib


def stable_sha256(value: str) -> str:
    """Return the same SHA-256 hex hash for the same input text every time."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()

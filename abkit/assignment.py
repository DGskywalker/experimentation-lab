"""Deterministic, sticky variant assignment by hashing (what a production SDK does)."""

from __future__ import annotations

import hashlib

import numpy as np


def assign(user_id: str, experiment: str, variants=("control", "treatment"), weights=None) -> str:
    """Map a user to a variant. Same user + experiment always gets the same variant.

    Salting with the experiment name makes assignments independent across
    experiments, so one test's split can't leak into another's.
    """
    weights = np.ones(len(variants)) if weights is None else np.asarray(weights, float)
    digest = hashlib.sha256(f"{experiment}:{user_id}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64  # uniform in [0, 1)
    idx = int(np.searchsorted(np.cumsum(weights) / weights.sum(), u, side="right"))
    return variants[min(idx, len(variants) - 1)]

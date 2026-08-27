"""Intraclass correlation (ICC) helpers for severity evaluation."""

from __future__ import annotations

import numpy as np


def icc21(matrix: np.ndarray) -> float | None:
    """ICC(2,1) two-way random, absolute agreement."""
    y = np.asarray(matrix, dtype=np.float64)
    n, k = y.shape
    if n < 3 or k < 2:
        return None
    m = y.mean(axis=1, keepdims=True)
    r = y.mean(axis=0, keepdims=True)
    g = y.mean()
    ss_b = k * np.sum((m - g) ** 2)
    ss_r = n * np.sum((r - g) ** 2)
    ss_t = np.sum((y - g) ** 2)
    ss_e = ss_t - ss_b - ss_r
    ms_b = ss_b / (n - 1)
    ms_r = ss_r / (k - 1)
    ms_e = ss_e / ((n - 1) * (k - 1))
    denom = ms_b + (k - 1) * ms_e + (k / n) * (ms_r - ms_e)
    if abs(denom) < 1e-12:
        return None
    return float((ms_b - ms_e) / denom)

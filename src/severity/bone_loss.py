"""Geometric alveolar bone loss severity calculation (paper Eq. 1)."""

from __future__ import annotations

import math
from typing import Iterable


def minimax_line_params(p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float]):
    """Compute gradient and intercept for the min-max line through three points."""
    a1, b1 = p1
    a2, b2 = p2
    a3, b3 = p3
    if abs(a3 - a1) < 1e-8:
        m = float("inf")
        c = a1
    else:
        m = (b3 - b1) / (a3 - a1)
        c = (b1 * (a2 + a3) + b2 * (a3 - a1) - b3 * (a1 + a2)) / (2 * (a3 - a1))
    return m, c


def project_point_to_line(x: float, y: float, m: float, c: float) -> tuple[float, float]:
    if math.isinf(m):
        return float(c), y
    if abs(m) < 1e-8:
        return x, c
    x_proj = (x + m * (y - c)) / (1 + m * m)
    y_proj = m * x_proj + c
    return x_proj, y_proj


def euclidean(p1: Iterable[float], p2: Iterable[float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def compute_bone_loss_severity(
    cej: tuple[float, float],
    intersection: tuple[float, float],
    apex: tuple[float, float],
) -> float | None:
    """Return bone loss severity percentage, or None if points are invalid."""
    if cej == (0.0, 0.0) or intersection == (0.0, 0.0) or apex == (0.0, 0.0):
        return None

    points = sorted([cej, intersection, apex], key=lambda p: p[0])
    p1, p2, p3 = points
    m, c = minimax_line_params(p1, p2, p3)

    cej_proj = project_point_to_line(cej[0], cej[1], m, c)
    inter_proj = project_point_to_line(intersection[0], intersection[1], m, c)
    apex_proj = project_point_to_line(apex[0], apex[1], m, c)

    num = euclidean(cej_proj, inter_proj)
    den = euclidean(cej_proj, apex_proj)
    if den < 1e-6:
        return None
    # Intersection should lie between CEJ and apex on the root axis.
    dx = apex_proj[0] - cej_proj[0]
    dy = apex_proj[1] - cej_proj[1]
    n2 = dx * dx + dy * dy
    if n2 > 1e-12:
        t = ((inter_proj[0] - cej_proj[0]) * dx + (inter_proj[1] - cej_proj[1]) * dy) / n2
        if t < -0.08 or t > 1.08:
            return None
    sev = (num / den) * 100.0
    if sev < 0.0:
        return None
    return min(sev, 100.0)

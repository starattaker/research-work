"""Axis-constrained bone-loss severity (projection onto tooth long axis)."""

from __future__ import annotations

import math
from enum import Enum
from typing import Iterable

import numpy as np

from src.preprocess.prepare_dataset import mask_pca_axis
from src.severity.bone_loss import compute_bone_loss_severity


class AxisSeverityMethod(str, Enum):
    PAPER_EQ1 = "paper_eq1"
    MASK_PCA = "mask_pca"
    CEJ_INT_MIDPOINT = "cej_int_midpoint"


def _mean_point(pts: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))


def _unit(dx: float, dy: float) -> tuple[float, float] | None:
    n = math.hypot(dx, dy)
    if n < 1e-8:
        return None
    return dx / n, dy / n


def axis_mask_pca(mask: np.ndarray | None) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if mask is None:
        return None
    out = mask_pca_axis(mask)
    if out is None:
        return None
    mean, direction = out
    d = _unit(float(direction[0]), float(direction[1]))
    if d is None:
        return None
    return (float(mean[0]), float(mean[1])), d


def axis_cej_int_midpoint(
    cej_pts: list[tuple[float, float]],
    int_pts: list[tuple[float, float]],
    bbox: list[float] | None = None,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Origin = CEJ midpoint; direction = unit(CEJ_mid → INT_mid)."""
    cej_mid = _mean_point(cej_pts)
    int_mid = _mean_point(int_pts)
    if cej_mid is None or int_mid is None:
        return None
    d = _unit(int_mid[0] - cej_mid[0], int_mid[1] - cej_mid[1])
    if d is None:
        if bbox is not None:
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            return (cej_mid[0], cej_mid[1]), (0.0, 1.0)
        return None
    return cej_mid, d


def project_scalar_along_axis(
    pt: tuple[float, float],
    origin: tuple[float, float],
    direction: tuple[float, float],
) -> float:
    ox, oy = origin
    dx, dy = direction
    return (pt[0] - ox) * dx + (pt[1] - oy) * dy


def severity_on_axis(
    cej: tuple[float, float],
    intersection: tuple[float, float],
    apex: tuple[float, float],
    origin: tuple[float, float],
    direction: tuple[float, float],
) -> float | None:
    if cej == (0.0, 0.0) or intersection == (0.0, 0.0) or apex == (0.0, 0.0):
        return None
    t_cej = project_scalar_along_axis(cej, origin, direction)
    t_int = project_scalar_along_axis(intersection, origin, direction)
    t_apex = project_scalar_along_axis(apex, origin, direction)
    den = abs(t_apex - t_cej)
    if den < 1e-6:
        return None
    # Intersection between CEJ and apex along axis
    lo, hi = min(t_cej, t_apex), max(t_cej, t_apex)
    margin = 0.08 * den
    if t_int < lo - margin or t_int > hi + margin:
        return None
    num = abs(t_int - t_cej)
    sev = (num / den) * 100.0
    if sev < 0.0:
        return None
    return min(sev, 100.0)


def severity_for_side_slots(
    cej_pts: list,
    inter_pts: list,
    apex_pts: list,
    slot: int,
    method: AxisSeverityMethod,
    *,
    mask: np.ndarray | None = None,
    bbox: list[float] | None = None,
    visible_cej: list[tuple[float, float]] | None = None,
    visible_int: list[tuple[float, float]] | None = None,
) -> float | None:
    from src.severity.gt_labels import point_from_slot

    c = point_from_slot(cej_pts, slot)
    t = point_from_slot(inter_pts, slot)
    a = point_from_slot(apex_pts, slot)
    if c == (0.0, 0.0) or t == (0.0, 0.0) or a == (0.0, 0.0):
        return None

    if method == AxisSeverityMethod.PAPER_EQ1:
        return compute_bone_loss_severity(c, t, a)

    cej_vis = visible_cej or []
    int_vis = visible_int or []
    if method == AxisSeverityMethod.MASK_PCA:
        axis = axis_mask_pca(mask)
        if axis is None:
            axis = axis_cej_int_midpoint(cej_vis, int_vis, bbox)
    else:
        axis = axis_cej_int_midpoint(cej_vis, int_vis, bbox)

    if axis is None:
        return None
    origin, direction = axis
    return severity_on_axis(c, t, a, origin, direction)


def severities_both_sides(
    cej_pts: list,
    inter_pts: list,
    apex_pts: list,
    method: AxisSeverityMethod,
    *,
    mask: np.ndarray | None = None,
    bbox: list[float] | None = None,
) -> list[tuple[int, float]]:
    from src.severity.gt_labels import point_from_slot

    visible_cej = [point_from_slot(cej_pts, i) for i in range(2)]
    visible_cej = [p for p in visible_cej if p != (0.0, 0.0)]
    visible_int = [point_from_slot(inter_pts, i) for i in range(2)]
    visible_int = [p for p in visible_int if p != (0.0, 0.0)]

    out: list[tuple[int, float]] = []
    for slot in (0, 1):
        sev = severity_for_side_slots(
            cej_pts,
            inter_pts,
            apex_pts,
            slot,
            method,
            mask=mask,
            bbox=bbox,
            visible_cej=visible_cej,
            visible_int=visible_int,
        )
        if sev is not None:
            out.append((slot, sev))
    return out

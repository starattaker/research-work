"""Assign CEJ / intersection / apex predictions to tooth sides for severity (Eq. 1)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np
import torch

from src.preprocess.prepare_dataset import mask_pca_axis
from src.severity.bone_loss import compute_bone_loss_severity

# Merge predicted apex pair when closer than this (px). User choice: 20px at inference.
DEFAULT_APEX_MERGE_PX = 20.0


class AxisMethod(str, Enum):
    MASK_PCA = "mask_pca"
    POINTS_AXIS = "points_axis"
    LR_POSITION = "lr_position"


@dataclass
class SideAssignment:
    """Two sides: index 0 tried before index 1 (matches GT slot 0 then slot 1)."""

    cej: tuple[float, float] | None
    intersection: tuple[float, float] | None
    apex: tuple[float, float] | None


@dataclass
class AxisInfo:
    method: AxisMethod
    origin: tuple[float, float]
    direction: tuple[float, float]
    reference_x: float | None = None


def slot_xy_from_tensor(kps: torch.Tensor | None, slot: int) -> tuple[float, float] | None:
    if kps is None or kps.numel() == 0 or slot >= kps.shape[0]:
        return None
    x, y, v = float(kps[slot, 0]), float(kps[slot, 1]), float(kps[slot, 2])
    if v <= 0 or (x == 0.0 and y == 0.0):
        return None
    return x, y


def visible_points_from_tensor(kps: torch.Tensor | None) -> list[tuple[float, float]]:
    if kps is None:
        return []
    out: list[tuple[float, float]] = []
    for i in range(kps.shape[0]):
        pt = slot_xy_from_tensor(kps, i)
        if pt is not None:
            out.append(pt)
    return out


def _unit_direction(dx: float, dy: float) -> tuple[float, float] | None:
    norm = math.hypot(dx, dy)
    if norm < 1e-8:
        return None
    return dx / norm, dy / norm


def _pca_side_sign(pt: tuple[float, float], origin: np.ndarray, direction: np.ndarray) -> int:
    rel = np.array([pt[0] - origin[0], pt[1] - origin[1]], dtype=np.float64)
    cross = direction[0] * rel[1] - direction[1] * rel[0]
    return 0 if cross < 0 else 1


def _side_from_x(pt: tuple[float, float], reference_x: float) -> int:
    return 0 if pt[0] < reference_x else 1


def axis_from_mask(mask: np.ndarray | None) -> AxisInfo | None:
    axis = mask_pca_axis(mask)
    if axis is None:
        return None
    mean, direction = axis
    return AxisInfo(
        method=AxisMethod.MASK_PCA,
        origin=(float(mean[0]), float(mean[1])),
        direction=(float(direction[0]), float(direction[1])),
    )


def axis_from_points(
    cej_pts: list[tuple[float, float]],
    int_pts: list[tuple[float, float]],
    bbox: list[float],
) -> AxisInfo | None:
    """PCA on CEJ + intersection points only (no apex)."""
    pts = list(cej_pts) + list(int_pts)
    if len(pts) < 2:
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        return AxisInfo(
            method=AxisMethod.POINTS_AXIS,
            origin=(cx, cy),
            direction=(0.0, 1.0),
            reference_x=cx,
        )
    coords = np.array(pts, dtype=np.float64)
    mean = coords.mean(axis=0)
    centered = coords - mean
    if len(pts) == 2:
        d = centered[1] - centered[0]
        direction = _unit_direction(float(d[0]), float(d[1]))
        if direction is None:
            direction = (0.0, 1.0)
    else:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        d = vt[0]
        norm = float(np.linalg.norm(d))
        direction = (float(d[0] / norm), float(d[1] / norm)) if norm > 1e-8 else (0.0, 1.0)
    return AxisInfo(
        method=AxisMethod.POINTS_AXIS,
        origin=(float(mean[0]), float(mean[1])),
        direction=direction,
    )


def axis_from_lr_position(
    cej_pts: list[tuple[float, float]],
    int_pts: list[tuple[float, float]],
    bbox: list[float],
) -> AxisInfo:
    """Left/right split by x; if only one CEJ + one INT, use bbox vertical centerline."""
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    if len(cej_pts) >= 2:
        reference_x = (min(p[0] for p in cej_pts) + max(p[0] for p in cej_pts)) / 2.0
    elif len(cej_pts) == 1 and len(int_pts) == 1:
        reference_x = cx
    elif len(int_pts) >= 2:
        reference_x = (min(p[0] for p in int_pts) + max(p[0] for p in int_pts)) / 2.0
    else:
        reference_x = cx
    return AxisInfo(
        method=AxisMethod.LR_POSITION,
        origin=(reference_x, cy),
        direction=(0.0, 1.0),
        reference_x=reference_x,
    )


def assign_point_to_side(pt: tuple[float, float], axis: AxisInfo) -> int:
    if axis.method == AxisMethod.LR_POSITION and axis.reference_x is not None:
        return _side_from_x(pt, axis.reference_x)
    origin = np.array(axis.origin, dtype=np.float64)
    direction = np.array(axis.direction, dtype=np.float64)
    return _pca_side_sign(pt, origin, direction)


def _pick_best_for_side(
    points: list[tuple[float, float]],
    side: int,
    axis: AxisInfo,
) -> tuple[float, float] | None:
    if not points:
        return None
    on_side = [p for p in points if assign_point_to_side(p, axis) == side]
    if not on_side:
        return None
    if len(on_side) == 1:
        return on_side[0]
    origin = np.array(axis.origin, dtype=np.float64)
    direction = np.array(axis.direction, dtype=np.float64)
    rel = lambda p: np.array([p[0] - origin[0], p[1] - origin[1]], dtype=np.float64)
    perp = lambda p: abs(direction[0] * rel(p)[1] - direction[1] * rel(p)[0])
    return min(on_side, key=perp)


def resolve_shared_apex(
    apex_pts: list[tuple[float, float]],
    merge_radius_px: float = DEFAULT_APEX_MERGE_PX,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Return (apex_for_side0, apex_for_side1); duplicate when only one apex."""
    if not apex_pts:
        return None, None
    if len(apex_pts) == 1:
        return apex_pts[0], apex_pts[0]
    d = math.hypot(apex_pts[0][0] - apex_pts[1][0], apex_pts[0][1] - apex_pts[1][1])
    if d < merge_radius_px:
        shared = (
            (apex_pts[0][0] + apex_pts[1][0]) / 2.0,
            (apex_pts[0][1] + apex_pts[1][1]) / 2.0,
        )
        return shared, shared
    return apex_pts[0], apex_pts[1]


def _dist_point_to_segment(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def _assign_singleton_by_anchor_lines(
    pt: tuple[float, float],
    anchor0: list[tuple[float, float]],
    anchor1: list[tuple[float, float]],
) -> int:
    """Pick side 0 or 1 by min distance to line between two anchors on that side."""
    def side_cost(anchors: list[tuple[float, float]]) -> float:
        if len(anchors) >= 2:
            return _dist_point_to_segment(pt, anchors[0], anchors[1])
        if len(anchors) == 1:
            return math.hypot(pt[0] - anchors[0][0], pt[1] - anchors[0][1])
        return float("inf")

    c0 = side_cost(anchor0)
    c1 = side_cost(anchor1)
    return 0 if c0 <= c1 else 1


def build_lr_side_assignments(
    cej_pts: list[tuple[float, float]],
    int_pts: list[tuple[float, float]],
    apex_pts: list[tuple[float, float]],
    bbox: list[float],
    merge_radius_px: float = DEFAULT_APEX_MERGE_PX,
) -> list[SideAssignment]:
    """LR by x when possible; line-distance fallback for 2+1 point patterns."""
    cx = (bbox[0] + bbox[2]) / 2.0
    sides = [SideAssignment(None, None, None), SideAssignment(None, None, None)]

    if len(cej_pts) >= 2:
        ordered = sorted(cej_pts, key=lambda p: p[0])
        sides[0].cej, sides[1].cej = ordered[0], ordered[1]
    elif len(cej_pts) == 1:
        s = 0 if cej_pts[0][0] < cx else 1
        sides[s].cej = cej_pts[0]

    if len(int_pts) >= 2:
        ordered = sorted(int_pts, key=lambda p: p[0])
        sides[0].intersection, sides[1].intersection = ordered[0], ordered[1]
    elif len(int_pts) == 1:
        a0 = [p for p in (sides[0].cej, sides[0].apex) if p]
        a1 = [p for p in (sides[1].cej, sides[1].apex) if p]
        if a0 or a1:
            s = _assign_singleton_by_anchor_lines(int_pts[0], a0, a1)
        else:
            s = 0 if int_pts[0][0] < cx else 1
        sides[s].intersection = int_pts[0]

    if not apex_pts:
        pass
    elif len(apex_pts) == 1:
        a0 = [p for p in (sides[0].cej, sides[0].intersection) if p]
        a1 = [p for p in (sides[1].cej, sides[1].intersection) if p]
        if a0 and a1:
            s = _assign_singleton_by_anchor_lines(apex_pts[0], a0, a1)
            sides[s].apex = apex_pts[0]
            sides[1 - s].apex = apex_pts[0]
        else:
            sides[0].apex = sides[1].apex = apex_pts[0]
    else:
        d = math.hypot(apex_pts[0][0] - apex_pts[1][0], apex_pts[0][1] - apex_pts[1][1])
        if d < merge_radius_px:
            shared = (
                (apex_pts[0][0] + apex_pts[1][0]) / 2.0,
                (apex_pts[0][1] + apex_pts[1][1]) / 2.0,
            )
            sides[0].apex = sides[1].apex = shared
        else:
            ordered = sorted(apex_pts, key=lambda p: p[0])
            sides[0].apex, sides[1].apex = ordered[0], ordered[1]

    # 1 CEJ + 1 INT: severity only if both on same side of bbox center
    if len(cej_pts) == 1 and len(int_pts) == 1 and len(apex_pts) <= 1:
        s_cej = 0 if cej_pts[0][0] < cx else 1
        s_int = 0 if int_pts[0][0] < cx else 1
        if s_cej != s_int:
            return [SideAssignment(None, None, None), SideAssignment(None, None, None)]
        apx = apex_pts[0] if apex_pts else None
        empty = SideAssignment(None, None, None)
        valid = SideAssignment(cej_pts[0], int_pts[0], apx)
        return [valid, empty] if s_cej == 0 else [empty, valid]

    return sides


def build_side_assignments(
    cej_kps: torch.Tensor | None,
    int_kps: torch.Tensor | None,
    apex_kps: torch.Tensor | None,
    axis: AxisInfo,
    merge_radius_px: float = DEFAULT_APEX_MERGE_PX,
) -> list[SideAssignment]:
    cej_pts = visible_points_from_tensor(cej_kps)
    int_pts = visible_points_from_tensor(int_kps)
    apex_pts = visible_points_from_tensor(apex_kps)

    if not apex_pts:
        apex_by_side: list[tuple[float, float] | None] = [None, None]
    elif len(apex_pts) == 1:
        apex_by_side = [apex_pts[0], apex_pts[0]]
    else:
        d = math.hypot(apex_pts[0][0] - apex_pts[1][0], apex_pts[0][1] - apex_pts[1][1])
        if d < merge_radius_px:
            shared = (
                (apex_pts[0][0] + apex_pts[1][0]) / 2.0,
                (apex_pts[0][1] + apex_pts[1][1]) / 2.0,
            )
            apex_by_side = [shared, shared]
        else:
            apex_by_side = [
                _pick_best_for_side(apex_pts, 0, axis),
                _pick_best_for_side(apex_pts, 1, axis),
            ]

    sides: list[SideAssignment] = []
    for side in (0, 1):
        sides.append(
            SideAssignment(
                cej=_pick_best_for_side(cej_pts, side, axis),
                intersection=_pick_best_for_side(int_pts, side, axis),
                apex=apex_by_side[side],
            )
        )
    return sides


def severity_from_sides(sides: Iterable[SideAssignment]) -> float | None:
    """First valid side (0 then 1), same convention as GT severity_from_slots."""
    for side in sides:
        if side.cej is None or side.intersection is None or side.apex is None:
            continue
        sev = compute_bone_loss_severity(side.cej, side.intersection, side.apex)
        if sev is not None:
            return sev
    return None


def severity_from_side_index(sides: list[SideAssignment], side_idx: int) -> float | None:
    """Severity for one PCA side only (0=neg, 1=pos in v6 labels)."""
    if side_idx < 0 or side_idx >= len(sides):
        return None
    side = sides[side_idx]
    if side.cej is None or side.intersection is None or side.apex is None:
        return None
    return compute_bone_loss_severity(side.cej, side.intersection, side.apex)


def flip_axis(axis: AxisInfo) -> AxisInfo:
    """Reverse PCA direction (tests sign ambiguity vs GT slot 0/1)."""
    return AxisInfo(
        method=axis.method,
        origin=axis.origin,
        direction=(-axis.direction[0], -axis.direction[1]),
        reference_x=axis.reference_x,
    )


def oracle_best_severity(
    cej_kps: torch.Tensor | None,
    int_kps: torch.Tensor | None,
    apex_kps: torch.Tensor | None,
    gt_sev: float,
) -> tuple[float | None, tuple[int, int, int] | None]:
    """Best of 8 tensor-slot combos vs GT (diagnostic ceiling; uses GT severity)."""
    best_sev = None
    best_slots = None
    best_err = float("inf")
    for cs in (0, 1):
        for is_ in (0, 1):
            for as_ in (0, 1):
                c = slot_xy_from_tensor(cej_kps, cs)
                t = slot_xy_from_tensor(int_kps, is_)
                a = slot_xy_from_tensor(apex_kps, as_)
                if c is None or t is None or a is None:
                    continue
                sev = compute_bone_loss_severity(c, t, a)
                if sev is None:
                    continue
                err = abs(sev - gt_sev)
                if err < best_err:
                    best_err = err
                    best_sev = sev
                    best_slots = (cs, is_, as_)
    return best_sev, best_slots


def first_valid_side_index(sides: list[SideAssignment]) -> int | None:
    for i, side in enumerate(sides):
        if side.cej and side.intersection and side.apex:
            if compute_bone_loss_severity(side.cej, side.intersection, side.apex) is not None:
                return i
    return None


def severity_with_axis_method(
    cej_kps: torch.Tensor | None,
    int_kps: torch.Tensor | None,
    apex_kps: torch.Tensor | None,
    method: AxisMethod,
    bbox: list[float],
    mask: np.ndarray | None = None,
    merge_radius_px: float = DEFAULT_APEX_MERGE_PX,
) -> tuple[float | None, AxisInfo | None, list[SideAssignment]]:
    cej_pts = visible_points_from_tensor(cej_kps)
    int_pts = visible_points_from_tensor(int_kps)

    axis: AxisInfo | None
    if method == AxisMethod.MASK_PCA:
        axis = axis_from_mask(mask)
        if axis is None:
            axis = axis_from_lr_position(cej_pts, int_pts, bbox)
    elif method == AxisMethod.POINTS_AXIS:
        axis = axis_from_points(cej_pts, int_pts, bbox)
    else:
        axis = axis_from_lr_position(cej_pts, int_pts, bbox)

    if method == AxisMethod.LR_POSITION:
        sides = build_lr_side_assignments(cej_pts, int_pts, visible_points_from_tensor(apex_kps), bbox, merge_radius_px)
    else:
        sides = build_side_assignments(cej_kps, int_kps, apex_kps, axis, merge_radius_px)
    return severity_from_sides(sides), axis, sides

"""Assign INT/APEX to CEJ anchors by permutation cost (no GT, no masks)."""

from __future__ import annotations

import math

import torch

from src.severity.bone_loss import compute_bone_loss_severity
from src.severity.paper_combine import slot_xy_from_tensor
from src.severity.side_details import SideDetail
from src.severity.slot_matching import DEFAULT_APEX_MERGE_PX, visible_points_from_tensor


def _d(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _best_assignment(
    anchors: list[tuple[float, float]],
    pts: list[tuple[float, float]],
) -> list[tuple[float, float] | None]:
    """Map each anchor to at most one point; unused anchors get None."""
    n_a, n_p = len(anchors), len(pts)
    assigned: list[tuple[float, float] | None] = [None] * n_a
    if n_a == 0 or n_p == 0:
        return assigned
    if n_a == 1:
        assigned[0] = min(pts, key=lambda p: _d(anchors[0], p))
        return assigned
    if n_p == 1:
        j = 0 if _d(anchors[0], pts[0]) <= _d(anchors[1], pts[0]) else 1
        assigned[j] = pts[0]
        return assigned
    # 2×2: try both permutations
    p0, p1 = pts[0], pts[1]
    cost_id = _d(anchors[0], p0) + _d(anchors[1], p1)
    cost_sw = _d(anchors[0], p1) + _d(anchors[1], p0)
    if cost_id <= cost_sw:
        assigned[0], assigned[1] = p0, p1
    else:
        assigned[0], assigned[1] = p1, p0
    return assigned


def pred_side_details_hungarian(
    cej_kps: torch.Tensor | None,
    int_kps: torch.Tensor | None,
    apex_kps: torch.Tensor | None,
    merge_radius_px: float = DEFAULT_APEX_MERGE_PX,
) -> list[SideDetail]:
    cejs = visible_points_from_tensor(cej_kps)
    ints = visible_points_from_tensor(int_kps)
    apices = visible_points_from_tensor(apex_kps)
    if not cejs or not ints or not apices:
        return []
    cejs = sorted(cejs, key=lambda p: p[0])[:2]
    int_asg = _best_assignment(cejs, ints[:2])
    if len(apices) == 1:
        apex_asg = [apices[0], apices[0]][: len(cejs)]
        if len(cejs) == 1:
            apex_asg = [apices[0]]
        elif len(cejs) == 2:
            apex_asg = [apices[0], apices[0]]
    else:
        a0, a1 = apices[0], apices[1]
        if _d(a0, a1) < merge_radius_px:
            shared = ((a0[0] + a1[0]) / 2.0, (a0[1] + a1[1]) / 2.0)
            apex_asg = [shared, shared][: len(cejs)]
            if len(cejs) == 2:
                apex_asg = [shared, shared]
        else:
            apex_asg = _best_assignment(cejs, apices[:2])

    out: list[SideDetail] = []
    for slot, cej in enumerate(cejs):
        t = int_asg[slot] if slot < len(int_asg) else None
        a = apex_asg[slot] if slot < len(apex_asg) else None
        if t is None or a is None:
            continue
        sev = compute_bone_loss_severity(cej, t, a)
        if sev is not None:
            out.append(SideDetail(slot=slot, severity=sev, cej=cej))
    return out

"""Predicted severity from keypoint tensors (no YOLO / albumentations deps)."""

from __future__ import annotations

import torch

from src.severity.bone_loss import compute_bone_loss_severity
from src.severity.paper_combine import (
    severity_geom_consistent,
    severity_paper_first_valid,
    severities_paper_aligned,
    slot_xy_from_tensor,
)


def severity_from_tensor_slots(
    cej_kps: torch.Tensor | None,
    inter_kps: torch.Tensor | None,
    apex_kps: torch.Tensor | None,
) -> float | None:
    if cej_kps is None or inter_kps is None or apex_kps is None:
        return None
    for slot in (0, 1):
        c = slot_xy_from_tensor(cej_kps, slot)
        t = slot_xy_from_tensor(inter_kps, slot)
        a = slot_xy_from_tensor(apex_kps, slot)
        if c is None or t is None or a is None:
            continue
        sev = compute_bone_loss_severity(c, t, a)
        if sev is not None:
            return sev
    return None


def pred_severity_from_tensors(
    cej_kps: torch.Tensor | None,
    int_kps: torch.Tensor | None,
    apex_kps: torch.Tensor | None,
    *,
    combine_mode: str = "tensor",
    merge_radius_px: float = 20.0,
) -> float | None:
    if combine_mode == "paper_x":
        return severity_paper_first_valid(cej_kps, int_kps, apex_kps, merge_radius_px)
    if combine_mode == "geom_consistent":
        return severity_geom_consistent(cej_kps, int_kps, apex_kps)
    return severity_from_tensor_slots(cej_kps, int_kps, apex_kps)


def pred_severities_from_tensors(
    cej_kps: torch.Tensor | None,
    int_kps: torch.Tensor | None,
    apex_kps: torch.Tensor | None,
    *,
    combine_mode: str = "tensor",
    merge_radius_px: float = 20.0,
) -> list[tuple[int, float]]:
    if combine_mode == "paper_x":
        return severities_paper_aligned(cej_kps, int_kps, apex_kps, merge_radius_px)
    if combine_mode == "geom_consistent":
        sev = severity_geom_consistent(cej_kps, int_kps, apex_kps)
        return [(0, sev)] if sev is not None else []
    out: list[tuple[int, float]] = []
    for slot in (0, 1):
        c = slot_xy_from_tensor(cej_kps, slot)
        t = slot_xy_from_tensor(int_kps, slot)
        a = slot_xy_from_tensor(apex_kps, slot)
        if c is None or t is None or a is None:
            continue
        sev = compute_bone_loss_severity(c, t, a)
        if sev is not None:
            out.append((slot, sev))
    return out

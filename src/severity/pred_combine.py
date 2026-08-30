"""Predicted severity from keypoint tensors (no YOLO / albumentations deps)."""

from __future__ import annotations

import torch

from src.severity.bone_loss import compute_bone_loss_severity
from src.severity.hungarian_assign import pred_side_details_hungarian
from src.severity.paper_combine import (
    severities_paper_aligned,
    slot_xy_from_tensor,
)
from src.severity.side_details import SideDetail
from src.severity.slot_matching import (
    axis_from_lr_position,
    axis_from_mask,
    build_lr_side_assignments,
    build_side_assignments,
    flip_axis,
    visible_points_from_tensor,
)


def severity_from_tensor_slots(
    cej_kps: torch.Tensor | None,
    inter_kps: torch.Tensor | None,
    apex_kps: torch.Tensor | None,
) -> float | None:
    sides = pred_side_details_from_tensors(cej_kps, inter_kps, apex_kps, combine_mode="tensor")
    return sides[0].severity if sides else None


def _details_from_lr_sides(sides, merge_radius_px: float) -> list[SideDetail]:
    out: list[SideDetail] = []
    for slot in (0, 1):
        side = sides[slot]
        if side.cej is None or side.intersection is None or side.apex is None:
            continue
        sev = compute_bone_loss_severity(side.cej, side.intersection, side.apex)
        if sev is not None:
            out.append(SideDetail(slot=slot, severity=sev, cej=side.cej))
    return out


def sort_visible_slots(kps: torch.Tensor) -> torch.Tensor:
    rows = []
    for i in range(kps.shape[0]):
        pt = slot_xy_from_tensor(kps, i)
        if pt is not None:
            rows.append((pt[0], i))
    if not rows:
        return kps
    rows.sort(key=lambda r: r[0])
    out = kps.clone()
    for new_i, (_, old_i) in enumerate(rows):
        out[new_i] = kps[old_i]
    return out


def pred_side_details_from_tensors(
    cej_kps: torch.Tensor | None,
    int_kps: torch.Tensor | None,
    apex_kps: torch.Tensor | None,
    *,
    combine_mode: str = "tensor",
    merge_radius_px: float = 20.0,
    bbox: list[float] | None = None,
    mask=None,
) -> list[SideDetail]:
    if cej_kps is None or int_kps is None or apex_kps is None:
        return []

    if combine_mode == "hungarian":
        return pred_side_details_hungarian(cej_kps, int_kps, apex_kps, merge_radius_px)

    if combine_mode == "mask_pca":
        cej_pts = visible_points_from_tensor(cej_kps)
        int_pts = visible_points_from_tensor(int_kps)
        axis = axis_from_mask(mask)
        if axis is None:
            axis = axis_from_lr_position(cej_pts, int_pts, bbox or [0.0, 0.0, 1.0, 1.0])
        sides = build_side_assignments(cej_kps, int_kps, apex_kps, axis, merge_radius_px)
        flipped = build_side_assignments(cej_kps, int_kps, apex_kps, flip_axis(axis), merge_radius_px)
        a = _details_from_lr_sides(sides, merge_radius_px)
        b = _details_from_lr_sides(flipped, merge_radius_px)
        return a if len(a) >= len(b) else b

    if combine_mode == "lr":
        if bbox is None:
            return []
        sides = build_lr_side_assignments(
            visible_points_from_tensor(cej_kps),
            visible_points_from_tensor(int_kps),
            visible_points_from_tensor(apex_kps),
            bbox,
            merge_radius_px,
        )
        return _details_from_lr_sides(sides, merge_radius_px)

    if combine_mode == "paper_x":
        raw = severities_paper_aligned(cej_kps, int_kps, apex_kps, merge_radius_px)
        out: list[SideDetail] = []
        sorted_cej = sort_visible_slots(cej_kps)
        for slot, sev in raw:
            c = slot_xy_from_tensor(sorted_cej, slot)
            if c is not None:
                out.append(SideDetail(slot=slot, severity=sev, cej=c))
        return out

    if combine_mode == "geom_consistent":
        from src.severity.paper_combine import _rank_combos_by_x_spread

        ranked = _rank_combos_by_x_spread(cej_kps, int_kps, apex_kps)
        out: list[SideDetail] = []
        used_cs: set[int] = set()
        for _spread, sev, cs, _is, _as in ranked:
            c = slot_xy_from_tensor(cej_kps, cs)
            if c is None:
                continue
            if cs in used_cs:
                continue
            used_cs.add(cs)
            out.append(SideDetail(slot=len(out), severity=sev, cej=c))
            if len(out) >= 2:
                break
        return out

    out = []
    for slot in (0, 1):
        c = slot_xy_from_tensor(cej_kps, slot)
        t = slot_xy_from_tensor(int_kps, slot)
        a = slot_xy_from_tensor(apex_kps, slot)
        if c is None or t is None or a is None:
            continue
        sev = compute_bone_loss_severity(c, t, a)
        if sev is not None:
            out.append(SideDetail(slot=slot, severity=sev, cej=c))
    return out


def pred_severity_from_tensors(
    cej_kps: torch.Tensor | None,
    int_kps: torch.Tensor | None,
    apex_kps: torch.Tensor | None,
    *,
    combine_mode: str = "tensor",
    merge_radius_px: float = 20.0,
    bbox: list[float] | None = None,
    mask=None,
) -> float | None:
    sides = pred_side_details_from_tensors(
        cej_kps, int_kps, apex_kps,
        combine_mode=combine_mode, merge_radius_px=merge_radius_px, bbox=bbox, mask=mask,
    )
    return sides[0].severity if sides else None


def pred_severities_from_tensors(
    cej_kps: torch.Tensor | None,
    int_kps: torch.Tensor | None,
    apex_kps: torch.Tensor | None,
    *,
    combine_mode: str = "tensor",
    merge_radius_px: float = 20.0,
    bbox: list[float] | None = None,
    mask=None,
) -> list[tuple[int, float]]:
    sides = pred_side_details_from_tensors(
        cej_kps, int_kps, apex_kps,
        combine_mode=combine_mode, merge_radius_px=merge_radius_px, bbox=bbox, mask=mask,
    )
    return [(s.slot, s.severity) for s in sides]

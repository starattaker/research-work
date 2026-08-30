"""Per-side severity with CEJ anchor for anatomical ICC pairing."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from src.severity.bone_loss import compute_bone_loss_severity
from src.severity.gt_labels import point_from_slot


@dataclass(frozen=True)
class SideDetail:
    slot: int
    severity: float
    cej: tuple[float, float]


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def gt_side_details_for_tooth(
    merged: dict,
    tooth_idx: int,
    *,
    slot_convention: str = "pca",
) -> list[SideDetail]:
    cej_list = merged["cej"][tooth_idx]
    inter_list = merged["intersection"][tooth_idx]
    apex_list = merged["apex"][tooth_idx]
    out: list[SideDetail] = []
    for slot in (0, 1):
        c = point_from_slot(cej_list, slot)
        t = point_from_slot(inter_list, slot)
        a = point_from_slot(apex_list, slot)
        if c == (0.0, 0.0):
            continue
        sev = compute_bone_loss_severity(c, t, a)
        if sev is not None:
            out.append(SideDetail(slot=slot, severity=sev, cej=c))
    return out


def pair_sides_by_cej(
    gt_details: list[SideDetail],
    pred_details: list[SideDetail],
) -> list[tuple[float, float]]:
    """Greedy one-to-one GT↔pred pairing by nearest CEJ (anatomical, not slot index)."""
    if not gt_details or not pred_details:
        return []
    pairs: list[tuple[float, float]] = []
    used_pred: set[int] = set()
    for gt in gt_details:
        best_j = None
        best_d = float("inf")
        for j, pred in enumerate(pred_details):
            if j in used_pred:
                continue
            d = _dist(gt.cej, pred.cej)
            if d < best_d:
                best_d = d
                best_j = j
        if best_j is not None:
            pred = pred_details[best_j]
            pairs.append((gt.severity, pred.severity))
            used_pred.add(best_j)
    return pairs


def pair_sides_by_slot_index(
    gt_details: list[SideDetail],
    pred_details: list[SideDetail],
) -> list[tuple[float, float]]:
    pred_map = {p.slot: p.severity for p in pred_details}
    pairs: list[tuple[float, float]] = []
    for gt in gt_details:
        pred_sev = pred_map.get(gt.slot)
        if pred_sev is not None:
            pairs.append((gt.severity, pred_sev))
    return pairs


def slot_flip_rate(gt_details: list[SideDetail], pred_details: list[SideDetail]) -> bool | None:
    """True if pred slot index closest to GT slot 0 CEJ is not slot 0."""
    if len(gt_details) < 2 or len(pred_details) < 2:
        return None
    gt0 = next((g for g in gt_details if g.slot == 0), None)
    if gt0 is None:
        return None
    best_slot = min(pred_details, key=lambda p: _dist(gt0.cej, p.cej)).slot
    return best_slot != 0

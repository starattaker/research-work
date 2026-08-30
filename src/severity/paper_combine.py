"""Paper-style keypoint combination (no masks): NMS detections + x-sorted slots.

From authors' Keypoint_R_CNN_Script.py / dataset.py:
  keypoints_original = [sorted(sublist, key=lambda x: x[0]) for sublist in keypoints_original]

Slot 0 = left (smaller x), slot 1 = right (larger x). Combine CEJ[i] + INT[i] + APEX[i].
"""

from __future__ import annotations

import math

import torch

from src.severity.bone_loss import compute_bone_loss_severity

DEFAULT_APEX_MERGE_PX = 20.0


def slot_xy_from_tensor(kps: torch.Tensor | None, slot: int) -> tuple[float, float] | None:
    if kps is None or kps.numel() == 0 or slot >= kps.shape[0]:
        return None
    x, y, v = float(kps[slot, 0]), float(kps[slot, 1]), float(kps[slot, 2])
    if v <= 0 or (x == 0.0 and y == 0.0):
        return None
    return x, y


def _visible_rows(kps: torch.Tensor | None) -> list[tuple[int, tuple[float, float]]]:
    if kps is None or kps.numel() == 0:
        return []
    rows: list[tuple[int, tuple[float, float]]] = []
    for i in range(kps.shape[0]):
        pt = slot_xy_from_tensor(kps, i)
        if pt is not None:
            rows.append((i, pt))
    return rows


def sort_keypoints_by_x(kps: torch.Tensor | None) -> torch.Tensor | None:
    """Reorder keypoint rows left→right by x (paper training convention)."""
    if kps is None or kps.numel() == 0:
        return kps
    rows = _visible_rows(kps)
    if not rows:
        return kps
    rows.sort(key=lambda r: r[1][0])
    out = kps.clone()
    for new_i, (_, pt) in enumerate(rows):
        out[new_i, 0] = pt[0]
        out[new_i, 1] = pt[1]
        out[new_i, 2] = 1.0
    for j in range(len(rows), kps.shape[0]):
        out[j] = 0.0
    return out


def _apex_for_slots(
    apex_kps: torch.Tensor | None,
    merge_radius_px: float,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """After x-sort: slot0/slot1 apex; merge if single or two very close."""
    apex = sort_keypoints_by_x(apex_kps)
    a0 = slot_xy_from_tensor(apex, 0) if apex is not None else None
    a1 = slot_xy_from_tensor(apex, 1) if apex is not None else None
    if a0 is None and a1 is None:
        return None, None
    if a0 is not None and a1 is None:
        return a0, a0
    if a1 is not None and a0 is None:
        return a1, a1
    assert a0 is not None and a1 is not None
    d = math.hypot(a0[0] - a1[0], a0[1] - a1[1])
    if d < merge_radius_px:
        shared = ((a0[0] + a1[0]) / 2.0, (a0[1] + a1[1]) / 2.0)
        return shared, shared
    return a0, a1


def severity_at_paper_slot(
    cej_kps: torch.Tensor | None,
    int_kps: torch.Tensor | None,
    apex_kps: torch.Tensor | None,
    slot: int,
    merge_radius_px: float = DEFAULT_APEX_MERGE_PX,
) -> float | None:
    cej = sort_keypoints_by_x(cej_kps)
    inter = sort_keypoints_by_x(int_kps)
    c = slot_xy_from_tensor(cej, slot)
    t = slot_xy_from_tensor(inter, slot)
    a0, a1 = _apex_for_slots(apex_kps, merge_radius_px)
    a = a0 if slot == 0 else a1
    if c is None or t is None or a is None:
        return None
    return compute_bone_loss_severity(c, t, a)


def severities_paper_aligned(
    cej_kps: torch.Tensor | None,
    int_kps: torch.Tensor | None,
    apex_kps: torch.Tensor | None,
    merge_radius_px: float = DEFAULT_APEX_MERGE_PX,
) -> list[tuple[int, float]]:
    """All valid side severities (slot 0 then 1) — paper Fig.9 both sides."""
    out: list[tuple[int, float]] = []
    for slot in (0, 1):
        sev = severity_at_paper_slot(cej_kps, int_kps, apex_kps, slot, merge_radius_px)
        if sev is not None:
            out.append((slot, sev))
    return out


def severities_paper_aligned_lists(
    cej_pts: list,
    inter_pts: list,
    apex_pts: list,
    merge_radius_px: float = DEFAULT_APEX_MERGE_PX,
) -> list[tuple[int, float]]:
    """List-based GT / sanity checks (same logic as tensor paper_x)."""
    cej = sort_list_by_x(cej_pts)
    inter = sort_list_by_x(inter_pts)
    apex = sort_list_by_x(apex_pts)
    out: list[tuple[int, float]] = []
    for slot in (0, 1):
        c = _list_pt(cej, slot)
        t = _list_pt(inter, slot)
        a0, a1 = _apex_list_slots(apex, merge_radius_px)
        a = a0 if slot == 0 else a1
        if c is None or t is None or a is None:
            continue
        sev = compute_bone_loss_severity(c, t, a)
        if sev is not None:
            out.append((slot, sev))
    return out


def sort_list_by_x(kps: list) -> list:
    visible: list[tuple[float, float]] = []
    for i in range(len(kps)):
        pt = _list_pt(kps, i)
        if pt is not None:
            visible.append(pt)
    visible.sort(key=lambda p: p[0])
    out = [[0.0, 0.0, 0] for _ in kps]
    for i, (x, y) in enumerate(visible):
        out[i] = [x, y, 2]
    return out


def _list_pt(kps: list, slot: int) -> tuple[float, float] | None:
    if slot >= len(kps):
        return None
    kp = kps[slot]
    if len(kp) > 2 and int(kp[2]) == 0:
        return None
    x, y = float(kp[0]), float(kp[1])
    if x == 0.0 and y == 0.0:
        return None
    return x, y


def _apex_list_slots(
    apex_pts: list,
    merge_radius_px: float,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    apex = sort_list_by_x(apex_pts)
    a0 = _list_pt(apex, 0)
    a1 = _list_pt(apex, 1)
    if a0 is None and a1 is None:
        return None, None
    if a0 is not None and a1 is None:
        return a0, a0
    if a1 is not None and a0 is None:
        return a1, a1
    assert a0 is not None and a1 is not None
    d = math.hypot(a0[0] - a1[0], a0[1] - a1[1])
    if d < merge_radius_px:
        shared = ((a0[0] + a1[0]) / 2.0, (a0[1] + a1[1]) / 2.0)
        return shared, shared
    return a0, a1


def severity_paper_first_valid(
    cej_kps: torch.Tensor | None,
    int_kps: torch.Tensor | None,
    apex_kps: torch.Tensor | None,
    merge_radius_px: float = DEFAULT_APEX_MERGE_PX,
) -> float | None:
    sides = severities_paper_aligned(cej_kps, int_kps, apex_kps, merge_radius_px)
    return sides[0][1] if sides else None


def _combo_points(
    cej_kps: torch.Tensor | None,
    int_kps: torch.Tensor | None,
    apex_kps: torch.Tensor | None,
    cs: int,
    is_: int,
    as_: int,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None, tuple[float, float] | None]:
    return (
        slot_xy_from_tensor(cej_kps, cs),
        slot_xy_from_tensor(int_kps, is_),
        slot_xy_from_tensor(apex_kps, as_),
    )


def severity_geom_consistent(
    cej_kps: torch.Tensor | None,
    int_kps: torch.Tensor | None,
    apex_kps: torch.Tensor | None,
) -> float | None:
    """Pick valid 8-combo with smallest x-spread (same-side geometry, no GT cheat)."""
    ranked = _rank_combos_by_x_spread(cej_kps, int_kps, apex_kps)
    return ranked[0][1] if ranked else None


def severities_geom_both_sides(
    cej_kps: torch.Tensor | None,
    int_kps: torch.Tensor | None,
    apex_kps: torch.Tensor | None,
) -> list[tuple[int, float]]:
    """Up to two sides: best x-coherent combo per distinct CEJ slot."""
    ranked = _rank_combos_by_x_spread(cej_kps, int_kps, apex_kps)
    if not ranked:
        return []
    out: list[tuple[int, float]] = [(0, ranked[0][1])]
    best_cs = ranked[0][2]
    for _spread, sev, cs, _is, _as in ranked[1:]:
        if cs != best_cs:
            out.append((1, sev))
            break
    return out


def _rank_combos_by_x_spread(
    cej_kps: torch.Tensor | None,
    int_kps: torch.Tensor | None,
    apex_kps: torch.Tensor | None,
) -> list[tuple[float, float, int, int, int]]:
    ranked: list[tuple[float, float, int, int, int]] = []
    for cs in (0, 1):
        for is_ in (0, 1):
            for as_ in (0, 1):
                c, t, a = _combo_points(cej_kps, int_kps, apex_kps, cs, is_, as_)
                if c is None or t is None or a is None:
                    continue
                sev = compute_bone_loss_severity(c, t, a)
                if sev is None:
                    continue
                ranked.append((float(np_std([c[0], t[0], a[0]])), sev, cs, is_, as_))
    ranked.sort(key=lambda row: row[0])
    return ranked


def np_std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

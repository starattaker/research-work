"""Ground-truth severity from preprocessed keypoint JSON (no separate expert severity file)."""

from __future__ import annotations

import json
from pathlib import Path

from src.severity.bone_loss import compute_bone_loss_severity
from src.severity.paper_combine import severities_paper_aligned_lists


def point_from_slot(keypoints: list, slot: int) -> tuple[float, float]:
    if slot >= len(keypoints):
        return 0.0, 0.0
    kp = keypoints[slot]
    if len(kp) > 2 and int(kp[2]) == 0:
        return 0.0, 0.0
    x, y = float(kp[0]), float(kp[1])
    if x == 0.0 and y == 0.0:
        return 0.0, 0.0
    return x, y


def sort_list_keypoints_by_x(kps: list) -> list:
    visible: list[tuple[float, float]] = []
    for i in range(len(kps)):
        pt = point_from_slot(kps, i)
        if pt != (0.0, 0.0):
            visible.append(pt)
    visible.sort(key=lambda p: p[0])
    out = [[0.0, 0.0, 0] for _ in kps]
    for i, (x, y) in enumerate(visible):
        out[i] = [x, y, 2]
    return out


def severity_from_slots_pca(cej_pts: list, inter_pts: list, apex_pts: list) -> float | None:
    """v6 labels: try PCA slot 0 then slot 1."""
    for slot in (0, 1):
        c = point_from_slot(cej_pts, slot)
        t = point_from_slot(inter_pts, slot)
        a = point_from_slot(apex_pts, slot)
        sev = compute_bone_loss_severity(c, t, a)
        if sev is not None:
            return sev
    return None


def severities_from_gt_lists(
    cej_pts: list,
    inter_pts: list,
    apex_pts: list,
    merge_radius_px: float = 20.0,
) -> list[tuple[int, float]]:
    """Paper convention: sort each keypoint type by x, then severity per side."""
    return severities_paper_aligned_lists(cej_pts, inter_pts, apex_pts, merge_radius_px)


def gt_severities_for_tooth(
    merged: dict,
    tooth_idx: int,
    *,
    slot_convention: str = "pca",
    merge_radius_px: float = 20.0,
) -> list[tuple[int, float]]:
    cej = merged["cej"][tooth_idx]
    inter = merged["intersection"][tooth_idx]
    apex = merged["apex"][tooth_idx]
    if slot_convention == "paper_x":
        return severities_from_gt_lists(cej, inter, apex, merge_radius_px)
    out: list[tuple[int, float]] = []
    for slot in (0, 1):
        c = point_from_slot(cej, slot)
        t = point_from_slot(inter, slot)
        a = point_from_slot(apex, slot)
        sev = compute_bone_loss_severity(c, t, a)
        if sev is not None:
            out.append((slot, sev))
    return out


def load_gt_annotations(data_root: Path, split: str, stem: str) -> dict | None:
    merged: dict = {"bboxes": None, "labels": None, "cej": [], "intersection": [], "apex": []}
    for kpt_type in ("cej", "intersection", "apex"):
        ann_path = data_root / "keypoints" / kpt_type / split / "annotations" / f"{stem}.json"
        if not ann_path.exists():
            return None
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        if merged["bboxes"] is None:
            merged["bboxes"] = data["bboxes"]
            merged["labels"] = data["labels"]
        merged[kpt_type] = data["keypoints"]
    return merged

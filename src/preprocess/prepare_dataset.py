"""Convert DenPAR raw annotations into training-ready formats."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm

SPLITS = ("Training", "Validation", "Testing")
SPLIT_ALIASES = {"Training": "train", "Validation": "val", "Testing": "test"}


@dataclass
class ToothRecord:
    bbox: list[float]
    label: int  # 1=single root, 2=double root
    cej: list[list[float]]
    intersection: list[list[float]]
    apex: list[list[float]]


def point_in_bbox(x: float, y: float, bbox: list[float], margin: float = 0.0) -> bool:
    x1, y1, x2, y2 = bbox
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def mask_center(mask: np.ndarray | None) -> tuple[float, float] | None:
    """Centroid of foreground pixels (v3/v4 tie-break)."""
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def tooth_anchor_center(
    index: int,
    bboxes: list[list[float]],
    masks: list[np.ndarray | None],
) -> tuple[float, float]:
    """Prefer mask centroid; fall back to bbox center if mask missing."""
    if index < len(masks):
        c = mask_center(masks[index])
        if c is not None:
            return c
    return bbox_center(bboxes[index])


def dist2(p1: Iterable[float], p2: Iterable[float]) -> float:
    return float((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def load_mask_contour(mask_path: Path) -> np.ndarray | None:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea).reshape(-1, 2)


def nearest_contour_point(contour: np.ndarray, target: tuple[float, float]) -> list[float]:
    pts = contour.astype(np.float32)
    d = ((pts[:, 0] - target[0]) ** 2 + (pts[:, 1] - target[1]) ** 2)
    idx = int(np.argmin(d))
    return [float(pts[idx, 0]), float(pts[idx, 1])]


def cross2(a, b) -> float:
    """2D scalar cross product (NumPy 2.x safe)."""
    return float(a[0] * b[1] - a[1] * b[0])


def ray_segment_intersection(origin, direction, q1, q2, eps: float = 1e-6):
    """Ray origin + t*direction (t >= 0) vs segment q1-q2."""
    o = np.array(origin, dtype=np.float64)
    d = np.array(direction, dtype=np.float64)
    d_norm = np.linalg.norm(d)
    if d_norm < eps:
        return None
    d = d / d_norm
    s = q2 - q1
    qmp = q1 - o
    rxs = cross2(d, s)
    if abs(rxs) < eps:
        return None
    t = cross2(qmp, s) / rxs
    u = cross2(qmp, d) / rxs
    if t >= -eps and 0.0 - eps <= u <= 1.0 + eps:
        return o + max(t, 0.0) * d
    return None


def _polyline_contour_hits(contour: np.ndarray, line_pts: list[list[float]]) -> list[np.ndarray]:
    hits: list[np.ndarray] = []
    for i in range(len(line_pts) - 1):
        p1 = np.array(line_pts[i], dtype=np.float32)
        p2 = np.array(line_pts[i + 1], dtype=np.float32)
        for j in range(len(contour)):
            q1 = contour[j].astype(np.float32)
            q2 = contour[(j + 1) % len(contour)].astype(np.float32)
            inter = segment_intersection(p1, p2, q1, q2)
            if inter is not None:
                hits.append(inter)
    return hits


def _endpoint_towards_tooth(line_pts: list[list[float]], tooth_bbox: list[float]) -> tuple[list[float], np.ndarray]:
    """Bone-line endpoint on the tooth side + outward extension direction."""
    center = np.array(bbox_center(tooth_bbox), dtype=np.float64)
    p0 = np.array(line_pts[0], dtype=np.float64)
    p1 = np.array(line_pts[-1], dtype=np.float64)
    if np.linalg.norm(p0 - center) <= np.linalg.norm(p1 - center):
        anchor = p0
        inward = p0 - np.array(line_pts[1], dtype=np.float64)
    else:
        anchor = p1
        inward = p1 - np.array(line_pts[-2], dtype=np.float64)
    if np.linalg.norm(inward) < 1e-8:
        inward = center - anchor
    return [float(anchor[0]), float(anchor[1])], inward


def line_contour_intersection(
    contour: np.ndarray,
    line_pts: list[list[float]],
    tooth_bbox: list[float],
) -> list[float] | None:
    if len(line_pts) < 2 or contour is None or len(contour) < 3:
        return None

    hits = _polyline_contour_hits(contour, line_pts)
    if hits:
        mid = line_pts[len(line_pts) // 2]
        best = min(hits, key=lambda p: dist2(p, mid))
        return [float(best[0]), float(best[1])]

    anchor, direction = _endpoint_towards_tooth(line_pts, tooth_bbox)
    extended_hits: list[np.ndarray] = []
    for j in range(len(contour)):
        q1 = contour[j].astype(np.float32)
        q2 = contour[(j + 1) % len(contour)].astype(np.float32)
        inter = ray_segment_intersection(anchor, direction, q1, q2)
        if inter is not None:
            extended_hits.append(inter)

    if extended_hits:
        best = min(extended_hits, key=lambda p: dist2(p, anchor))
        return [float(best[0]), float(best[1])]

    return nearest_contour_point(contour, (anchor[0], anchor[1]))


def segment_intersection(p1, p2, q1, q2, eps: float = 1e-6):
    r = p2 - p1
    s = q2 - q1
    rxs = cross2(r, s)
    qmp = q1 - p1
    if abs(rxs) < eps:
        return None
    t = cross2(qmp, s) / rxs
    u = cross2(qmp, r) / rxs
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return p1 + t * r
    return None


def assign_points_to_teeth(
    points: list[list[float]],
    bboxes: list[list[float]],
    margin: float = 0.0,
) -> list[list[list[float]]]:
    assigned: list[list[list[float]]] = [[] for _ in bboxes]
    for pt in points:
        candidates = [
            i for i, bb in enumerate(bboxes) if point_in_bbox(pt[0], pt[1], bb, margin=margin)
        ]
        if not candidates:
            i = int(np.argmin([dist2(pt, bbox_center(bb)) for bb in bboxes]))
        else:
            i = min(candidates, key=lambda idx: dist2(pt, bbox_center(bboxes[idx])))
        assigned[i].append(pt)
    return assigned


def load_tooth_mask(mask_path: Path) -> np.ndarray | None:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return (mask > 0).astype(np.uint8)


def distance_to_mask(px: float, py: float, mask: np.ndarray) -> float:
    """Distance from point to nearest foreground pixel in mask."""
    h, w = mask.shape
    xi = int(round(px))
    yi = int(round(py))
    if 0 <= xi < w and 0 <= yi < h and mask[yi, xi] > 0:
        return 0.0
    inv = (mask == 0).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    if 0 <= xi < w and 0 <= yi < h:
        return float(dist[yi, xi])
    # Off-image: approximate from border
    xi = min(max(xi, 0), w - 1)
    yi = min(max(yi, 0), h - 1)
    return float(dist[yi, xi])


def assign_points_to_teeth_mask(
    points: list[list[float]],
    bboxes: list[list[float]],
    mask_paths: list[Path],
    grace_px: float = 4.0,
) -> list[list[list[float]]]:
    """v3: mask containment, then nearest mask within grace_px, else drop."""
    masks = [load_tooth_mask(p) for p in mask_paths]
    assigned: list[list[list[float]]] = [[] for _ in bboxes]
    for pt in points:
        on_mask = [
            i
            for i, m in enumerate(masks)
            if m is not None and distance_to_mask(pt[0], pt[1], m) == 0.0
        ]
        if len(on_mask) == 1:
            assigned[on_mask[0]].append(pt)
            continue
        if len(on_mask) > 1:
            i = min(on_mask, key=lambda idx: dist2(pt, tooth_anchor_center(idx, bboxes, masks)))
            assigned[i].append(pt)
            continue

        candidates: list[tuple[float, int]] = []
        for i, m in enumerate(masks):
            if m is None:
                continue
            d = distance_to_mask(pt[0], pt[1], m)
            candidates.append((d, i))
        if not candidates:
            continue
        best_d, best_i = min(
            candidates,
            key=lambda x: (x[0], dist2(pt, tooth_anchor_center(x[1], bboxes, masks))),
        )
        if best_d <= grace_px:
            assigned[best_i].append(pt)
    return assigned


def assign_points_to_teeth_mask_region_grow(
    points: list[list[float]],
    bboxes: list[list[float]],
    mask_paths: list[Path],
    step_px: int = 1,
    max_radius_px: int = 8,
) -> list[list[list[float]]]:
    """v4: on-mask first, then assign in outward rings (step_px) up to max_radius_px; else drop."""
    masks = [load_tooth_mask(p) for p in mask_paths]
    assigned: list[list[list[float]]] = [[] for _ in bboxes]
    pending = [list(pt) for pt in points]
    step_px = max(1, int(step_px))
    max_radius_px = max(0, int(max_radius_px))

    radii = list(range(0, max_radius_px + 1, step_px))
    if not radii or radii[-1] != max_radius_px:
        radii.append(max_radius_px)

    for r in radii:
        if not pending:
            break
        next_pending: list[list[float]] = []
        for pt in pending:
            candidates: list[int] = []
            for i, m in enumerate(masks):
                if m is None:
                    continue
                d = distance_to_mask(pt[0], pt[1], m)
                if r == 0:
                    if d == 0.0:
                        candidates.append(i)
                else:
                    inner = float(r - step_px)
                    if d <= float(r) and d > inner:
                        candidates.append(i)
            if len(candidates) == 1:
                assigned[candidates[0]].append(pt)
            elif len(candidates) > 1:
                i = min(
                    candidates,
                    key=lambda idx: dist2(pt, tooth_anchor_center(idx, bboxes, masks)),
                )
                assigned[i].append(pt)
            else:
                next_pending.append(pt)
        pending = next_pending
    return assigned


def line_contour_intersection_midpoint_fallback(
    contour: np.ndarray,
    line_pts: list[list[float]],
    tooth_bbox: list[float],
) -> list[float] | None:
    """v1 intersection fallback: nearest contour point to line midpoint."""
    if len(line_pts) < 2 or contour is None or len(contour) < 3:
        return None
    hits = _polyline_contour_hits(contour, line_pts)
    if hits:
        mid = line_pts[len(line_pts) // 2]
        best = min(hits, key=lambda p: dist2(p, mid))
        return [float(best[0]), float(best[1])]
    mid = line_pts[len(line_pts) // 2]
    return nearest_contour_point(contour, (mid[0], mid[1]))


def compute_intersections_v1(
    bboxes: list[list[float]],
    bone_lines: list[list[list[float]]],
    mask_paths: list[Path],
) -> list[list[list[float]]]:
    intersections: list[list[list[float]]] = [[] for _ in bboxes]
    contours = [load_mask_contour(p) for p in mask_paths]
    for line in bone_lines:
        pair = adjacent_teeth_for_bone_line(bboxes, line)
        if pair is None:
            continue
        for tooth_i in pair:
            if contours[tooth_i] is None:
                continue
            pt = line_contour_intersection_midpoint_fallback(contours[tooth_i], line, bboxes[tooth_i])
            if pt is not None:
                intersections[tooth_i].append(pt)
    return intersections


def pad_keypoints(points: list[list[float]], count: int = 2) -> list[list[float]]:
    out = [p[:2] for p in sorted(points, key=lambda p: p[0])[:count]]
    while len(out) < count:
        out.append([0.0, 0.0])
    return out


def keypoints_with_visibility(points: list[list[float]], count: int = 2) -> list[list[float]]:
    padded = pad_keypoints(points, count)
    result = []
    for pt in padded:
        vis = 2 if pt != [0.0, 0.0] else 0
        result.append([pt[0], pt[1], vis])
    return result


def infer_root_label(apex_points: list[list[float]]) -> int:
    visible = [p for p in apex_points if p != [0.0, 0.0]]
    return 2 if len(visible) >= 2 else 1


def adjacent_teeth_for_bone_line(bboxes: list[list[float]], line_pts: list[list[float]]) -> tuple[int, int] | None:
    if len(bboxes) < 2:
        return None
    cx = float(np.mean([p[0] for p in line_pts]))
    order = sorted(range(len(bboxes)), key=lambda i: bbox_center(bboxes[i])[0])
    for left_idx, right_idx in zip(order[:-1], order[1:]):
        left_c = bbox_center(bboxes[left_idx])[0]
        right_c = bbox_center(bboxes[right_idx])[0]
        if left_c <= cx <= right_c:
            return left_idx, right_idx
    nearest = int(np.argmin([dist2((cx, np.mean([p[1] for p in line_pts])), bbox_center(bb)) for bb in bboxes]))
    if nearest == 0:
        return 0, 1
    if nearest == len(bboxes) - 1:
        return len(bboxes) - 2, len(bboxes) - 1
    return nearest - 1, nearest


def compute_intersections(
    bboxes: list[list[float]],
    bone_lines: list[list[list[float]]],
    mask_paths: list[Path],
) -> list[list[list[float]]]:
    intersections: list[list[list[float]]] = [[] for _ in bboxes]
    contours = [load_mask_contour(p) for p in mask_paths]

    for line in bone_lines:
        pair = adjacent_teeth_for_bone_line(bboxes, line)
        if pair is None:
            continue
        left_i, right_i = pair
        for tooth_i in (left_i, right_i):
            if contours[tooth_i] is None:
                continue
            pt = line_contour_intersection(contours[tooth_i], line, bboxes[tooth_i])
            if pt is not None:
                intersections[tooth_i].append(pt)
    return intersections


def nearest_tooth_for_point(point: list[float], bboxes: list[list[float]]) -> int:
    return int(np.argmin([dist2(point, bbox_center(bb)) for bb in bboxes]))


def nearest_tooth_for_point_by_mask(
    point: list[float],
    masks: list[np.ndarray | None],
) -> int:
    """Assign point to tooth with smallest distance to its segmentation mask."""
    candidates: list[tuple[float, int]] = []
    for i, mask in enumerate(masks):
        if mask is None:
            continue
        candidates.append((distance_to_mask(point[0], point[1], mask), i))
    if not candidates:
        return 0
    return min(candidates, key=lambda x: (x[0], x[1]))[1]


def compute_intersections_endpoints(
    bboxes: list[list[float]],
    bone_lines: list[list[list[float]]],
    mask_paths: list[Path],
) -> list[list[list[float]]]:
    """v5: bone-line endpoints assigned to nearest tooth mask (same CEJ/apex as v4)."""
    masks = [load_tooth_mask(p) for p in mask_paths]
    intersections: list[list[list[float]]] = [[] for _ in bboxes]
    for line in bone_lines:
        if len(line) < 2:
            continue
        for endpoint in (line[0], line[-1]):
            tooth_i = nearest_tooth_for_point_by_mask(endpoint, masks)
            intersections[tooth_i].append([float(endpoint[0]), float(endpoint[1])])
    return intersections


def build_tooth_records(
    kp_json: dict,
    bone_json: dict | None,
    mask_dir: Path | None,
    strategy: str = "v2",
    bbox_margin: float = 0.0,
    grace_px: float = 4.0,
    grace_step_px: int = 1,
    max_grace_px: int = 8,
) -> list[ToothRecord]:
    bboxes = kp_json["bboxes"]
    mask_paths = sorted(mask_dir.glob("*.png")) if mask_dir and mask_dir.exists() else []
    use_masks = len(mask_paths) == len(bboxes)

    if strategy in ("v4", "v5") and use_masks:
        cej_assign = assign_points_to_teeth_mask_region_grow(
            kp_json.get("CEJ_Points", []),
            bboxes,
            mask_paths,
            step_px=grace_step_px,
            max_radius_px=max_grace_px,
        )
        apex_assign = assign_points_to_teeth_mask_region_grow(
            kp_json.get("Apex_Points", []),
            bboxes,
            mask_paths,
            step_px=grace_step_px,
            max_radius_px=max_grace_px,
        )
    elif strategy == "v3" and use_masks:
        cej_assign = assign_points_to_teeth_mask(
            kp_json.get("CEJ_Points", []), bboxes, mask_paths, grace_px=grace_px
        )
        apex_assign = assign_points_to_teeth_mask(
            kp_json.get("Apex_Points", []), bboxes, mask_paths, grace_px=grace_px
        )
    else:
        margin = 8.0 if strategy == "v1" else bbox_margin
        cej_assign = assign_points_to_teeth(kp_json.get("CEJ_Points", []), bboxes, margin=margin)
        apex_assign = assign_points_to_teeth(kp_json.get("Apex_Points", []), bboxes, margin=margin)

    intersections = [[] for _ in bboxes]
    if bone_json:
        if strategy == "v5":
            if not use_masks:
                raise ValueError(
                    f"v5 requires tooth masks (stem={mask_dir.name if mask_dir else '?'}, "
                    f"bboxes={len(bboxes)}, masks={len(mask_paths)})"
                )
            intersections = compute_intersections_endpoints(
                bboxes, bone_json.get("Bone_Lines", []), mask_paths
            )
        elif use_masks:
            if strategy == "v1":
                intersections = compute_intersections_v1(
                    bboxes, bone_json.get("Bone_Lines", []), mask_paths
                )
            else:
                intersections = compute_intersections(
                    bboxes, bone_json.get("Bone_Lines", []), mask_paths
                )

    records: list[ToothRecord] = []
    for i, bbox in enumerate(bboxes):
        apex = pad_keypoints(apex_assign[i], 2)
        record = ToothRecord(
            bbox=[float(v) for v in bbox],
            label=infer_root_label(apex),
            cej=pad_keypoints(cej_assign[i], 2),
            intersection=pad_keypoints(intersections[i], 2),
            apex=apex,
        )
        records.append(record)
    return records


def to_keypoint_json(records: list[ToothRecord], keypoint_type: str) -> dict:
    bboxes = [r.bbox for r in records]
    labels = [r.label for r in records]
    keypoints = []
    for r in records:
        if keypoint_type == "cej":
            pts = keypoints_with_visibility(r.cej, 2)
        elif keypoint_type == "intersection":
            pts = keypoints_with_visibility(r.intersection, 2)
        elif keypoint_type == "apex":
            pts = keypoints_with_visibility(r.apex, 2)
        else:
            raise ValueError(f"Unknown keypoint type: {keypoint_type}")
        keypoints.append(pts)
    return {"bboxes": bboxes, "labels": labels, "keypoints": keypoints}


def bbox_to_yolo_line(bbox: list[float], label: int, img_w: int, img_h: int) -> str:
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    cls = 0 if label == 1 else 1
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def process_split(
    raw_root: Path,
    processed_root: Path,
    split: str,
    strategy: str = "v2",
    grace_px: float = 4.0,
    grace_step_px: int = 1,
    max_grace_px: int = 8,
) -> dict:
    split_alias = SPLIT_ALIASES[split]
    images_dir = raw_root / split / "Images"
    kp_dir = raw_root / split / "Key Points Annotations"
    bone_dir = raw_root / split / "Bone Level Annotations"
    mask_root = raw_root / split / "Masks (Tooth-wise)"

    yolo_images = processed_root / "yolo_detection" / split_alias / "images"
    yolo_labels = processed_root / "yolo_detection" / split_alias / "labels"
    yolo_images.mkdir(parents=True, exist_ok=True)
    yolo_labels.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "teeth": 0, "missing_bone": 0, "mask_mismatch": 0, "mask_mismatch_stems": []}

    for kp_path in tqdm(sorted(kp_dir.glob("*.json")), desc=f"Preprocess {split_alias}"):
        stem = kp_path.stem
        image_name = f"{stem}.jpg"
        src_image = images_dir / image_name
        if not src_image.exists():
            continue

        kp_json = json.loads(kp_path.read_text(encoding="utf-8"))
        bboxes = kp_json["bboxes"]
        mask_dir = mask_root / stem
        mask_paths = sorted(mask_dir.glob("*.png")) if mask_dir.exists() else []
        if strategy in ("v3", "v4", "v5") and len(mask_paths) != len(bboxes):
            stats["mask_mismatch"] += 1
            stats["mask_mismatch_stems"].append(
                f"{stem}(bboxes={len(bboxes)},masks={len(mask_paths)})"
            )
            continue

        img = cv2.imread(str(src_image))
        if img is None:
            continue
        h, w = img.shape[:2]

        bone_path = bone_dir / kp_path.name
        bone_json = json.loads(bone_path.read_text(encoding="utf-8")) if bone_path.exists() else None
        if bone_json is None:
            stats["missing_bone"] += 1

        records = build_tooth_records(
            kp_json,
            bone_json,
            mask_dir,
            strategy=strategy,
            grace_px=grace_px,
            grace_step_px=grace_step_px,
            max_grace_px=max_grace_px,
        )
        if not records:
            continue

        dst_image = yolo_images / image_name
        if not dst_image.exists():
            shutil.copy2(src_image, dst_image)

        yolo_lines = [bbox_to_yolo_line(r.bbox, r.label, w, h) for r in records]
        (yolo_labels / f"{stem}.txt").write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

        for kpt_type in ("cej", "intersection", "apex"):
            out_dir = processed_root / "keypoints" / kpt_type / split_alias
            (out_dir / "images").mkdir(parents=True, exist_ok=True)
            (out_dir / "annotations").mkdir(parents=True, exist_ok=True)
            dst_img = out_dir / "images" / image_name
            if not dst_img.exists():
                shutil.copy2(src_image, dst_img)
            ann = to_keypoint_json(records, kpt_type)
            (out_dir / "annotations" / f"{stem}.json").write_text(
                json.dumps(ann, indent=2), encoding="utf-8"
            )

        stats["images"] += 1
        stats["teeth"] += len(records)

    return stats


def write_yolo_data_yaml(processed_root: Path) -> None:
    yaml_path = processed_root / "yolo_detection" / "data.yaml"
    content = f"""path: {processed_root.as_posix()}/yolo_detection
train: train/images
val: val/images
test: test/images

names:
  0: single
  1: double
"""
    yaml_path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess DenPAR dataset")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/DenPAR/Dataset"),
        help="Path to extracted DenPAR Dataset folder",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--strategy",
        choices=["v1", "v2", "v3", "v4", "v5"],
        default="v2",
        help="v1=8px margin, v2=strict bbox, v3=mask+grace, v4=region-growing, v5=v4 CEJ/apex + bone endpoints -> nearest mask",
    )
    parser.add_argument("--grace-px", type=float, default=4.0, help="v3: max distance to mask (px)")
    parser.add_argument("--grace-step-px", type=int, default=1, help="v4: ring step size (px)")
    parser.add_argument("--max-grace-px", type=int, default=8, help="v4: max outward rings (px)")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in SPLITS:
        summary[split] = process_split(
            args.raw_root,
            args.output_root,
            split,
            args.strategy,
            args.grace_px,
            args.grace_step_px,
            args.max_grace_px,
        )

    write_yolo_data_yaml(args.output_root)
    summary_path = args.output_root / "preprocess_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    for split, st in summary.items():
        if st.get("mask_mismatch"):
            print(f"  {split}: skipped {st['mask_mismatch']} mask/bbox mismatches: {st.get('mask_mismatch_stems', [])}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()

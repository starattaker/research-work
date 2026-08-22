"""QA: bone-line × tooth-mask intersection logic (v2+ algorithm, not v1–v4 ablation)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import scripts._bootstrap  # noqa: F401

from src.preprocess.prepare_dataset import (
    SPLIT_ALIASES,
    _endpoint_towards_tooth,
    _polyline_contour_hits,
    adjacent_teeth_for_bone_line,
    dist2,
    load_mask_contour,
    load_tooth_mask,
    nearest_contour_point,
    ray_segment_intersection,
)


PANEL_W = 360
TOOTH_COLORS = [
    (60, 60, 255),
    (60, 200, 60),
    (255, 160, 40),
    (200, 60, 200),
    (40, 200, 200),
    (120, 120, 255),
    (80, 255, 120),
    (255, 80, 180),
    (180, 255, 80),
    (255, 120, 120),
    (100, 180, 255),
    (255, 200, 100),
]


def tooth_color(i: int) -> tuple[int, int, int]:
    return TOOTH_COLORS[i % len(TOOTH_COLORS)]


def intersection_detail(
    contour: np.ndarray,
    line_pts: list[list[float]],
    tooth_bbox: list[float],
) -> tuple[list[float] | None, str, tuple | None]:
    """Return (point, method, highlight) where highlight marks the winning hit."""
    if len(line_pts) < 2 or contour is None or len(contour) < 3:
        return None, "none", None

    hits = _polyline_contour_hits(contour, line_pts)
    if hits:
        mid = line_pts[len(line_pts) // 2]
        best = min(hits, key=lambda p: dist2(p, mid))
        return [float(best[0]), float(best[1])], "direct", None

    anchor, direction = _endpoint_towards_tooth(line_pts, tooth_bbox)
    extended_hits: list[tuple[np.ndarray, int]] = []
    for j in range(len(contour)):
        q1 = contour[j].astype(np.float32)
        q2 = contour[(j + 1) % len(contour)].astype(np.float32)
        inter = ray_segment_intersection(anchor, direction, q1, q2)
        if inter is not None:
            extended_hits.append((inter, j))
    if extended_hits:
        best_pt, seg_j = min(extended_hits, key=lambda t: dist2(t[0], anchor))
        q1 = contour[seg_j]
        q2 = contour[(seg_j + 1) % len(contour)]
        return [float(best_pt[0]), float(best_pt[1])], "ray", (anchor, direction, q1, q2)

    pt = nearest_contour_point(contour, (anchor[0], anchor[1]))
    return pt, "nearest", None


def draw_legend_panel(h: int) -> np.ndarray:
    panel = np.full((h, PANEL_W, 3), 24, dtype=np.uint8)
    y = 28
    cv2.putText(panel, "Intersection logic", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    y += 28
    lines = [
        "1. Expert bone line (green polyline)",
        "   spans crest between two teeth.",
        "",
        "2. Pair teeth by x of line midpoint:",
        "   left & right bbox centers bracket cx.",
        "",
        "3. Per tooth in pair, find point on",
        "   THAT tooth's mask contour:",
        "",
        "   DIRECT (D): bone segment hits",
        "   contour segment; pick hit nearest",
        "   to bone-line midpoint.",
        "",
        "   RAY (R): extend from tooth-side",
        "   endpoint along inward vector until",
        "   ray hits contour (orange arrow).",
        "",
        "   NEAREST (N): last resort — closest",
        "   contour point to anchor.",
        "",
        "Colors: each tooth T{i} has one color",
        "for mask fill, contour, and its",
        "intersection dot (white ring).",
    ]
    for line in lines:
        cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA)
        y += 17

    # Mini schematic
    sy = min(y + 20, h - 130)
    sx = 24
    cv2.line(panel, (sx, sy + 50), (sx + 120, sy + 50), (0, 220, 0), 2, cv2.LINE_AA)
    cv2.ellipse(panel, (sx + 30, sy + 70), (22, 35), 0, 0, 360, (60, 60, 255), -1, cv2.LINE_AA)
    cv2.ellipse(panel, (sx + 90, sy + 70), (22, 35), 0, 0, 360, (60, 200, 60), -1, cv2.LINE_AA)
    cv2.circle(panel, (sx + 30, sy + 50), 5, (60, 60, 255), -1, cv2.LINE_AA)
    cv2.circle(panel, (sx + 90, sy + 50), 5, (60, 200, 60), -1, cv2.LINE_AA)
    cv2.putText(panel, "bone", (sx + 40, sy + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 0), 1, cv2.LINE_AA)
    cv2.putText(panel, "schematic", (sx, sy + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1, cv2.LINE_AA)

    # Color key samples
    ky = h - 90
    for i, col in enumerate(TOOTH_COLORS[:4]):
        cv2.circle(panel, (24 + i * 42, ky), 8, col, -1, cv2.LINE_AA)
        cv2.putText(panel, f"T{i}", (16 + i * 42, ky + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
    return panel


def draw_sample(
    image_path: Path,
    kp_json_path: Path,
    bone_json_path: Path,
    mask_dir: Path,
    out_path: Path,
) -> dict:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(image_path)

    kp_json = json.loads(kp_json_path.read_text(encoding="utf-8"))
    bone_json = json.loads(bone_json_path.read_text(encoding="utf-8"))
    bboxes = kp_json["bboxes"]
    bone_lines = bone_json.get("Bone_Lines", [])

    mask_paths = sorted(mask_dir.glob("*.png"))
    if len(mask_paths) != len(bboxes):
        raise ValueError(f"mask count {len(mask_paths)} != bbox count {len(bboxes)}")

    masks = [load_tooth_mask(p) for p in mask_paths]
    contours = [load_mask_contour(p) for p in mask_paths]

    stats = {"teeth": len(bboxes), "bone_lines": len(bone_lines), "direct": 0, "ray": 0, "nearest": 0, "intersection_points": 0}

    h, w = img.shape[:2]
    overlay = img.copy()

    # Mask fill (per-tooth color) + contour
    for i, (mask, contour) in enumerate(zip(masks, contours)):
        col = tooth_color(i)
        if mask is not None:
            tint = np.zeros_like(overlay)
            tint[mask > 0] = col
            cv2.addWeighted(tint, 0.32, overlay, 0.68, 0, overlay)
        if contour is not None:
            cv2.polylines(overlay, [contour.astype(np.int32)], True, col, 2, cv2.LINE_AA)

    # Bone lines
    for li, line in enumerate(bone_lines):
        pts = np.array(line, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], False, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.polylines(overlay, [pts], False, (0, 230, 0), 2, cv2.LINE_AA)
        pair = adjacent_teeth_for_bone_line(bboxes, line)
        if pair is not None:
            mid = line[len(line) // 2]
            label = f"B{li} -> T{pair[0]},T{pair[1]}"
            cv2.putText(overlay, label, (int(mid[0]), int(mid[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(overlay, label, (int(mid[0]), int(mid[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 0), 1, cv2.LINE_AA)

    # Intersections — one color per tooth
    for line in bone_lines:
        pair = adjacent_teeth_for_bone_line(bboxes, line)
        if pair is None:
            continue
        for tooth_i in pair:
            if contours[tooth_i] is None:
                continue
            pt, method, extra = intersection_detail(contours[tooth_i], line, bboxes[tooth_i])
            if pt is None:
                continue
            stats[method] += 1
            stats["intersection_points"] += 1
            col = tooth_color(tooth_i)
            x, y = int(pt[0]), int(pt[1])

            if method == "ray" and extra is not None:
                anchor, direction, q1, q2 = extra
                end = anchor + direction / (np.linalg.norm(direction) + 1e-8) * 50
                cv2.arrowedLine(
                    overlay,
                    (int(anchor[0]), int(anchor[1])),
                    (int(end[0]), int(end[1])),
                    (0, 180, 255),
                    2,
                    tipLength=0.2,
                )
                cv2.line(overlay, tuple(map(int, q1)), tuple(map(int, q2)), (0, 180, 255), 3, cv2.LINE_AA)

            cv2.circle(overlay, (x, y), 12, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(overlay, (x, y), 9, col, -1, cv2.LINE_AA)
            tag = {"direct": "D", "ray": "R", "nearest": "N"}.get(method, "?")
            cv2.putText(overlay, tag, (x - 5, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(overlay, tag, (x - 5, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(overlay, f"T{tooth_i}", (x + 12, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(overlay, f"T{tooth_i}", (x + 12, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)

    # Tooth labels
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, bbox)
        col = tooth_color(i)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), col, 1, cv2.LINE_AA)
        cv2.putText(overlay, f"T{i}", (x1 + 3, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, f"T{i}", (x1 + 3, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    panel = draw_legend_panel(h)
    canvas = np.zeros((h, w + PANEL_W, 3), dtype=np.uint8)
    canvas[:, :w] = overlay
    canvas[:, w:] = panel

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return stats


def main():
    p = argparse.ArgumentParser(description="Visualize bone-line × mask intersection logic")
    p.add_argument("--raw-root", type=Path, default=Path("data/DenPAR/Dataset"))
    p.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--all", action="store_true", help="All images in split")
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("research_log/figures/intersection_logic"))
    args = p.parse_args()

    split_raw = {v: k for k, v in SPLIT_ALIASES.items()}[args.split]
    kp_dir = args.raw_root / split_raw / "Key Points Annotations"
    img_dir = args.processed_root / "yolo_detection" / args.split / "images"
    stems = sorted(p.stem for p in kp_dir.glob("*.json"))
    if not args.all:
        stems = stems[: args.n] if args.n > 0 else stems[:12]

    summary = []
    for stem in tqdm(stems, desc="intersection_logic"):
        img_p = img_dir / f"{stem}.jpg"
        if not img_p.exists():
            img_p = args.raw_root / split_raw / "Images" / f"{stem}.jpg"
        mask_dir = args.raw_root / split_raw / "Masks (Tooth-wise)" / stem
        bone_p = args.raw_root / split_raw / "Bone Level Annotations" / f"{stem}.json"
        kp_p = kp_dir / f"{stem}.json"
        if not all(x.exists() for x in (kp_p, bone_p, mask_dir)):
            continue
        try:
            stats = draw_sample(img_p, kp_p, bone_p, mask_dir, args.out_dir / f"{args.split}_{stem}.jpg")
            summary.append({"image": stem, **stats})
        except ValueError as e:
            summary.append({"image": stem, "error": str(e)})

    (args.out_dir / f"summary_{args.split}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ok = sum(1 for s in summary if "error" not in s)
    print(f"Wrote {ok}/{len(summary)} -> {args.out_dir}")


if __name__ == "__main__":
    main()

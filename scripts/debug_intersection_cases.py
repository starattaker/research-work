"""Debug specific intersection cases — step-by-step ray/direct analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import scripts._bootstrap  # noqa: F401

from src.preprocess.prepare_dataset import (
    SPLIT_ALIASES,
    _endpoint_towards_tooth,
    _polyline_contour_hits,
    adjacent_teeth_for_bone_line,
    bbox_center,
    compute_intersections,
    line_contour_intersection,
    load_mask_contour,
    load_tooth_mask,
    ray_segment_intersection,
)

TOOTH_COLORS = [
    (60, 60, 255), (60, 200, 60), (255, 160, 40), (200, 60, 200),
    (40, 200, 200), (120, 120, 255), (80, 255, 120), (255, 80, 180),
]


def analyze(stem: str, split: str, tooth_i: int) -> dict:
    split_raw = {v: k for k, v in SPLIT_ALIASES.items()}[split]
    raw = Path("data/DenPAR/Dataset") / split_raw
    kp = json.loads((raw / "Key Points Annotations" / f"{stem}.json").read_text())
    bone = json.loads((raw / "Bone Level Annotations" / f"{stem}.json").read_text())
    mask_dir = raw / "Masks (Tooth-wise)" / stem
    bboxes = kp["bboxes"]
    lines = bone["Bone_Lines"]
    mask_paths = sorted(mask_dir.glob("*.png"))
    contours = [load_mask_contour(p) for p in mask_paths]

    report = {"stem": stem, "tooth": tooth_i, "bone_lines": []}
    for li, line in enumerate(lines):
        pair = adjacent_teeth_for_bone_line(bboxes, line)
        if pair is None or tooth_i not in pair:
            continue
        hits = _polyline_contour_hits(contours[tooth_i], line)
        anchor, direction = _endpoint_towards_tooth(line, bboxes[tooth_i])
        pt = line_contour_intersection(contours[tooth_i], line, bboxes[tooth_i])
        dn = direction / (np.linalg.norm(direction) + 1e-8)
        entry = {
            "line_idx": li,
            "pair": pair,
            "line_pts": line,
            "direct_hits": [[float(x), float(y)] for h in hits for x, y in [h]],
            "anchor": anchor,
            "ray_dir": [float(dn[0]), float(dn[1])],
            "method": "direct" if hits else "ray",
            "result": pt,
        }
        if not hits:
            c = contours[tooth_i]
            rh = []
            for j in range(len(c)):
                q1 = c[j].astype(float)
                q2 = c[(j + 1) % len(c)].astype(float)
                inter = ray_segment_intersection(anchor, direction, q1, q2)
                if inter is not None:
                    d = float(np.linalg.norm(inter - np.array(anchor)))
                    rh.append({"seg": j, "pt": [float(inter[0]), float(inter[1])], "dist": d})
            rh.sort(key=lambda x: x["dist"])
            entry["ray_hits"] = rh[:8]
        report["bone_lines"].append(entry)

    report["all_intersections"] = compute_intersections(bboxes, lines, mask_paths)[tooth_i]
    return report


def draw_debug(stem: str, split: str, tooth_i: int, line_idx: int, out_path: Path):
    split_raw = {v: k for k, v in SPLIT_ALIASES.items()}[split]
    raw = Path("data/DenPAR/Dataset") / split_raw
    img_p = Path("data/processed/yolo_detection") / split / "images" / f"{stem}.jpg"
    if not img_p.exists():
        img_p = raw / "Images" / f"{stem}.jpg"
    img = cv2.imread(str(img_p))
    kp = json.loads((raw / "Key Points Annotations" / f"{stem}.json").read_text())
    bone = json.loads((raw / "Bone Level Annotations" / f"{stem}.json").read_text())
    mask_dir = raw / "Masks (Tooth-wise)" / stem
    bboxes = kp["bboxes"]
    lines = bone["Bone_Lines"]
    line = lines[line_idx]
    mask_paths = sorted(mask_dir.glob("*.png"))
    masks = [load_tooth_mask(p) for p in mask_paths]
    contours = [load_mask_contour(p) for p in mask_paths]

    # Crop around tooth + line
    xs, ys = [], []
    for p in line:
        xs.append(p[0]); ys.append(p[1])
    x1, y1, x2, y2 = bboxes[tooth_i]
    xs.extend([x1, x2]); ys.extend([y1, y2])
    margin = 60
    cx1 = max(0, int(min(xs)) - margin)
    cy1 = max(0, int(min(ys)) - margin)
    cx2 = min(img.shape[1], int(max(xs)) + margin)
    cy2 = min(img.shape[0], int(max(ys)) + margin)
    crop = img[cy1:cy2, cx1:cx2].copy()

    def tx(p):
        return int(p[0] - cx1), int(p[1] - cy1)

    col = TOOTH_COLORS[tooth_i % len(TOOTH_COLORS)]
    m = masks[tooth_i]
    if m is not None:
        mc = m[cy1:cy2, cx1:cx2]
        tint = np.zeros_like(crop)
        tint[mc > 0] = col
        cv2.addWeighted(tint, 0.35, crop, 0.65, 0, crop)
    c = contours[tooth_i]
    if c is not None:
        cc = (c - np.array([cx1, cy1])).astype(np.int32)
        cv2.polylines(crop, [cc], True, col, 2, cv2.LINE_AA)

    pts = np.array([[tx(p) for p in line]], dtype=np.int32)
    cv2.polylines(crop, pts, False, (0, 230, 0), 2, cv2.LINE_AA)
    for j, p in enumerate(line):
        cv2.circle(crop, tx(p), 5, (0, 255, 0), -1)
        cv2.putText(crop, f"p{j}", (tx(p)[0] + 6, tx(p)[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    hits = _polyline_contour_hits(contours[tooth_i], line)
    anchor, direction = _endpoint_towards_tooth(line, bboxes[tooth_i])
    dn = direction / (np.linalg.norm(direction) + 1e-8)
    result = line_contour_intersection(contours[tooth_i], line, bboxes[tooth_i])

    for h in hits:
        cv2.circle(crop, tx(h), 7, (255, 255, 0), 2)
        cv2.putText(crop, "direct hit", (tx(h)[0] + 8, tx(h)[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

    a = tx(anchor)
    end = (int(a[0] + dn[0] * 80), int(a[1] + dn[1] * 80))
    cv2.circle(crop, a, 6, (0, 180, 255), -1)
    cv2.putText(crop, "anchor", (a[0] + 8, a[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 255), 1)
    cv2.arrowedLine(crop, a, end, (0, 180, 255), 2, tipLength=0.12)
    cv2.putText(crop, "ray dir", (end[0], end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 255), 1)

    if result:
        cv2.circle(crop, tx(result), 10, (255, 255, 255), 2)
        cv2.circle(crop, tx(result), 8, col, -1)
        method = "D" if hits else "R"
        cv2.putText(crop, f"chosen {method}", (tx(result)[0] + 10, tx(result)[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2)

    title = f"test_{stem} T{tooth_i} B{line_idx} ({'direct' if hits else 'RAY fallback'})"
    cv2.putText(crop, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(crop, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)


def find_overloaded_teeth(split: str) -> list[dict]:
    """Teeth with >3 total raw CEJ + apex + intersection points (v2 bbox assign)."""
    from src.preprocess.prepare_dataset import assign_points_to_teeth

    split_raw = {v: k for k, v in SPLIT_ALIASES.items()}[split]
    raw = Path("data/DenPAR/Dataset") / split_raw
    kp_dir = raw / "Key Points Annotations"
    bone_dir = raw / "Bone Level Annotations"
    mask_root = raw / "Masks (Tooth-wise)"
    out = []
    for kp_p in sorted(kp_dir.glob("*.json")):
        stem = kp_p.stem
        bone_p = bone_dir / f"{stem}.json"
        mask_dir = mask_root / stem
        if not bone_p.exists() or not mask_dir.exists():
            continue
        kp = json.loads(kp_p.read_text())
        bone = json.loads(bone_p.read_text())
        bboxes = kp["bboxes"]
        mask_paths = sorted(mask_dir.glob("*.png"))
        if len(mask_paths) != len(bboxes):
            continue
        cej_assign = assign_points_to_teeth(kp.get("CEJ_Points", []), bboxes, margin=8.0)
        apex_assign = assign_points_to_teeth(kp.get("Apex_Points", []), bboxes, margin=8.0)
        ints = compute_intersections(bboxes, bone.get("Bone_Lines", []), mask_paths)
        for i in range(len(bboxes)):
            n_cej = len(cej_assign[i])
            n_apex = len(apex_assign[i])
            n_int = len(ints[i])
            total = n_cej + n_apex + n_int
            if total > 3:
                out.append({
                    "image": stem,
                    "tooth": i,
                    "cej": n_cej,
                    "apex": n_apex,
                    "intersection": n_int,
                    "total": total,
                    "cej_pts": cej_assign[i],
                    "apex_pts": apex_assign[i],
                    "int_pts": ints[i],
                })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--out-dir", type=Path, default=Path("research_log/figures/intersection_debug"))
    args = p.parse_args()

    cases = [("51", 1), ("5", 1)]
    reports = []
    for stem, tooth_i in cases:
        r = analyze(stem, args.split, tooth_i)
        reports.append(r)
        print(json.dumps(r, indent=2))
        for bl in r["bone_lines"]:
            if bl["method"] == "ray":
                draw_debug(stem, args.split, tooth_i, bl["line_idx"], args.out_dir / f"debug_{stem}_T{tooth_i}_B{bl['line_idx']}.jpg")

    overloaded = find_overloaded_teeth(args.split)
    (args.out_dir / "overloaded_teeth.json").write_text(json.dumps(overloaded, indent=2), encoding="utf-8")
    print(f"\nTeeth with >3 keypoints: {len(overloaded)}")
    for o in overloaded[:20]:
        print(f"  test_{o['image']} T{o['tooth']}: cej={o['cej']} apex={o['apex']} int={o['intersection']} total={o['total']}")

    (args.out_dir / "debug_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

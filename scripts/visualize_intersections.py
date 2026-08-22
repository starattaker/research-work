"""Visualize derived intersection keypoints (bone line × tooth mask contour)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from src.preprocess.prepare_dataset import (
    SPLIT_ALIASES,
    adjacent_teeth_for_bone_line,
    build_tooth_records,
    compute_intersections,
    load_mask_contour,
)


def draw_sample(
    image_path: Path,
    kp_json_path: Path,
    bone_json_path: Path,
    mask_dir: Path,
    out_path: Path,
) -> dict:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read {image_path}")

    kp_json = json.loads(kp_json_path.read_text(encoding="utf-8"))
    bone_json = json.loads(bone_json_path.read_text(encoding="utf-8"))
    bboxes = kp_json["bboxes"]
    bone_lines = bone_json.get("Bone_Lines", [])

    mask_paths = sorted(mask_dir.glob("*.png"))
    contours = [load_mask_contour(p) for p in mask_paths] if len(mask_paths) == len(bboxes) else []
    intersections = (
        compute_intersections(bboxes, bone_lines, mask_paths)
        if contours
        else [[] for _ in bboxes]
    )
    records = build_tooth_records(kp_json, bone_json, mask_dir)

    stats = {
        "teeth": len(bboxes),
        "bone_lines": len(bone_lines),
        "intersection_points": 0,
        "teeth_with_intersection": 0,
    }

    overlay = img.copy()

    # Bone lines (expert annotation)
    for line in bone_lines:
        pts = np.array(line, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], False, (0, 255, 0), 2, cv2.LINE_AA)
        pair = adjacent_teeth_for_bone_line(bboxes, line)
        if pair is not None:
            mid = line[len(line) // 2]
            cv2.putText(
                overlay,
                f"L{pair[0]}-{pair[1]}",
                (int(mid[0]), int(mid[1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )

    # Tooth contours from masks
    for i, contour in enumerate(contours):
        if contour is not None:
            cv2.polylines(
                overlay,
                [contour.astype(np.int32)],
                True,
                (255, 200, 0),
                1,
                cv2.LINE_AA,
            )

    # Tooth bboxes + derived intersection points
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (180, 180, 180), 1)
        cv2.putText(overlay, str(i), (x1 + 4, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        tooth_hits = 0
        for pt in intersections[i]:
            cv2.circle(overlay, (int(pt[0]), int(pt[1])), 7, (0, 255, 255), -1)
            cv2.circle(overlay, (int(pt[0]), int(pt[1])), 9, (0, 0, 0), 1)
            stats["intersection_points"] += 1
            tooth_hits += 1

        for pt in records[i].intersection:
            if pt != [0.0, 0.0]:
                cv2.circle(overlay, (int(pt[0]), int(pt[1])), 4, (255, 0, 255), -1)

        if tooth_hits:
            stats["teeth_with_intersection"] += 1

    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    cv2.putText(
        img,
        "green=bone line | cyan=tooth mask | yellow=line x contour | magenta=padded (max 2) | fallback=extend endpoint",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/DenPAR/Dataset"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=Path("research_log/figures/intersection_qa"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    split_raw = {v: k for k, v in SPLIT_ALIASES.items()}[args.split]
    img_dir = args.processed_root / "yolo_detection" / args.split / "images"

    candidates = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        stem = img_path.stem
        bone_path = args.raw_root / split_raw / "Bone Level Annotations" / f"{stem}.json"
        mask_dir = args.raw_root / split_raw / "Masks (Tooth-wise)" / stem
        if bone_path.exists() and mask_dir.exists():
            bone = json.loads(bone_path.read_text(encoding="utf-8"))
            if bone.get("Bone_Lines"):
                candidates.append(img_path)

    chosen = rng.sample(candidates, min(args.n, len(candidates)))
    summary = []
    for img_path in chosen:
        stem = img_path.stem
        stats = draw_sample(
            img_path,
            args.raw_root / split_raw / "Key Points Annotations" / f"{stem}.json",
            args.raw_root / split_raw / "Bone Level Annotations" / f"{stem}.json",
            args.raw_root / split_raw / "Masks (Tooth-wise)" / stem,
            args.out_dir / f"{args.split}_{stem}.jpg",
        )
        summary.append({"image": stem, **stats})
        print(stem, stats)

    (args.out_dir / f"summary_{args.split}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Saved figures to {args.out_dir}")


if __name__ == "__main__":
    main()

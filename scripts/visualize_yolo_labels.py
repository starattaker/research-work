"""Visualize YOLO labels + apex-based root typing for QA (single vs double)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from src.preprocess.prepare_dataset import (
    SPLIT_ALIASES,
    assign_points_to_teeth,
    build_tooth_records,
    infer_root_label,
    pad_keypoints,
)


def load_yolo_labels(label_path: Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, cx, cy, w, h = map(float, line.split())
        rows.append((int(cls), (cx, cy, w, h)))
    return rows


def yolo_to_xyxy(box, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    cx, cy, w, h = box
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2


def draw_sample(
    image_path: Path,
    label_path: Path,
    kp_json_path: Path | None,
    bone_json_path: Path | None,
    mask_dir: Path | None,
    out_path: Path,
) -> dict:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read {image_path}")
    h, w = img.shape[:2]
    labels = load_yolo_labels(label_path)

    stats = {"single": 0, "double": 0, "mismatch": 0}
    apex_info = []

    if kp_json_path and kp_json_path.exists():
        kp_json = json.loads(kp_json_path.read_text(encoding="utf-8"))
        bone_json = (
            json.loads(bone_json_path.read_text(encoding="utf-8"))
            if bone_json_path and bone_json_path.exists()
            else None
        )
        records = build_tooth_records(kp_json, bone_json, mask_dir)
        apex_assign = assign_points_to_teeth(kp_json.get("Apex_Points", []), kp_json["bboxes"])
    else:
        records = None
        apex_assign = None

    for i, (cls, box) in enumerate(labels):
        x1, y1, x2, y2 = yolo_to_xyxy(box, w, h)
        color = (0, 200, 0) if cls == 0 else (0, 0, 255)  # single=green, double=red
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        tag = "single" if cls == 0 else "double"
        stats[tag] += 1

        if apex_assign is not None and i < len(apex_assign):
            apex_pts = pad_keypoints(apex_assign[i], 2)
            inferred = infer_root_label(apex_pts)
            inferred_cls = 0 if inferred == 1 else 1
            n_apex = sum(1 for p in apex_pts if p != [0.0, 0.0])
            if inferred_cls != cls:
                stats["mismatch"] += 1
                cv2.putText(img, f"MISMATCH n_apex={n_apex}", (x1, max(y1 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            for pt in apex_pts:
                if pt != [0.0, 0.0]:
                    cv2.circle(img, (int(pt[0]), int(pt[1])), 4, (255, 0, 255), -1)
            apex_info.append({"tooth": i, "yolo": tag, "n_apex": n_apex, "inferred": "single" if inferred == 1 else "double"})

        cv2.putText(img, f"{tag}", (x1, min(y2 + 14, h - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/DenPAR/Dataset"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=12, help="Number of random images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefer-double", action="store_true", help="Prefer images with double-root teeth")
    parser.add_argument("--out-dir", type=Path, default=Path("research_log/figures/yolo_qa"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    img_dir = args.processed_root / "yolo_detection" / args.split / "images"
    lbl_dir = args.processed_root / "yolo_detection" / args.split / "labels"
    split_raw = {v: k for k, v in SPLIT_ALIASES.items()}[args.split]

    images = sorted(img_dir.glob("*.jpg"))
    if args.prefer_double:
        double_imgs = []
        for img_path in images:
            lbl = lbl_dir / f"{img_path.stem}.txt"
            if any(line.startswith("1 ") for line in lbl.read_text(encoding="utf-8").splitlines()):
                double_imgs.append(img_path)
        images = double_imgs or images

    chosen = rng.sample(images, min(args.n, len(images)))
    summary = []

    for img_path in chosen:
        stem = img_path.stem
        kp_path = args.raw_root / split_raw / "Key Points Annotations" / f"{stem}.json"
        bone_path = args.raw_root / split_raw / "Bone Level Annotations" / f"{stem}.json"
        mask_dir = args.raw_root / split_raw / "Masks (Tooth-wise)" / stem
        out_path = args.out_dir / f"{args.split}_{stem}.jpg"
        stats = draw_sample(
            img_path,
            lbl_dir / f"{stem}.txt",
            kp_path,
            bone_path,
            mask_dir if mask_dir.exists() else None,
            out_path,
        )
        summary.append({"image": stem, **stats})
        print(stem, stats)

    (args.out_dir / f"summary_{args.split}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved figures to {args.out_dir}")


if __name__ == "__main__":
    main()

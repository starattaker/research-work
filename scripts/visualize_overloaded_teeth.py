"""Visualize teeth with >3 total CEJ + apex + intersection keypoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import scripts._bootstrap  # noqa: F401

from scripts.debug_intersection_cases import find_overloaded_teeth
from src.preprocess.prepare_dataset import SPLIT_ALIASES, load_mask_contour, load_tooth_mask


TOOTH_COLORS = [
    (60, 60, 255), (60, 200, 60), (255, 160, 40), (200, 60, 200),
    (40, 200, 200), (120, 120, 255), (80, 255, 120), (255, 80, 180),
    (180, 255, 80), (255, 120, 120), (100, 180, 255), (255, 200, 100),
]

CEJ_COL = (0, 255, 255)
APEX_COL = (255, 0, 255)
INT_COL = (0, 200, 255)


def draw_overloaded(entry: dict, split: str, out_path: Path):
    split_raw = {v: k for k, v in SPLIT_ALIASES.items()}[split]
    raw = Path("data/DenPAR/Dataset") / split_raw
    stem = entry["image"]
    tooth_i = entry["tooth"]
    img_p = Path("data/processed/yolo_detection") / split / "images" / f"{stem}.jpg"
    if not img_p.exists():
        img_p = raw / "Images" / f"{stem}.jpg"
    img = cv2.imread(str(img_p))
    kp = json.loads((raw / "Key Points Annotations" / f"{stem}.json").read_text())
    bone = json.loads((raw / "Bone Level Annotations" / f"{stem}.json").read_text())
    mask_dir = raw / "Masks (Tooth-wise)" / stem
    bboxes = kp["bboxes"]
    mask_paths = sorted(mask_dir.glob("*.png"))
    masks = [load_tooth_mask(p) for p in mask_paths]
    contours = [load_mask_contour(p) for p in mask_paths]

    overlay = img.copy()
    for i, (mask, contour) in enumerate(zip(masks, contours)):
        col = TOOTH_COLORS[i % len(TOOTH_COLORS)]
        if mask is not None:
            tint = np.zeros_like(overlay)
            tint[mask > 0] = col
            cv2.addWeighted(tint, 0.22, overlay, 0.78, 0, overlay)
        if contour is not None:
            cv2.polylines(overlay, [contour.astype(np.int32)], True, col, 1, cv2.LINE_AA)

    for line in bone.get("Bone_Lines", []):
        pts = np.array(line, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], False, (0, 200, 0), 1, cv2.LINE_AA)

    x1, y1, x2, y2 = map(int, bboxes[tooth_i])
    col = TOOTH_COLORS[tooth_i % len(TOOTH_COLORS)]
    cv2.rectangle(overlay, (x1, y1), (x2, y2), col, 2)

    for pt in entry["cej_pts"]:
        cv2.circle(overlay, (int(pt[0]), int(pt[1])), 7, CEJ_COL, 2)
        cv2.putText(overlay, "CEJ", (int(pt[0]) + 8, int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CEJ_COL, 1)
    for pt in entry["apex_pts"]:
        cv2.circle(overlay, (int(pt[0]), int(pt[1])), 7, APEX_COL, 2)
        cv2.putText(overlay, "A", (int(pt[0]) + 8, int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, APEX_COL, 1)
    for j, pt in enumerate(entry["int_pts"]):
        cv2.circle(overlay, (int(pt[0]), int(pt[1])), 9, (255, 255, 255), 2)
        cv2.circle(overlay, (int(pt[0]), int(pt[1])), 7, INT_COL, -1)
        cv2.putText(overlay, f"I{j}", (int(pt[0]) + 8, int(pt[1]) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, INT_COL, 1)

    title = (
        f"test_{stem} T{tooth_i} | CEJ={entry['cej']} apex={entry['apex']} "
        f"int={entry['intersection']} total={entry['total']} (>3, pad keeps 2 each)"
    )
    cv2.putText(overlay, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(overlay, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--out-dir", type=Path, default=Path("research_log/figures/overloaded_teeth"))
    args = p.parse_args()

    entries = find_overloaded_teeth(args.split)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"summary_{args.split}.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    for e in tqdm(entries, desc="overloaded"):
        draw_overloaded(e, args.split, args.out_dir / f"{args.split}_{e['image']}_T{e['tooth']}.jpg")
    print(f"Wrote {len(entries)} images -> {args.out_dir}")


if __name__ == "__main__":
    main()

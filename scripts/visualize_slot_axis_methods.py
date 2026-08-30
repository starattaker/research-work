"""Visualize three slot-axis methods (PCA / points / LR) on sample teeth."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

import scripts._bootstrap  # noqa: F401

from scripts.compare_slot_axis_icc import (
    collect_predictions,
    image_paths,
    mask_for_tooth,
    resolve_yolo_weights,
)
from src.denpar_paths import DEFAULT_DENPAR_ROOT
from src.severity.inference_pipeline import SeverityPipeline, load_gt_annotations
from src.severity.slot_matching import AxisMethod, severity_with_axis_method

METHOD_COLORS = {
    AxisMethod.MASK_PCA: (255, 120, 0),
    AxisMethod.POINTS_AXIS: (0, 200, 255),
    AxisMethod.LR_POSITION: (200, 80, 255),
}
SIDE_COLORS = [(0, 255, 0), (0, 180, 255)]


def draw_axis(img: np.ndarray, origin: tuple[float, float], direction: tuple[float, float], color, label: str):
    ox, oy = int(origin[0]), int(origin[1])
    scale = 80
    ex = int(ox + direction[0] * scale)
    ey = int(oy + direction[1] * scale)
    cv2.arrowedLine(img, (ox, oy), (ex, ey), color, 2, tipLength=0.2)
    cv2.circle(img, (ox, oy), 4, color, -1)
    cv2.putText(img, label, (ex + 4, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_lr_line(img: np.ndarray, reference_x: float, y1: float, y2: float, color):
    x = int(reference_x)
    cv2.line(img, (x, int(y1)), (x, int(y2)), color, 1, cv2.LINE_AA)
    cv2.putText(img, "LR split", (x + 3, int(y1) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def draw_side_points(panel: np.ndarray, sides, method_name: str, sev: float | None):
    for si, side in enumerate(sides):
        col = SIDE_COLORS[si]
        for name, pt in (("C", side.cej), ("I", side.intersection), ("A", side.apex)):
            if pt is None:
                continue
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(panel, (x, y), 5, col, -1)
            cv2.putText(panel, f"{name}{si}", (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
    title = f"{method_name}  sev={sev:.1f}%" if sev is not None else f"{method_name}  sev=—"
    cv2.putText(panel, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(panel, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)


def main():
    parser = argparse.ArgumentParser(description="Draw PCA / points / LR axis on sample teeth")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed_v6"))
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_DENPAR_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--yolo-weights", type=Path, default=None)
    parser.add_argument("--cej-weights", type=Path, required=True)
    parser.add_argument("--intersection-weights", type=Path, required=True)
    parser.add_argument("--apex-weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-images", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge-radius", type=float, default=20.0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("research_log/figures/slot_axis_methods"),
    )
    args = parser.parse_args()

    paths = image_paths(args.data_root, args.split)
    rng = random.Random(args.seed)
    picks = rng.sample(paths, min(args.n_images, len(paths)))

    pipeline = SeverityPipeline(
        yolo_weights=resolve_yolo_weights(args.yolo_weights),
        cej_weights=args.cej_weights,
        intersection_weights=args.intersection_weights,
        apex_weights=args.apex_weights,
        device=args.device,
        inference_mode="full",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in picks:
        stem = img_path.stem
        merged = load_gt_annotations(args.data_root, args.split, stem)
        if merged is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        kps_by_tooth = collect_predictions(pipeline, img_path, merged)
        saved = False

        for tooth_idx, bbox in enumerate(merged["bboxes"]):
            if saved or tooth_idx not in kps_by_tooth:
                continue
            k = kps_by_tooth[tooth_idx]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            pad = 20
            crop = img[
                max(0, y1 - pad) : min(img.shape[0], y2 + pad),
                max(0, x1 - pad) : min(img.shape[1], x2 + pad),
            ].copy()
            ox, oy = max(0, x1 - pad), max(0, y1 - pad)

            panels = []
            for method in AxisMethod:
                panel = crop.copy()
                mask = mask_for_tooth(args.raw_root, args.split, stem, tooth_idx)
                if mask is not None:
                    mh, mw = mask.shape
                    if mh == img.shape[0] and mw == img.shape[1]:
                        mask_crop = mask[max(0, y1 - pad) : min(mh, y2 + pad), max(0, x1 - pad) : min(mw, x2 + pad)]
                    else:
                        mask_crop = mask
                    overlay = panel.copy()
                    overlay[mask_crop > 0] = (overlay[mask_crop > 0] * 0.6 + np.array([80, 80, 180]) * 0.4).astype(
                        np.uint8
                    )
                    panel = overlay

                sev, axis, sides = severity_with_axis_method(
                    k.get("cej"),
                    k.get("intersection"),
                    k.get("apex"),
                    method,
                    bbox,
                    mask,
                    args.merge_radius,
                )
                for side in sides:
                    for pt in (side.cej, side.intersection, side.apex):
                        if pt is None:
                            continue
                        lx, ly = int(pt[0] - ox), int(pt[1] - oy)
                        cv2.circle(panel, (lx, ly), 3, (180, 180, 180), -1)

                color = METHOD_COLORS[method]
                if method == AxisMethod.LR_POSITION and axis.reference_x is not None:
                    draw_lr_line(panel, axis.reference_x - ox, 0, panel.shape[0], color)
                else:
                    draw_axis(
                        panel,
                        (axis.origin[0] - ox, axis.origin[1] - oy),
                        axis.direction,
                        color,
                        method.value[:6],
                    )
                local_sides = []
                for side in sides:
                    local_sides.append(
                        type(side)(
                            cej=(side.cej[0] - ox, side.cej[1] - oy) if side.cej else None,
                            intersection=(side.intersection[0] - ox, side.intersection[1] - oy)
                            if side.intersection
                            else None,
                            apex=(side.apex[0] - ox, side.apex[1] - oy) if side.apex else None,
                        )
                    )
                draw_side_points(panel, local_sides, method.value, sev)
                panels.append(panel)

            if not panels:
                continue
            row = np.hstack(panels)
            out_path = args.out_dir / f"{stem}_tooth{tooth_idx}.jpg"
            cv2.imwrite(str(out_path), row)
            saved = True

    print(f"Figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()

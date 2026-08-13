"""Run YOLO best.pt on sample images — pred vs ground-truth overlay for visual QA.

Use in a separate terminal while keypoint training runs. Does not affect training.

Example (Linux):
  export PYTHONPATH=.
  python scripts/test_yolo_detection.py \\
    --weights runs/detect/runs/detection/yolov8x_tooth/weights/best.pt \\
    --n 10 --seed 7 --compare-gt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


CLASS_NAMES = {0: "single", 1: "double"}
CLASS_COLORS = {
    "pred_single": (60, 60, 255),   # red-ish BGR
    "pred_double": (0, 140, 255),   # orange
    "gt_single": (0, 200, 0),       # green
    "gt_double": (255, 180, 0),     # cyan
}


def load_yolo_labels(label_path: Path) -> list[tuple[int, np.ndarray]]:
    rows = []
    if not label_path.exists():
        return rows
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, cx, cy, w, h = map(float, line.split())
        rows.append((int(cls), np.array([cx, cy, w, h], dtype=np.float32)))
    return rows


def yolo_norm_to_xyxy(box: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    cx, cy, w, h = box
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def match_boxes(
    gt_boxes: list[tuple[int, np.ndarray]],
    pred_boxes: list[tuple[int, np.ndarray, float]],
    iou_thresh: float = 0.5,
) -> dict:
    used_pred = set()
    matched_ious = []
    class_correct = 0
    missed = 0

    for gt_cls, gt_xyxy in gt_boxes:
        best_iou = 0.0
        best_j = -1
        for j, (pred_cls, pred_xyxy, _conf) in enumerate(pred_boxes):
            if j in used_pred:
                continue
            iou = box_iou(gt_xyxy, pred_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_thresh:
            used_pred.add(best_j)
            matched_ious.append(best_iou)
            if pred_boxes[best_j][0] == gt_cls:
                class_correct += 1
        else:
            missed += 1

    extra = len(pred_boxes) - len(used_pred)
    return {
        "n_gt": len(gt_boxes),
        "n_pred": len(pred_boxes),
        "n_matched": len(matched_ious),
        "n_missed_gt": missed,
        "n_extra_pred": extra,
        "mean_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "class_acc_on_matched": (
            class_correct / len(matched_ious) if matched_ious else 0.0
        ),
    }


def draw_box(
    img: np.ndarray,
    xyxy: np.ndarray,
    label: str,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        img,
        label,
        (x1, max(y1 - 6, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def resolve_weights(path: Path | None) -> Path:
    candidates = [
        path,
        Path("runs/detect/runs/detection/yolov8x_tooth/weights/best.pt"),
        Path("runs/detection/yolov8x_tooth/weights/best.pt"),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Weights not found. Pass --weights path/to/best.pt "
        "(expected under runs/detect/.../best.pt)."
    )


def main():
    parser = argparse.ArgumentParser(description="Visual YOLO detection test on N images")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to best.pt (auto-detects common run folders if omitted)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/processed/yolo_detection"),
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=10, help="Number of images")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--compare-gt",
        action="store_true",
        help="Overlay ground-truth boxes (green/cyan) under predictions",
    )
    parser.add_argument(
        "--prefer-double",
        action="store_true",
        help="Sample images that contain at least one double-root GT box",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("research_log/figures/yolo_inference"),
    )
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    weights = resolve_weights(args.weights)
    img_dir = args.data_root / args.split / "images"
    lbl_dir = args.data_root / args.split / "labels"
    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No images in {img_dir}")

    rng = random.Random(args.seed)
    if args.prefer_double:
        double_imgs = []
        for img_path in images:
            lbl = lbl_dir / f"{img_path.stem}.txt"
            if any(line.startswith("1 ") for line in lbl.read_text(encoding="utf-8").splitlines()):
                double_imgs.append(img_path)
        images = double_imgs or images

    chosen = rng.sample(images, min(args.n, len(images)))
    model = YOLO(str(weights))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    print(f"Weights: {weights}")
    print(f"Images:  {args.split} x {len(chosen)}  ->  {args.out_dir}\n")

    for img_path in chosen:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"SKIP (unreadable): {img_path.name}")
            continue
        h, w = img.shape[:2]
        canvas = img.copy()

        gt_rows = load_yolo_labels(lbl_dir / f"{img_path.stem}.txt")
        gt_boxes = [(cls, yolo_norm_to_xyxy(box, w, h)) for cls, box in gt_rows]

        if args.compare_gt:
            for cls, xyxy in gt_boxes:
                name = CLASS_NAMES.get(cls, str(cls))
                color = CLASS_COLORS["gt_single"] if cls == 0 else CLASS_COLORS["gt_double"]
                draw_box(canvas, xyxy, f"GT {name}", color, thickness=2)

        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]

        pred_boxes: list[tuple[int, np.ndarray, float]] = []
        if results.boxes is not None and len(results.boxes):
            for box in results.boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                xyxy = box.xyxy.cpu().numpy().reshape(4)
                pred_boxes.append((cls, xyxy, conf))

        for cls, xyxy, conf in pred_boxes:
            name = CLASS_NAMES.get(cls, str(cls))
            color = CLASS_COLORS["pred_single"] if cls == 0 else CLASS_COLORS["pred_double"]
            draw_box(canvas, xyxy, f"PRED {name} {conf:.2f}", color, thickness=2)

        stats = match_boxes(gt_boxes, pred_boxes)
        stats["image"] = img_path.stem
        summary.append(stats)

        header = (
            f"{img_path.stem} | GT={stats['n_gt']} PRED={stats['n_pred']} "
            f"matched={stats['n_matched']} missed={stats['n_missed_gt']} "
            f"extra={stats['n_extra_pred']} meanIoU={stats['mean_iou']:.3f} "
            f"clsAcc={stats['class_acc_on_matched']:.2f}"
        )
        cv2.putText(
            canvas,
            header[:120],
            (8, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        out_path = args.out_dir / f"{args.split}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), canvas)
        print(header)
        print(f"  -> {out_path}")

    if summary:
        mean_iou = float(np.mean([s["mean_iou"] for s in summary if s["n_matched"]]))
        print(
            f"\nAverage mean IoU (matched boxes): {mean_iou:.3f} over {len(summary)} images"
        )

    summary_path = args.out_dir / f"summary_{args.split}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")
    if args.compare_gt:
        print("Legend: green/cyan = GT | red/orange = predictions")


if __name__ == "__main__":
    main()

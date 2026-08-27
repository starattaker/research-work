"""End-to-end severity ICC: YOLO + 3× Keypoint R-CNN + paper Eq. 1 vs GT labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import scripts._bootstrap  # noqa: F401

from src.severity.icc import icc21
from src.severity.inference_pipeline import SeverityPipeline, load_gt_annotations


def resolve_yolo_weights(path: Path | None) -> Path:
    candidates = [
        path,
        Path("runs/detect/runs/detection/yolov8x_tooth/weights/best.pt"),
        Path("runs/detection/yolov8x_tooth/weights/best.pt"),
        Path("runs/detect/train/weights/best.pt"),
    ]
    for c in candidates:
        if c and c.exists():
            return c
    raise FileNotFoundError("YOLO weights not found. Pass --yolo-weights.")


def image_paths(data_root: Path, split: str) -> list[Path]:
    img_dir = data_root / "yolo_detection" / split / "images"
    if not img_dir.exists():
        img_dir = data_root / "keypoints" / "cej" / split / "images"
    return sorted(img_dir.glob("*.jpg"))


def main():
    parser = argparse.ArgumentParser(description="End-to-end severity ICC on DenPAR")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed_v6"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--yolo-weights", type=Path, default=None)
    parser.add_argument("--cej-weights", type=Path, required=True)
    parser.add_argument("--intersection-weights", type=Path, required=True)
    parser.add_argument("--apex-weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--nms-thresh", type=float, default=0.6, help="Paper NMS IoU")
    parser.add_argument("--match-iou", type=float, default=0.5, help="GT↔YOLO IoU")
    parser.add_argument(
        "--keypoint-match-iou",
        type=float,
        default=0.5,
        help="GT/CEJ-box ↔ Keypoint R-CNN det IoU",
    )
    parser.add_argument(
        "--no-require-yolo",
        action="store_true",
        help="Include teeth even when YOLO misses (upper-bound diagnostic)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("research_log/severity_icc_end_to_end.json"),
    )
    args = parser.parse_args()

    yolo_weights = resolve_yolo_weights(args.yolo_weights)
    pipeline = SeverityPipeline(
        yolo_weights=yolo_weights,
        cej_weights=args.cej_weights,
        intersection_weights=args.intersection_weights,
        apex_weights=args.apex_weights,
        device=args.device,
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
        match_iou=args.match_iou,
        keypoint_match_iou=args.keypoint_match_iou,
        require_yolo=not args.no_require_yolo,
    )

    gt_vals: list[float] = []
    pred_vals: list[float] = []
    per_image = []
    stats = {
        "teeth_total": 0,
        "yolo_matched": 0,
        "both_valid": 0,
        "gt_only_valid": 0,
        "pred_only_valid": 0,
    }

    paths = image_paths(args.data_root, args.split)
    if not paths:
        raise FileNotFoundError(f"No images for split={args.split} under {args.data_root}")

    for img_path in tqdm(paths, desc=f"ICC {args.split}"):
        stem = img_path.stem
        merged = load_gt_annotations(args.data_root, args.split, stem)
        if merged is None:
            continue
        rows = pipeline.predict_image_severities(img_path, merged)
        img_stats = {"image": stem, "teeth": len(rows), "pairs": 0}
        for row in rows:
            stats["teeth_total"] += 1
            if row["yolo_matched"]:
                stats["yolo_matched"] += 1
            gt_sev = row["gt_severity"]
            pred_sev = row["pred_severity"]
            if gt_sev is not None:
                stats["gt_only_valid"] += 1
            if pred_sev is not None:
                stats["pred_only_valid"] += 1
            if gt_sev is not None and pred_sev is not None:
                gt_vals.append(gt_sev)
                pred_vals.append(pred_sev)
                stats["both_valid"] += 1
                img_stats["pairs"] += 1
        per_image.append(img_stats)

    icc = None
    mae = None
    if len(gt_vals) >= 3:
        mat = np.column_stack([gt_vals, pred_vals])
        icc = icc21(mat)
        mae = float(np.mean(np.abs(mat[:, 0] - mat[:, 1])))

    results = {
        "paper_target_icc": 0.801,
        "split": args.split,
        "data_root": args.data_root.as_posix(),
        "yolo_weights": yolo_weights.as_posix(),
        "cej_weights": args.cej_weights.as_posix(),
        "intersection_weights": args.intersection_weights.as_posix(),
        "apex_weights": args.apex_weights.as_posix(),
        "score_thresh": args.score_thresh,
        "nms_thresh": args.nms_thresh,
        "match_iou": args.match_iou,
        "keypoint_match_iou": args.keypoint_match_iou,
        "require_yolo": not args.no_require_yolo,
        "icc": icc,
        "mae_pct": mae,
        "n_pairs": len(gt_vals),
        "stats": stats,
        "note": (
            "GT severity from v6 label JSON (aligned slots). "
            "Pred: CEJ det anchored on GT box; int/apex tied to CEJ box; YOLO gate optional."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

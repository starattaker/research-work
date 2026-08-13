"""YOLOv8x tooth detection training script (paper hyperparameters).

Replication policy: only --batch and --workers may differ from the paper (hardware).
All augmentation and training hyperparameters use Ultralytics defaults (same as
authors' official_repo/Detection Tasks/yolov8-detect.py — they do not override mosaic).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=Path("data/processed/yolo_detection/data.yaml"),
    )
    parser.add_argument("--model", default="yolov8x.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--project", default="runs/detection")
    parser.add_argument("--name", default="yolov8x_tooth")
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to last.pt to resume (e.g. runs/detect/runs/detection/yolov8x_tooth/weights/last.pt)",
    )
    args = parser.parse_args()

    weights = str(args.resume) if args.resume else args.model
    model = YOLO(weights)
    model.train(
        data=str(args.data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        lr0=args.lr0,
        optimizer="Adam",
        patience=args.patience,
        cos_lr=True,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()

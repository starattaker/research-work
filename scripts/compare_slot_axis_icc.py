"""Compare ICC across three tooth-axis / slot-matching methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import scripts._bootstrap  # noqa: F401

from src.keypoint.inference_utils import predict_keypoints_for_boxes
from src.denpar_paths import DEFAULT_DENPAR_ROOT, denpar_mask_path, resolve_denpar_root
from src.severity.icc import icc21
from src.severity.inference_pipeline import (
    SeverityPipeline,
    gt_severity_for_tooth,
    load_gt_annotations,
    load_image_tensor,
    yolo_boxes,
)
from src.severity.slot_matching import AxisMethod, severity_with_axis_method

def image_paths(data_root: Path, split: str) -> list[Path]:
    img_dir = data_root / "yolo_detection" / split / "images"
    if not img_dir.exists():
        img_dir = data_root / "keypoints" / "cej" / split / "images"
    return sorted(img_dir.glob("*.jpg"))


def resolve_yolo_weights(path: Path | None) -> Path:
    for c in (
        path,
        Path("runs/detect/runs/detection/yolov8x_tooth/weights/best.pt"),
        Path("runs/detection/yolov8x_tooth/weights/best.pt"),
    ):
        if c and c.exists():
            return c
    raise FileNotFoundError("YOLO weights not found")


def mask_for_tooth(raw_root: Path | None, split: str, stem: str, tooth_idx: int):
    mask_path = denpar_mask_path(raw_root, split, stem, tooth_idx)
    if not mask_path.exists():
        return None
    from src.preprocess.prepare_dataset import load_tooth_mask

    return load_tooth_mask(mask_path)


def metric_icc_mae(gt: list[float], pred: list[float]) -> dict:
    if len(gt) < 3:
        return {"icc": None, "mae_pct": None, "n_pairs": len(gt)}
    mat = np.column_stack([gt, pred])
    return {
        "icc": icc21(mat),
        "mae_pct": float(np.mean(np.abs(mat[:, 0] - mat[:, 1]))),
        "n_pairs": len(gt),
    }


def collect_predictions(pipeline: SeverityPipeline, img_path: Path, merged: dict) -> dict[int, dict]:
    yolo_result = pipeline.yolo.predict(
        source=str(img_path), imgsz=640, conf=pipeline.score_thresh, verbose=False
    )[0]
    yolo_t = yolo_boxes(yolo_result)
    image_tensor = load_image_tensor(img_path, pipeline.transform)
    gt_boxes = torch.tensor(merged["bboxes"], dtype=torch.float32)
    proposals, _ = pipeline._proposal_boxes_for_teeth(gt_boxes, yolo_t)
    valid_idx = [i for i, b in enumerate(proposals) if b is not None]
    kps_by_tooth: dict[int, dict] = {}
    if not valid_idx:
        return kps_by_tooth
    prop_tensor = torch.stack([proposals[i] for i in valid_idx])
    labels = torch.tensor(merged["labels"], dtype=torch.int64)[valid_idx]
    for name, model in pipeline.models.items():
        kps_list = predict_keypoints_for_boxes(
            model,
            image_tensor,
            prop_tensor,
            pipeline.device,
            "full",
            pipeline.score_thresh,
            pipeline.nms_thresh,
            labels,
            pipeline.keypoint_match_iou,
        )
        for row, tooth_idx in enumerate(valid_idx):
            kps_by_tooth.setdefault(tooth_idx, {})[name] = kps_list[row]
    return kps_by_tooth


def main():
    parser = argparse.ArgumentParser(description="ICC for mask PCA vs points-axis vs LR slot methods")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed_v6"))
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_DENPAR_ROOT, help="DenPAR root (…/Dataset)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--yolo-weights", type=Path, default=None)
    parser.add_argument("--cej-weights", type=Path, required=True)
    parser.add_argument("--intersection-weights", type=Path, required=True)
    parser.add_argument("--apex-weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--merge-radius", type=float, default=70.0, help="Merge two apex preds if closer (px); GT p10≈70")
    parser.add_argument("--out", type=Path, default=Path("research_log/slot_axis_icc_comparison.json"))
    args = parser.parse_args()

    paths = image_paths(args.data_root, args.split)
    pipeline = SeverityPipeline(
        yolo_weights=resolve_yolo_weights(args.yolo_weights),
        cej_weights=args.cej_weights,
        intersection_weights=args.intersection_weights,
        apex_weights=args.apex_weights,
        device=args.device,
        match_iou=args.match_iou,
        inference_mode="full",
    )

    method_results = {m.value: {"gt": [], "pred": []} for m in AxisMethod}
    mask_hits = mask_miss = 0

    for img_path in tqdm(paths, desc="slot-axis ICC"):
        stem = img_path.stem
        merged = load_gt_annotations(args.data_root, args.split, stem)
        if merged is None:
            continue
        kps_by_tooth = collect_predictions(pipeline, img_path, merged)

        for i, bbox in enumerate(merged["bboxes"]):
            gt_sev = gt_severity_for_tooth(merged, i)
            if gt_sev is None or i not in kps_by_tooth:
                continue
            k = kps_by_tooth[i]
            mask = mask_for_tooth(args.raw_root, args.split, stem, i)
            if mask is not None:
                mask_hits += 1
            else:
                mask_miss += 1

            for method in AxisMethod:
                pred_sev, _, _ = severity_with_axis_method(
                    k.get("cej"),
                    k.get("intersection"),
                    k.get("apex"),
                    method,
                    bbox,
                    mask,
                    args.merge_radius,
                )
                if pred_sev is None:
                    continue
                method_results[method.value]["gt"].append(gt_sev)
                method_results[method.value]["pred"].append(pred_sev)

    report = {
        "paper_target_icc": 0.801,
        "split": args.split,
        "data_root": args.data_root.as_posix(),
        "raw_root": resolve_denpar_root(args.raw_root).as_posix(),
        "merge_radius_px": args.merge_radius,
        "mask_available": {"hits": mask_hits, "misses": mask_miss},
        "methods": {},
        "notes": {
            "mask_pca": "Mask PCA axis; LR fallback if mask missing",
            "points_axis": "PCA on pred CEJ + intersection (no apex)",
            "lr_position": "Left/right by x; bbox center if 1 CEJ + 1 INT",
            "apex_rule": "One apex → both sides; two close apices → merged; else per side",
            "severity_rule": "Try side 0 then side 1; all three points required per side",
        },
    }
    print("=" * 60)
    print("SLOT-AXIS ICC COMPARISON")
    print("=" * 60)
    for method in AxisMethod:
        m = metric_icc_mae(
            method_results[method.value]["gt"],
            method_results[method.value]["pred"],
        )
        report["methods"][method.value] = m
        icc = m["icc"]
        icc_s = f"{icc:.4f}" if icc is not None else "n/a"
        print(f"  {method.value:14s}  ICC={icc_s}  n={m['n_pairs']}")
    if not resolve_denpar_root(args.raw_root).joinpath("Testing").is_dir():
        print("\n  Warning: DenPAR Testing/ not found under raw-root; mask_pca falls back to LR.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report: {args.out}")


if __name__ == "__main__":
    main()

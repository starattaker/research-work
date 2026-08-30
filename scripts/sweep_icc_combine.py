"""Sweep combine modes × inference modes for severity ICC (friend GPU)."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import scripts._bootstrap  # noqa: F401

from src.severity.icc import icc21
from src.severity.icc_pairs import collect_severity_pairs
from src.severity.inference_pipeline import SeverityPipeline, load_gt_annotations


def image_paths(data_root: Path, split: str) -> list[Path]:
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("data/processed_v6"))
    p.add_argument("--split", default="test")
    p.add_argument("--yolo-weights", type=Path, default=None)
    p.add_argument("--cej-weights", type=Path, required=True)
    p.add_argument("--intersection-weights", type=Path, required=True)
    p.add_argument("--apex-weights", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=Path("research_log/icc_combine_sweep.json"))
    args = p.parse_args()

    paths = image_paths(args.data_root, args.split)
    yolo = resolve_yolo_weights(args.yolo_weights)

    combos = list(
        itertools.product(
            ["roi", "full"],
            ["paper_x", "tensor", "geom_consistent", "lr", "mask_pca"],
            ["one_per_tooth", "both_sides"],
            ["paper_x", "pca"],
        )
    )
    results = []

    for inf_mode, combine, protocol, gt_conv in combos:
        pipeline = SeverityPipeline(
            yolo_weights=yolo,
            cej_weights=args.cej_weights,
            intersection_weights=args.intersection_weights,
            apex_weights=args.apex_weights,
            device=args.device,
            inference_mode=inf_mode,
            combine_mode=combine,
            gt_slot_convention=gt_conv,
            severity_protocol=protocol,
        )
        gt_vals: list[float] = []
        pred_vals: list[float] = []
        for img_path in tqdm(paths, desc=f"{inf_mode}/{combine}/{protocol}", leave=False):
            merged = load_gt_annotations(args.data_root, args.split, img_path.stem)
            if merged is None:
                continue
            rows = pipeline.predict_image_severities(img_path, merged, split=args.split, stem=stem)
            g, pr = collect_severity_pairs(rows, protocol)
            gt_vals.extend(g)
            pred_vals.extend(pr)

        icc = mae = None
        if len(gt_vals) >= 3:
            mat = np.column_stack([gt_vals, pred_vals])
            icc = icc21(mat)
            mae = float(np.mean(np.abs(mat[:, 0] - mat[:, 1])))
        row = {
            "inference_mode": inf_mode,
            "combine_mode": combine,
            "severity_protocol": protocol,
            "gt_slot_convention": gt_conv,
            "icc": icc,
            "mae_pct": mae,
            "n_pairs": len(gt_vals),
        }
        results.append(row)
        icc_s = f"{icc:.4f}" if icc is not None else "n/a"
        print(f"{inf_mode:4s} {combine:16s} {protocol:14s} gt={gt_conv:7s}  ICC={icc_s}  n={len(gt_vals)}")

    best = max((r for r in results if r["icc"] is not None), key=lambda r: r["icc"], default=None)
    report = {"paper_target": 0.801, "split": args.split, "best": best, "all": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nBest: {best}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

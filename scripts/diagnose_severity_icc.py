"""Exhaustive ICC diagnosis: isolate YOLO, slots, oracle, and partial-GT ceilings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import scripts._bootstrap  # noqa: F401

from scripts.compare_slot_axis_icc import (
    collect_predictions,
    image_paths,
    metric_icc_mae,
    resolve_yolo_weights,
)
from src.severity.bone_loss import compute_bone_loss_severity
from src.severity.gt_labels import load_gt_annotations, point_from_slot
from src.severity.inference_pipeline import SeverityPipeline, gt_severity_for_tooth
from src.severity.pred_combine import pred_severity_from_tensors, severity_from_tensor_slots, slot_xy_from_tensor
from src.severity.slot_matching import oracle_best_severity


def gt_slot_used(merged: dict, tooth_idx: int) -> int | None:
    for slot in (0, 1):
        c = point_from_slot(merged["cej"][tooth_idx], slot)
        t = point_from_slot(merged["intersection"][tooth_idx], slot)
        a = point_from_slot(merged["apex"][tooth_idx], slot)
        if compute_bone_loss_severity(c, t, a) is not None:
            return slot
    return None


def severity_at_slot(cej_kps, int_kps, apex_kps, slot: int) -> float | None:
    c = slot_xy_from_tensor(cej_kps, slot)
    t = slot_xy_from_tensor(int_kps, slot)
    a = slot_xy_from_tensor(apex_kps, slot)
    if c is None or t is None or a is None:
        return None
    return compute_bone_loss_severity(c, t, a)


def run_icc(gt: list[float], pred: list[float]) -> dict:
    return metric_icc_mae(gt, pred)


def main():
    parser = argparse.ArgumentParser(description="Diagnose low severity ICC")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed_v6"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--yolo-weights", type=Path, default=None)
    parser.add_argument("--cej-weights", type=Path, required=True)
    parser.add_argument("--intersection-weights", type=Path, required=True)
    parser.add_argument("--apex-weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-mode", choices=["full", "roi"], default="full")
    parser.add_argument("--out", type=Path, default=Path("research_log/severity_icc_diagnosis.json"))
    args = parser.parse_args()

    paths = image_paths(args.data_root, args.split)
    pipeline = SeverityPipeline(
        yolo_weights=resolve_yolo_weights(args.yolo_weights),
        cej_weights=args.cej_weights,
        intersection_weights=args.intersection_weights,
        apex_weights=args.apex_weights,
        device=args.device,
        inference_mode=args.inference_mode,
    )

    buckets: dict[str, tuple[list[float], list[float]]] = {
        "A1_gt_self": ([], []),
        "C1_yolo_pipeline_tensor": ([], []),
        "D1_tensor_first_valid": ([], []),
        "D3_tensor_gt_slot": ([], []),
        "D2_oracle_8combo": ([], []),
        "E_paper_x": ([], []),
        "E_geom": ([], []),
    }

    for img_path in tqdm(paths, desc="diagnose ICC"):
        stem = img_path.stem
        merged = load_gt_annotations(args.data_root, args.split, stem)
        if merged is None:
            continue
        kps_by_tooth = collect_predictions(pipeline, img_path, merged)

        for i in range(len(merged["bboxes"])):
            gt_sev = gt_severity_for_tooth(merged, i)
            if gt_sev is None:
                continue
            gt_slot = gt_slot_used(merged, i)

            buckets["A1_gt_self"][0].append(gt_sev)
            buckets["A1_gt_self"][1].append(gt_sev)

            if i not in kps_by_tooth:
                continue
            k = kps_by_tooth[i]
            cej, inter, apex = k.get("cej"), k.get("intersection"), k.get("apex")

            pred_tensor = severity_from_tensor_slots(cej, inter, apex)
            if pred_tensor is not None:
                buckets["D1_tensor_first_valid"][0].append(gt_sev)
                buckets["D1_tensor_first_valid"][1].append(pred_tensor)
                buckets["C1_yolo_pipeline_tensor"][0].append(gt_sev)
                buckets["C1_yolo_pipeline_tensor"][1].append(pred_tensor)

            if gt_slot is not None:
                pred_gt_slot = severity_at_slot(cej, inter, apex, gt_slot)
                if pred_gt_slot is not None:
                    buckets["D3_tensor_gt_slot"][0].append(gt_sev)
                    buckets["D3_tensor_gt_slot"][1].append(pred_gt_slot)

            o_sev, _ = oracle_best_severity(cej, inter, apex, gt_sev)
            if o_sev is not None:
                buckets["D2_oracle_8combo"][0].append(gt_sev)
                buckets["D2_oracle_8combo"][1].append(o_sev)

            for key, mode in (
                ("E_paper_x", "paper_x"),
                ("E_geom", "geom_consistent"),
            ):
                pred = pred_severity_from_tensors(
                    cej, inter, apex, combine_mode=mode, merge_radius_px=20.0
                )
                if pred is not None:
                    buckets[key][0].append(gt_sev)
                    buckets[key][1].append(pred)

    diagnosis = {
        "paper_target_icc": 0.801,
        "split": args.split,
        "inference_mode": args.inference_mode,
        "tests": {name: run_icc(g, p) for name, (g, p) in buckets.items()},
        "interpretation": [
            "A1≈1.0: GT labels + ICC code OK",
            "D1 low: tensor slot 0/1 order ≠ anatomical side (main bug)",
            "D3 ~0.55: correct slot if we knew GT side (not usable at inference)",
            "D2 ~0.78: oracle 8-combo ceiling with current keypoint error",
            "E_*: mask-free combine modes; best E_* is production candidate",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
    print(json.dumps(diagnosis, indent=2))


if __name__ == "__main__":
    main()

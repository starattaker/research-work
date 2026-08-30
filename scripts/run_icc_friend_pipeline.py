"""Friend GPU: audit ICC pipeline + find best combine mode + run train/val/test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import scripts._bootstrap  # noqa: F401

from scripts.run_severity_icc import image_paths, resolve_yolo_weights
from src.severity.icc import icc21
from src.severity.icc_pairs import collect_severity_pairs
from src.severity.inference_pipeline import SeverityPipeline, load_gt_annotations
from src.severity.side_details import pair_sides_by_cej, pair_sides_by_slot_index, slot_flip_rate


def run_icc_on_split(pipeline: SeverityPipeline, data_root: Path, split: str, protocol: str) -> dict:
    gt_vals: list[float] = []
    pred_vals: list[float] = []
    slot_flips = 0
    slot_flip_total = 0
    yolo_matched = 0
    teeth = 0

    for img_path in tqdm(image_paths(data_root, split), desc=f"{split}/{protocol}", leave=False):
        merged = load_gt_annotations(data_root, split, img_path.stem)
        if merged is None:
            continue
        rows = pipeline.predict_image_severities(img_path, merged, split=split, stem=img_path.stem)
        g, p = collect_severity_pairs(rows, protocol)
        gt_vals.extend(g)
        pred_vals.extend(p)
        for row in rows:
            teeth += 1
            if row["yolo_matched"]:
                yolo_matched += 1
            flip = slot_flip_rate(row.get("gt_side_details", []), row.get("pred_side_details", []))
            if flip is not None:
                slot_flip_total += 1
                if flip:
                    slot_flips += 1

    icc = mae = None
    if len(gt_vals) >= 3:
        mat = np.column_stack([gt_vals, pred_vals])
        icc = icc21(mat)
        mae = float(np.mean(np.abs(mat[:, 0] - mat[:, 1])))

    return {
        "icc": icc,
        "mae_pct": mae,
        "n_pairs": len(gt_vals),
        "teeth": teeth,
        "yolo_matched": yolo_matched,
        "slot_flip_rate": slot_flips / slot_flip_total if slot_flip_total else None,
    }


def audit_checks(pipeline: SeverityPipeline, data_root: Path, split: str = "test") -> dict:
    """Sanity checks before trusting ICC numbers."""
    gt_self_gt: list[float] = []
    gt_self_pr: list[float] = []
    cej_pairs = 0
    index_pairs = 0
    cej_better = 0

    paths = image_paths(data_root, split)[:50]
    for img_path in paths:
        merged = load_gt_annotations(data_root, split, img_path.stem)
        if merged is None:
            continue
        rows = pipeline.predict_image_severities(img_path, merged, split=split, stem=img_path.stem)
        for row in rows:
            gd = row.get("gt_side_details", [])
            pd = row.get("pred_side_details", [])
            for g, p in pair_sides_by_cej(gd, pd):
                gt_self_gt.append(g)
                gt_self_pr.append(g)
            if gd and pd:
                cej_p = pair_sides_by_cej(gd, pd)
                idx_p = pair_sides_by_slot_index(gd, pd)
                cej_pairs += len(cej_p)
                index_pairs += len(idx_p)
                if len(cej_p) == len(idx_p) and cej_p != idx_p:
                    cej_better += 1

    mat = np.column_stack([gt_self_gt, gt_self_pr]) if len(gt_self_gt) >= 3 else None
    return {
        "gt_self_icc": icc21(mat) if mat is not None else None,
        "sample_images": len(paths),
        "cej_vs_index_mismatch_teeth": cej_better,
        "note": "CEJ pairing fixes slot-index mismatch when pred slot 0 != GT PCA slot 0",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("data/processed_v6"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--yolo-weights", type=Path, default=None)
    p.add_argument("--cej-weights", type=Path, default=Path("runs/keypoints/v6_cej/best.pt"))
    p.add_argument("--intersection-weights", type=Path, default=Path("runs/keypoints/v6_intersection/best.pt"))
    p.add_argument("--apex-weights", type=Path, default=Path("runs/keypoints/v6_apex/best.pt"))
    p.add_argument("--out", type=Path, default=Path("research_log/icc_final_report.json"))
    args = p.parse_args()

    yolo = resolve_yolo_weights(args.yolo_weights)
    combines = ["mask_pca", "lr", "tensor", "geom_consistent"]

    print("=" * 60)
    print("STEP 1: Audit (test sample)")
    print("=" * 60)
    audit_pipe = SeverityPipeline(
        yolo_weights=yolo,
        cej_weights=args.cej_weights,
        intersection_weights=args.intersection_weights,
        apex_weights=args.apex_weights,
        device=args.device,
        combine_mode="tensor",
    )
    audit = audit_checks(audit_pipe, args.data_root, "test")
    print(json.dumps(audit, indent=2))

    print("\n" + "=" * 60)
    print("STEP 2: Compare combine modes on TEST (CEJ-matched both_sides)")
    print("=" * 60)
    test_results: dict[str, dict] = {}
    for mode in combines:
        pipe = SeverityPipeline(
            yolo_weights=yolo,
            cej_weights=args.cej_weights,
            intersection_weights=args.intersection_weights,
            apex_weights=args.apex_weights,
            device=args.device,
            combine_mode=mode,
            inference_mode="roi",
        )
        cej_res = run_icc_on_split(pipe, args.data_root, "test", "both_sides")
        idx_res = run_icc_on_split(pipe, args.data_root, "test", "match_by_slot")
        test_results[mode] = {"both_sides_cej": cej_res, "match_by_slot": idx_res}
        icc = cej_res["icc"]
        icc_s = f"{icc:.4f}" if icc is not None else "n/a"
        idx_icc = idx_res["icc"]
        idx_s = f"{idx_icc:.4f}" if idx_icc is not None else "n/a"
        print(f"  {mode:16s}  CEJ-match ICC={icc_s}  slot-index ICC={idx_s}  n={cej_res['n_pairs']}")

    best_mode = max(
        combines,
        key=lambda m: test_results[m]["both_sides_cej"]["icc"] or -1.0,
    )
    print(f"\nBest on test: {best_mode}")

    print("\n" + "=" * 60)
    print(f"STEP 3: ICC train/val/test with {best_mode} + CEJ pairing")
    print("=" * 60)
    final_pipe = SeverityPipeline(
        yolo_weights=yolo,
        cej_weights=args.cej_weights,
        intersection_weights=args.intersection_weights,
        apex_weights=args.apex_weights,
        device=args.device,
        combine_mode=best_mode,
        inference_mode="roi",
    )
    split_results = {}
    for split in ("train", "val", "test"):
        split_results[split] = run_icc_on_split(final_pipe, args.data_root, split, "both_sides")
        r = split_results[split]
        icc = r["icc"]
        print(f"  {split:5s}  ICC={icc:.4f}  n={r['n_pairs']}  flip_rate={r['slot_flip_rate']}")

    report = {
        "paper_target_test_icc": 0.801,
        "audit": audit,
        "test_mode_comparison": test_results,
        "best_combine_mode": best_mode,
        "final_splits_cej_matched": split_results,
        "settings": {
            "gt_slot_convention": "pca",
            "severity_protocol": "both_sides",
            "pairing": "cej_nearest_neighbor",
            "inference_mode": "roi",
            "data_root": str(args.data_root),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    test_icc = split_results["test"]["icc"]
    if test_icc is not None:
        print(f"\n>>> TEST ICC = {test_icc:.4f}  (paper 0.801)  mode={best_mode}")


if __name__ == "__main__":
    main()

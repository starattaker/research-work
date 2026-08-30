"""Friend GPU: one GPU pass per split, then CPU sweep of pairing/combine.

Selects protocol on VAL (not test), then reports train/val/test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import scripts._bootstrap  # noqa: F401

from scripts.compare_slot_axis_icc import collect_predictions, mask_for_tooth
from scripts.run_severity_icc import image_paths, resolve_yolo_weights
from src.denpar_paths import DEFAULT_DENPAR_ROOT
from src.severity.icc import icc21
from src.severity.inference_pipeline import SeverityPipeline, load_gt_annotations
from src.severity.pred_combine import pred_side_details_from_tensors
from src.severity.side_details import (
    gt_side_details_for_tooth,
    pair_sides_by_cej,
    pair_sides_by_slot_index,
)


COMBINES = ("tensor", "lr", "hungarian", "paper_x", "mask_pca")
PROTOCOLS = ("match_by_slot", "both_sides", "one_per_tooth")


def _icc_mae(gt: list[float], pred: list[float]) -> dict:
    if len(gt) < 3:
        return {"icc": None, "mae_pct": None, "n_pairs": len(gt), "pred_p99": None, "pred_max": None}
    mat = np.column_stack([gt, pred])
    return {
        "icc": icc21(mat),
        "mae_pct": float(np.mean(np.abs(mat[:, 0] - mat[:, 1]))),
        "n_pairs": len(gt),
        "pred_p99": float(np.percentile(mat[:, 1], 99)),
        "pred_max": float(np.max(mat[:, 1])),
        "gt_mean": float(np.mean(mat[:, 0])),
        "pred_mean": float(np.mean(mat[:, 1])),
    }


def _pair(gt_d, pred_d, protocol: str) -> list[tuple[float, float]]:
    if protocol == "match_by_slot":
        return pair_sides_by_slot_index(gt_d, pred_d)
    if protocol == "both_sides":
        return pair_sides_by_cej(gt_d, pred_d)
    if gt_d and pred_d:
        return [(gt_d[0].severity, pred_d[0].severity)]
    return []


def cache_split(
    pipeline: SeverityPipeline,
    data_root: Path,
    split: str,
    raw_root,
) -> list[dict]:
    rows: list[dict] = []
    for img_path in tqdm(image_paths(data_root, split), desc=f"GPU {split}"):
        merged = load_gt_annotations(data_root, split, img_path.stem)
        if merged is None:
            continue
        kps = collect_predictions(pipeline, img_path, merged)
        for i in range(len(merged["bboxes"])):
            gt_d = gt_side_details_for_tooth(merged, i, slot_convention="pca")
            k = kps.get(i, {})
            rows.append(
                {
                    "stem": img_path.stem,
                    "split": split,
                    "tooth_idx": i,
                    "bbox": merged["bboxes"][i],
                    "gt_details": gt_d,
                    "cej": k.get("cej"),
                    "intersection": k.get("intersection"),
                    "apex": k.get("apex"),
                }
            )
    return rows


def eval_cached(cache: list[dict], combine: str, protocol: str, raw_root=None) -> dict:
    gt_vals: list[float] = []
    pred_vals: list[float] = []
    for row in cache:
        mask = None
        if combine == "mask_pca":
            mask = mask_for_tooth(raw_root, row["split"], row["stem"], row["tooth_idx"])
        pred_d = pred_side_details_from_tensors(
            row["cej"],
            row["intersection"],
            row["apex"],
            combine_mode=combine,
            bbox=row["bbox"],
            mask=mask,
        )
        for g, p in _pair(row["gt_details"], pred_d, protocol):
            gt_vals.append(g)
            pred_vals.append(p)
    return _icc_mae(gt_vals, pred_vals)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("data/processed_v6"))
    p.add_argument("--raw-root", type=Path, default=DEFAULT_DENPAR_ROOT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--yolo-weights", type=Path, default=None)
    p.add_argument("--cej-weights", type=Path, default=Path("runs/keypoints/v6_cej/best.pt"))
    p.add_argument("--intersection-weights", type=Path, default=Path("runs/keypoints/v6_intersection/best.pt"))
    p.add_argument("--apex-weights", type=Path, default=Path("runs/keypoints/v6_apex/best.pt"))
    p.add_argument("--out", type=Path, default=Path("research_log/icc_final_report.json"))
    args = p.parse_args()

    yolo = resolve_yolo_weights(args.yolo_weights)
    pipeline = SeverityPipeline(
        yolo_weights=yolo,
        cej_weights=args.cej_weights,
        intersection_weights=args.intersection_weights,
        apex_weights=args.apex_weights,
        device=args.device,
        inference_mode="roi",
        combine_mode="tensor",
        raw_root=args.raw_root,
    )

    caches = {}
    for split in ("val", "test", "train"):
        caches[split] = cache_split(pipeline, args.data_root, split, args.raw_root)

    print("=" * 60)
    print("VAL sweep (choose protocol here, not on test)")
    print("=" * 60)
    val_table = []
    for combine in COMBINES:
        for protocol in PROTOCOLS:
            m = eval_cached(caches["val"], combine, protocol, args.raw_root)
            val_table.append({"combine": combine, "protocol": protocol, **m})
            icc = m["icc"]
            icc_s = f"{icc:.4f}" if icc is not None else "n/a"
            print(f"  {combine:12s} {protocol:14s} ICC={icc_s}  n={m['n_pairs']}  mae={m['mae_pct']}")

    ranked = [r for r in val_table if r["icc"] is not None]
    best = max(ranked, key=lambda r: r["icc"]) if ranked else None
    print(f"\nBest on VAL: {best}")

    print("\n" + "=" * 60)
    print("TEST all combos (report only; winner locked from val)")
    print("=" * 60)
    test_table = []
    for combine in COMBINES:
        for protocol in PROTOCOLS:
            m = eval_cached(caches["test"], combine, protocol, args.raw_root)
            test_table.append({"combine": combine, "protocol": protocol, **m})
            icc = m["icc"]
            icc_s = f"{icc:.4f}" if icc is not None else "n/a"
            print(f"  {combine:12s} {protocol:14s} ICC={icc_s}  n={m['n_pairs']}")

    print("\n" + "=" * 60)
    print("Locked protocol on train / val / test")
    print("=" * 60)
    split_final = {}
    if best:
        for split in ("train", "val", "test"):
            split_final[split] = eval_cached(caches[split], best["combine"], best["protocol"], args.raw_root)
            r = split_final[split]
            icc = r["icc"]
            print(
                f"  {split:5s}  ICC={icc:.4f}  n={r['n_pairs']}  mae={r['mae_pct']:.1f}  "
                f"pred_max={r['pred_max']:.1f}"
                if icc is not None
                else f"  {split}: n/a"
            )

    report = {
        "paper_target_test_icc": 0.801,
        "severity_clipped_0_100": True,
        "best_on_val": best,
        "val_sweep": val_table,
        "test_sweep": test_table,
        "locked_splits": split_final,
        "notes": [
            "GT = processed_v6 PCA slots.",
            "both_sides pairs by nearest CEJ; match_by_slot uses slot index.",
            "hungarian assigns INT/APEX to CEJs by min distance (no masks, no GT).",
            "Winner chosen on VAL only.",
            "Oracle 8-combo ~0.79 is a cheat ceiling, not a production method.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {args.out}")
    if best and split_final.get("test", {}).get("icc") is not None:
        print(
            f"\n>>> VAL-locked {best['combine']}+{best['protocol']}  "
            f"TEST ICC={split_final['test']['icc']:.4f}  (paper 0.801)"
        )


if __name__ == "__main__":
    main()

"""ICC for axis-constrained severity: PCA mask vs CEJ–INT midpoint vs paper Eq.1.

Computes severity on GT keypoints and on predicted keypoints (end-to-end),
pairing by slot index (match_by_slot).
"""

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
    mask_for_tooth,
    metric_icc_mae,
    resolve_yolo_weights,
)
from src.denpar_paths import DEFAULT_DENPAR_ROOT, resolve_denpar_root
from src.severity.axis_severity import AxisSeverityMethod, severities_both_sides
from src.severity.icc import icc21
from src.severity.inference_pipeline import SeverityPipeline, load_gt_annotations
from src.severity.icc_pairs import pair_sides_by_slot_index
from src.severity.side_details import SideDetail


METHODS = (
    AxisSeverityMethod.PAPER_EQ1,
    AxisSeverityMethod.MASK_PCA,
    AxisSeverityMethod.CEJ_INT_MIDPOINT,
)


def gt_details_axis(merged, tooth_idx: int, method: AxisSeverityMethod, mask, bbox) -> list[SideDetail]:
    from src.severity.gt_labels import point_from_slot

    cej_pts = merged["cej"][tooth_idx]
    sides = severities_both_sides(
        cej_pts,
        merged["intersection"][tooth_idx],
        merged["apex"][tooth_idx],
        method,
        mask=mask,
        bbox=bbox,
    )
    return [
        SideDetail(slot=s, severity=sev, cej=point_from_slot(cej_pts, s))
        for s, sev in sides
    ]


def pred_details_axis(kps: dict, method: AxisSeverityMethod, mask, bbox) -> list[SideDetail]:
    from src.severity.pred_combine import pred_side_details_from_tensors

    if method == AxisSeverityMethod.PAPER_EQ1:
        # tensor slot order + standard Eq.1
        return pred_side_details_from_tensors(
            kps.get("cej"), kps.get("intersection"), kps.get("apex"), combine_mode="tensor"
        )

    # Convert tensors to list format for axis module
    from src.severity.slot_matching import visible_points_from_tensor

    cej_t, int_t, apex_t = kps.get("cej"), kps.get("intersection"), kps.get("apex")
    if cej_t is None or int_t is None or apex_t is None:
        return []

    def tensor_to_lists(t):
        rows = []
        for i in range(t.shape[0]):
            x, y, v = float(t[i, 0]), float(t[i, 1]), float(t[i, 2])
            rows.append([x, y, 2 if v > 0 else 0])
        while len(rows) < 2:
            rows.append([0.0, 0.0, 0])
        return rows

    from src.severity.gt_labels import point_from_slot

    cej_list = tensor_to_lists(cej_t)
    sides = severities_both_sides(
        cej_list,
        tensor_to_lists(int_t),
        tensor_to_lists(apex_t),
        method,
        mask=mask,
        bbox=bbox,
    )
    return [
        SideDetail(slot=s, severity=sev, cej=point_from_slot(cej_list, s))
        for s, sev in sides
    ]


def cache_predictions_split(
    pipeline: SeverityPipeline,
    data_root: Path,
    split: str,
) -> list[tuple[Path, dict, dict[int, dict]]]:
    rows: list[tuple[Path, dict, dict[int, dict]]] = []
    for img_path in tqdm(image_paths(data_root, split), desc=f"GPU cache {split}"):
        merged = load_gt_annotations(data_root, split, img_path.stem)
        if merged is None:
            continue
        kps = collect_predictions(pipeline, img_path, merged)
        rows.append((img_path, merged, kps))
    return rows


def pairs_from_cache(
    cache: list[tuple[Path, dict, dict[int, dict]]],
    raw_root: Path,
    split: str,
    gt_method: AxisSeverityMethod,
    pred_method: AxisSeverityMethod,
) -> tuple[list[float], list[float]]:
    gt_vals: list[float] = []
    pred_vals: list[float] = []
    for img_path, merged, kps_by_tooth in cache:
        for i, bbox in enumerate(merged["bboxes"]):
            mask = mask_for_tooth(raw_root, split, img_path.stem, i)
            gt_d = gt_details_axis(merged, i, gt_method, mask, bbox)
            if i not in kps_by_tooth:
                continue
            pred_d = pred_details_axis(kps_by_tooth[i], pred_method, mask, bbox)
            for g, p in pair_sides_by_slot_index(gt_d, pred_d):
                gt_vals.append(g)
                pred_vals.append(p)
    return gt_vals, pred_vals


def main():
    p = argparse.ArgumentParser(description="Axis-constrained severity ICC")
    p.add_argument("--data-root", type=Path, default=Path("data/processed_v6"))
    p.add_argument("--raw-root", type=Path, default=DEFAULT_DENPAR_ROOT)
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--yolo-weights", type=Path, default=None)
    p.add_argument("--cej-weights", type=Path, default=Path("runs/keypoints/v6_cej/best.pt"))
    p.add_argument("--intersection-weights", type=Path, default=Path("runs/keypoints/v6_intersection/best.pt"))
    p.add_argument("--apex-weights", type=Path, default=Path("runs/keypoints/v6_apex/best.pt"))
    p.add_argument("--out", type=Path, default=Path("research_log/axis_severity_icc.json"))
    args = p.parse_args()

    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    raw_root = resolve_denpar_root(args.raw_root)

    pipeline = SeverityPipeline(
        yolo_weights=resolve_yolo_weights(args.yolo_weights),
        cej_weights=args.cej_weights,
        intersection_weights=args.intersection_weights,
        apex_weights=args.apex_weights,
        device=args.device,
        inference_mode="roi",
    )

    report: dict = {
        "paper_target_icc": 0.801,
        "data_root": args.data_root.as_posix(),
        "splits": {},
        "gt_consistency": {},
    }

    # GT-only: compare severity definitions on GT keypoints (no YOLO)
    for split in splits:
        gt_pairs: dict[str, list[tuple[float, float]]] = {}
        for img_path in tqdm(image_paths(args.data_root, split), desc=f"GT consistency {split}"):
            merged = load_gt_annotations(args.data_root, split, img_path.stem)
            if merged is None:
                continue
            for i, bbox in enumerate(merged["bboxes"]):
                mask = mask_for_tooth(raw_root, split, img_path.stem, i)
                sides_eq1 = {s: v for s, v in severities_both_sides(
                    merged["cej"][i], merged["intersection"][i], merged["apex"][i],
                    AxisSeverityMethod.PAPER_EQ1, mask=mask, bbox=bbox)}
                sides_pca = {s: v for s, v in severities_both_sides(
                    merged["cej"][i], merged["intersection"][i], merged["apex"][i],
                    AxisSeverityMethod.MASK_PCA, mask=mask, bbox=bbox)}
                sides_mid = {s: v for s, v in severities_both_sides(
                    merged["cej"][i], merged["intersection"][i], merged["apex"][i],
                    AxisSeverityMethod.CEJ_INT_MIDPOINT, mask=mask, bbox=bbox)}
                for slot in sides_eq1:
                    if slot in sides_pca:
                        gt_pairs.setdefault("paper_eq1_vs_mask_pca", []).append(
                            (sides_eq1[slot], sides_pca[slot])
                        )
                    if slot in sides_mid:
                        gt_pairs.setdefault("paper_eq1_vs_cej_int_mid", []).append(
                            (sides_eq1[slot], sides_mid[slot])
                        )
                    if slot in sides_pca and slot in sides_mid:
                        gt_pairs.setdefault("mask_pca_vs_cej_int_mid", []).append(
                            (sides_pca[slot], sides_mid[slot])
                        )

        gt_consistency_metrics = {}
        for key, pairs in gt_pairs.items():
            if len(pairs) < 3:
                gt_consistency_metrics[key] = {"icc": None, "n_pairs": len(pairs)}
            else:
                mat = np.array(pairs)
                gt_consistency_metrics[key] = {
                    "icc": icc21(mat),
                    "n_pairs": len(pairs),
                    "mae_pct": float(np.mean(np.abs(mat[:, 0] - mat[:, 1]))),
                }
        report["gt_consistency"][split] = gt_consistency_metrics

        cache = cache_predictions_split(pipeline, args.data_root, split)
        split_report: dict = {}
        for pred_m in METHODS:
            gt_v, pred_v = pairs_from_cache(cache, raw_root, split, pred_m, pred_m)
            split_report[pred_m.value] = metric_icc_mae(gt_v, pred_v)
        report["splits"][split] = split_report

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=" * 60)
    print("AXIS-CONSTRAINED SEVERITY ICC")
    print("=" * 60)
    for split in splits:
        print(f"\n[{split}] end-to-end (GT method = pred method, match_by_slot)")
        for m in METHODS:
            r = report["splits"][split][m.value]
            icc = r.get("icc")
            icc_s = f"{icc:.4f}" if icc is not None else "n/a"
            print(f"  {m.value:20s}  ICC={icc_s}  n={r['n_pairs']}")
        print(f"  GT consistency (paper vs PCA): {report['gt_consistency'][split].get('paper_eq1_vs_mask_pca')}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
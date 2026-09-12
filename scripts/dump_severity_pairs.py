"""Dump per-tooth severity pairs to CSV (for Bland-Altman / bootstrap CI).

Two tables are written:

1. ``severity_pairs_endtoend.csv`` - GT severity vs PREDICTED severity, per tooth
   side, for each axis method. This is the same pairing used by
   ``compare_axis_severity_icc.py`` (GT method = pred method, match_by_slot),
   so the aggregate ICC recomputed from this CSV must reproduce
   ``research_log/axis_severity_icc.json`` exactly.

2. ``severity_pairs_gtonly.csv`` - formula vs formula on the SAME GT keypoints
   (no model in the loop). Isolates the intrinsic reproducibility of each
   severity definition from learned-component error.

Aggregates only were persisted previously; the raw pairs were discarded. This
script keeps them so agreement statistics (limits of agreement, confidence
intervals) can be computed without a GPU.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tqdm import tqdm

import scripts._bootstrap  # noqa: F401

from scripts.compare_axis_severity_icc import (
    METHODS,
    gt_details_axis,
    pred_details_axis,
)
from scripts.compare_slot_axis_icc import (
    collect_predictions,
    image_paths,
    mask_for_tooth,
    resolve_yolo_weights,
)
from src.denpar_paths import DEFAULT_DENPAR_ROOT, resolve_denpar_root
from src.severity.axis_severity import AxisSeverityMethod, severities_both_sides
from src.severity.inference_pipeline import SeverityPipeline, load_gt_annotations

ENDTOEND_FIELDS = ["split", "method", "stem", "tooth_idx", "slot", "gt_severity", "pred_severity"]
GTONLY_FIELDS = ["split", "stem", "tooth_idx", "slot", "paper_eq1", "mask_pca", "cej_int_midpoint"]


def dump_gt_only(data_root: Path, raw_root: Path, split: str) -> list[dict]:
    """Severity under all three formulas on identical GT keypoints."""
    rows: list[dict] = []
    for img_path in tqdm(image_paths(data_root, split), desc=f"GT-only {split}"):
        merged = load_gt_annotations(data_root, split, img_path.stem)
        if merged is None:
            continue
        for i, bbox in enumerate(merged["bboxes"]):
            mask = mask_for_tooth(raw_root, split, img_path.stem, i)
            per_method = {}
            for method in METHODS:
                per_method[method.value] = dict(
                    severities_both_sides(
                        merged["cej"][i],
                        merged["intersection"][i],
                        merged["apex"][i],
                        method,
                        mask=mask,
                        bbox=bbox,
                    )
                )
            slots = set()
            for table in per_method.values():
                slots.update(table.keys())
            for slot in sorted(slots):
                rows.append(
                    {
                        "split": split,
                        "stem": img_path.stem,
                        "tooth_idx": i,
                        "slot": slot,
                        "paper_eq1": per_method[AxisSeverityMethod.PAPER_EQ1.value].get(slot),
                        "mask_pca": per_method[AxisSeverityMethod.MASK_PCA.value].get(slot),
                        "cej_int_midpoint": per_method[
                            AxisSeverityMethod.CEJ_INT_MIDPOINT.value
                        ].get(slot),
                    }
                )
    return rows


def dump_end_to_end(
    pipeline: SeverityPipeline,
    data_root: Path,
    raw_root: Path,
    split: str,
) -> list[dict]:
    """GT vs predicted severity, per method, pairing by slot index."""
    rows: list[dict] = []
    for img_path in tqdm(image_paths(data_root, split), desc=f"end-to-end {split}"):
        merged = load_gt_annotations(data_root, split, img_path.stem)
        if merged is None:
            continue
        kps_by_tooth = collect_predictions(pipeline, img_path, merged)
        for i, bbox in enumerate(merged["bboxes"]):
            if i not in kps_by_tooth:
                continue
            mask = mask_for_tooth(raw_root, split, img_path.stem, i)
            for method in METHODS:
                gt_d = gt_details_axis(merged, i, method, mask, bbox)
                pred_d = pred_details_axis(kps_by_tooth[i], method, mask, bbox)
                pred_map = {p.slot: p.severity for p in pred_d}
                for gt in gt_d:
                    pred_sev = pred_map.get(gt.slot)
                    if pred_sev is None:
                        continue
                    rows.append(
                        {
                            "split": split,
                            "method": method.value,
                            "stem": img_path.stem,
                            "tooth_idx": i,
                            "slot": gt.slot,
                            "gt_severity": gt.severity,
                            "pred_severity": pred_sev,
                        }
                    )
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}  ({len(rows)} rows)")


def main() -> None:
    p = argparse.ArgumentParser(description="Dump per-tooth severity pairs to CSV")
    p.add_argument("--data-root", type=Path, default=Path("data/processed_v6"))
    p.add_argument("--raw-root", type=Path, default=DEFAULT_DENPAR_ROOT)
    p.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--yolo-weights", type=Path, default=None)
    p.add_argument("--cej-weights", type=Path, default=Path("runs/keypoints/v6_cej/best.pt"))
    p.add_argument(
        "--intersection-weights",
        type=Path,
        default=Path("runs/keypoints/v6_intersection/best.pt"),
    )
    p.add_argument("--apex-weights", type=Path, default=Path("runs/keypoints/v6_apex/best.pt"))
    p.add_argument("--out-dir", type=Path, default=Path("research_log/severity_pairs"))
    p.add_argument(
        "--gt-only",
        action="store_true",
        help="Skip the model pass; write only the formula-vs-formula table (no GPU needed)",
    )
    args = p.parse_args()

    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    raw_root = resolve_denpar_root(args.raw_root)

    gt_rows: list[dict] = []
    for split in splits:
        gt_rows.extend(dump_gt_only(args.data_root, raw_root, split))
    write_csv(args.out_dir / "severity_pairs_gtonly.csv", gt_rows, GTONLY_FIELDS)

    if args.gt_only:
        print("--gt-only set: skipping end-to-end model pass.")
        return

    pipeline = SeverityPipeline(
        yolo_weights=resolve_yolo_weights(args.yolo_weights),
        cej_weights=args.cej_weights,
        intersection_weights=args.intersection_weights,
        apex_weights=args.apex_weights,
        device=args.device,
        inference_mode="roi",
    )

    e2e_rows: list[dict] = []
    for split in splits:
        e2e_rows.extend(dump_end_to_end(pipeline, args.data_root, raw_root, split))
    write_csv(args.out_dir / "severity_pairs_endtoend.csv", e2e_rows, ENDTOEND_FIELDS)


if __name__ == "__main__":
    main()

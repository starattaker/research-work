"""Oracle vs mask-PCA EDA: why ICC stays low despite GT masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import scripts._bootstrap  # noqa: F401

from scripts.compare_slot_axis_icc import (
    collect_predictions,
    image_paths,
    mask_for_tooth,
    metric_icc_mae,
    resolve_yolo_weights,
)
from scripts.diagnose_severity_icc import gt_slot_used, severity_at_slot
from src.denpar_paths import DEFAULT_DENPAR_ROOT, resolve_denpar_root
from src.severity.inference_pipeline import (
    SeverityPipeline,
    gt_severity_for_tooth,
    load_gt_annotations,
    severity_from_tensor_slots,
)
from src.severity.slot_matching import (
    AxisMethod,
    build_side_assignments,
    first_valid_side_index,
    flip_axis,
    oracle_best_severity,
    severity_from_side_index,
    severity_from_sides,
    severity_with_axis_method,
)


def main():
    parser = argparse.ArgumentParser(description="Oracle slot EDA vs mask PCA")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed_v6"))
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_DENPAR_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--yolo-weights", type=Path, default=None)
    parser.add_argument("--cej-weights", type=Path, required=True)
    parser.add_argument("--intersection-weights", type=Path, required=True)
    parser.add_argument("--apex-weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--merge-radius", type=float, default=20.0)
    parser.add_argument("--out-dir", type=Path, default=Path("research_log/oracle_slot_eda"))
    args = parser.parse_args()

    paths = image_paths(args.data_root, args.split)
    pipeline = SeverityPipeline(
        yolo_weights=resolve_yolo_weights(args.yolo_weights),
        cej_weights=args.cej_weights,
        intersection_weights=args.intersection_weights,
        apex_weights=args.apex_weights,
        device=args.device,
        inference_mode="full",
    )

    methods = {
        "broken_tensor_slots": [],
        "mask_pca_first_valid": [],
        "mask_pca_gt_side": [],
        "mask_pca_flipped_axis": [],
        "tensor_gt_slot": [],
        "oracle_8combo": [],
    }
    gt_all: list[float] = []

    records: list[dict] = []
    mask_hits = mask_miss = 0
    wrong_side_first = 0
    wrong_side_total = 0
    oracle_mismatch_slots = 0
    oracle_total = 0

    for img_path in tqdm(paths, desc="oracle EDA"):
        stem = img_path.stem
        merged = load_gt_annotations(args.data_root, args.split, stem)
        if merged is None:
            continue
        kps_by_tooth = collect_predictions(pipeline, img_path, merged)

        for i, bbox in enumerate(merged["bboxes"]):
            gt_sev = gt_severity_for_tooth(merged, i)
            gt_slot = gt_slot_used(merged, i)
            if gt_sev is None or gt_slot is None or i not in kps_by_tooth:
                continue
            k = kps_by_tooth[i]
            cej, inter, apex = k.get("cej"), k.get("intersection"), k.get("apex")
            mask = mask_for_tooth(args.raw_root, args.split, stem, i)
            if mask is not None:
                mask_hits += 1
            else:
                mask_miss += 1

            preds = {
                "broken_tensor_slots": severity_from_tensor_slots(cej, inter, apex),
                "mask_pca_first_valid": severity_with_axis_method(
                    cej, inter, apex, AxisMethod.MASK_PCA, bbox, mask, args.merge_radius
                )[0],
                "tensor_gt_slot": severity_at_slot(cej, inter, apex, gt_slot),
                "oracle_8combo": oracle_best_severity(cej, inter, apex, gt_sev)[0],
            }

            _, axis, sides = severity_with_axis_method(
                cej, inter, apex, AxisMethod.MASK_PCA, bbox, mask, args.merge_radius
            )
            preds["mask_pca_gt_side"] = severity_from_side_index(sides, gt_slot)

            if axis is not None:
                flipped_sides = build_side_assignments(cej, inter, apex, flip_axis(axis), args.merge_radius)
                preds["mask_pca_flipped_axis"] = severity_from_sides(flipped_sides)
            else:
                preds["mask_pca_flipped_axis"] = None

            chosen = first_valid_side_index(sides)
            if chosen is not None:
                wrong_side_total += 1
                if chosen != gt_slot:
                    wrong_side_first += 1

            o_sev, o_slots = oracle_best_severity(cej, inter, apex, gt_sev)
            if o_slots is not None:
                oracle_total += 1
                if o_slots != (gt_slot, gt_slot, gt_slot):
                    oracle_mismatch_slots += 1

            gt_all.append(gt_sev)
            for name in methods:
                methods[name].append(preds[name])

            records.append(
                {
                    "stem": stem,
                    "tooth_idx": i,
                    "gt_severity": gt_sev,
                    "gt_slot": gt_slot,
                    "mask_found": mask is not None,
                    "chosen_pca_side": chosen,
                    "oracle_slots": o_slots,
                    "abs_err_mask_pca": abs(preds["mask_pca_first_valid"] - gt_sev)
                    if preds["mask_pca_first_valid"] is not None
                    else None,
                    "abs_err_oracle": abs(o_sev - gt_sev) if o_sev is not None else None,
                    **{f"pred_{k}": v for k, v in preds.items()},
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    icc_report = {}
    for name, pred_list in methods.items():
        pairs_gt = [g for g, p in zip(gt_all, pred_list) if p is not None]
        pairs_pr = [p for p in pred_list if p is not None]
        icc_report[name] = metric_icc_mae(pairs_gt, pairs_pr)

    summary = {
        "data_root": args.data_root.as_posix(),
        "raw_root": resolve_denpar_root(args.raw_root).as_posix(),
        "mask_available": {"hits": mask_hits, "misses": mask_miss},
        "pca_first_valid_side_neq_gt_slot_rate": wrong_side_first / max(wrong_side_total, 1),
        "oracle_slots_neq_aligned_gt_slot_rate": oracle_mismatch_slots / max(oracle_total, 1),
        "icc_by_method": icc_report,
        "interpretation": [],
    }

    mp = icc_report.get("mask_pca_first_valid", {}).get("icc")
    mg = icc_report.get("mask_pca_gt_side", {}).get("icc")
    orc = icc_report.get("oracle_8combo", {}).get("icc")
    tg = icc_report.get("tensor_gt_slot", {}).get("icc")

    if mg is not None and mp is not None and mg > mp + 0.15:
        summary["interpretation"].append(
            "PCA buckets OK but 'first valid side' picks wrong root — use GT-side selection or better INT pairing."
        )
    if tg is not None and mg is not None and abs(tg - mg) < 0.05:
        summary["interpretation"].append(
            "mask_pca_gt_side ≈ tensor_gt_slot → PCA axis matches GT; tensor order was the old bug."
        )
    if orc is not None and mg is not None and orc > mg + 0.15:
        summary["interpretation"].append(
            "Oracle >> mask_pca_gt_side → models need independent per-model slot pairing (8-combo logic without GT cheat)."
        )
    flip_icc = icc_report.get("mask_pca_flipped_axis", {}).get("icc")
    if flip_icc is not None and mp is not None and flip_icc > mp + 0.1:
        summary["interpretation"].append("PCA axis SIGN may be flipped vs v6 labels — fix direction convention.")

    (args.out_dir / "oracle_slot_eda.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "per_tooth_records.json").write_text(
        json.dumps(records[:500], indent=2), encoding="utf-8"
    )

    # ICC bar chart
    names = list(icc_report.keys())
    iccs = [icc_report[n]["icc"] or 0 for n in names]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(names, iccs, color="#3b82f6")
    ax.axvline(0.801, color="red", linestyle="--", label="paper 0.801")
    ax.set_xlabel("ICC")
    ax.set_title("Severity ICC by combination method (YOLO boxes)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "icc_by_method.png", dpi=150)
    plt.close(fig)

    # Error scatter mask_pca vs oracle
    xs, ys = [], []
    for r in records:
        if r["abs_err_mask_pca"] is not None and r["abs_err_oracle"] is not None:
            xs.append(r["abs_err_mask_pca"])
            ys.append(r["abs_err_oracle"])
    if xs:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(xs, ys, alpha=0.3, s=8)
        ax.plot([0, max(xs)], [0, max(xs)], "r--", label="y=x")
        ax.set_xlabel("|pred - GT| mask_pca")
        ax.set_ylabel("|pred - GT| oracle")
        ax.set_title("Per-tooth error: mask PCA vs oracle")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out_dir / "error_scatter_mask_vs_oracle.png", dpi=150)
        plt.close(fig)

    print("=" * 60)
    print("ORACLE / MASK-PCA EDA")
    print("=" * 60)
    print(f"mask_available: hits={mask_hits} misses={mask_miss}")
    print(f"  (full detail in research_log/slot_axis_icc_comparison.json too)")
    print(f"PCA first-valid side != GT slot: {100*summary['pca_first_valid_side_neq_gt_slot_rate']:.1f}%")
    print(f"Oracle uses non-aligned slots: {100*summary['oracle_slots_neq_aligned_gt_slot_rate']:.1f}%")
    print()
    for name in names:
        icc = icc_report[name]["icc"]
        n = icc_report[name]["n_pairs"]
        icc_s = f"{icc:.4f}" if icc is not None else "n/a"
        print(f"  {name:28s} ICC={icc_s}  n={icc_report[name]['n_pairs']}")
    print()
    for line in summary["interpretation"]:
        print(f"  • {line}")
    print(f"\nReports: {args.out_dir}/")


if __name__ == "__main__":
    main()

"""GT keypoints as predictions: ICC ceiling per combine mode (no GPU weights)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import scripts._bootstrap  # noqa: F401

from src.severity.gt_labels import gt_severities_for_tooth, load_gt_annotations
from src.severity.icc import icc21
from src.severity.icc_pairs import collect_severity_pairs
from src.severity.pred_combine import pred_severities_from_tensors


def list_to_tensor(kps: list) -> torch.Tensor:
    rows = []
    for kp in kps:
        if len(kp) > 2:
            rows.append([float(kp[0]), float(kp[1]), float(kp[2])])
        else:
            rows.append([float(kp[0]), float(kp[1]), 2.0 if kp != [0.0, 0.0] else 0.0])
    while len(rows) < 2:
        rows.append([0.0, 0.0, 0.0])
    return torch.tensor(rows[:2], dtype=torch.float32)


def image_paths(data_root: Path, split: str) -> list[Path]:
    return sorted((data_root / "keypoints" / "cej" / split / "images").glob("*.jpg"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("data/processed"))
    p.add_argument("--split", default="test")
    p.add_argument("--gt-slot-convention", default="paper_x", choices=["paper_x", "pca"])
    p.add_argument("--out", type=Path, default=Path("research_log/icc_gt_as_pred_ceiling.json"))
    args = p.parse_args()

    modes = ["paper_x", "tensor", "geom_consistent"]
    buckets = {m: ([], []) for m in modes}
    protocol = "both_sides"

    for img_path in tqdm(image_paths(args.data_root, args.split), desc="GT-as-pred"):
        merged = load_gt_annotations(args.data_root, args.split, img_path.stem)
        if merged is None:
            continue
        for i in range(len(merged["bboxes"])):
            gt_sides = gt_severities_for_tooth(merged, i, slot_convention=args.gt_slot_convention)
            if not gt_sides:
                continue
            cej = list_to_tensor(merged["cej"][i])
            inter = list_to_tensor(merged["intersection"][i])
            apex = list_to_tensor(merged["apex"][i])
            row = {
                "gt_sides": gt_sides,
                "gt_severity": gt_sides[0][1],
                "pred_sides": [],
                "pred_severity": None,
            }
            for mode in modes:
                row["pred_sides"] = pred_severities_from_tensors(cej, inter, apex, combine_mode=mode)
                row["pred_severity"] = row["pred_sides"][0][1] if row["pred_sides"] else None
                g, pr = collect_severity_pairs([row], protocol)
                buckets[mode][0].extend(g)
                buckets[mode][1].extend(pr)

    report = {"split": args.split, "gt_slot_convention": args.gt_slot_convention, "protocol": protocol, "modes": {}}
    for mode, (g, pr) in buckets.items():
        icc = mae = None
        if len(g) >= 3:
            mat = np.column_stack([g, pr])
            icc = icc21(mat)
            mae = float(np.mean(np.abs(mat[:, 0] - mat[:, 1])))
        report["modes"][mode] = {"icc": icc, "mae_pct": mae, "n_pairs": len(g)}
        print(f"{mode}: ICC={icc} n={len(g)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

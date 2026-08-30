"""GT-only ICC sanity: no model weights. Verifies label + combine logic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import scripts._bootstrap  # noqa: F401

from src.severity.gt_labels import gt_severities_for_tooth, load_gt_annotations
from src.severity.icc import icc21


def image_paths(data_root: Path, split: str) -> list[Path]:
    img_dir = data_root / "keypoints" / "cej" / split / "images"
    return sorted(img_dir.glob("*.jpg"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("data/processed"))
    p.add_argument("--split", default="test")
    p.add_argument("--out", type=Path, default=Path("research_log/icc_gt_sanity.json"))
    args = p.parse_args()

    gt_pca: list[float] = []
    gt_paper: list[float] = []
    paths = image_paths(args.data_root, args.split)

    for img_path in tqdm(paths, desc="GT sanity"):
        merged = load_gt_annotations(args.data_root, args.split, img_path.stem)
        if merged is None:
            continue
        for i in range(len(merged["bboxes"])):
            pca = gt_severities_for_tooth(merged, i, slot_convention="pca")
            paper = gt_severities_for_tooth(merged, i, slot_convention="paper_x")
            if pca and paper:
                gt_pca.append(pca[0][1])
                gt_paper.append(paper[0][1])

    mat = np.column_stack([gt_pca, gt_paper]) if len(gt_pca) >= 3 else None
    icc_pca_vs_paper = icc21(mat) if mat is not None else None

    report = {
        "note": "Compares first-valid-side severity: v6 PCA slots vs paper x-sort on same GT keypoints",
        "split": args.split,
        "data_root": str(args.data_root),
        "n_teeth": len(gt_pca),
        "icc_pca_slot0_vs_paper_x": icc_pca_vs_paper,
        "mae_pct": float(np.mean(np.abs(mat[:, 0] - mat[:, 1]))) if mat is not None else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

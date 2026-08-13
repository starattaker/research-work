"""Validate processed dataset counts and sample annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def count_split(root: Path, split: str) -> dict:
    images = list((root / "yolo_detection" / split / "images").glob("*.jpg"))
    labels = list((root / "yolo_detection" / split / "labels").glob("*.txt"))
    stats = {"images": len(images), "labels": len(labels), "teeth": 0}
    for label_file in labels:
        stats["teeth"] += sum(1 for _ in label_file.read_text(encoding="utf-8").splitlines() if _.strip())
    for kpt in ("cej", "intersection", "apex"):
        ann_dir = root / "keypoints" / kpt / split / "annotations"
        stats[f"{kpt}_annotations"] = len(list(ann_dir.glob("*.json")))
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    summary = {}
    for split in ("train", "val", "test"):
        summary[split] = count_split(args.processed_root, split)

    sample = args.processed_root / "keypoints" / "cej" / "train" / "annotations"
    first = next(iter(sorted(sample.glob("*.json"))), None)
    if first:
        summary["sample_cej_annotation"] = json.loads(first.read_text(encoding="utf-8"))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

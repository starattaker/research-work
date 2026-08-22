"""Intersection QA for v1–v4 (side-by-side panels)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import scripts._bootstrap  # noqa: F401

from src.preprocess.prepare_dataset import SPLIT_ALIASES, build_tooth_records


def draw_panel(img, records, title: str) -> np.ndarray:
    out = img.copy()
    for i, r in enumerate(records):
        x1, y1, x2, y2 = map(int, r.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (120, 120, 120), 1)
        for pt in r.intersection:
            if pt == [0.0, 0.0]:
                continue
            cv2.circle(out, (int(pt[0]), int(pt[1])), 7, (0, 255, 255), -1)
            cv2.circle(out, (int(pt[0]), int(pt[1])), 9, (0, 0, 0), 1)
        cv2.putText(out, str(i), (x1 + 3, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(out, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(out, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, default=Path("data/DenPAR/Dataset"))
    p.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--all", action="store_true", help="All images in split")
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("research_log/figures/intersection_qa_all"))
    args = p.parse_args()

    versions = [("v1", "v1"), ("v2", "v2"), ("v3", "v3"), ("v4", "v4")]
    split_raw = {v: k for k, v in SPLIT_ALIASES.items()}[args.split]
    kp_dir = args.raw_root / split_raw / "Key Points Annotations"
    stems = sorted(p.stem for p in kp_dir.glob("*.json"))
    if not args.all and args.n > 0:
        stems = stems[: args.n]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for stem in tqdm(stems, desc="intersection"):
        img = cv2.imread(str(args.processed_root / "yolo_detection" / args.split / "images" / f"{stem}.jpg"))
        if img is None:
            img_p = args.raw_root / split_raw / "Images" / f"{stem}.jpg"
            img = cv2.imread(str(img_p))
        if img is None:
            continue
        kp_p = args.raw_root / split_raw / "Key Points Annotations" / f"{stem}.json"
        bone_p = args.raw_root / split_raw / "Bone Level Annotations" / f"{stem}.json"
        mask_dir = args.raw_root / split_raw / "Masks (Tooth-wise)" / stem
        if not (kp_p.exists() and mask_dir.exists()):
            continue
        kp = json.loads(kp_p.read_text())
        bone = json.loads(bone_p.read_text()) if bone_p.exists() else None
        panels = []
        stats = {"image": stem}
        for label, strat in versions:
            recs = build_tooth_records(kp, bone, mask_dir, strategy=strat)
            n_hit = sum(1 for r in recs for pt in r.intersection if pt != [0.0, 0.0])
            stats[f"{label}_hits"] = n_hit
            panels.append(draw_panel(img, recs, f"{label}: {n_hit} pts"))
        h, w = panels[0].shape[:2]
        grid = np.hstack(panels)
        cv2.imwrite(str(args.out_dir / f"{args.split}_{stem}.jpg"), grid)
        summary.append(stats)

    (args.out_dir / f"summary_{args.split}.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {len(summary)} -> {args.out_dir}")


if __name__ == "__main__":
    main()

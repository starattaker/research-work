"""CEJ drops / issues with v4 grace band overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import scripts._bootstrap  # noqa: F401

from scripts.visualize_v3_v4_grace import (
    assigned_set,
    draw_points,
    draw_tooth_bands,
    grace_band_masks,
)
from src.preprocess.prepare_dataset import (
    SPLIT_ALIASES,
    assign_points_to_teeth_mask,
    assign_points_to_teeth_mask_region_grow,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, default=Path("data/DenPAR/Dataset"))
    p.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--all", action="store_true")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--max-grace-px", type=int, default=8)
    p.add_argument("--out-dir", type=Path, default=Path("research_log/figures/cej_issues_qa"))
    args = p.parse_args()

    split_raw = {v: k for k, v in SPLIT_ALIASES.items()}[args.split]
    img_dir = args.processed_root / "yolo_detection" / args.split / "images"
    stems = sorted(x.stem for x in img_dir.glob("*.jpg"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    saved = 0

    for stem in tqdm(stems, desc="cej"):
        kp_p = args.raw_root / split_raw / "Key Points Annotations" / f"{stem}.json"
        mask_root = args.raw_root / split_raw / "Masks (Tooth-wise)" / stem
        img_p = img_dir / f"{stem}.jpg"
        if not (kp_p.exists() and mask_root.exists() and img_p.exists()):
            continue
        kp = json.loads(kp_p.read_text())
        cej_pts = kp.get("CEJ_Points", [])
        if not cej_pts:
            continue
        mask_paths = sorted(mask_root.glob("*.png"))
        bboxes = kp["bboxes"]
        if len(mask_paths) != len(bboxes):
            continue
        assign_v4 = assign_points_to_teeth_mask_region_grow(
            cej_pts, bboxes, mask_paths, step_px=1, max_radius_px=args.max_grace_px
        )
        assign_v3 = assign_points_to_teeth_mask(cej_pts, bboxes, mask_paths, grace_px=4.0)
        kept4 = assigned_set(assign_v4)
        kept3 = assigned_set(assign_v3)
        dropped = [
            pt
            for pt in cej_pts
            if (round(pt[0], 2), round(pt[1], 2)) not in kept4
            or (round(pt[0], 2), round(pt[1], 2)) not in kept3
        ]
        if not dropped:
            continue
        if not args.all and saved >= args.n:
            break

        img = cv2.imread(str(img_p))
        draw_tooth_bands(img, mask_paths, False, True, True)
        draw_points(img, cej_pts, kept4, (40, 200, 40), (40, 40, 220), "CEJ")
        cv2.putText(
            img,
            f"CEJ dropped {len(dropped)}/{len(cej_pts)} | pink=5-8px grace",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
        out = args.out_dir / f"{args.split}_{stem}.jpg"
        cv2.imwrite(str(out), img)
        summary.append({"image": stem, "dropped": len(dropped), "total": len(cej_pts)})
        saved += 1

    (args.out_dir / f"summary_{args.split}.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved {saved} issue images -> {args.out_dir}")


if __name__ == "__main__":
    main()

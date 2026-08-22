"""Find teeth with CEJ>2, apex>2, intersection>2, or total>6."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import scripts._bootstrap  # noqa: F401

from src.preprocess.prepare_dataset import SPLIT_ALIASES, assign_points_to_teeth, compute_intersections


def scan(split_filter: str | None = None) -> list[dict]:
    raw = Path("data/DenPAR/Dataset")
    anomalies = []
    for split_raw, split in SPLIT_ALIASES.items():
        if split_filter and split != split_filter:
            continue
        kp_dir = raw / split_raw / "Key Points Annotations"
        bone_dir = raw / split_raw / "Bone Level Annotations"
        mask_root = raw / split_raw / "Masks (Tooth-wise)"
        for kp_p in sorted(kp_dir.glob("*.json")):
            stem = kp_p.stem
            bone_p = bone_dir / f"{stem}.json"
            mask_dir = mask_root / stem
            if not bone_p.exists() or not mask_dir.exists():
                continue
            kp = json.loads(kp_p.read_text(encoding="utf-8"))
            bone = json.loads(bone_p.read_text(encoding="utf-8"))
            bboxes = kp["bboxes"]
            mask_paths = sorted(mask_dir.glob("*.png"))
            if len(mask_paths) != len(bboxes):
                continue
            cej = assign_points_to_teeth(kp.get("CEJ_Points", []), bboxes, margin=8.0)
            apex = assign_points_to_teeth(kp.get("Apex_Points", []), bboxes, margin=8.0)
            ints = compute_intersections(bboxes, bone.get("Bone_Lines", []), mask_paths)
            for i in range(len(bboxes)):
                nc, na, ni = len(cej[i]), len(apex[i]), len(ints[i])
                total = nc + na + ni
                flags = {
                    "cej_gt2": nc > 2,
                    "apex_gt2": na > 2,
                    "int_gt2": ni > 2,
                    "total_gt6": total > 6,
                }
                if not any(flags.values()):
                    continue
                anomalies.append({
                    "split": split,
                    "image": stem,
                    "tooth": i,
                    "cej": nc,
                    "apex": na,
                    "intersection": ni,
                    "total": total,
                    **flags,
                    "cej_pts": cej[i],
                    "apex_pts": apex[i],
                    "int_pts": ints[i],
                })
    return anomalies


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default=None, choices=["train", "val", "test"])
    p.add_argument("--out", type=Path, default=Path("research_log/figures/keypoint_anomalies/summary_all_splits.json"))
    args = p.parse_args()

    anomalies = scan(args.split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(anomalies, indent=2), encoding="utf-8")

    c = Counter()
    for a in anomalies:
        for k in ("cej_gt2", "apex_gt2", "int_gt2", "total_gt6"):
            if a[k]:
                c[k] += 1

    print(f"Teeth with any anomaly: {len(anomalies)}")
    print(f"  CEJ > 2:         {c['cej_gt2']}")
    print(f"  Apex > 2:        {c['apex_gt2']}")
    print(f"  Intersection >2: {c['int_gt2']}")
    print(f"  Total > 6:       {c['total_gt6']}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

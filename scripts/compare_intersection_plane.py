"""Compare intersection labels: v5 endpoints vs v5 + vertical midplane filter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import scripts._bootstrap  # noqa: F401

from src.preprocess.prepare_dataset import (
    SPLIT_ALIASES,
    compute_intersections_endpoints,
    dist2,
    load_tooth_mask,
    mask_center,
    pad_keypoints,
)


def side_of_plane(x: float, plane_x: float) -> str:
    return "left" if x < plane_x else "right"


def filter_by_midplane(
    points: list[list[float]],
    plane_x: float,
    max_per_side: int = 1,
) -> list[list[float]]:
    """Keep nearest-to-plane point per side (max 2 total)."""
    buckets: dict[str, list[list[float]]] = {"left": [], "right": []}
    for pt in points:
        buckets[side_of_plane(pt[0], plane_x)].append(pt)
    out: list[list[float]] = []
    for side in ("left", "right"):
        pts = buckets[side]
        if not pts:
            continue
        pts.sort(key=lambda p: abs(p[0] - plane_x))
        out.extend(pts[:max_per_side])
    out.sort(key=lambda p: p[0])
    return out


def intersections_with_plane(
    bboxes: list[list[float]],
    bone_lines: list[list[list[float]]],
    masks: list[np.ndarray | None],
) -> list[list[list[float]]]:
    raw = compute_intersections_endpoints(bboxes, bone_lines)
    filtered: list[list[list[float]]] = []
    for i, pts in enumerate(raw):
        if not pts:
            filtered.append([])
            continue
        c = mask_center(masks[i]) if i < len(masks) else None
        plane_x = c[0] if c else bboxes[i][0] + (bboxes[i][2] - bboxes[i][0]) / 2
        filtered.append(filter_by_midplane(pts, plane_x))
    return filtered


def compare_split(split: str) -> tuple[list[dict], dict]:
    split_raw = {v: k for k, v in SPLIT_ALIASES.items()}[split]
    raw = Path("data/DenPAR/Dataset") / split_raw
    img_dir = Path("data/processed/yolo_detection") / split / "images"
    diffs = []
    stats = {
        "teeth": 0,
        "diff_teeth": 0,
        "only_noplane": 0,
        "only_plane": 0,
        "both_differ": 0,
    }

    for kp_p in sorted((raw / "Key Points Annotations").glob("*.json")):
        stem = kp_p.stem
        bone_p = raw / "Bone Level Annotations" / f"{stem}.json"
        mask_dir = raw / "Masks (Tooth-wise)" / stem
        if not bone_p.exists() or not mask_dir.exists():
            continue
        kp = json.loads(kp_p.read_text(encoding="utf-8"))
        bone = json.loads(bone_p.read_text(encoding="utf-8"))
        bboxes = kp["bboxes"]
        mask_paths = sorted(mask_dir.glob("*.png"))
        if len(mask_paths) != len(bboxes):
            continue
        masks = [load_tooth_mask(p) for p in mask_paths]
        bone_lines = bone.get("Bone_Lines", [])

        raw_int = compute_intersections_endpoints(bboxes, bone_lines)
        plane_int = intersections_with_plane(bboxes, bone_lines, masks)
        for i in range(len(bboxes)):
            stats["teeth"] += 1
            noplane = pad_keypoints(raw_int[i], 2)
            with_plane = pad_keypoints(plane_int[i], 2)
            # ignore padding zeros for comparison
            def visible(pts):
                return [p for p in pts if p != [0.0, 0.0]]

            v0, v1 = visible(noplane), visible(with_plane)
            if len(v0) != len(v1) or any(
                min((dist2(a, b) for b in v1), default=1e9) > 1.0 for a in v0
            ):
                stats["diff_teeth"] += 1
                c = mask_center(masks[i])
                plane_x = c[0] if c else (bboxes[i][0] + bboxes[i][2]) / 2
                diffs.append({
                    "split": split,
                    "image": stem,
                    "tooth": i,
                    "plane_x": plane_x,
                    "raw_count": len(raw_int[i]),
                    "plane_count": len(plane_int[i]),
                    "without_plane": noplane,
                    "with_plane": with_plane,
                    "raw_all": raw_int[i],
                    "plane_all": plane_int[i],
                })

    return diffs, stats


def draw_diff(entry: dict, split: str, out_path: Path):
    split_raw = {v: k for k, v in SPLIT_ALIASES.items()}[split]
    raw = Path("data/DenPAR/Dataset") / split_raw
    stem = entry["image"]
    tooth_i = entry["tooth"]
    img_p = Path("data/processed/yolo_detection") / split / "images" / f"{stem}.jpg"
    if not img_p.exists():
        img_p = raw / "Images" / f"{stem}.jpg"
    img = cv2.imread(str(img_p))
    kp = json.loads((raw / "Key Points Annotations" / f"{stem}.json").read_text())
    bone = json.loads((raw / "Bone Level Annotations" / f"{stem}.json").read_text())
    mask_dir = raw / "Masks (Tooth-wise)" / stem
    mask = load_tooth_mask(sorted(mask_dir.glob("*.png"))[tooth_i])
    bboxes = kp["bboxes"]
    x1, y1, x2, y2 = map(int, bboxes[tooth_i])
    plane_x = int(entry["plane_x"])

    overlay = img.copy()
    if mask is not None:
        tint = np.zeros_like(overlay)
        tint[mask > 0] = (80, 200, 80)
        cv2.addWeighted(tint, 0.3, overlay, 0.7, 0, overlay)
    cv2.line(overlay, (plane_x, y1 - 20), (plane_x, y2 + 20), (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(overlay, "midplane", (plane_x + 4, y1 - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    for line in bone.get("Bone_Lines", []):
        if len(line) >= 2:
            pts = np.array(line, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [pts], False, (0, 200, 0), 1, cv2.LINE_AA)

    for pt in entry["raw_all"]:
        cv2.circle(overlay, (int(pt[0]), int(pt[1])), 6, (180, 180, 180), 2)

    for pt in entry["without_plane"]:
        if pt != [0.0, 0.0]:
            cv2.circle(overlay, (int(pt[0]), int(pt[1])), 9, (0, 140, 255), 2)
            cv2.putText(overlay, "no-plane", (int(pt[0]) + 8, int(pt[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 140, 255), 1)

    for pt in entry["with_plane"]:
        if pt != [0.0, 0.0]:
            cv2.circle(overlay, (int(pt[0]), int(pt[1])), 9, (255, 80, 255), 2)
            cv2.putText(overlay, "plane", (int(pt[0]) + 8, int(pt[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 80, 255), 1)

    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 1)
    title = f"{split}_{stem} T{tooth_i} | raw={entry['raw_count']} plane={entry['plane_count']}"
    cv2.putText(overlay, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(overlay, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, "orange=no-plane kept  magenta=plane kept  gray=all raw endpoints", (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--out-dir", type=Path, default=Path("research_log/figures/intersection_plane_compare"))
    args = p.parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    all_diffs = []
    total_stats = {"teeth": 0, "diff_teeth": 0}
    for split in splits:
        diffs, stats = compare_split(split)
        all_diffs.extend(diffs)
        total_stats["teeth"] += stats["teeth"]
        total_stats["diff_teeth"] += stats["diff_teeth"]
        print(f"{split}: {stats['diff_teeth']}/{stats['teeth']} teeth differ")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "total_teeth": total_stats["teeth"],
        "diff_teeth": total_stats["diff_teeth"],
        "pct": round(100 * total_stats["diff_teeth"] / max(1, total_stats["teeth"]), 2),
        "diffs": [{k: v for k, v in d.items() if k not in ("raw_all", "plane_all")} for d in all_diffs],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for d in tqdm(all_diffs, desc="viz"):
        draw_diff(d, d["split"], args.out_dir / f"{d['split']}_{d['image']}_T{d['tooth']}.jpg")

    print(f"Wrote {len(all_diffs)} diff images -> {args.out_dir}")
    print(f"Overall: {total_stats['diff_teeth']}/{total_stats['teeth']} ({summary['pct']}%)")


if __name__ == "__main__":
    main()

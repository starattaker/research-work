"""Grace-radius sweep until full assignment + bbox-distance outlier analysis.

Extends analyze_grace_radius_sweep.py:
  - CEJ + Apex (+ optional bone-line endpoints as pseudo-points)
  - Sweep 0..max_radius (default 48px) until ~100% assigned
  - Recommend max_grace_px and max_bbox_margin_px for v7 preprocess
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import scripts._bootstrap  # noqa: F401

from scripts.analyze_grace_radius_sweep import (
    discover_denpar_split_dirs,
    eval_grace_v3,
    eval_grace_v4_max_ring,
    find_elbow,
    to_curves,
    true_nearest_tooth,
)
from src.denpar_paths import DEFAULT_DENPAR_ROOT, resolve_denpar_root
from src.preprocess.prepare_dataset import (
    assign_points_to_teeth_mask_region_grow,
    distance_to_mask,
    load_tooth_mask,
    resolve_tooth_masks,
)


def bbox_margin_distance(pt: list[float], bbox: list[float]) -> float:
    """Min distance from point to bbox rectangle (0 if inside)."""
    x, y = pt[0], pt[1]
    x1, y1, x2, y2 = bbox
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return math.hypot(dx, dy)


def collect_bone_line_endpoints(kp_json: dict) -> list[list[float]]:
    pts: list[list[float]] = []
    for line in kp_json.get("Bone_Lines", []) or []:
        if len(line) >= 2:
            pts.append([float(line[0][0]), float(line[0][1])])
            pts.append([float(line[-1][0]), float(line[-1][1])])
    return pts


def sweep_split(
    raw_root: Path,
    split: str,
    max_radius: int,
    include_bone_endpoints: bool,
) -> dict:
    discovered = discover_denpar_split_dirs(raw_root, split)
    if discovered is None:
        raise FileNotFoundError(f"DenPAR split not found: {split} under {raw_root}")
    kp_dir, mask_root = discovered

    point_keys = ("CEJ_Points", "Apex_Points")
    totals_v4 = {
        r: {"assigned": 0, "unassigned": 0, "wrong_tooth": 0, "total_points": 0}
        for r in range(max_radius + 1)
    }
    bbox_margins_at_assign: dict[int, list[float]] = {r: [] for r in range(max_radius + 1)}
    mask_dist_at_assign: dict[int, list[float]] = {r: [] for r in range(max_radius + 1)}

    for kp_path in tqdm(sorted(kp_dir.glob("*.json")), desc=f"assign {split}"):
        kp_json = json.loads(kp_path.read_text(encoding="utf-8"))
        bboxes = kp_json.get("bboxes", [])
        if not bboxes:
            continue
        mask_dir = mask_root / kp_path.stem
        mask_paths = resolve_tooth_masks(mask_dir, len(bboxes))
        if not any(mask_paths):
            continue
        masks = [load_tooth_mask(p) for p in mask_paths]

        all_pts: list[list[float]] = []
        for key in point_keys:
            all_pts.extend(kp_json.get(key, []) or [])
        if include_bone_endpoints:
            all_pts.extend(collect_bone_line_endpoints(kp_json))

        for r in range(max_radius + 1):
            s4 = eval_grace_v4_max_ring(all_pts, bboxes, mask_paths, max_radius_px=r)
            for k in ("assigned", "unassigned", "wrong_tooth", "total_points"):
                totals_v4[r][k] += s4[k]

            # Per-point margins for assigned points at this radius
            assigned_buckets = assign_points_to_teeth_mask_region_grow(
                all_pts, bboxes, mask_paths, max_radius_px=r
            )
            for tooth_i, bucket in enumerate(assigned_buckets):
                bbox = bboxes[tooth_i]
                m = masks[tooth_i]
                for pt in bucket:
                    bbox_margins_at_assign[r].append(bbox_margin_distance(pt, bbox))
                    if m is not None:
                        mask_dist_at_assign[r].append(distance_to_mask(pt[0], pt[1], m))

    return {
        "v4_max_ring": totals_v4,
        "bbox_margins": bbox_margins_at_assign,
        "mask_dists": mask_dist_at_assign,
        "kp_dir": kp_dir.as_posix(),
    }


def first_radius_at_pct(curves: dict, target_pct: float = 99.5) -> int | None:
    for r, pct in zip(curves["radii_px"], curves["pct_assigned"]):
        if pct >= target_pct:
            return int(r)
    return None


def recommend_bbox_cutoff(margins: list[float], wrong_rate_target: float = 0.02) -> dict:
    if not margins:
        return {}
    arr = np.array(margins, dtype=np.float64)
    pctiles = {f"p{p}": float(np.percentile(arr, p)) for p in (90, 95, 99, 99.5)}
    # Conservative: p99 of margin among assigned — points farther are likely cross-tooth
    return {
        "n": int(len(arr)),
        "percentiles_px": pctiles,
        "suggested_max_bbox_margin_px": round(pctiles["p99"], 1),
        "note": "Drop assigned points with bbox_margin > suggested value before training v7",
    }


def plot_full(v4: dict, margins_by_r: dict, out_path: Path, max_radius: int, markers: dict):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(v4["radii_px"], v4["pct_assigned"], color="#10b981", linewidth=2, label="v4 assigned %")
    axes[0, 0].axhline(100, color="gray", linestyle=":", alpha=0.6)
    for label, px, color in (
        ("v4 default 8px", 8, "red"),
        ("v3 default 4px", 4, "orange"),
        ("100% radius", markers.get("full_assign_px"), "purple"),
    ):
        if px is not None:
            axes[0, 0].axvline(px, color=color, linestyle="--", alpha=0.85, label=label)
    axes[0, 0].set_xlabel("Max region-grow radius (px)")
    axes[0, 0].set_ylabel("% points assigned")
    axes[0, 0].set_title("Assignment yield (grow 1px rings until max)")
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(v4["radii_px"], v4["pct_wrong_tooth"], color="#ef4444", linewidth=2)
    axes[0, 1].set_xlabel("Radius (px)")
    axes[0, 1].set_ylabel("% assigned → wrong tooth")
    axes[0, 1].set_title("Cross-tooth contamination (nearest-mask oracle)")
    axes[0, 1].grid(True, alpha=0.3)

    # Bbox margin distribution at recommended radius
    rec_r = markers.get("recommended_grace_px", 8)
    margs = margins_by_r.get(rec_r, [])
    if margs:
        axes[1, 0].hist(margs, bins=50, color="#6366f1", edgecolor="white", alpha=0.9)
        cut = markers.get("bbox_cutoff_px")
        if cut:
            axes[1, 0].axvline(cut, color="red", linestyle="--", label=f"cutoff {cut}px")
        axes[1, 0].set_xlabel("Distance to assigned tooth bbox (px)")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].set_title(f"Bbox margin at radius={rec_r}px (assigned points)")
        axes[1, 0].legend(fontsize=8)
    else:
        axes[1, 0].text(0.5, 0.5, "No assigned points", ha="center", va="center")

    axes[1, 1].plot(v4["radii_px"], v4["wrong_absolute"], color="#f59e0b", linewidth=2)
    axes[1, 1].set_xlabel("Radius (px)")
    axes[1, 1].set_ylabel("Wrong-tooth count")
    axes[1, 1].set_title("Absolute mis-assignments vs radius")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"Point assignment sweep 0–{max_radius}px (CEJ + Apex + bone endpoints)", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Full point-assignment sweep + outlier guidance")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_DENPAR_ROOT)
    parser.add_argument("--split", default="all", help="Training | Validation | Testing | all")
    parser.add_argument("--max-radius", type=int, default=48, help="Sweep 0..N px (was 24 in earlier run)")
    parser.add_argument("--include-bone-endpoints", action="store_true", default=True)
    parser.add_argument("--out-dir", type=Path, default=Path("research_log/figures/point_assignment_full"))
    args = parser.parse_args()

    splits = ("Training", "Validation", "Testing") if args.split == "all" else (args.split,)
    merged_v4 = {r: {"assigned": 0, "unassigned": 0, "wrong_tooth": 0, "total_points": 0} for r in range(args.max_radius + 1)}
    merged_margins: dict[int, list[float]] = {r: [] for r in range(args.max_radius + 1)}

    for sp in splits:
        raw = sweep_split(resolve_denpar_root(args.raw_root), sp, args.max_radius, args.include_bone_endpoints)
        for r in range(args.max_radius + 1):
            for k in merged_v4[r]:
                merged_v4[r][k] += raw["v4_max_ring"][r][k]
            merged_margins[r].extend(raw["bbox_margins"][r])

    v4_curves = to_curves(merged_v4)
    elbow = find_elbow(v4_curves["radii_px"], v4_curves["pct_assigned"])
    full_assign = first_radius_at_pct(v4_curves, 99.9)
    at_8 = {k: v4_curves[k][8] for k in ("pct_assigned", "pct_unassigned", "pct_wrong_tooth")} if len(v4_curves["radii_px"]) > 8 else {}
    at_24 = {k: v4_curves[k][24] for k in ("pct_assigned", "pct_unassigned", "pct_wrong_tooth")} if len(v4_curves["radii_px"]) > 24 else {}

    # Pick grace: elbow, or 12 if 8 leaves >2% unassigned and wrong rate acceptable
    rec_grace = elbow or 8
    if at_8 and at_8["pct_assigned"] < 98.0 and len(v4_curves["radii_px"]) > 12:
        if v4_curves["pct_wrong_tooth"][12] < 3.0:
            rec_grace = 12

    bbox_rec = recommend_bbox_cutoff(merged_margins.get(rec_grace, []))
    markers = {
        "full_assign_px": full_assign,
        "recommended_grace_px": rec_grace,
        "bbox_cutoff_px": bbox_rec.get("suggested_max_bbox_margin_px"),
    }
    plot_full(v4_curves, merged_margins, args.out_dir / "point_assignment_full.png", args.max_radius, markers)

    report = {
        "raw_root": str(args.raw_root),
        "splits": list(splits),
        "max_radius_px": args.max_radius,
        "include_bone_endpoints": args.include_bone_endpoints,
        "at_8px_v4": at_8,
        "at_24px_v4": at_24,
        "full_assignment_radius_px": full_assign,
        "elbow_px": elbow,
        "recommended_for_v7_preprocess": {
            "max_grace_px": rec_grace,
            "bbox_outlier_margin_px": bbox_rec.get("suggested_max_bbox_margin_px", 24.0),
            "strategy": "v6 region-grow + drop points with bbox_margin > cutoff after assign",
        },
        "bbox_margin_at_recommended_grace": bbox_rec,
        "curves_v4": v4_curves,
        "interpretation": {
            "graph_location": str(args.out_dir / "point_assignment_full.png"),
            "prior_24px_run": "scripts/analyze_grace_radius_sweep.py --max-radius 24 → research_log/figures/grace_radius_sweep/",
            "why_not_grow_forever": (
                "pct_assigned→100% but wrong_tooth rises; use recommended grace + bbox outlier filter"
            ),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "point_assignment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Training script reads this path
    train_cfg = Path("research_log/point_assignment_report.json")
    train_cfg.write_text(json.dumps(report["recommended_for_v7_preprocess"], indent=2), encoding="utf-8")

    print("=" * 60)
    print("POINT ASSIGNMENT FULL SWEEP")
    print("=" * 60)
    if at_8:
        print(f"  At 8px:  assigned={at_8['pct_assigned']:.1f}%  wrong={at_8['pct_wrong_tooth']:.2f}%")
    if at_24:
        print(f"  At 24px: assigned={at_24['pct_assigned']:.1f}%  wrong={at_24['pct_wrong_tooth']:.2f}%")
    print(f"  100% assigned at: {full_assign}px")
    print(f"  Recommended max_grace_px: {rec_grace}")
    print(f"  Recommended bbox outlier cutoff: {bbox_rec.get('suggested_max_bbox_margin_px')} px")
    print(f"  Figure: {args.out_dir / 'point_assignment_full.png'}")
    print(f"  Report: {args.out_dir / 'point_assignment_report.json'}")
    print(f"  Training config: {train_cfg}")


if __name__ == "__main__":
    main()

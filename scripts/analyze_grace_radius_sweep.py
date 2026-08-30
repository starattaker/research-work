"""Sweep mask grace radius: assignment yield vs cross-tooth contamination.

Answers: why 8px (v4 max_grace)? What if we grow until every point is assigned?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import scripts._bootstrap  # noqa: F401

from src.preprocess.prepare_dataset import (
    SPLITS,
    SPLIT_ALIASES,
    assign_points_to_teeth_mask,
    assign_points_to_teeth_mask_region_grow,
    distance_to_mask,
    load_tooth_mask,
    resolve_tooth_masks,
    tooth_anchor_center,
)

POINT_TYPES = ("CEJ_Points", "Apex_Points")


def true_nearest_tooth(pt: list[float], masks: list, bboxes: list) -> int | None:
    """Tooth index with smallest mask distance (oracle for correctness)."""
    best_i = None
    best_d = float("inf")
    for i, m in enumerate(masks):
        if m is None:
            continue
        d = distance_to_mask(pt[0], pt[1], m)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _pt_eq(a: list[float], b: list[float]) -> bool:
    return abs(a[0] - b[0]) < 1e-6 and abs(a[1] - b[1]) < 1e-6


def _bucket_index(pt: list[float], assigned: list[list[list[float]]]) -> int | None:
    for i, bucket in enumerate(assigned):
        for p in bucket:
            if _pt_eq(p, pt):
                return i
    return None


def eval_grace_v3(
    points: list[list[float]],
    bboxes: list,
    mask_paths: list[Path],
    grace_px: float,
) -> dict:
    masks = [load_tooth_mask(p) for p in mask_paths]
    assigned = assign_points_to_teeth_mask(points, bboxes, mask_paths, grace_px=grace_px)
    assigned_n = sum(len(a) for a in assigned)
    wrong = 0
    unassigned = 0
    for pt in points:
        truth = true_nearest_tooth(pt, masks, bboxes)
        if truth is None:
            continue
        found_i = _bucket_index(pt, assigned)
        if found_i is not None:
            if found_i != truth:
                wrong += 1
        else:
            unassigned += 1
    return {
        "assigned": assigned_n,
        "unassigned": unassigned,
        "wrong_tooth": wrong,
        "total_points": len(points),
    }


def eval_grace_v4_max_ring(
    points: list[list[float]],
    bboxes: list,
    mask_paths: list[Path],
    max_radius_px: int,
    step_px: int = 1,
) -> dict:
    masks = [load_tooth_mask(p) for p in mask_paths]
    assigned = assign_points_to_teeth_mask_region_grow(
        points, bboxes, mask_paths, step_px=step_px, max_radius_px=max_radius_px
    )
    assigned_n = sum(len(a) for a in assigned)
    wrong = 0
    unassigned = 0
    for pt in points:
        truth = true_nearest_tooth(pt, masks, bboxes)
        if truth is None:
            continue
        found_i = _bucket_index(pt, assigned)
        if found_i is not None:
            if found_i != truth:
                wrong += 1
        else:
            unassigned += 1
    return {
        "assigned": assigned_n,
        "unassigned": unassigned,
        "wrong_tooth": wrong,
        "total_points": len(points),
    }


def discover_denpar_split_dirs(raw_root: Path, split: str) -> tuple[Path, Path] | None:
    """Find Key Points Annotations + Masks folders (DenPAR naming variants)."""
    split_raw = split if split in SPLITS else SPLIT_ALIASES.get(split, split)
    if split_raw not in SPLITS:
        alias_inv = {v: k for k, v in SPLIT_ALIASES.items()}
        split_raw = alias_inv.get(split, split)

    bases = [raw_root, raw_root / "DenPAR", raw_root / "denpar"]
    split_names = [split_raw, split_raw.lower(), split.lower()]
    for base in bases:
        for name in split_names:
            kp_dir = base / name / "Key Points Annotations"
            mask_root = base / name / "Masks (Tooth-wise)"
            if kp_dir.is_dir() and any(kp_dir.glob("*.json")):
                return kp_dir, mask_root
    return None


def sweep_raw_split(raw_root: Path, split: str, max_radius: int) -> dict:
    discovered = discover_denpar_split_dirs(raw_root, split)
    if discovered is None:
        raise FileNotFoundError(
            f"DenPAR not found under {raw_root}. "
            "Set RAW_ROOT to the folder containing Training/, Validation/, Testing/. "
            "Example: export RAW_ROOT=~/data/DenPAR_dataset"
        )
    kp_dir, mask_root = discovered

    totals_v3 = {r: {"assigned": 0, "unassigned": 0, "wrong_tooth": 0, "total_points": 0} for r in range(max_radius + 1)}
    totals_v4 = {r: {"assigned": 0, "unassigned": 0, "wrong_tooth": 0, "total_points": 0} for r in range(max_radius + 1)}

    for kp_path in tqdm(sorted(kp_dir.glob("*.json")), desc=f"sweep {kp_dir.parent.name}"):
        kp_json = json.loads(kp_path.read_text(encoding="utf-8"))
        bboxes = kp_json.get("bboxes", [])
        if not bboxes:
            continue
        mask_dir = mask_root / kp_path.stem
        mask_paths = resolve_tooth_masks(mask_dir, len(bboxes))
        if not any(mask_paths):
            continue

        for key in POINT_TYPES:
            pts = kp_json.get(key, [])
            if not pts:
                continue
            for r in range(max_radius + 1):
                s3 = eval_grace_v3(pts, bboxes, mask_paths, grace_px=float(r))
                s4 = eval_grace_v4_max_ring(pts, bboxes, mask_paths, max_radius_px=r)
                for bucket, s in ((totals_v3, s3), (totals_v4, s4)):
                    for k in ("assigned", "unassigned", "wrong_tooth", "total_points"):
                        bucket[r][k] += s[k]

    return {"v3_grace": totals_v3, "v4_max_ring": totals_v4, "kp_dir": kp_dir.as_posix()}


def to_curves(totals: dict) -> dict:
    radii = sorted(totals.keys())
    total_pts = max(totals[radii[0]]["total_points"], 1)
    return {
        "radii_px": radii,
        "pct_assigned": [100.0 * totals[r]["assigned"] / total_pts for r in radii],
        "pct_unassigned": [100.0 * totals[r]["unassigned"] / total_pts for r in radii],
        "pct_wrong_tooth": [
            100.0 * totals[r]["wrong_tooth"] / max(totals[r]["assigned"], 1) for r in radii
        ],
        "wrong_absolute": [totals[r]["wrong_tooth"] for r in radii],
    }


def find_elbow(radii: list[int], pct_assigned: list[float], threshold: float = 0.5) -> int | None:
    """Smallest radius where assigned gain per +1px drops below threshold."""
    if len(radii) < 3:
        return None
    for i in range(1, len(radii)):
        gain = pct_assigned[i] - pct_assigned[i - 1]
        if gain < threshold and pct_assigned[i] >= 95.0:
            return radii[i]
    return None


def plot_curves(v3: dict, v4: dict, out_path: Path, max_radius: int):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for curves, label, color in ((v3, "v3 grace", "#3b82f6"), (v4, "v4 max ring", "#10b981")):
        axes[0, 0].plot(curves["radii_px"], curves["pct_assigned"], label=label, color=color, linewidth=2)
    axes[0, 0].axvline(4, color="orange", linestyle="--", alpha=0.8, label="v3 default 4px")
    axes[0, 0].axvline(8, color="red", linestyle="--", alpha=0.8, label="v4 default 8px")
    axes[0, 0].set_xlabel("Radius (px)")
    axes[0, 0].set_ylabel("% points assigned")
    axes[0, 0].set_title("Assignment yield vs grace radius")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    for curves, label, color in ((v3, "v3", "#3b82f6"), (v4, "v4", "#10b981")):
        axes[0, 1].plot(curves["radii_px"], curves["pct_unassigned"], label=label, color=color, linewidth=2)
    axes[0, 1].set_xlabel("Radius (px)")
    axes[0, 1].set_ylabel("% still unassigned")
    axes[0, 1].set_title("Dropped points vs radius")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    for curves, label, color in ((v3, "v3", "#3b82f6"), (v4, "v4", "#10b981")):
        axes[1, 0].plot(curves["radii_px"], curves["pct_wrong_tooth"], label=label, color=color, linewidth=2)
    axes[1, 0].set_xlabel("Radius (px)")
    axes[1, 0].set_ylabel("% of assigned → wrong tooth")
    axes[1, 0].set_title("Cross-tooth contamination (nearest-mask oracle)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(v3["radii_px"], v3["wrong_absolute"], label="v3 wrong count", color="#3b82f6")
    axes[1, 1].plot(v4["radii_px"], v4["wrong_absolute"], label="v4 wrong count", color="#10b981")
    axes[1, 1].axvline(8, color="red", linestyle="--", alpha=0.8)
    axes[1, 1].set_xlabel("Radius (px)")
    axes[1, 1].set_ylabel("Wrong-tooth assignments (count)")
    axes[1, 1].set_title("Absolute mis-assignments if you keep growing grace")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"Grace radius sweep (0–{max_radius}px) — CEJ + Apex points", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Grace radius sweep for preprocessing justification")
    parser.add_argument("--raw-root", type=Path, required=True, help="DenPAR raw root")
    parser.add_argument("--split", default="Testing", help="Training | Validation | Testing")
    parser.add_argument("--max-radius", type=int, default=24, help="Max px to sweep (avoid neighbor bleed)")
    parser.add_argument("--out-dir", type=Path, default=Path("research_log/figures/grace_radius_sweep"))
    args = parser.parse_args()

    raw = sweep_raw_split(args.raw_root, args.split, args.max_radius)
    if raw["v3_grace"][0]["total_points"] == 0:
        raise RuntimeError(f"No points processed from {raw['kp_dir']} — check masks exist per image.")

    v3_curves = to_curves(raw["v3_grace"])
    v4_curves = to_curves(raw["v4_max_ring"])

    elbow_v3 = find_elbow(v3_curves["radii_px"], v3_curves["pct_assigned"])
    elbow_v4 = find_elbow(v4_curves["radii_px"], v4_curves["pct_assigned"])

    report = {
        "raw_root": args.raw_root.as_posix(),
        "split": args.split,
        "max_radius_px": args.max_radius,
        "kp_dir": raw["kp_dir"],
        "at_4px_v3": {k: v3_curves[k][4] for k in ("pct_assigned", "pct_unassigned", "pct_wrong_tooth")},
        "at_8px_v3": {k: v3_curves[k][8] for k in ("pct_assigned", "pct_unassigned", "pct_wrong_tooth")},
        "at_8px_v4": {k: v4_curves[k][8] for k in ("pct_assigned", "pct_unassigned", "pct_wrong_tooth")},
        "elbow_assigned_pct": {"v3_px": elbow_v3, "v4_px": elbow_v4},
        "interpretation": {
            "why_8px": (
                "v4 uses outward rings 0–8px: captures CEJ/apex just outside mask boundary "
                "without growing so far that neighboring tooth masks steal points."
            ),
            "if_grow_until_all_assigned": (
                "pct_assigned → 100% but pct_wrong_tooth rises — labels get noisier; "
                "see bottom-right plot for wrong-tooth count vs radius."
            ),
        },
        "curves": {"v3": v3_curves, "v4": v4_curves},
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_curves(v3_curves, v4_curves, args.out_dir / "grace_radius_sweep.png", args.max_radius)
    (args.out_dir / "grace_radius_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 60)
    print("GRACE RADIUS SWEEP")
    print("=" * 60)
    print(f"  At 4px (v3 default): assigned={report['at_4px_v3']['pct_assigned']:.1f}%  wrong={report['at_4px_v3']['pct_wrong_tooth']:.2f}%")
    print(f"  At 8px (v4 default): assigned={report['at_8px_v4']['pct_assigned']:.1f}%  wrong={report['at_8px_v4']['pct_wrong_tooth']:.2f}%")
    print(f"  Figure: {args.out_dir / 'grace_radius_sweep.png'}")


if __name__ == "__main__":
    main()

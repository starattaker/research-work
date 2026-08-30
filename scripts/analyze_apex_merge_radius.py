"""Distribution of apex–apex distance on double-root teeth (GT v6 labels).

Use output to choose merge_radius_px for shared apex at inference.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import scripts._bootstrap  # noqa: F401

from src.severity.inference_pipeline import load_gt_annotations, point_from_slot

SPLITS = ("train", "val", "test")


def collect_apex_distances(data_root: Path, split: str) -> list[dict]:
    ann_dir = data_root / "keypoints" / "apex" / split / "annotations"
    rows: list[dict] = []
    for ann_path in sorted(ann_dir.glob("*.json")):
        merged = load_gt_annotations(data_root, split, ann_path.stem)
        if merged is None:
            continue
        for i, label in enumerate(merged["labels"]):
            if int(label) != 2:
                continue
            a0 = point_from_slot(merged["apex"][i], 0)
            a1 = point_from_slot(merged["apex"][i], 1)
            if a0 == (0.0, 0.0) or a1 == (0.0, 0.0):
                continue
            d = math.hypot(a0[0] - a1[0], a0[1] - a1[1])
            rows.append(
                {
                    "stem": ann_path.stem,
                    "tooth_idx": i,
                    "distance_px": d,
                    "apex0": a0,
                    "apex1": a1,
                }
            )
    return rows


def summarize(distances: list[float]) -> dict:
    if not distances:
        return {}
    arr = np.array(distances, dtype=np.float64)
    pct = [5, 10, 25, 50, 75, 90, 95, 99]
    return {
        "n": int(len(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "percentiles": {f"p{p}": float(np.percentile(arr, p)) for p in pct},
    }


def suggest_merge_range(stats: dict) -> dict:
    """Conservative merge: below p10; safe separate: above p5."""
    if not stats:
        return {}
    p5 = stats["percentiles"]["p5"]
    p10 = stats["percentiles"]["p10"]
    p25 = stats["percentiles"]["p25"]
    return {
        "merge_below_px": f"≤ {p10:.1f} (p10) — likely same root / duplicate preds",
        "separate_above_px": f"≥ {p25:.1f} (p25) — clearly two roots",
        "recommended_search_range": [round(p5, 1), round(p25, 1)],
        "default_12px_note": "Compare to fixed 12px default in slot_matching.py",
    }


def plot_histogram(distances: list[float], out_path: Path, stats: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(distances, bins=40, color="#3b82f6", edgecolor="white", alpha=0.9)
    axes[0].axvline(12, color="red", linestyle="--", label="default 12px")
    if stats:
        axes[0].axvline(stats["percentiles"]["p10"], color="orange", linestyle=":", label="p10")
        axes[0].axvline(stats["percentiles"]["p25"], color="green", linestyle=":", label="p25")
    axes[0].set_xlabel("Apex–apex distance (px)")
    axes[0].set_ylabel("Count (double-root teeth)")
    axes[0].set_title("GT v6: distance between two apex slots")
    axes[0].legend(fontsize=8)

    sorted_d = np.sort(distances)
    cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    axes[1].plot(sorted_d, cdf, color="#2563eb", linewidth=2)
    axes[1].axvline(12, color="red", linestyle="--", label="12px")
    if stats:
        for p, c in (("p10", "orange"), ("p25", "green")):
            v = stats["percentiles"][p]
            axes[1].axvline(v, color=c, linestyle=":", label=p)
    axes[1].set_xlabel("Distance (px)")
    axes[1].set_ylabel("CDF")
    axes[1].set_title("Cumulative distribution")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Apex merge radius analysis from GT double-root teeth")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed_v6"))
    parser.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    parser.add_argument("--out-dir", type=Path, default=Path("research_log/figures/apex_merge_analysis"))
    args = parser.parse_args()

    splits = SPLITS if args.split == "all" else (args.split,)
    all_rows: list[dict] = []
    for sp in splits:
        all_rows.extend(collect_apex_distances(args.data_root, sp))

    distances = [r["distance_px"] for r in all_rows]
    stats = summarize(distances)
    suggestion = suggest_merge_range(stats)

    report = {
        "data_root": args.data_root.as_posix(),
        "splits": list(splits),
        "double_root_teeth_with_two_apex": stats,
        "merge_radius_guidance": suggestion,
        "samples": all_rows[:20],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_histogram(distances, args.out_dir / "apex_distance_hist_cdf.png", stats)
    (args.out_dir / "apex_distance_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("=" * 60)
    print("APEX MERGE RADIUS ANALYSIS (GT double-root, 2 apex slots)")
    print("=" * 60)
    if not stats:
        print("No double-root teeth with two valid apex slots found.")
        return
    print(f"  n = {stats['n']}")
    print(f"  range: {stats['min']:.1f} – {stats['max']:.1f} px")
    print(f"  mean ± std: {stats['mean']:.1f} ± {stats['std']:.1f} px")
    for p, v in stats["percentiles"].items():
        print(f"  {p}: {v:.1f} px")
    print()
    print("Suggested merge radius band:")
    for k, v in suggestion.items():
        print(f"  {k}: {v}")
    print(f"\nFigure: {args.out_dir / 'apex_distance_hist_cdf.png'}")
    print(f"Report: {args.out_dir / 'apex_distance_report.json'}")


if __name__ == "__main__":
    main()

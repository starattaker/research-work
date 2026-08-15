"""Compare preprocessing strategies v1 / v2 / v3 on DenPAR (no full export required)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import scripts._bootstrap  # noqa: F401

from tqdm import tqdm

from src.preprocess.prepare_dataset import SPLITS, build_tooth_records


def count_records(records) -> dict:
    teeth = len(records)
    single = sum(1 for r in records if r.label == 1)
    double = teeth - single
    apex_counts = {"0": 0, "1": 0, "2plus": 0}
    for r in records:
        n = sum(1 for p in r.apex if p != [0.0, 0.0])
        if n == 0:
            apex_counts["0"] += 1
        elif n == 1:
            apex_counts["1"] += 1
        else:
            apex_counts["2plus"] += 1
    return {
        "teeth": teeth,
        "single": single,
        "double": double,
        "zero_apex_teeth": apex_counts["0"],
        "one_apex_teeth": apex_counts["1"],
        "two_plus_apex_teeth": apex_counts["2plus"],
        "pct_zero_apex": round(100.0 * apex_counts["0"] / max(teeth, 1), 1),
    }


def aggregate(stats_list: list[dict]) -> dict:
    total = {"teeth": 0, "single": 0, "double": 0, "zero_apex_teeth": 0, "one_apex_teeth": 0, "two_plus_apex_teeth": 0}
    for s in stats_list:
        for k in total:
            total[k] += s[k]
    total["pct_zero_apex"] = round(100.0 * total["zero_apex_teeth"] / max(total["teeth"], 1), 1)
    return total


def run_strategy(raw_root: Path, strategy: str, grace_px: float) -> dict:
    per_split = {}
    all_stats = []
    for split in SPLITS:
        kp_dir = raw_root / split / "Key Points Annotations"
        if not kp_dir.exists():
            raise FileNotFoundError(
                f"Missing DenPAR annotations: {kp_dir}\n"
                f"Pass --raw-root pointing to DenPAR/Dataset (folder with Training/, Validation/, Testing/)."
            )
        bone_dir = raw_root / split / "Bone Level Annotations"
        mask_root = raw_root / split / "Masks (Tooth-wise)"
        split_stats = []
        for kp_path in tqdm(sorted(kp_dir.glob("*.json")), desc=f"{strategy} {split}"):
            kp_json = json.loads(kp_path.read_text(encoding="utf-8"))
            bone_path = bone_dir / kp_path.name
            bone_json = json.loads(bone_path.read_text(encoding="utf-8")) if bone_path.exists() else None
            records = build_tooth_records(
                kp_json, bone_json, mask_root / kp_path.stem, strategy=strategy, grace_px=grace_px
            )
            if records:
                split_stats.append(count_records(records))
        per_split[split] = aggregate(split_stats) if split_stats else {}
        if split_stats:
            all_stats.extend(split_stats)
    return {"total": aggregate(all_stats), "splits": per_split}


def markdown_table(rows: list[tuple]) -> str:
    lines = ["| Strategy | Teeth | Single | Double | 0 apex | 1 apex | >=2 apex | % 0 apex |",
             "|----------|------:|-------:|-------:|-------:|-------:|--------:|---------:|"]
    for row in rows:
        lines.append(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} | {row[7]} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/DenPAR/Dataset"))
    parser.add_argument("--grace-px", type=float, default=4.0)
    parser.add_argument("--out-json", type=Path, default=Path("research_log/preprocessing_comparison.json"))
    parser.add_argument("--out-md", type=Path, default=Path("research_log/preprocessing_comparison.md"))
    args = parser.parse_args()

    if not args.raw_root.exists():
        print(f"ERROR: --raw-root not found: {args.raw_root}", file=sys.stderr)
        print("Example: --raw-root data/DenPAR/Dataset", file=sys.stderr)
        sys.exit(1)

    sample = args.raw_root / "Training" / "Key Points Annotations"
    if not sample.exists():
        print(f"ERROR: DenPAR layout not found under {args.raw_root}", file=sys.stderr)
        print("Expected: Training/Key Points Annotations/*.json", file=sys.stderr)
        sys.exit(1)

    results = {}
    for strategy in ("v1", "v2", "v3"):
        results[strategy] = run_strategy(args.raw_root, strategy, args.grace_px)

    rows = []
    for name, label in [("v1", "v1 (8px margin)"), ("v2", "v2 (strict bbox)"), ("v3", "v3 (mask+grace)")]:
        t = results[name]["total"]
        rows.append(
            (
                label,
                t["teeth"],
                t["single"],
                t["double"],
                t["zero_apex_teeth"],
                t["one_apex_teeth"],
                t["two_plus_apex_teeth"],
                f"{t['pct_zero_apex']}%",
            )
        )

    table = markdown_table(rows)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md = f"""# Preprocessing comparison (v1 / v2 / v3)

Generated by `scripts/compare_preprocessing.py`.

**BBox note:** All strategies use the same DenPAR tooth **bounding boxes** ({results['v2']['total']['teeth']} teeth). Strategies differ in **apex/CEJ assignment** and intersection fallback, which changes inferred single/double labels and 0-apex counts.

{table}

## Strategy summary

| Version | Apex/CEJ assignment | Intersection fallback |
|---------|---------------------|------------------------|
| v1 | Point in bbox + **8 px margin** | Line × contour, else nearest to **midpoint** |
| v2 | **Strict** DenPAR bbox only | Line × contour, else **endpoint extension** |
| v3 | **Tooth mask** + nearest within **{args.grace_px}px** grace | Same as v2 |

## Keypoint training note

Keypoint R-CNN is trained on **GT bboxes from JSON**, not YOLO. Many red boxes in visual QA are **unfiltered model proposals** (apply score + NMS 0.6 for clean viz).
"""
    args.out_md.write_text(md, encoding="utf-8")
    print(table)
    print(f"\nWrote {args.out_json}\nWrote {args.out_md}")


if __name__ == "__main__":
    main()

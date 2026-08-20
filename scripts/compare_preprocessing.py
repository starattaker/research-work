"""Compare preprocessing strategies v1 / v2 / v3 / v4 on DenPAR (no full export required)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import scripts._bootstrap  # noqa: F401

from tqdm import tqdm

from src.preprocess.prepare_dataset import SPLITS, build_tooth_records


def count_visible_kpts(records, field: str) -> int:
    total = 0
    for r in records:
        pts = getattr(r, field)
        total += sum(1 for p in pts if p != [0.0, 0.0])
    return total


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
        "cej_assigned": count_visible_kpts(records, "cej"),
        "intersection_assigned": count_visible_kpts(records, "intersection"),
        "apex_assigned": count_visible_kpts(records, "apex"),
    }


def aggregate(stats_list: list[dict]) -> dict:
    total = {
        "teeth": 0,
        "single": 0,
        "double": 0,
        "zero_apex_teeth": 0,
        "one_apex_teeth": 0,
        "two_plus_apex_teeth": 0,
        "cej_assigned": 0,
        "intersection_assigned": 0,
        "apex_assigned": 0,
        "raw_cej": 0,
        "raw_apex": 0,
    }
    for s in stats_list:
        for k in total:
            total[k] += s.get(k, 0)
    total["pct_zero_apex"] = round(100.0 * total["zero_apex_teeth"] / max(total["teeth"], 1), 1)
    total["cej_drop_pct"] = round(
        100.0 * max(total["raw_cej"] - total["cej_assigned"], 0) / max(total["raw_cej"], 1), 1
    )
    total["apex_drop_pct"] = round(
        100.0 * max(total["raw_apex"] - total["apex_assigned"], 0) / max(total["raw_apex"], 1), 1
    )
    return total


def run_strategy(
    raw_root: Path,
    strategy: str,
    grace_px: float,
    grace_step_px: int,
    max_grace_px: int,
) -> dict:
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
                kp_json,
                bone_json,
                mask_root / kp_path.stem,
                strategy=strategy,
                grace_px=grace_px,
                grace_step_px=grace_step_px,
                max_grace_px=max_grace_px,
            )
            if records:
                row = count_records(records)
                row["raw_cej"] = len(kp_json.get("CEJ_Points", []))
                row["raw_apex"] = len(kp_json.get("Apex_Points", []))
                split_stats.append(row)
        per_split[split] = aggregate(split_stats) if split_stats else {}
        if split_stats:
            all_stats.extend(split_stats)
    return {"total": aggregate(all_stats), "splits": per_split}


def markdown_table(rows: list[tuple]) -> str:
    lines = [
        "| Strategy | Teeth | Single | Double | 0 apex | 1 apex | >=2 apex | % 0 apex |",
        "|----------|------:|-------:|-------:|-------:|-------:|--------:|---------:|",
    ]
    for row in rows:
        lines.append(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} | {row[7]} |"
        )
    return "\n".join(lines)


def point_table(rows: list[tuple]) -> str:
    lines = [
        "| Strategy | Raw CEJ | CEJ kept | CEJ drop % | Raw apex | Apex kept | Apex drop % |",
        "|----------|--------:|---------:|-----------:|---------:|----------:|------------:|",
    ]
    for row in rows:
        lines.append(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} |"
        )
    return "\n".join(lines)


def chi2_independence(a_single: int, a_double: int, b_single: int, b_double: int) -> tuple[float, float] | None:
    try:
        from scipy.stats import chi2_contingency
    except ImportError:
        return None
    table = [[a_single, a_double], [b_single, b_double]]
    chi2, p, _, _ = chi2_contingency(table)
    return float(chi2), float(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/DenPAR/Dataset"))
    parser.add_argument("--grace-px", type=float, default=4.0)
    parser.add_argument("--grace-step-px", type=int, default=1)
    parser.add_argument("--max-grace-px", type=int, default=8)
    parser.add_argument("--out-json", type=Path, default=Path("research_log/preprocessing_comparison.json"))
    parser.add_argument("--out-md", type=Path, default=Path("research_log/preprocessing_comparison.md"))
    args = parser.parse_args()

    if not args.raw_root.exists():
        print(f"ERROR: --raw-root not found: {args.raw_root}", file=sys.stderr)
        sys.exit(1)

    sample = args.raw_root / "Training" / "Key Points Annotations"
    if not sample.exists():
        print(f"ERROR: DenPAR layout not found under {args.raw_root}", file=sys.stderr)
        sys.exit(1)

    configs = [
        ("v1", "v1 (8px margin)", {}),
        ("v2", "v2 (strict bbox)", {}),
        ("v3", f"v3 (mask + {args.grace_px}px grace)", {}),
        (
            "v4",
            f"v4 (region grow 1-{args.max_grace_px}px rings)",
            {"grace_step_px": args.grace_step_px, "max_grace_px": args.max_grace_px},
        ),
    ]

    results = {}
    for strategy, _, extra in configs:
        results[strategy] = run_strategy(
            args.raw_root,
            strategy,
            args.grace_px,
            extra.get("grace_step_px", args.grace_step_px),
            extra.get("max_grace_px", args.max_grace_px),
        )

    rows = []
    pt_rows = []
    for strategy, label, _ in configs:
        t = results[strategy]["total"]
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
        pt_rows.append(
            (
                label,
                t["raw_cej"],
                t["cej_assigned"],
                f"{t['cej_drop_pct']}%",
                t["raw_apex"],
                t["apex_assigned"],
                f"{t['apex_drop_pct']}%",
            )
        )

    table = markdown_table(rows)
    pt_tbl = point_table(pt_rows)

    stats_lines = ["## Single vs double (vs v2 baseline)", ""]
    v2 = results["v2"]["total"]
    for strategy, label, _ in configs:
        if strategy == "v2":
            continue
        t = results[strategy]["total"]
        test = chi2_independence(t["single"], t["double"], v2["single"], v2["double"])
        if test:
            chi2, p = test
            stats_lines.append(
                f"- **{label}** vs v2: chi2={chi2:.2f}, p={p:.4g} "
                f"(single/double {t['single']}/{t['double']} vs {v2['single']}/{v2['double']})"
            )
        else:
            stats_lines.append(f"- **{label}** vs v2: install scipy for chi-square test")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "grace_px": args.grace_px,
        "grace_step_px": args.grace_step_px,
        "max_grace_px": args.max_grace_px,
        "strategies": results,
    }
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = f"""# Preprocessing comparison (v1 / v2 / v3 / v4)

Generated by `scripts/compare_preprocessing.py`.

**BBox note:** All strategies use the same DenPAR tooth **bounding boxes** ({results['v2']['total']['teeth']} teeth). Strategies differ in **apex/CEJ assignment** and intersection fallback.

## Tooth / root label counts

{table}

## Expert click retention (CEJ & apex)

Raw = clicks in DenPAR JSON. Kept = clicks assigned to a tooth and exported (may exceed raw when overlap rules duplicate assignment is prevented — here counts visible keypoints on teeth).

{pt_tbl}

{chr(10).join(stats_lines)}

## Strategy summary

| Version | Apex/CEJ assignment | Intersection fallback |
|---------|---------------------|------------------------|
| v1 | Point in bbox + **8 px margin** | Line x contour, else nearest to **midpoint** |
| v2 | **Strict** DenPAR bbox only | Line x contour, else **endpoint extension** |
| v3 | **Tooth mask** + nearest within **{args.grace_px}px** grace; tie-break **mask centroid** | Same as v2 |
| v4 | **Region growing:** on-mask, then rings **+1..{args.max_grace_px}px**; tie-break **mask centroid** | Same as v2 |

## Why v3 used 4 px (and why v4)

**4 px** was an initial tolerance matching the scale of v1's **8 px bbox margin** — not from the original paper. Defensible only as a **pilot constant**.

**v4 (region growing)** is the paper-ready variant: assign on the segmentation first, then expand tolerance in **1 px rings up to 8 px**, dropping only what remains outside. This is a **sensitivity-style rule** you can defend as "nearest mask band" rather than a single magic number.

**Final judge:** test **OKS after keypoint training** (v2 CEJ 0.843, etc.) — label statistics alone do not prove clinical correctness.

## Keypoint training note

Keypoint R-CNN is trained on **GT bboxes from JSON**, not YOLO.
"""
    args.out_md.write_text(md, encoding="utf-8")
    print(table)
    print()
    print(pt_tbl)
    print(f"\nWrote {args.out_json}\nWrote {args.out_md}")


if __name__ == "__main__":
    main()

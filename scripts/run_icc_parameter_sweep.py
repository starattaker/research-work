"""Val-locked ICC grid: combine × protocol × apex_merge (+ optional YOLO/NMS GPU pass).

One GPU cache per split (default thresholds), then CPU sweep over pairing knobs.
Optional second GPU pass sweeps score_thresh × nms_thresh on val only.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import scripts._bootstrap  # noqa: F401

from scripts.compare_slot_axis_icc import collect_predictions, mask_for_tooth
from scripts.run_icc_friend_pipeline import (
    COMBINES,
    PROTOCOLS,
    _icc_mae,
    _pair,
    cache_split,
)
from scripts.run_severity_icc import image_paths, resolve_yolo_weights
from src.denpar_paths import DEFAULT_DENPAR_ROOT
from src.severity.inference_pipeline import SeverityPipeline, load_gt_annotations
from src.severity.pred_combine import pred_side_details_from_tensors
from src.severity.side_details import gt_side_details_for_tooth

COMBINES_EXT = COMBINES + ("geom_consistent",)
DEFAULT_APEX_MERGE = (8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0)
DEFAULT_SCORE_THRESH = (0.4, 0.5, 0.6)
DEFAULT_NMS_THRESH = (0.5, 0.6, 0.7)


def eval_cached(
    cache: list[dict],
    combine: str,
    protocol: str,
    apex_merge_px: float,
    raw_root=None,
) -> dict:
    gt_vals: list[float] = []
    pred_vals: list[float] = []
    for row in cache:
        mask = None
        if combine == "mask_pca":
            mask = mask_for_tooth(raw_root, row["split"], row["stem"], row["tooth_idx"])
        pred_d = pred_side_details_from_tensors(
            row["cej"],
            row["intersection"],
            row["apex"],
            combine_mode=combine,
            merge_radius_px=apex_merge_px,
            bbox=row["bbox"],
            mask=mask,
        )
        for g, p in _pair(row["gt_details"], pred_d, protocol):
            gt_vals.append(g)
            pred_vals.append(p)
    out = _icc_mae(gt_vals, pred_vals)
    out["combine"] = combine
    out["protocol"] = protocol
    out["apex_merge_px"] = apex_merge_px
    return out


def parse_float_list(s: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in s.split(",") if x.strip())


def main():
    p = argparse.ArgumentParser(description="ICC parameter grid (val-locked)")
    p.add_argument("--data-root", type=Path, default=Path("data/processed_v6"))
    p.add_argument("--raw-root", type=Path, default=DEFAULT_DENPAR_ROOT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--yolo-weights", type=Path, default=None)
    p.add_argument("--cej-weights", type=Path, default=Path("runs/keypoints/v6_cej/best.pt"))
    p.add_argument("--intersection-weights", type=Path, default=Path("runs/keypoints/v6_intersection/best.pt"))
    p.add_argument("--apex-weights", type=Path, default=Path("runs/keypoints/v6_apex/best.pt"))
    p.add_argument("--apex-merge-grid", default=",".join(str(x) for x in DEFAULT_APEX_MERGE))
    p.add_argument("--combines", default=",".join(COMBINES_EXT))
    p.add_argument("--protocols", default=",".join(PROTOCOLS))
    p.add_argument("--sweep-yolo-nms", action="store_true", help="Extra GPU pass: score_thresh × nms on val")
    p.add_argument("--score-thresh-grid", default=",".join(str(x) for x in DEFAULT_SCORE_THRESH))
    p.add_argument("--nms-thresh-grid", default=",".join(str(x) for x in DEFAULT_NMS_THRESH))
    p.add_argument("--cache-dir", type=Path, default=Path("research_log/icc_sweep_cache"))
    p.add_argument("--out", type=Path, default=Path("research_log/icc_parameter_sweep.json"))
    args = p.parse_args()

    apex_grid = parse_float_list(args.apex_merge_grid)
    combines = tuple(c.strip() for c in args.combines.split(",") if c.strip())
    protocols = tuple(pr.strip() for pr in args.protocols.split(",") if pr.strip())

    yolo = resolve_yolo_weights(args.yolo_weights)
    pipeline = SeverityPipeline(
        yolo_weights=yolo,
        cej_weights=args.cej_weights,
        intersection_weights=args.intersection_weights,
        apex_weights=args.apex_weights,
        device=args.device,
        inference_mode="roi",
        combine_mode="tensor",
        raw_root=args.raw_root,
    )

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    caches: dict[str, list[dict]] = {}
    for split in ("val", "test", "train"):
        caches[split] = cache_split(pipeline, args.data_root, split, args.raw_root)
        print(f"Cached {split}: {len(caches[split])} teeth")

    print("=" * 70)
    print("CPU grid: combine × protocol × apex_merge_px  (VAL)")
    print("=" * 70)
    val_rows: list[dict] = []
    grid = list(itertools.product(combines, protocols, apex_grid))
    for combine, protocol, apex_px in tqdm(grid, desc="val grid"):
        m = eval_cached(caches["val"], combine, protocol, apex_px, args.raw_root)
        val_rows.append(m)

    ranked = [r for r in val_rows if r.get("icc") is not None]
    best = max(ranked, key=lambda r: r["icc"]) if ranked else None

    # Top 10 on val for quick view
    top10 = sorted(ranked, key=lambda r: r["icc"], reverse=True)[:10]
    print("\nTop 10 on VAL:")
    for r in top10:
        print(
            f"  {r['combine']:16s} {r['protocol']:14s} apex={r['apex_merge_px']:4.0f}  "
            f"ICC={r['icc']:.4f}  n={r['n_pairs']}  mae={r['mae_pct']:.2f}"
        )
    print(f"\nBest on VAL: {best}")

    print("\n" + "=" * 70)
    print("Locked winner on train / val / test")
    print("=" * 70)
    split_final: dict[str, dict] = {}
    test_top: list[dict] = []
    if best:
        for split in ("train", "val", "test"):
            split_final[split] = eval_cached(
                caches[split],
                best["combine"],
                best["protocol"],
                best["apex_merge_px"],
                args.raw_root,
            )
            r = split_final[split]
            if r.get("icc") is not None:
                print(
                    f"  {split:5s}  ICC={r['icc']:.4f}  n={r['n_pairs']}  "
                    f"mae={r['mae_pct']:.1f}  apex={best['apex_merge_px']}"
                )
        # Also report best-on-test among same grid (not for selection)
        for combine, protocol, apex_px in grid:
            test_top.append(eval_cached(caches["test"], combine, protocol, apex_px, args.raw_root))
        test_best = max([r for r in test_top if r.get("icc")], key=lambda r: r["icc"], default=None)
        if test_best:
            print(
                f"\n  (oracle test peek) best test ICC={test_best['icc']:.4f}  "
                f"{test_best['combine']}+{test_best['protocol']} apex={test_best['apex_merge_px']}"
            )

    yolo_nms_rows: list[dict] = []
    if args.sweep_yolo_nms:
        print("\n" + "=" * 70)
        print("GPU mini-sweep: score_thresh × nms_thresh (val only, locked combine/protocol/apex)")
        print("=" * 70)
        if not best:
            print("  skipped — no val winner")
        else:
            score_grid = parse_float_list(args.score_thresh_grid)
            nms_grid = parse_float_list(args.nms_thresh_grid)
            for score_t, nms_t in itertools.product(score_grid, nms_grid):
                pl = SeverityPipeline(
                    yolo_weights=yolo,
                    cej_weights=args.cej_weights,
                    intersection_weights=args.intersection_weights,
                    apex_weights=args.apex_weights,
                    device=args.device,
                    score_thresh=score_t,
                    nms_thresh=nms_t,
                    inference_mode="roi",
                    combine_mode="tensor",
                    raw_root=args.raw_root,
                )
                val_cache = cache_split(pl, args.data_root, "val", args.raw_root)
                m = eval_cached(
                    val_cache,
                    best["combine"],
                    best["protocol"],
                    best["apex_merge_px"],
                    args.raw_root,
                )
                m["score_thresh"] = score_t
                m["nms_thresh"] = nms_t
                yolo_nms_rows.append(m)
                icc = m.get("icc")
                icc_s = f"{icc:.4f}" if icc is not None else "n/a"
                print(f"  score={score_t} nms={nms_t}  ICC={icc_s}  n={m['n_pairs']}")

    report = {
        "paper_target_test_icc": 0.801,
        "grid": {
            "combines": list(combines),
            "protocols": list(protocols),
            "apex_merge_px": list(apex_grid),
            "n_configs": len(grid),
        },
        "best_on_val": best,
        "val_sweep": val_rows,
        "locked_splits": split_final,
        "test_sweep_all": test_top if best else [],
        "yolo_nms_val_sweep": yolo_nms_rows,
        "recommended_defaults": {
            "combine_mode": best["combine"] if best else "tensor",
            "severity_protocol": best["protocol"] if best else "match_by_slot",
            "apex_merge_px": best["apex_merge_px"] if best else 20.0,
        },
        "notes": [
            "Winner chosen on VAL only.",
            "apex_merge_px affects double-root apex pairing at inference.",
            "Use --sweep-yolo-nms for detection threshold tuning (slow).",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {args.out}")
    if best and split_final.get("test", {}).get("icc"):
        print(
            f"\n>>> VAL-locked  TEST ICC={split_final['test']['icc']:.4f}  "
            f"({best['combine']}+{best['protocol']} apex={best['apex_merge_px']})  paper 0.801"
        )


if __name__ == "__main__":
    main()

"""Visualize v3 (4px mask grace) vs v4 (region-growing to 8px) on DenPAR X-rays."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

import scripts._bootstrap  # noqa: F401

from src.preprocess.prepare_dataset import (
    SPLIT_ALIASES,
    assign_points_to_teeth_mask,
    assign_points_to_teeth_mask_region_grow,
    build_tooth_records,
    distance_to_mask,
    load_tooth_mask,
    pad_keypoints,
)


def grace_band_masks(
    mask: np.ndarray,
    inner_px: float,
    outer_px: float,
) -> np.ndarray:
    """Pixels with distance to mask in (inner_px, outer_px], mask interior excluded when inner=0."""
    inv = (mask == 0).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    if inner_px <= 0:
        return ((dist > 0) & (dist <= outer_px)).astype(np.uint8)
    return ((dist > inner_px) & (dist <= outer_px)).astype(np.uint8)


def tint_overlay(base: np.ndarray, band: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    if not band.any():
        return
    layer = base.copy()
    layer[band > 0] = color
    cv2.addWeighted(layer, alpha, base, 1.0 - alpha, 0, base)


def draw_mask_outline(img: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], thickness: int = 1) -> None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(img, contours, -1, color, thickness, cv2.LINE_AA)


def assigned_set(assign: list[list[list[float]]]) -> set[tuple[float, float]]:
    out: set[tuple[float, float]] = set()
    for tooth_pts in assign:
        for pt in tooth_pts:
            out.add((round(pt[0], 2), round(pt[1], 2)))
    return out


def draw_points(
    img: np.ndarray,
    points: list[list[float]],
    assigned: set[tuple[float, float]],
    kept_color: tuple[int, int, int],
    dropped_color: tuple[int, int, int],
    label_prefix: str,
) -> tuple[int, int]:
    kept = dropped = 0
    for pt in points:
        key = (round(pt[0], 2), round(pt[1], 2))
        x, y = int(round(pt[0])), int(round(pt[1]))
        if key in assigned:
            cv2.circle(img, (x, y), 6, kept_color, -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), 8, (0, 0, 0), 1, cv2.LINE_AA)
            kept += 1
        else:
            cv2.drawMarker(img, (x, y), dropped_color, cv2.MARKER_TILTED_CROSS, 10, 2, cv2.LINE_AA)
            dropped += 1
    return kept, dropped


def draw_tooth_bands(
    img: np.ndarray,
    mask_paths: list[Path],
    show_v3_band: bool,
    show_v4_extra_band: bool,
    show_full_8: bool,
) -> None:
    for path in mask_paths:
        mask = load_tooth_mask(path)
        if mask is None:
            continue
        draw_mask_outline(img, mask, (255, 220, 0), 1)
        if show_full_8:
            band8 = grace_band_masks(mask, 0.0, 8.0)
            tint_overlay(img, band8, (180, 80, 255), 0.35)
        if show_v3_band:
            band4 = grace_band_masks(mask, 0.0, 4.0)
            tint_overlay(img, band4, (255, 180, 60), 0.30)
        if show_v4_extra_band:
            band_extra = grace_band_masks(mask, 4.0, 8.0)
            tint_overlay(img, band_extra, (220, 60, 220), 0.35)


def annotate(img: np.ndarray, lines: list[str]) -> None:
    y = 24
    for line in lines:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
        y += 22


def build_panel(
    base_img: np.ndarray,
    mask_paths: list[Path],
    kp_json: dict,
    strategy: str,
    grace_px: float,
    max_grace_px: int,
    grace_step_px: int,
    mode: str,
) -> tuple[np.ndarray, dict]:
    img = base_img.copy()
    bboxes = kp_json["bboxes"]

    if mode == "reference":
        draw_tooth_bands(img, mask_paths, False, False, True)
        annotate(
            img,
            [
                "8 px grace envelope (purple) around each tooth mask",
                "yellow outline = expert tooth mask | compare width at CEJ/apex",
            ],
        )
        return img, {}

    if strategy == "v3":
        draw_tooth_bands(img, mask_paths, True, False, False)
        records = build_tooth_records(kp_json, None, mask_paths[0].parent, strategy="v3", grace_px=grace_px)
        cej_assign = assign_points_to_teeth_mask(
            kp_json.get("CEJ_Points", []), bboxes, mask_paths, grace_px=grace_px
        )
        apex_assign = assign_points_to_teeth_mask(
            kp_json.get("Apex_Points", []), bboxes, mask_paths, grace_px=grace_px
        )
        title = f"v3: mask + {grace_px:g}px grace (orange band)"
    else:
        draw_tooth_bands(img, mask_paths, False, True, True)
        records = build_tooth_records(
            kp_json,
            None,
            mask_paths[0].parent,
            strategy="v4",
            grace_step_px=grace_step_px,
            max_grace_px=max_grace_px,
        )
        cej_assign = assign_points_to_teeth_mask_region_grow(
            kp_json.get("CEJ_Points", []),
            bboxes,
            mask_paths,
            step_px=grace_step_px,
            max_radius_px=max_grace_px,
        )
        apex_assign = assign_points_to_teeth_mask_region_grow(
            kp_json.get("Apex_Points", []),
            bboxes,
            mask_paths,
            step_px=grace_step_px,
            max_radius_px=max_grace_px,
        )
        title = f"v4: region-growing rings 0-{max_grace_px}px (purple=full, pink=5-8px extra vs v3)"

    cej_pts = kp_json.get("CEJ_Points", [])
    apex_pts = kp_json.get("Apex_Points", [])
    cej_kept, cej_drop = draw_points(
        img, cej_pts, assigned_set(cej_assign), (40, 200, 40), (40, 40, 220), "CEJ"
    )
    apex_kept, apex_drop = draw_points(
        img, apex_pts, assigned_set(apex_assign), (40, 160, 255), (180, 40, 180), "Apex"
    )

    zero_apex = sum(1 for r in records if all(p == [0.0, 0.0] for p in r.apex))
    annotate(
        img,
        [
            title,
            f"CEJ kept {cej_kept}/{len(cej_pts)} | Apex kept {apex_kept}/{len(apex_pts)} | teeth w/ 0 apex: {zero_apex}/{len(records)}",
            "green/blue dot=kept | X=dropped",
        ],
    )
    stats = {
        "cej_kept": cej_kept,
        "cej_total": len(cej_pts),
        "apex_kept": apex_kept,
        "apex_total": len(apex_pts),
        "zero_apex_teeth": zero_apex,
        "teeth": len(records),
    }
    return img, stats


def draw_sample(
    image_path: Path,
    kp_json_path: Path,
    mask_dir: Path,
    out_path: Path,
    grace_px: float,
    max_grace_px: int,
    grace_step_px: int,
) -> dict:
    base = cv2.imread(str(image_path))
    if base is None:
        raise ValueError(f"Cannot read {image_path}")
    kp_json = json.loads(kp_json_path.read_text(encoding="utf-8"))
    mask_paths = sorted(mask_dir.glob("*.png"))
    if len(mask_paths) != len(kp_json["bboxes"]):
        raise ValueError(f"Mask count mismatch for {kp_json_path.name}")

    ref, _ = build_panel(base, mask_paths, kp_json, "v3", grace_px, max_grace_px, grace_step_px, "reference")
    v3, s3 = build_panel(base, mask_paths, kp_json, "v3", grace_px, max_grace_px, grace_step_px, "v3")
    v4, s4 = build_panel(base, mask_paths, kp_json, "v4", grace_px, max_grace_px, grace_step_px, "v4")

    h, w = base.shape[:2]
    gap = 8
    header_h = 36
    canvas = np.full((h * 3 + gap * 2 + header_h, w, 3), 32, dtype=np.uint8)
    canvas[header_h : header_h + h, 0:w] = ref
    canvas[header_h + h + gap : header_h + 2 * h + gap, 0:w] = v3
    canvas[header_h + 2 * h + 2 * gap : header_h + 3 * h + 2 * gap, 0:w] = v4

    title = f"{kp_json_path.stem} | v3 vs v4 grace QA"
    cv2.putText(canvas, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return {"image": kp_json_path.stem, "v3": s3, "v4": s4}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/DenPAR/Dataset"))
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed"),
        help="Use YOLO-cropped images from processed data (matches other QA figures)",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--ids",
        nargs="*",
        default=[],
        help="Specific image stems (e.g. test_1065) — overrides random sampling",
    )
    parser.add_argument("--grace-px", type=float, default=4.0)
    parser.add_argument("--max-grace-px", type=int, default=8)
    parser.add_argument("--grace-step-px", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("research_log/figures/v3_v4_grace_qa"))
    args = parser.parse_args()

    split_name = next(k for k, v in SPLIT_ALIASES.items() if v == args.split)
    kp_dir = args.raw_root / split_name / "Key Points Annotations"
    img_dir = args.processed_root / "yolo_detection" / args.split / "images"
    mask_root = args.raw_root / split_name / "Masks (Tooth-wise)"

    def normalize_stem(raw: str) -> str:
        for prefix in (f"{args.split}_", "test_", "val_", "train_"):
            if raw.startswith(prefix):
                return raw[len(prefix) :]
        return raw

    if args.ids:
        stems = [normalize_stem(s) for s in args.ids]
    else:
        all_stems = sorted(p.stem for p in img_dir.glob("*.jpg"))
        rng = random.Random(args.seed)
        stems = rng.sample(all_stems, min(args.n, len(all_stems)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for stem in stems:
        kp_path = kp_dir / f"{stem}.json"
        img_path = img_dir / f"{stem}.jpg"
        mask_dir = mask_root / stem
        if not (kp_path.exists() and img_path.exists() and mask_dir.exists()):
            print(f"skip {stem}: missing kp/image/mask")
            continue
        stats = draw_sample(
            img_path,
            kp_path,
            mask_dir,
            args.out_dir / f"{args.split}_{stem}.jpg",
            args.grace_px,
            args.max_grace_px,
            args.grace_step_px,
        )
        summaries.append(stats)
        print(stem, stats["v3"], stats["v4"])

    summary_path = args.out_dir / f"summary_{args.split}.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Wrote {len(summaries)} panels to {args.out_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

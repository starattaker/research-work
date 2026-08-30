"""Step-by-step figures: raw keypoints → mask PCA / points / LR → GT compare."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

import scripts._bootstrap  # noqa: F401

from scripts.compare_slot_axis_icc import collect_predictions, image_paths, mask_for_tooth, resolve_yolo_weights
from scripts.diagnose_severity_icc import gt_slot_used
from src.denpar_paths import DEFAULT_DENPAR_ROOT
from src.severity.inference_pipeline import SeverityPipeline, gt_severity_for_tooth, load_gt_annotations, point_from_slot
from src.severity.slot_matching import AxisMethod, oracle_best_severity, severity_with_axis_method

STEP_DIRS = {
    1: "step1_raw_predictions",
    2: "step2_gt_keypoints",
    3: "step3_mask_pca",
    4: "step4_points_axis",
    5: "step5_lr_position",
    6: "step6_oracle_summary",
}

MODEL_COLORS = {"cej": (0, 0, 255), "intersection": (0, 200, 0), "apex": (255, 120, 0)}
SIDE_COLORS = [(0, 255, 255), (255, 0, 255)]


def ensure_dirs(root: Path) -> dict[int, Path]:
    out = {}
    for k, name in STEP_DIRS.items():
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        out[k] = d
    return out


def crop_tooth(img: np.ndarray, bbox: list[float], pad: int = 24):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    y1p, y2p = max(0, y1 - pad), min(img.shape[0], y2 + pad)
    x1p, x2p = max(0, x1 - pad), min(img.shape[1], x2 + pad)
    return img[y1p:y2p, x1p:x2p].copy(), x1p, y1p


def draw_raw_preds(panel, kps: dict, ox: int, oy: int):
    for name, color in MODEL_COLORS.items():
        t = kps.get(name)
        if t is None:
            continue
        for si in range(min(2, t.shape[0])):
            x, y, v = float(t[si, 0]), float(t[si, 1]), float(t[si, 2])
            if v <= 0:
                continue
            lx, ly = int(x - ox), int(y - oy)
            cv2.circle(panel, (lx, ly), 5, color, -1)
            cv2.putText(panel, f"{name[0].upper()}{si}", (lx + 4, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)


def draw_gt(panel, merged, tooth_idx: int, ox: int, oy: int, gt_slot: int | None):
    for slot in (0, 1):
        col = SIDE_COLORS[slot]
        for key, letter in (("cej", "C"), ("intersection", "I"), ("apex", "A")):
            pt = point_from_slot(merged[key][tooth_idx], slot)
            if pt == (0.0, 0.0):
                continue
            lx, ly = int(pt[0] - ox), int(pt[1] - oy)
            cv2.circle(panel, (lx, ly), 5, col, 2 if slot == gt_slot else 1)
            cv2.putText(panel, f"G{letter}{slot}", (lx + 3, ly - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1)


def draw_method_panel(panel, method, kps, bbox, mask, merge_r, ox, oy, title: str):
    sev, axis, sides = severity_with_axis_method(
        kps.get("cej"), kps.get("intersection"), kps.get("apex"), method, bbox, mask, merge_r
    )
    if mask is not None and mask.shape[:2] == panel.shape[:2]:
        overlay = panel.copy()
        overlay[mask > 0] = (overlay[mask > 0] * 0.55 + np.array([100, 100, 220]) * 0.45).astype(np.uint8)
        panel[:] = overlay
    if method == AxisMethod.LR_POSITION and axis.reference_x is not None:
        x = int(axis.reference_x - ox)
        cv2.line(panel, (x, 0), (x, panel.shape[0]), (200, 80, 255), 1)
    else:
        ox_a, oy_a = int(axis.origin[0] - ox), int(axis.origin[1] - oy)
        ex = int(ox_a + axis.direction[0] * 70)
        ey = int(oy_a + axis.direction[1] * 70)
        cv2.arrowedLine(panel, (ox_a, oy_a), (ex, ey), (255, 120, 0), 2, tipLength=0.2)
    for si, side in enumerate(sides):
        col = SIDE_COLORS[si]
        for letter, pt in (("C", side.cej), ("I", side.intersection), ("A", side.apex)):
            if pt is None:
                continue
            lx, ly = int(pt[0] - ox), int(pt[1] - oy)
            cv2.circle(panel, (lx, ly), 5, col, -1)
            cv2.putText(panel, f"{letter}{si}", (lx + 3, ly - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1)
    sub = f"sev={sev:.1f}%" if sev is not None else "sev=—"
    cv2.putText(panel, f"{title} {sub}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
    cv2.putText(panel, f"{title} {sub}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
    return sev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/processed_v6"))
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_DENPAR_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--yolo-weights", type=Path, default=None)
    parser.add_argument("--cej-weights", type=Path, required=True)
    parser.add_argument("--intersection-weights", type=Path, required=True)
    parser.add_argument("--apex-weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-images", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge-radius", type=float, default=20.0)
    parser.add_argument("--out-dir", type=Path, default=Path("research_log/figures/slot_axis_steps"))
    args = parser.parse_args()

    dirs = ensure_dirs(args.out_dir)
    paths = image_paths(args.data_root, args.split)
    picks = random.Random(args.seed).sample(paths, min(args.n_images, len(paths)))

    pipeline = SeverityPipeline(
        yolo_weights=resolve_yolo_weights(args.yolo_weights),
        cej_weights=args.cej_weights,
        intersection_weights=args.intersection_weights,
        apex_weights=args.apex_weights,
        device=args.device,
        inference_mode="full",
    )

    summaries = []
    for img_path in picks:
        stem = img_path.stem
        merged = load_gt_annotations(args.data_root, args.split, stem)
        if merged is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        kps_by_tooth = collect_predictions(pipeline, img_path, merged)

        for tooth_idx, bbox in enumerate(merged["bboxes"]):
            if tooth_idx not in kps_by_tooth:
                continue
            gt_sev = gt_severity_for_tooth(merged, tooth_idx)
            if gt_sev is None:
                continue
            k = kps_by_tooth[tooth_idx]
            mask_full = mask_for_tooth(args.raw_root, args.split, stem, tooth_idx)
            crop, ox, oy = crop_tooth(img, bbox)
            mask = None
            if mask_full is not None:
                mask = mask_full[oy : oy + crop.shape[0], ox : ox + crop.shape[1]]

            tag = f"{stem}_t{tooth_idx}"
            p1 = crop.copy()
            cv2.rectangle(p1, (2, 2), (p1.shape[1] - 2, p1.shape[0] - 2), (255, 255, 0), 1)
            draw_raw_preds(p1, k, ox, oy)
            cv2.putText(p1, "step1 raw preds (R=CEJ G=INT B=apex)", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
            cv2.imwrite(str(dirs[1] / f"{tag}.jpg"), p1)

            p2 = crop.copy()
            gt_slot = gt_slot_used(merged, tooth_idx)
            draw_gt(p2, merged, tooth_idx, ox, oy, gt_slot)
            cv2.putText(p2, f"step2 GT slots (bold=used slot {gt_slot}) GTsev={gt_sev:.1f}%", (4, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 2)
            cv2.imwrite(str(dirs[2] / f"{tag}.jpg"), p2)

            p3 = crop.copy()
            sev_m = draw_method_panel(
                p3, AxisMethod.MASK_PCA, k, bbox, mask, args.merge_radius, ox, oy, "mask_pca"
            )
            cv2.imwrite(str(dirs[3] / f"{tag}.jpg"), p3)

            p4 = crop.copy()
            sev_p = draw_method_panel(
                p4, AxisMethod.POINTS_AXIS, k, bbox, mask, args.merge_radius, ox, oy, "points_axis"
            )
            cv2.imwrite(str(dirs[4] / f"{tag}.jpg"), p4)

            p5 = crop.copy()
            sev_l = draw_method_panel(
                p5, AxisMethod.LR_POSITION, k, bbox, mask, args.merge_radius, ox, oy, "lr_position"
            )
            cv2.imwrite(str(dirs[5] / f"{tag}.jpg"), p5)

            o_sev, o_slots = oracle_best_severity(k.get("cej"), k.get("intersection"), k.get("apex"), gt_sev)
            p6 = np.hstack([p3, p2])
            sm = f"{sev_m:.1f}" if sev_m is not None else "—"
            so = f"{o_sev:.1f}" if o_sev is not None else "—"
            txt = f"GT={gt_sev:.1f}% mask_pca={sm}% oracle={so}% slots={o_slots}"
            cv2.putText(p6, txt, (4, p6.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
            cv2.imwrite(str(dirs[6] / f"{tag}.jpg"), p6)

            summaries.append(
                {
                    "tag": tag,
                    "gt_sev": gt_sev,
                    "mask_pca": sev_m,
                    "points_axis": sev_p,
                    "lr_position": sev_l,
                    "oracle": o_sev,
                    "oracle_slots": o_slots,
                }
            )
            break  # one tooth per image for clarity

    (args.out_dir / "summaries" / "step_index.json").parent.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summaries" / "step_index.json").write_text(
        json.dumps({"steps": STEP_DIRS, "samples": summaries}, indent=2), encoding="utf-8"
    )
    print(f"Step figures: {args.out_dir}/")
    for k, v in STEP_DIRS.items():
        print(f"  step{k}: {v}/")


if __name__ == "__main__":
    main()

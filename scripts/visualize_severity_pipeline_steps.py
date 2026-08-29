"""Step-by-step visual QA for the severity / ICC pipeline (no ICC computed)."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch

import scripts._bootstrap  # noqa: F401

from src.severity.bone_loss import compute_bone_loss_severity, minimax_line_params, project_point_to_line
from src.severity.inference_pipeline import (
    SeverityPipeline,
    best_match_idx,
    gt_severity_for_tooth,
    load_gt_annotations,
    point_from_slot,
    severity_from_tensor_slots,
    slot_xy_from_tensor,
    yolo_boxes,
)

TOOTH_COLORS = [
    (60, 60, 255),
    (60, 200, 60),
    (255, 160, 40),
    (200, 60, 200),
    (40, 200, 200),
    (120, 120, 255),
    (80, 255, 120),
    (255, 80, 180),
    (180, 255, 80),
    (255, 120, 120),
    (100, 180, 255),
    (255, 200, 100),
]

PANEL_W = 420

STEP_FOLDERS = {
    1: "step1_gt_labels",
    2: "step2_yolo",
    3: "step3_gt_yolo_match",
    4: "step4_keypoints_pred",
    5: "step5_severity_compare",
}


def step_dirs(out_dir: Path) -> dict[int, Path]:
    dirs = {}
    for step, name in STEP_FOLDERS.items():
        d = out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        dirs[step] = d
    summary_dir = out_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    dirs["summaries"] = summary_dir
    return dirs


def tooth_color(i: int) -> tuple[int, int, int]:
    return TOOTH_COLORS[i % len(TOOTH_COLORS)]


def load_bgr(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    return img


def vis_pt(kps: list, slot: int) -> tuple[float, float] | None:
    if slot >= len(kps):
        return None
    kp = kps[slot]
    if len(kp) > 2 and int(kp[2]) == 0:
        return None
    x, y = float(kp[0]), float(kp[1])
    if x == 0.0 and y == 0.0:
        return None
    return x, y


def gt_slot_used(merged: dict, tooth_idx: int) -> int | None:
    for slot in (0, 1):
        c = point_from_slot(merged["cej"][tooth_idx], slot)
        t = point_from_slot(merged["intersection"][tooth_idx], slot)
        a = point_from_slot(merged["apex"][tooth_idx], slot)
        if compute_bone_loss_severity(c, t, a) is not None:
            return slot
    return None


def pred_slot_used(cej_kps, int_kps, apex_kps) -> int | None:
    for slot in (0, 1):
        c = slot_xy_from_tensor(cej_kps, slot) if cej_kps is not None else None
        t = slot_xy_from_tensor(int_kps, slot) if int_kps is not None else None
        a = slot_xy_from_tensor(apex_kps, slot) if apex_kps is not None else None
        if c and t and a and compute_bone_loss_severity(c, t, a) is not None:
            return slot
    return None


def draw_severity_line(
    canvas: np.ndarray,
    cej: tuple[float, float],
    inter: tuple[float, float],
    apex: tuple[float, float],
    color: tuple[int, int, int],
) -> None:
    pts = sorted([cej, inter, apex], key=lambda p: p[0])
    m, c = minimax_line_params(pts[0], pts[1], pts[2])
    if math.isinf(m):
        x0, x1 = int(c), int(c)
        ys = [int(cej[1]), int(inter[1]), int(apex[1])]
        y0, y1 = min(ys) - 20, max(ys) + 20
    else:
        all_x = [cej[0], inter[0], apex[0]]
        x0, x1 = int(min(all_x) - 15), int(max(all_x) + 15)
        y0 = int(m * x0 + c)
        y1 = int(m * x1 + c)
    cv2.line(canvas, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
    for pt, mk in ((cej, 0), (inter, 1), (apex, 2)):
        proj = project_point_to_line(pt[0], pt[1], m, c)
        cv2.circle(canvas, (int(proj[0]), int(proj[1])), 4, color, -1, cv2.LINE_AA)


def draw_kp(canvas, pt: tuple[float, float], color, shape: str, label: str = "") -> None:
    x, y = int(pt[0]), int(pt[1])
    if shape == "cej":
        cv2.circle(canvas, (x, y), 7, color, 2, cv2.LINE_AA)
    elif shape == "inter":
        cv2.rectangle(canvas, (x - 6, y - 6), (x + 6, y + 6), color, 2, cv2.LINE_AA)
    else:
        pts = np.array([[x, y - 8], [x - 7, y + 6], [x + 7, y + 6]], np.int32)
        cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)
    if label:
        cv2.putText(canvas, label, (x + 8, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def legend_panel(title: str, lines: list[str], height: int) -> np.ndarray:
    panel = np.full((height, PANEL_W, 3), 24, dtype=np.uint8)
    y = 28
    cv2.putText(panel, title, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    y += 28
    for line in lines:
        cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA)
        y += 17
    return panel


def compose(canvas: np.ndarray, title: str, legend_lines: list[str]) -> np.ndarray:
    h = canvas.shape[0]
    panel = legend_panel(title, legend_lines, h)
    out = np.hstack([canvas, panel])
    cv2.putText(out, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def image_paths(data_root: Path, split: str) -> list[Path]:
    img_dir = data_root / "yolo_detection" / split / "images"
    if not img_dir.exists():
        img_dir = data_root / "keypoints" / "cej" / split / "images"
    return sorted(img_dir.glob("*.jpg"))


def resolve_yolo_weights(path: Path | None) -> Path:
    candidates = [
        path,
        Path("runs/detect/runs/detection/yolov8x_tooth/weights/best.pt"),
        Path("runs/detection/yolov8x_tooth/weights/best.pt"),
    ]
    for c in candidates:
        if c and c.exists():
            return c
    raise FileNotFoundError("YOLO weights not found")


def step1_gt_labels(img_bgr: np.ndarray, merged: dict) -> tuple[np.ndarray, list[dict]]:
    canvas = img_bgr.copy()
    tooth_rows = []
    for i, bbox in enumerate(merged["bboxes"]):
        color = tooth_color(i)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        root = "dbl" if merged["labels"][i] == 2 else "sgl"
        slot = gt_slot_used(merged, i)
        gt_sev = gt_severity_for_tooth(merged, i)
        for s in (0, 1):
            for pts, shape, tag in (
                (merged["cej"][i], "cej", "C"),
                (merged["intersection"][i], "inter", "I"),
                (merged["apex"][i], "apex", "A"),
            ):
                pt = vis_pt(pts, s)
                if pt:
                    draw_kp(canvas, pt, color, shape, f"{tag}{s}")
        if slot is not None:
            c = point_from_slot(merged["cej"][i], slot)
            t = point_from_slot(merged["intersection"][i], slot)
            a = point_from_slot(merged["apex"][i], slot)
            draw_severity_line(canvas, c, t, a, color)
        label = f"T{i} {root} GT={gt_sev:.1f}% s{slot}" if gt_sev is not None else f"T{i} {root} GT=NA"
        cv2.putText(canvas, label, (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        tooth_rows.append(
            {
                "tooth_idx": i,
                "root_type": root,
                "gt_severity": gt_sev,
                "gt_slot_used": slot,
                "bbox": bbox,
            }
        )
    legend = [
        "STEP 1 — Ground truth (reference column for ICC)",
        "",
        "Per tooth from processed_v6 JSON:",
        "  • Colored box = GT tooth bbox",
        "  • C0/C1 = CEJ slots (circle)",
        "  • I0/I1 = intersection slots (square)",
        "  • A0/A1 = apex slots (triangle)",
        "  • Line = min-max line for chosen slot",
        "  • GT % = Eq. 1 severity (slot 0, else slot 1)",
        "",
        "sgl/dbl = single / double root label",
    ]
    return compose(canvas, "Step 1: GT labels + severity", legend), tooth_rows


def step2_yolo(img_bgr: np.ndarray, yolo_result) -> tuple[np.ndarray, list[list[float]]]:
    canvas = img_bgr.copy()
    boxes = yolo_boxes(yolo_result)
    box_list = boxes.tolist() if boxes.numel() else []
    for j, bbox in enumerate(box_list):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
        conf = float(yolo_result.boxes.conf[j]) if yolo_result.boxes is not None else 0.0
        cv2.putText(
            canvas,
            f"Y{j} {conf:.2f}",
            (x1, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    legend = [
        "STEP 2 — YOLOv8x tooth detection",
        "",
        f"Detections: {len(box_list)}",
        "Yellow boxes = predicted tooth bboxes",
        "(conf shown). Raw image, imgsz=640.",
        "",
        "These boxes become ROI proposals",
        "for Keypoint R-CNN in Step 4.",
    ]
    return compose(canvas, "Step 2: YOLO detection", legend), box_list


def step3_match(
    img_bgr: np.ndarray,
    merged: dict,
    yolo_box_list: list[list[float]],
    match_iou: float,
) -> tuple[np.ndarray, list[dict]]:
    canvas = img_bgr.copy()
    gt_boxes = torch.tensor(merged["bboxes"], dtype=torch.float32)
    yolo_t = torch.tensor(yolo_box_list, dtype=torch.float32) if yolo_box_list else torch.empty(0, 4)
    rows = []
    for i, bbox in enumerate(merged["bboxes"]):
        color = tooth_color(i)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        gt_box = gt_boxes[i]
        yolo_idx = best_match_idx(gt_box, yolo_t, match_iou)
        if yolo_idx is not None:
            yb = yolo_box_list[yolo_idx]
            cx_gt = (x1 + x2) // 2
            cy_gt = (y1 + y2) // 2
            cx_y = int((yb[0] + yb[2]) / 2)
            cy_y = int((yb[1] + yb[3]) / 2)
            cv2.line(canvas, (cx_gt, cy_gt), (cx_y, cy_y), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(canvas, (int(yb[0]), int(yb[1])), (int(yb[2]), int(yb[3])), (0, 255, 255), 2, cv2.LINE_AA)
            iou = float(torchvision_iou(gt_box, yolo_t[yolo_idx]))
            status = f"match Y{yolo_idx} IoU={iou:.2f}"
        else:
            status = "NO YOLO MATCH"
            cv2.putText(canvas, "X", (x1 + 4, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"T{i} {status}", (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        rows.append({"tooth_idx": i, "yolo_idx": yolo_idx, "yolo_matched": yolo_idx is not None})
    legend = [
        "STEP 3 — GT tooth ↔ YOLO match",
        "",
        "Colored = GT box per tooth T{i}",
        "Yellow = matched YOLO box",
        "Green line = GT center → YOLO center",
        "Red X = no YOLO IoU >= threshold",
        "",
        f"IoU threshold: {match_iou}",
    ]
    return compose(canvas, "Step 3: GT-YOLO matching", legend), rows


def torchvision_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    from torchvision.ops import box_iou

    return box_iou(a.unsqueeze(0), b.unsqueeze(0))[0, 0]


def step4_keypoints(
    img_bgr: np.ndarray,
    merged: dict,
    kps_by_tooth: dict[int, tuple],
) -> np.ndarray:
    canvas = img_bgr.copy()
    for i, bbox in enumerate(merged["bboxes"]):
        if i not in kps_by_tooth:
            continue
        color = tooth_color(i)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        cej_kps, int_kps, apex_kps = kps_by_tooth[i]
        for kps, shape, prefix in ((cej_kps, "cej", "pC"), (int_kps, "inter", "pI"), (apex_kps, "apex", "pA")):
            if kps is None:
                continue
            for s in range(kps.shape[0]):
                pt = slot_xy_from_tensor(kps, s)
                if pt:
                    draw_kp(canvas, pt, (255, 255, 255), shape, f"{prefix}{s}")
                    draw_kp(canvas, pt, color, shape, "")
    legend = [
        "STEP 4 — Predicted keypoints (3 models)",
        "",
        "ROI = YOLO-matched box (Step 3)",
        "Keypoint R-CNN ×3 on each ROI:",
        "  pC = predicted CEJ",
        "  pI = predicted intersection",
        "  pA = predicted apex",
        "Slots 0/1 from each model independently",
        "(CLAHE preproc, score>=0.5, NMS 0.6)",
    ]
    return compose(canvas, "Step 4: Predicted keypoints", legend)


def step5_severity(
    img_bgr: np.ndarray,
    merged: dict,
    tooth_rows: list[dict],
    kps_by_tooth: dict,
) -> tuple[np.ndarray, list[dict]]:
    canvas = img_bgr.copy()
    compare_rows = []
    for row in tooth_rows:
        i = row["tooth_idx"]
        color = tooth_color(i)
        bbox = merged["bboxes"][i]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        gt_sev = row["gt_severity"]
        gt_slot = row["gt_slot_used"]
        pred_sev = None
        pred_slot = None
        if i in kps_by_tooth:
            cej_kps, int_kps, apex_kps = kps_by_tooth[i]
            pred_sev = severity_from_tensor_slots(cej_kps, int_kps, apex_kps)
            pred_slot = pred_slot_used(cej_kps, int_kps, apex_kps)
            if pred_slot is not None and pred_sev is not None:
                c = slot_xy_from_tensor(cej_kps, pred_slot)
                t = slot_xy_from_tensor(int_kps, pred_slot)
                a = slot_xy_from_tensor(apex_kps, pred_slot)
                if c and t and a:
                    draw_severity_line(canvas, c, t, a, (255, 255, 0))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        if gt_sev is not None and pred_sev is not None:
            delta = pred_sev - gt_sev
            txt = f"T{i} GT={gt_sev:.1f}% P={pred_sev:.1f}% d={delta:+.1f}"
        elif gt_sev is not None:
            txt = f"T{i} GT={gt_sev:.1f}% P=NA"
        else:
            txt = f"T{i} GT=NA P={pred_sev:.1f}%" if pred_sev else f"T{i} no severity"
        cv2.putText(canvas, txt, (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        compare_rows.append(
            {
                "tooth_idx": i,
                "gt_severity": gt_sev,
                "gt_slot": gt_slot,
                "pred_severity": pred_sev,
                "pred_slot": pred_slot,
                "delta": (pred_sev - gt_sev) if gt_sev is not None and pred_sev is not None else None,
            }
        )
    legend = [
        "STEP 5 — Severity compare (no ICC yet)",
        "",
        "Yellow line = pred min-max (pred slot)",
        "GT % from Step 1, Pred % from",
        "aligned model slots (0 then 1).",
        "",
        "d = pred - GT (per tooth)",
        "ICC will aggregate all teeth later.",
    ]
    return compose(canvas, "Step 5: GT vs pred severity", legend), compare_rows


def process_image(
    pipeline: SeverityPipeline,
    image_path: Path,
    data_root: Path,
    split: str,
    out_dir: Path,
    match_iou: float,
    dirs: dict,
) -> dict:
    stem = image_path.stem
    merged = load_gt_annotations(data_root, split, stem)
    if merged is None:
        raise FileNotFoundError(f"Missing annotations for {stem}")

    img_bgr = load_bgr(image_path)

    s1_img, tooth_rows = step1_gt_labels(img_bgr, merged)
    cv2.imwrite(str(dirs[1] / f"{stem}.jpg"), s1_img)

    yolo_result = pipeline.yolo.predict(
        source=str(image_path), imgsz=640, conf=pipeline.score_thresh, verbose=False
    )[0]
    s2_img, yolo_box_list = step2_yolo(img_bgr, yolo_result)
    cv2.imwrite(str(dirs[2] / f"{stem}.jpg"), s2_img)

    s3_img, match_rows = step3_match(img_bgr, merged, yolo_box_list, match_iou)
    cv2.imwrite(str(dirs[3] / f"{stem}.jpg"), s3_img)

    from src.severity.inference_pipeline import load_image_tensor

    image_tensor = load_image_tensor(image_path, pipeline.transform)
    gt_boxes = torch.tensor(merged["bboxes"], dtype=torch.float32)
    yolo_t = torch.tensor(yolo_box_list, dtype=torch.float32) if yolo_box_list else torch.empty(0, 4)
    proposals, _ = pipeline._proposal_boxes_for_teeth(gt_boxes, yolo_t)
    valid_idx = [i for i, b in enumerate(proposals) if b is not None]
    kps_by_tooth: dict = {}
    if valid_idx:
        from src.keypoint.inference_utils import predict_keypoints_on_proposals

        prop_tensor = torch.stack([proposals[i] for i in valid_idx])
        labels = torch.tensor(merged["labels"], dtype=torch.int64)[valid_idx]
        cej_list = predict_keypoints_on_proposals(
            pipeline.models["cej"], image_tensor, prop_tensor, pipeline.device,
            pipeline.score_thresh, pipeline.nms_thresh, labels,
        )
        int_list = predict_keypoints_on_proposals(
            pipeline.models["intersection"], image_tensor, prop_tensor, pipeline.device,
            pipeline.score_thresh, pipeline.nms_thresh, labels,
        )
        apex_list = predict_keypoints_on_proposals(
            pipeline.models["apex"], image_tensor, prop_tensor, pipeline.device,
            pipeline.score_thresh, pipeline.nms_thresh, labels,
        )
        for row, tooth_idx in enumerate(valid_idx):
            kps_by_tooth[tooth_idx] = (cej_list[row], int_list[row], apex_list[row])

    s4_img = step4_keypoints(img_bgr, merged, kps_by_tooth)
    cv2.imwrite(str(dirs[4] / f"{stem}.jpg"), s4_img)

    s5_img, compare_rows = step5_severity(img_bgr, merged, tooth_rows, kps_by_tooth)
    cv2.imwrite(str(dirs[5] / f"{stem}.jpg"), s5_img)

    summary = {
        "image": stem,
        "n_teeth": len(merged["bboxes"]),
        "n_yolo": len(yolo_box_list),
        "outputs": {
            "step1": (dirs[1] / f"{stem}.jpg").as_posix(),
            "step2": (dirs[2] / f"{stem}.jpg").as_posix(),
            "step3": (dirs[3] / f"{stem}.jpg").as_posix(),
            "step4": (dirs[4] / f"{stem}.jpg").as_posix(),
            "step5": (dirs[5] / f"{stem}.jpg").as_posix(),
        },
        "step1_gt": tooth_rows,
        "step3_match": match_rows,
        "step5_compare": compare_rows,
    }
    (dirs["summaries"] / f"{stem}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Visualize severity pipeline steps 1-5")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed_v6"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-images", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--yolo-weights", type=Path, default=None)
    parser.add_argument("--cej-weights", type=Path, default=None)
    parser.add_argument("--intersection-weights", type=Path, default=None)
    parser.add_argument("--apex-weights", type=Path, default=None)
    parser.add_argument(
        "--step1-only",
        action="store_true",
        help="Only Step 1 (GT labels) — no model weights needed",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--nms-thresh", type=float, default=0.6)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("research_log/figures/severity_pipeline_steps"),
    )
    args = parser.parse_args()

    paths = image_paths(args.data_root, args.split)
    if len(paths) < args.n_images:
        raise ValueError(f"Only {len(paths)} images in {args.split}, need {args.n_images}")

    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(paths, args.n_images), key=lambda p: p.stem)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dirs = step_dirs(args.out_dir)
    manifest = {
        "seed": args.seed,
        "images": [],
        "out_dir": args.out_dir.as_posix(),
        "folders": {f"step{k}": v for k, v in STEP_FOLDERS.items()},
        "summaries": "summaries",
    }

    if args.step1_only:
        for img_path in chosen:
            stem = img_path.stem
            merged = load_gt_annotations(args.data_root, args.split, stem)
            if merged is None:
                continue
            img_bgr = load_bgr(img_path)
            s1_img, tooth_rows = step1_gt_labels(img_bgr, merged)
            out_path = dirs[1] / f"{stem}.jpg"
            cv2.imwrite(str(out_path), s1_img)
            summary = {"image": stem, "n_teeth": len(merged["bboxes"]), "step1_gt": tooth_rows}
            (dirs["summaries"] / f"{stem}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            manifest["images"].append(summary)
            print(f"  step1_gt_labels/{stem}.jpg")
    else:
        if not all([args.cej_weights, args.intersection_weights, args.apex_weights]):
            raise ValueError("Steps 2-5 require --cej/intersection/apex-weights (or use --step1-only)")
        pipeline = SeverityPipeline(
            yolo_weights=resolve_yolo_weights(args.yolo_weights),
            cej_weights=args.cej_weights,
            intersection_weights=args.intersection_weights,
            apex_weights=args.apex_weights,
            device=args.device,
            score_thresh=args.score_thresh,
            nms_thresh=args.nms_thresh,
            match_iou=args.match_iou,
        )
        for img_path in chosen:
            print(f"Processing {img_path.stem} (steps 1-5)...")
            summary = process_image(
                pipeline, img_path, args.data_root, args.split, args.out_dir, args.match_iou, dirs
            )
            manifest["images"].append(summary)

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nDone. Outputs in {args.out_dir}/")
    for step, folder in STEP_FOLDERS.items():
        print(f"  {folder}/  ({args.n_images} images)")
    print(f"  summaries/  ({len(manifest['images'])} json files)")


if __name__ == "__main__":
    main()

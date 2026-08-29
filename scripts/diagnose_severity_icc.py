"""Exhaustive ICC pipeline diagnosis — find which stage breaks before fixing."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import scripts._bootstrap  # noqa: F401

from src.keypoint.dataset import KeypointDataset
from src.keypoint.inference_utils import predict_keypoints_for_boxes
from src.keypoint.model import get_keypoint_model
from src.keypoint.train import build_transforms
from src.keypoint.train_utils import evaluate_oks
from src.severity.bone_loss import compute_bone_loss_severity
from src.severity.icc import icc21
from src.severity.inference_pipeline import (
    SeverityPipeline,
    best_match_idx,
    gt_severity_for_tooth,
    load_gt_annotations,
    load_image_tensor,
    point_from_slot,
    severity_from_tensor_slots,
    slot_xy_from_tensor,
    yolo_boxes,
)


@dataclass
class ToothDiag:
    stem: str
    tooth_idx: int
    label: int
    gt_severity: float | None
    gt_slot: int | None
    yolo_matched: bool
    yolo_iou: float | None
    cej_yolo: torch.Tensor | None = None
    int_yolo: torch.Tensor | None = None
    apex_yolo: torch.Tensor | None = None
    cej_gt: torch.Tensor | None = None
    int_gt: torch.Tensor | None = None
    apex_gt: torch.Tensor | None = None


def image_paths(data_root: Path, split: str) -> list[Path]:
    img_dir = data_root / "yolo_detection" / split / "images"
    if not img_dir.exists():
        img_dir = data_root / "keypoints" / "cej" / split / "images"
    return sorted(img_dir.glob("*.jpg"))


def resolve_yolo_weights(path: Path | None) -> Path:
    for c in (
        path,
        Path("runs/detect/runs/detection/yolov8x_tooth/weights/best.pt"),
        Path("runs/detection/yolov8x_tooth/weights/best.pt"),
    ):
        if c and c.exists():
            return c
    raise FileNotFoundError("YOLO weights not found")


def gt_slot_used(merged: dict, tooth_idx: int) -> int | None:
    for slot in (0, 1):
        c = point_from_slot(merged["cej"][tooth_idx], slot)
        t = point_from_slot(merged["intersection"][tooth_idx], slot)
        a = point_from_slot(merged["apex"][tooth_idx], slot)
        if compute_bone_loss_severity(c, t, a) is not None:
            return slot
    return None


def severity_at_slot(
    cej: torch.Tensor | None,
    inter: torch.Tensor | None,
    apex: torch.Tensor | None,
    slot: int,
) -> float | None:
    if cej is None or inter is None or apex is None:
        return None
    c = slot_xy_from_tensor(cej, slot)
    t = slot_xy_from_tensor(inter, slot)
    a = slot_xy_from_tensor(apex, slot)
    if c is None or t is None or a is None:
        return None
    return compute_bone_loss_severity(c, t, a)


def oracle_best_severity(
    cej: torch.Tensor | None,
    inter: torch.Tensor | None,
    apex: torch.Tensor | None,
    gt_sev: float,
) -> tuple[float | None, tuple[int, int, int] | None]:
    """Best severity over all 8 slot triplets (ceiling if combination were perfect)."""
    best_sev = None
    best_slots = None
    best_err = float("inf")
    for cs in (0, 1):
        for is_ in (0, 1):
            for as_ in (0, 1):
                c = slot_xy_from_tensor(cej, cs) if cej is not None else None
                t = slot_xy_from_tensor(inter, is_) if inter is not None else None
                a = slot_xy_from_tensor(apex, as_) if apex is not None else None
                if c is None or t is None or a is None:
                    continue
                sev = compute_bone_loss_severity(c, t, a)
                if sev is None:
                    continue
                err = abs(sev - gt_sev)
                if err < best_err:
                    best_err = err
                    best_sev = sev
                    best_slots = (cs, is_, as_)
    return best_sev, best_slots


def oracle_aligned_slots(
    cej: torch.Tensor | None,
    inter: torch.Tensor | None,
    apex: torch.Tensor | None,
    slot: int,
) -> float | None:
    return severity_at_slot(cej, inter, apex, slot)


def metric_icc_mae(gt: list[float], pred: list[float]) -> dict:
    if len(gt) < 3:
        return {"icc": None, "mae_pct": None, "n_pairs": len(gt)}
    mat = np.column_stack([gt, pred])
    return {
        "icc": icc21(mat),
        "mae_pct": float(np.mean(np.abs(mat[:, 0] - mat[:, 1]))),
        "n_pairs": len(gt),
    }


def mean_kp_dist(
    cej: torch.Tensor | None,
    inter: torch.Tensor | None,
    apex: torch.Tensor | None,
    merged: dict,
    tooth_idx: int,
    slot: int,
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for name, kps, pts in (
        ("cej", cej, merged["cej"][tooth_idx]),
        ("intersection", inter, merged["intersection"][tooth_idx]),
        ("apex", apex, merged["apex"][tooth_idx]),
    ):
        pred = slot_xy_from_tensor(kps, slot) if kps is not None else None
        gt = point_from_slot(pts, slot)
        if pred is None or gt == (0.0, 0.0):
            out[name] = None
        else:
            out[name] = float(np.hypot(pred[0] - gt[0], pred[1] - gt[1]))
    return out


def collect_tooth_records(
    pipeline: SeverityPipeline,
    data_root: Path,
    split: str,
    paths: list[Path],
) -> list[ToothDiag]:
    records: list[ToothDiag] = []
    for img_path in tqdm(paths, desc="collect"):
        stem = img_path.stem
        merged = load_gt_annotations(data_root, split, stem)
        if merged is None:
            continue

        yolo_result = pipeline.yolo.predict(
            source=str(img_path), imgsz=640, conf=pipeline.score_thresh, verbose=False
        )[0]
        yolo_t = yolo_boxes(yolo_result)
        image_tensor = load_image_tensor(img_path, pipeline.transform)
        gt_boxes = torch.tensor(merged["bboxes"], dtype=torch.float32)
        labels = torch.tensor(merged["labels"], dtype=torch.int64)

        # Predict on all GT boxes (full forward)
        gt_kps: dict[str, list] = {}
        for name, model in pipeline.models.items():
            gt_kps[name] = predict_keypoints_for_boxes(
                model,
                image_tensor,
                gt_boxes,
                pipeline.device,
                "full",
                pipeline.score_thresh,
                pipeline.nms_thresh,
                labels,
                pipeline.keypoint_match_iou,
            )

        proposals, yolo_flags = pipeline._proposal_boxes_for_teeth(gt_boxes, yolo_t)
        valid_idx = [i for i, b in enumerate(proposals) if b is not None]
        yolo_kps: dict[int, dict[str, torch.Tensor | None]] = {}
        if valid_idx:
            prop_tensor = torch.stack([proposals[i] for i in valid_idx])
            label_tensor = labels[valid_idx]
            for name, model in pipeline.models.items():
                kps_list = predict_keypoints_for_boxes(
                    model,
                    image_tensor,
                    prop_tensor,
                    pipeline.device,
                    "full",
                    pipeline.score_thresh,
                    pipeline.nms_thresh,
                    label_tensor,
                    pipeline.keypoint_match_iou,
                )
                for row, tooth_idx in enumerate(valid_idx):
                    yolo_kps.setdefault(tooth_idx, {})[name] = kps_list[row]

        for i in range(len(merged["bboxes"])):
            gt_sev = gt_severity_for_tooth(merged, i)
            gt_slot = gt_slot_used(merged, i)
            yolo_iou = None
            if yolo_t.numel():
                j = best_match_idx(gt_boxes[i], yolo_t, pipeline.match_iou)
                if j is not None:
                    from torchvision.ops import box_iou

                    yolo_iou = float(box_iou(gt_boxes[i].unsqueeze(0), yolo_t[j].unsqueeze(0))[0, 0])

            yk = yolo_kps.get(i, {})
            records.append(
                ToothDiag(
                    stem=stem,
                    tooth_idx=i,
                    label=int(merged["labels"][i]),
                    gt_severity=gt_sev,
                    gt_slot=gt_slot,
                    yolo_matched=yolo_flags[i],
                    yolo_iou=yolo_iou,
                    cej_yolo=yk.get("cej"),
                    int_yolo=yk.get("intersection"),
                    apex_yolo=yk.get("apex"),
                    cej_gt=gt_kps["cej"][i] if i < len(gt_kps["cej"]) else None,
                    int_gt=gt_kps["intersection"][i] if i < len(gt_kps["intersection"]) else None,
                    apex_gt=gt_kps["apex"][i] if i < len(gt_kps["apex"]) else None,
                )
            )
    return records


def severity_hybrid_at_gt_slot(
    r: ToothDiag,
    merged: dict,
    replace: set[str],
) -> float | None:
    """Pred keypoints at GT slot; optionally replace model(s) with GT coordinates."""
    if r.gt_slot is None:
        return None
    slot = r.gt_slot
    preds = {
        "cej": r.cej_gt,
        "intersection": r.int_gt,
        "apex": r.apex_gt,
    }
    pts = {
        "cej": merged["cej"][r.tooth_idx],
        "intersection": merged["intersection"][r.tooth_idx],
        "apex": merged["apex"][r.tooth_idx],
    }
    coords: list[tuple[float, float] | None] = []
    for name in ("cej", "intersection", "apex"):
        if name in replace:
            coords.append(point_from_slot(pts[name], slot))
        else:
            kps = preds[name]
            coords.append(slot_xy_from_tensor(kps, slot) if kps is not None else None)
    if any(c is None for c in coords):
        return None
    return compute_bone_loss_severity(coords[0], coords[1], coords[2])


def preds_hybrid(
    records: list[ToothDiag],
    merged_by_image: dict[str, dict],
    replace: set[str],
) -> tuple[list[float], list[float]]:
    gt_out: list[float] = []
    pred_out: list[float] = []
    for r in records:
        if r.gt_severity is None:
            continue
        m = merged_by_image.get(r.stem)
        if m is None:
            continue
        pred = severity_hybrid_at_gt_slot(r, m, replace)
        if pred is None:
            continue
        gt_out.append(r.gt_severity)
        pred_out.append(pred)
    return gt_out, pred_out


def preds_from_records(
    records: list[ToothDiag],
    source: str,
    combine: str,
    merged_by_image: dict[str, dict] | None = None,
) -> tuple[list[float], list[float]]:
    gt_out: list[float] = []
    pred_out: list[float] = []
    for r in records:
        if r.gt_severity is None:
            continue
        if source == "yolo":
            cej, inter, apex = r.cej_yolo, r.int_yolo, r.apex_yolo
        elif source == "gt":
            cej, inter, apex = r.cej_gt, r.int_gt, r.apex_gt
        else:
            raise ValueError(source)

        if combine == "current":
            pred = severity_from_tensor_slots(cej, inter, apex)
        elif combine == "gt_slot":
            pred = oracle_aligned_slots(cej, inter, apex, r.gt_slot) if r.gt_slot is not None else None
        elif combine == "oracle_best":
            pred, _ = oracle_best_severity(cej, inter, apex, r.gt_severity)
        elif combine == "gt_keypoints":
            if merged_by_image is None:
                raise ValueError("merged_by_image required")
            m = merged_by_image[r.stem]
            pred = gt_severity_for_tooth(m, r.tooth_idx)
        else:
            raise ValueError(combine)

        if pred is None:
            continue
        gt_out.append(r.gt_severity)
        pred_out.append(pred)
    return gt_out, pred_out


def per_model_oks(
    data_root: Path,
    split: str,
    weights: dict[str, Path],
    device: torch.device,
) -> dict[str, float]:
    out = {}
    for kpt_type, wpath in weights.items():
        ds = KeypointDataset(
            data_root / "keypoints" / kpt_type / split,
            transform=build_transforms(False),
        )
        loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=ds.collate_fn)
        model = get_keypoint_model(num_keypoints=2, weights_path=str(wpath)).to(device)
        out[kpt_type] = round(evaluate_oks(model, loader, device), 4)
    return out


def roi_yolo_icc(
    pipeline: SeverityPipeline,
    data_root: Path,
    split: str,
    paths: list[Path],
) -> dict:
    gt_vals, pred_vals = [], []
    for img_path in tqdm(paths, desc="roi-yolo"):
        merged = load_gt_annotations(data_root, split, img_path.stem)
        if merged is None:
            continue
        yolo_result = pipeline.yolo.predict(
            source=str(img_path), imgsz=640, conf=pipeline.score_thresh, verbose=False
        )[0]
        yolo_t = yolo_boxes(yolo_result)
        image_tensor = load_image_tensor(img_path, pipeline.transform)
        gt_boxes = torch.tensor(merged["bboxes"], dtype=torch.float32)
        proposals, _ = pipeline._proposal_boxes_for_teeth(gt_boxes, yolo_t)
        valid_idx = [i for i, b in enumerate(proposals) if b is not None]
        kps_by_tooth: dict = {}
        if valid_idx:
            prop_tensor = torch.stack([proposals[i] for i in valid_idx])
            labels = torch.tensor(merged["labels"], dtype=torch.int64)[valid_idx]
            for name, model in pipeline.models.items():
                kps_list = predict_keypoints_for_boxes(
                    model,
                    image_tensor,
                    prop_tensor,
                    pipeline.device,
                    "roi",
                    pipeline.score_thresh,
                    pipeline.nms_thresh,
                    labels,
                    pipeline.keypoint_match_iou,
                )
                for row, tooth_idx in enumerate(valid_idx):
                    if tooth_idx not in kps_by_tooth:
                        kps_by_tooth[tooth_idx] = {}
                    kps_by_tooth[tooth_idx][name] = kps_list[row]
        for i in range(len(merged["bboxes"])):
            gt_sev = gt_severity_for_tooth(merged, i)
            if gt_sev is None or i not in kps_by_tooth:
                continue
            k = kps_by_tooth[i]
            pred = severity_from_tensor_slots(k.get("cej"), k.get("intersection"), k.get("apex"))
            if pred is None:
                continue
            gt_vals.append(gt_sev)
            pred_vals.append(pred)
    m = metric_icc_mae(gt_vals, pred_vals)
    m["description"] = "ROI-on-YOLO (old path)"
    return m


def build_diagnosis(tests: dict) -> list[str]:
    lines = []
    a1 = tests["A1_icc_code_sanity"]["icc"]
    if a1 is not None and a1 > 0.99:
        lines.append("OK: ICC implementation is correct (identity test ~1.0).")
    else:
        lines.append("FAIL: ICC code sanity check — fix icc21 before anything else.")

    a2 = tests["A2_gt_labels_self"]["icc"]
    if a2 is not None and a2 > 0.99:
        lines.append("OK: GT severity labels + Eq.1 are self-consistent.")
    else:
        lines.append("FAIL: GT severity path inconsistent — check v6 labels / Eq.1.")

    oks = tests["B_per_model_oks_test"]
    lines.append(
        f"Per-model test OKS (full forward, GT boxes in loader): "
        f"CEJ={oks['cej']}, INT={oks['intersection']}, APEX={oks['apex']} (paper ~0.95/0.91/0.82)."
    )

    c1 = tests["C1_end_to_end_yolo_full"]["icc"]
    c2 = tests["C2_end_to_end_gt_boxes_full"]["icc"]
    if c1 is not None and c2 is not None and abs(c1 - c2) < 0.05:
        lines.append(
            "YOLO vs GT boxes: ICC similar → YOLO localization is NOT the main bottleneck."
        )
    elif c2 is not None and c1 is not None and c2 > c1 + 0.1:
        lines.append("YOLO box shift hurts ICC noticeably — improve detection or use tighter boxes.")
    else:
        lines.append("Compare C1 vs C2 ICC to see YOLO impact.")

    d_oracle = tests["D2_oracle_best_slot_combo_gt_boxes"]["icc"]
    d_current = tests["D1_current_slot_logic_gt_boxes"]["icc"]
    if d_oracle is not None and d_current is not None:
        if d_oracle > d_current + 0.15:
            lines.append(
                f"MAIN ISSUE: Slot/combination logic. Oracle ICC={d_oracle:.3f} vs "
                f"current={d_current:.3f}. Implement PCA slot matching across 3 models."
            )
        elif d_oracle < 0.3:
            lines.append(
                f"Even oracle slot matching ICC={d_oracle:.3f} is low on GT boxes → "
                "keypoint accuracy or severity formula vs paper may be wrong."
            )
        else:
            lines.append("Slot matching helps somewhat; also check keypoint error magnitudes (section E).")

    d_gt_slot = tests["D3_use_gt_slot_index_gt_boxes"]["icc"]
    if d_gt_slot is not None and d_current is not None and d_gt_slot > d_current + 0.05:
        lines.append("Using GT slot index for all 3 models improves ICC — pred picks wrong slot.")

    h_all = tests.get("H0_gt_slot_aligned_gt_boxes", {}).get("icc")
    for key, label in (
        ("H1_replace_cej_with_gt", "CEJ"),
        ("H2_replace_intersection_with_gt", "intersection"),
        ("H3_replace_apex_with_gt", "apex"),
    ):
        h = tests.get(key, {}).get("icc")
        if h_all is not None and h is not None and h > h_all + 0.1:
            lines.append(f"Replacing pred {label} with GT at GT slot lifts ICC a lot — focus on that model.")

    mismatch = tests["D4_pred_slot_neq_gt_slot_rate"]["rate"]
    lines.append(f"Pred slot (current logic) != GT slot on {mismatch*100:.1f}% of paired teeth.")

    lines.append(
        "Read tests C/D/E in JSON. Fix the stage with the largest ICC gap to oracle ceiling first."
    )
    return lines


def main():
    parser = argparse.ArgumentParser(description="Diagnose where severity ICC pipeline fails")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed_v6"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--yolo-weights", type=Path, default=None)
    parser.add_argument("--cej-weights", type=Path, required=True)
    parser.add_argument("--intersection-weights", type=Path, required=True)
    parser.add_argument("--apex-weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=0, help="0 = all images")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("research_log/severity_icc_diagnosis.json"),
    )
    args = parser.parse_args()

    paths = image_paths(args.data_root, args.split)
    if args.max_images > 0:
        paths = paths[: args.max_images]

    pipeline = SeverityPipeline(
        yolo_weights=resolve_yolo_weights(args.yolo_weights),
        cej_weights=args.cej_weights,
        intersection_weights=args.intersection_weights,
        apex_weights=args.apex_weights,
        device=args.device,
        match_iou=args.match_iou,
        inference_mode="full",
    )
    device = pipeline.device

    merged_cache: dict[str, dict] = {}
    for p in paths:
        m = load_gt_annotations(args.data_root, args.split, p.stem)
        if m:
            merged_cache[p.stem] = m

    records = collect_tooth_records(pipeline, args.data_root, args.split, paths)

    tests: dict = {}

    # A — sanity (no model uncertainty)
    gt_all = [r.gt_severity for r in records if r.gt_severity is not None]
    tests["A1_icc_code_sanity"] = {
        **metric_icc_mae(gt_all, gt_all),
        "description": "ICC(gt, gt) — must be ~1.0",
        "expected": "~1.0",
    }
    gt_kp, pred_kp = preds_from_records(records, "gt", "gt_keypoints", merged_cache)
    tests["A2_gt_labels_self"] = {
        **metric_icc_mae(gt_kp, pred_kp),
        "description": "ICC(gt_severity, severity from same GT keypoints)",
        "expected": "~1.0",
    }

    # B — per-model OKS (official eval path)
    tests["B_per_model_oks_test"] = per_model_oks(
        args.data_root,
        args.split,
        {
            "cej": args.cej_weights,
            "intersection": args.intersection_weights,
            "apex": args.apex_weights,
        },
        device,
    )

    # C — end-to-end severity ICC variants
    tests["C1_end_to_end_yolo_full"] = {
        **metric_icc_mae(*preds_from_records(records, "yolo", "current")),
        "description": "Production: YOLO box + full inference + current slot logic",
    }
    tests["C2_end_to_end_gt_boxes_full"] = {
        **metric_icc_mae(*preds_from_records(records, "gt", "current")),
        "description": "GT boxes + full inference + current slot logic (no YOLO shift)",
    }
    tests["C3_end_to_end_roi_yolo"] = roi_yolo_icc(pipeline, args.data_root, args.split, paths)

    # D — slot / combination analysis (on GT box predictions)
    tests["D1_current_slot_logic_gt_boxes"] = {
        **metric_icc_mae(*preds_from_records(records, "gt", "current")),
        "description": "Same as C2 — baseline for slot tests",
    }
    tests["D2_oracle_best_slot_combo_gt_boxes"] = {
        **metric_icc_mae(*preds_from_records(records, "gt", "oracle_best")),
        "description": "CEILING: best (c,i,a) slot combo per tooth vs GT severity",
    }
    tests["D3_use_gt_slot_index_gt_boxes"] = {
        **metric_icc_mae(*preds_from_records(records, "gt", "gt_slot")),
        "description": "Use GT slot index for CEJ+int+apex (aligned slots)",
    }

    slot_mismatch = 0
    slot_total = 0
    for r in records:
        if r.gt_severity is None or r.gt_slot is None:
            continue
        pred = severity_from_tensor_slots(r.cej_gt, r.int_gt, r.apex_gt)
        if pred is None:
            continue
        pred_slot = None
        for s in (0, 1):
            if severity_at_slot(r.cej_gt, r.int_gt, r.apex_gt, s) == pred:
                pred_slot = s
                break
        slot_total += 1
        if pred_slot is not None and pred_slot != r.gt_slot:
            slot_mismatch += 1
    tests["D4_pred_slot_neq_gt_slot_rate"] = {
        "rate": slot_mismatch / max(slot_total, 1),
        "n": slot_total,
        "description": "Current logic picks different slot than GT",
    }

    # H — which model hurts most (GT slot aligned, GT boxes)
    tests["H0_gt_slot_aligned_gt_boxes"] = {
        **metric_icc_mae(*preds_from_records(records, "gt", "gt_slot")),
        "description": "Baseline: pred kps at GT slot index (same as D3)",
    }
    tests["H1_replace_cej_with_gt"] = {
        **metric_icc_mae(*preds_hybrid(records, merged_cache, {"cej"})),
        "description": "GT slot + replace only CEJ with GT coordinates",
    }
    tests["H2_replace_intersection_with_gt"] = {
        **metric_icc_mae(*preds_hybrid(records, merged_cache, {"intersection"})),
        "description": "GT slot + replace only intersection with GT",
    }
    tests["H3_replace_apex_with_gt"] = {
        **metric_icc_mae(*preds_hybrid(records, merged_cache, {"apex"})),
        "description": "GT slot + replace only apex with GT",
    }
    tests["H4_replace_all_kps_with_gt"] = {
        **metric_icc_mae(gt_kp, pred_kp),
        "description": "All GT keypoints (ceiling for labels+formula)",
    }

    # E — keypoint pixel error at GT slot (GT box preds)
    dists = {"cej": [], "intersection": [], "apex": []}
    for r in records:
        if r.gt_slot is None:
            continue
        m = merged_cache.get(r.stem)
        if m is None:
            continue
        d = mean_kp_dist(r.cej_gt, r.int_gt, r.apex_gt, m, r.tooth_idx, r.gt_slot)
        for k, v in d.items():
            if v is not None:
                dists[k].append(v)
    tests["E_mean_px_error_gt_box_preds"] = {
        k: {"mean_px": float(np.mean(v)), "n": len(v)} if v else {"mean_px": None, "n": 0}
        for k, v in dists.items()
    }

    # F — by root type
    for label, name in ((1, "single_root"), (2, "double_root")):
        sub = [r for r in records if r.label == label]
        gt_l, pr_l = preds_from_records(sub, "yolo", "current")
        tests[f"F_icc_{name}_yolo_full"] = {
            **metric_icc_mae(gt_l, pr_l),
            "description": f"ICC for label={label} teeth only",
        }

    # G — YOLO IoU vs severity error
    ious, errs = [], []
    for r in records:
        if r.gt_severity is None or r.yolo_iou is None:
            continue
        pred = severity_from_tensor_slots(r.cej_yolo, r.int_yolo, r.apex_yolo)
        if pred is None:
            continue
        ious.append(r.yolo_iou)
        errs.append(abs(pred - r.gt_severity))
    if len(ious) >= 3:
        tests["G_yolo_iou_vs_severity_abs_error"] = {
            "pearson_r": float(np.corrcoef(ious, errs)[0, 1]),
            "n": len(ious),
            "description": "Negative r → worse YOLO IoU = worse severity",
        }
    else:
        tests["G_yolo_iou_vs_severity_abs_error"] = {"pearson_r": None, "n": len(ious)}

    diagnosis = build_diagnosis(tests)
    report = {
        "paper_target_icc": 0.801,
        "split": args.split,
        "data_root": args.data_root.as_posix(),
        "n_images": len(paths),
        "n_teeth_records": len(records),
        "tests": tests,
        "diagnosis": diagnosis,
        "how_to_read": {
            "A": "Sanity — must pass before trusting other tests",
            "B": "Single-model keypoint quality (OKS)",
            "C": "End-to-end severity ICC under different box/inference paths",
            "D": "Is slot/combination logic the bottleneck? Compare D1 vs D2 vs D3",
            "E": "Mean pixel error per keypoint type at GT slot",
            "F": "ICC split by single vs double root",
            "G": "Does YOLO IoU correlate with severity error?",
            "H": "Per-model blame: replace one pred keypoint with GT at GT slot",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 60)
    print("SEVERITY ICC DIAGNOSIS")
    print("=" * 60)
    for line in diagnosis:
        print(f"  • {line}")
    print()
    print("Key ICC values:")
    for key in sorted(tests.keys()):
        if isinstance(tests[key], dict) and "icc" in tests[key]:
            icc = tests[key]["icc"]
            if icc is not None:
                print(f"  {key}: ICC={icc:.4f}  n={tests[key].get('n_pairs', '?')}")
    print(f"\nFull report: {args.out}")


if __name__ == "__main__":
    main()

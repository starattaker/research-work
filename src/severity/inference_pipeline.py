"""End-to-end severity inference: YOLO + 3× Keypoint R-CNN + bone loss formula."""

from __future__ import annotations

import json
from pathlib import Path

import albumentations as A
import cv2
import torch
from torchvision.ops import box_iou
from torchvision.transforms import functional as F
from ultralytics import YOLO

from src.keypoint.inference_utils import filter_keypoint_output
from src.keypoint.model import get_keypoint_model
from src.severity.bone_loss import compute_bone_loss_severity


def build_clahe_transform():
    return A.Compose([A.CLAHE(clip_limit=40.0, tile_grid_size=(8, 8), p=1.0)])


def point_from_slot(keypoints: list, slot: int) -> tuple[float, float]:
    if slot >= len(keypoints):
        return 0.0, 0.0
    kp = keypoints[slot]
    if len(kp) > 2 and int(kp[2]) == 0:
        return 0.0, 0.0
    x, y = float(kp[0]), float(kp[1])
    if x == 0.0 and y == 0.0:
        return 0.0, 0.0
    return x, y


def severity_from_slots(
    cej_pts: list,
    inter_pts: list,
    apex_pts: list,
) -> float | None:
    """v6: try aligned slot 0 then slot 1 (PCA / L-R slots)."""
    for slot in (0, 1):
        sev = compute_bone_loss_severity(
            point_from_slot(cej_pts, slot),
            point_from_slot(inter_pts, slot),
            point_from_slot(apex_pts, slot),
        )
        if sev is not None:
            return sev
    return None


def slot_xy_from_tensor(kps: torch.Tensor, slot: int) -> tuple[float, float] | None:
    if slot >= kps.shape[0]:
        return None
    x, y, v = float(kps[slot, 0]), float(kps[slot, 1]), float(kps[slot, 2])
    if v <= 0 or (x == 0.0 and y == 0.0):
        return None
    return x, y


def severity_from_tensor_slots(
    cej_kps: torch.Tensor,
    inter_kps: torch.Tensor,
    apex_kps: torch.Tensor,
) -> float | None:
    for slot in (0, 1):
        c = slot_xy_from_tensor(cej_kps, slot)
        t = slot_xy_from_tensor(inter_kps, slot)
        a = slot_xy_from_tensor(apex_kps, slot)
        if c is None or t is None or a is None:
            continue
        sev = compute_bone_loss_severity(c, t, a)
        if sev is not None:
            return sev
    return None


def load_gt_annotations(data_root: Path, split: str, stem: str) -> dict | None:
    """Merge CEJ / intersection / apex GT JSON for one image."""
    merged: dict = {"bboxes": None, "labels": None, "cej": [], "intersection": [], "apex": []}
    for kpt_type in ("cej", "intersection", "apex"):
        ann_path = data_root / "keypoints" / kpt_type / split / "annotations" / f"{stem}.json"
        if not ann_path.exists():
            return None
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        if merged["bboxes"] is None:
            merged["bboxes"] = data["bboxes"]
            merged["labels"] = data["labels"]
        merged[kpt_type] = data["keypoints"]
    return merged


def gt_severity_for_tooth(merged: dict, tooth_idx: int) -> float | None:
    return severity_from_slots(
        merged["cej"][tooth_idx],
        merged["intersection"][tooth_idx],
        merged["apex"][tooth_idx],
    )


def load_image_tensor(image_path: Path, transform) -> torch.Tensor:
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if transform is not None:
        img = transform(image=img)["image"]
    return F.to_tensor(img)


def yolo_boxes(result) -> torch.Tensor:
    if result.boxes is None or len(result.boxes) == 0:
        return torch.empty(0, 4)
    return result.boxes.xyxy.cpu()


def best_match_idx(ref: torch.Tensor, boxes: torch.Tensor, min_iou: float) -> int | None:
    if boxes.numel() == 0:
        return None
    ious = box_iou(ref.unsqueeze(0), boxes)[0]
    j = int(ious.argmax().item())
    if float(ious[j]) < min_iou:
        return None
    return j


def keypoints_for_tooth(
    outputs: dict,
    anchor_box: torch.Tensor,
    device: torch.device,
    min_iou: float,
) -> torch.Tensor | None:
    boxes = outputs.get("boxes", torch.empty(0, device=device))
    kps = outputs.get("keypoints", torch.empty(0, device=device))
    if boxes.numel() == 0:
        return None
    j = best_match_idx(anchor_box.to(device), boxes, min_iou)
    if j is None:
        return None
    return kps[j]


@torch.no_grad()
def run_keypoint_model(
    model,
    image_tensor: torch.Tensor,
    device: torch.device,
    score_thresh: float,
    nms_thresh: float,
) -> dict:
    model.eval()
    out = model([image_tensor.to(device)])[0]
    return filter_keypoint_output(out, score_thresh, nms_thresh)


class SeverityPipeline:
    def __init__(
        self,
        yolo_weights: Path,
        cej_weights: Path,
        intersection_weights: Path,
        apex_weights: Path,
        device: str = "cuda",
        score_thresh: float = 0.5,
        nms_thresh: float = 0.6,
        match_iou: float = 0.5,
        keypoint_match_iou: float = 0.5,
        require_yolo: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.match_iou = match_iou
        self.keypoint_match_iou = keypoint_match_iou
        self.require_yolo = require_yolo
        self.yolo = YOLO(str(yolo_weights))
        self.transform = build_clahe_transform()
        self.models = {
            "cej": get_keypoint_model(num_keypoints=2, weights_path=str(cej_weights)).to(self.device),
            "intersection": get_keypoint_model(
                num_keypoints=2, weights_path=str(intersection_weights)
            ).to(self.device),
            "apex": get_keypoint_model(num_keypoints=2, weights_path=str(apex_weights)).to(self.device),
        }
        for m in self.models.values():
            m.eval()

    def predict_image_severities(
        self,
        image_path: Path,
        gt_merged: dict,
    ) -> list[dict]:
        """Per-tooth GT vs predicted severity. Keypoints matched via CEJ det box (same as OKS eval anchor)."""
        yolo_result = self.yolo.predict(
            source=str(image_path),
            imgsz=640,
            conf=self.score_thresh,
            verbose=False,
        )[0]
        yolo_boxes_xyxy = yolo_boxes(yolo_result)
        image_tensor = load_image_tensor(image_path, self.transform)

        cej_out = run_keypoint_model(
            self.models["cej"], image_tensor, self.device, self.score_thresh, self.nms_thresh
        )
        int_out = run_keypoint_model(
            self.models["intersection"],
            image_tensor,
            self.device,
            self.score_thresh,
            self.nms_thresh,
        )
        apex_out = run_keypoint_model(
            self.models["apex"], image_tensor, self.device, self.score_thresh, self.nms_thresh
        )

        rows = []
        gt_boxes = torch.tensor(gt_merged["bboxes"], dtype=torch.float32)
        for i in range(len(gt_merged["bboxes"])):
            gt_sev = gt_severity_for_tooth(gt_merged, i)
            gt_box = gt_boxes[i]

            yolo_idx = best_match_idx(gt_box, yolo_boxes_xyxy, self.match_iou)
            yolo_matched = yolo_idx is not None
            if self.require_yolo and not yolo_matched:
                rows.append(
                    {
                        "tooth_idx": i,
                        "gt_severity": gt_sev,
                        "pred_severity": None,
                        "yolo_matched": False,
                    }
                )
                continue

            cej_boxes = cej_out.get("boxes", torch.empty(0))
            if cej_boxes.numel() == 0:
                rows.append(
                    {
                        "tooth_idx": i,
                        "gt_severity": gt_sev,
                        "pred_severity": None,
                        "yolo_matched": yolo_matched,
                    }
                )
                continue
            cej_j = best_match_idx(gt_box, cej_boxes.cpu(), self.keypoint_match_iou)
            if cej_j is None:
                rows.append(
                    {
                        "tooth_idx": i,
                        "gt_severity": gt_sev,
                        "pred_severity": None,
                        "yolo_matched": yolo_matched,
                    }
                )
                continue

            cej_kps = cej_out["keypoints"][cej_j]
            cej_box = cej_boxes[cej_j]
            int_kps = keypoints_for_tooth(int_out, cej_box, self.device, self.keypoint_match_iou)
            apex_kps = keypoints_for_tooth(apex_out, cej_box, self.device, self.keypoint_match_iou)
            if int_kps is None or apex_kps is None:
                pred_sev = None
            else:
                pred_sev = severity_from_tensor_slots(cej_kps, int_kps, apex_kps)

            rows.append(
                {
                    "tooth_idx": i,
                    "gt_severity": gt_sev,
                    "pred_severity": pred_sev,
                    "yolo_matched": yolo_matched,
                }
            )
        return rows

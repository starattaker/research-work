"""End-to-end severity inference: YOLO + 3× Keypoint R-CNN + bone loss formula."""

from __future__ import annotations

import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torchvision.ops import box_iou
from torchvision.transforms import functional as F
from ultralytics import YOLO

from src.keypoint.inference_utils import filter_keypoint_output
from src.keypoint.model import get_keypoint_model
from src.severity.bone_loss import compute_bone_loss_severity


def build_clahe_transform():
    return A.Compose([A.CLAHE(clip_limit=40.0, tile_grid_size=(8, 8), p=1.0)])


def first_visible_xy(keypoints: list) -> tuple[float, float]:
    """First visible slot; fall back to (0,0) if none."""
    for kp in keypoints:
        if len(kp) > 2 and int(kp[2]) == 0:
            continue
        x, y = float(kp[0]), float(kp[1])
        if x == 0.0 and y == 0.0:
            continue
        return x, y
    return 0.0, 0.0


def tensor_keypoints_to_xy(kps: torch.Tensor) -> tuple[float, float]:
    """Pick first visible keypoint from model output (2, 3)."""
    rows = []
    for i in range(kps.shape[0]):
        x, y, v = float(kps[i, 0]), float(kps[i, 1]), float(kps[i, 2])
        if v > 0 and not (x == 0.0 and y == 0.0):
            rows.append((x, y))
    if not rows:
        return 0.0, 0.0
    return rows[0]


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
    cej = first_visible_xy(merged["cej"][tooth_idx])
    inter = first_visible_xy(merged["intersection"][tooth_idx])
    apex = first_visible_xy(merged["apex"][tooth_idx])
    return compute_bone_loss_severity(cej, inter, apex)


def load_image_tensor(image_path: Path, transform) -> torch.Tensor:
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if transform is not None:
        img = transform(image=img)["image"]
    return F.to_tensor(img)


def yolo_boxes(result, img_w: int, img_h: int) -> torch.Tensor:
    if result.boxes is None or len(result.boxes) == 0:
        return torch.empty(0, 4)
    xyxy = result.boxes.xyxy.cpu()
    return xyxy


def best_match_idx(ref: torch.Tensor, boxes: torch.Tensor, min_iou: float) -> int | None:
    if boxes.numel() == 0:
        return None
    ious = box_iou(ref.unsqueeze(0), boxes)[0]
    j = int(ious.argmax().item())
    if float(ious[j]) < min_iou:
        return None
    return j


def xy_from_filtered_output(
    output: dict,
    ref_box: torch.Tensor,
    device: torch.device,
    min_iou: float,
) -> tuple[float, float]:
    boxes = output.get("boxes", torch.empty(0, device=device))
    kps = output.get("keypoints", torch.empty(0, device=device))
    if boxes.numel() == 0:
        return 0.0, 0.0
    j = best_match_idx(ref_box.to(device), boxes, min_iou)
    if j is None:
        return 0.0, 0.0
    return tensor_keypoints_to_xy(kps[j])


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
        keypoint_match_iou: float = 0.3,
    ):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.match_iou = match_iou
        self.keypoint_match_iou = keypoint_match_iou
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
        """Return per-tooth GT vs predicted severity (matched by GT box index)."""
        img_bgr = cv2.imread(str(image_path))
        h, w = img_bgr.shape[:2]
        yolo_result = self.yolo.predict(
            source=str(image_path),
            imgsz=640,
            conf=self.score_thresh,
            verbose=False,
        )[0]
        yolo_boxes_xyxy = yolo_boxes(yolo_result, w, h)
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
            if yolo_idx is None:
                rows.append(
                    {
                        "tooth_idx": i,
                        "gt_severity": gt_sev,
                        "pred_severity": None,
                        "yolo_matched": False,
                    }
                )
                continue
            ref_box = yolo_boxes_xyxy[yolo_idx]
            cej = xy_from_filtered_output(
                cej_out, ref_box, self.device, self.keypoint_match_iou
            )
            inter = xy_from_filtered_output(
                int_out, ref_box, self.device, self.keypoint_match_iou
            )
            apex = xy_from_filtered_output(
                apex_out, ref_box, self.device, self.keypoint_match_iou
            )
            pred_sev = compute_bone_loss_severity(cej, inter, apex)
            rows.append(
                {
                    "tooth_idx": i,
                    "gt_severity": gt_sev,
                    "pred_severity": pred_sev,
                    "yolo_matched": True,
                }
            )
        return rows

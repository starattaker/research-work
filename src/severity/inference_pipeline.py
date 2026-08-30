"""End-to-end severity inference: YOLO + 3× Keypoint R-CNN + bone loss formula."""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import torch
from torchvision.ops import box_iou
from torchvision.transforms import functional as F
from ultralytics import YOLO

from src.keypoint.inference_utils import predict_keypoints_for_boxes
from src.keypoint.model import get_keypoint_model
from src.severity.gt_labels import (
    gt_severities_for_tooth,
    load_gt_annotations,
    point_from_slot,
    severities_from_gt_lists,
    severity_from_slots_pca,
)
from src.severity.pred_combine import (
    pred_severities_from_tensors,
    pred_severity_from_tensors,
    severity_from_tensor_slots,
    slot_xy_from_tensor,
)

__all__ = [
    "SeverityPipeline",
    "load_gt_annotations",
    "gt_severity_for_tooth",
    "gt_severities_for_tooth",
    "severity_from_slots",
    "severity_from_tensor_slots",
    "point_from_slot",
    "slot_xy_from_tensor",
    "pred_severity_from_tensors",
    "pred_severities_from_tensors",
]


def build_clahe_transform():
    return A.Compose([A.CLAHE(clip_limit=40.0, tile_grid_size=(8, 8), p=1.0)])


def severity_from_slots(
    cej_pts: list,
    inter_pts: list,
    apex_pts: list,
    *,
    slot_convention: str = "pca",
    merge_radius_px: float = 20.0,
) -> float | None:
    if slot_convention == "paper_x":
        sides = severities_from_gt_lists(cej_pts, inter_pts, apex_pts, merge_radius_px)
        return sides[0][1] if sides else None
    return severity_from_slots_pca(cej_pts, inter_pts, apex_pts)


def gt_severity_for_tooth(
    merged: dict,
    tooth_idx: int,
    *,
    slot_convention: str = "pca",
) -> float | None:
    return severity_from_slots(
        merged["cej"][tooth_idx],
        merged["intersection"][tooth_idx],
        merged["apex"][tooth_idx],
        slot_convention=slot_convention,
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
        inference_mode: str = "roi",
        require_yolo: bool = True,
        gt_proposals: bool = False,
        combine_mode: str = "paper_x",
        gt_slot_convention: str = "paper_x",
        severity_protocol: str = "both_sides",
        apex_merge_px: float = 20.0,
    ):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.match_iou = match_iou
        self.keypoint_match_iou = keypoint_match_iou
        self.inference_mode = inference_mode
        self.require_yolo = require_yolo
        self.gt_proposals = gt_proposals
        self.combine_mode = combine_mode
        self.gt_slot_convention = gt_slot_convention
        self.severity_protocol = severity_protocol
        self.apex_merge_px = apex_merge_px
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

    def _proposal_boxes_for_teeth(
        self,
        gt_boxes: torch.Tensor,
        yolo_boxes_xyxy: torch.Tensor,
    ) -> tuple[list[torch.Tensor | None], list[bool]]:
        proposals: list[torch.Tensor | None] = []
        yolo_matched: list[bool] = []
        for gt_box in gt_boxes:
            yolo_idx = best_match_idx(gt_box, yolo_boxes_xyxy, self.match_iou)
            yolo_matched.append(yolo_idx is not None)
            if self.gt_proposals:
                proposals.append(gt_box)
            elif yolo_idx is not None:
                proposals.append(yolo_boxes_xyxy[yolo_idx])
            elif not self.require_yolo:
                proposals.append(gt_box)
            else:
                proposals.append(None)
        return proposals, yolo_matched

    def predict_image_severities(
        self,
        image_path: Path,
        gt_merged: dict,
    ) -> list[dict]:
        yolo_result = self.yolo.predict(
            source=str(image_path),
            imgsz=640,
            conf=self.score_thresh,
            verbose=False,
        )[0]
        yolo_boxes_xyxy = yolo_boxes(yolo_result)
        image_tensor = load_image_tensor(image_path, self.transform)

        gt_boxes = torch.tensor(gt_merged["bboxes"], dtype=torch.float32)
        tooth_proposals, yolo_flags = self._proposal_boxes_for_teeth(gt_boxes, yolo_boxes_xyxy)

        valid_idx = [i for i, box in enumerate(tooth_proposals) if box is not None]
        kps_by_tooth: dict[int, tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]] = {}

        if valid_idx:
            prop_tensor = torch.stack([tooth_proposals[i] for i in valid_idx])
            gt_labels = torch.tensor(gt_merged["labels"], dtype=torch.int64)
            label_tensor = gt_labels[valid_idx]
            cej_list = predict_keypoints_for_boxes(
                self.models["cej"], image_tensor, prop_tensor, self.device,
                self.inference_mode, self.score_thresh, self.nms_thresh, label_tensor, self.keypoint_match_iou,
            )
            int_list = predict_keypoints_for_boxes(
                self.models["intersection"], image_tensor, prop_tensor, self.device,
                self.inference_mode, self.score_thresh, self.nms_thresh, label_tensor, self.keypoint_match_iou,
            )
            apex_list = predict_keypoints_for_boxes(
                self.models["apex"], image_tensor, prop_tensor, self.device,
                self.inference_mode, self.score_thresh, self.nms_thresh, label_tensor, self.keypoint_match_iou,
            )
            for row, tooth_idx in enumerate(valid_idx):
                kps_by_tooth[tooth_idx] = (cej_list[row], int_list[row], apex_list[row])

        rows = []
        for i in range(len(gt_merged["bboxes"])):
            gt_sides = gt_severities_for_tooth(
                gt_merged, i, slot_convention=self.gt_slot_convention, merge_radius_px=self.apex_merge_px
            )
            pred_sides: list[tuple[int, float]] = []
            if i in kps_by_tooth:
                cej_kps, int_kps, apex_kps = kps_by_tooth[i]
                if cej_kps is not None and int_kps is not None and apex_kps is not None:
                    pred_sides = pred_severities_from_tensors(
                        cej_kps,
                        int_kps,
                        apex_kps,
                        combine_mode=self.combine_mode,
                        merge_radius_px=self.apex_merge_px,
                    )
            gt_sev = gt_sides[0][1] if gt_sides else None
            pred_sev = pred_sides[0][1] if pred_sides else None
            rows.append(
                {
                    "tooth_idx": i,
                    "gt_severity": gt_sev,
                    "pred_severity": pred_sev,
                    "gt_sides": gt_sides,
                    "pred_sides": pred_sides,
                    "yolo_matched": yolo_flags[i],
                }
            )
        return rows

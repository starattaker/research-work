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

from src.keypoint.inference_utils import predict_keypoints_on_proposals
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

    def _proposal_boxes_for_teeth(
        self,
        gt_boxes: torch.Tensor,
        yolo_boxes_xyxy: torch.Tensor,
    ) -> tuple[list[torch.Tensor | None], list[bool]]:
        proposals: list[torch.Tensor | None] = []
        yolo_matched: list[bool] = []
        for gt_box in gt_boxes:
            yolo_idx = best_match_idx(gt_box, yolo_boxes_xyxy, self.match_iou)
            if yolo_idx is not None:
                proposals.append(yolo_boxes_xyxy[yolo_idx])
                yolo_matched.append(True)
            elif not self.require_yolo:
                proposals.append(gt_box)
                yolo_matched.append(False)
            else:
                proposals.append(None)
                yolo_matched.append(False)
        return proposals, yolo_matched

    def predict_image_severities(
        self,
        image_path: Path,
        gt_merged: dict,
    ) -> list[dict]:
        """Per-tooth GT vs predicted severity using YOLO boxes as Keypoint R-CNN proposals."""
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
            cej_list = predict_keypoints_on_proposals(
                self.models["cej"],
                image_tensor,
                prop_tensor,
                self.device,
                self.score_thresh,
                self.nms_thresh,
                self.keypoint_match_iou,
            )
            int_list = predict_keypoints_on_proposals(
                self.models["intersection"],
                image_tensor,
                prop_tensor,
                self.device,
                self.score_thresh,
                self.nms_thresh,
                self.keypoint_match_iou,
            )
            apex_list = predict_keypoints_on_proposals(
                self.models["apex"],
                image_tensor,
                prop_tensor,
                self.device,
                self.score_thresh,
                self.nms_thresh,
                self.keypoint_match_iou,
            )
            for row, tooth_idx in enumerate(valid_idx):
                kps_by_tooth[tooth_idx] = (cej_list[row], int_list[row], apex_list[row])

        rows = []
        for i in range(len(gt_merged["bboxes"])):
            gt_sev = gt_severity_for_tooth(gt_merged, i)
            pred_sev = None
            if i in kps_by_tooth:
                cej_kps, int_kps, apex_kps = kps_by_tooth[i]
                if cej_kps is not None and int_kps is not None and apex_kps is not None:
                    pred_sev = severity_from_tensor_slots(cej_kps, int_kps, apex_kps)
            rows.append(
                {
                    "tooth_idx": i,
                    "gt_severity": gt_sev,
                    "pred_severity": pred_sev,
                    "yolo_matched": yolo_flags[i],
                }
            )
        return rows

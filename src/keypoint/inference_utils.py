"""Filter Keypoint R-CNN inference outputs for visualization and metrics."""

from __future__ import annotations

import torch
from torchvision.ops import box_iou, nms


def filter_keypoint_output(
    output: dict,
    score_thresh: float = 0.5,
    nms_thresh: float = 0.6,
    max_detections: int = 100,
) -> dict:
    """Keep high-confidence boxes and apply NMS (paper inference IoU = 0.6)."""
    boxes = output.get("boxes", torch.empty(0))
    if boxes.numel() == 0:
        return output

    scores = output["scores"]
    keep = scores >= score_thresh
    if keep.sum() == 0:
        return {k: v[:0] if isinstance(v, torch.Tensor) else v for k, v in output.items()}

    boxes = boxes[keep]
    scores = scores[keep]
    labels = output["labels"][keep]
    keypoints = output["keypoints"][keep]

    order = nms(boxes, scores, nms_thresh)
    order = order[:max_detections]
    filtered = {
        "boxes": boxes[order],
        "scores": scores[order],
        "labels": labels[order],
        "keypoints": keypoints[order],
    }
    for key, value in output.items():
        if key not in filtered:
            filtered[key] = value
    return filtered


def keypoints_for_proposal(
    output: dict,
    proposal_box: torch.Tensor,
    min_iou: float,
) -> torch.Tensor | None:
    """Pick keypoints from ROI-head output that best overlaps a fixed proposal box."""
    boxes = output.get("boxes")
    keypoints = output.get("keypoints")
    if boxes is None or keypoints is None or boxes.numel() == 0:
        return None
    device = boxes.device
    ious = box_iou(proposal_box.unsqueeze(0).to(device), boxes)[0]
    j = int(ious.argmax().item())
    if float(ious[j]) < min_iou:
        return None
    return keypoints[j]


@torch.no_grad()
def predict_keypoints_on_proposals(
    model,
    image_tensor: torch.Tensor,
    proposal_boxes: torch.Tensor,
    device: torch.device,
    score_thresh: float = 0.5,
    nms_thresh: float = 0.6,
    match_iou: float = 0.3,
) -> list[torch.Tensor | None]:
    """
    Paper inference path: fixed boxes (YOLO) → backbone + ROI heads only (no RPN).

    Returns one (num_keypoints, 3) tensor per proposal, or None when no valid detection.
    """
    if proposal_boxes.numel() == 0:
        return []

    model.eval()
    boxes = proposal_boxes.to(device).float()
    labels = torch.ones(len(boxes), dtype=torch.int64, device=device)
    targets = [{"boxes": boxes, "labels": labels}]
    images = [image_tensor.to(device)]

    images_t, targets_t = model.transform(images, targets)
    features = model.backbone(images_t.tensors)
    proposals = [targets_t[0]["boxes"]]
    detections, _ = model.roi_heads(features, proposals, images_t.image_sizes, None)
    filtered = filter_keypoint_output(detections[0], score_thresh, nms_thresh)

    prop = proposals[0]
    return [keypoints_for_proposal(filtered, prop[j], match_iou) for j in range(len(prop))]

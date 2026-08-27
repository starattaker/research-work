"""Filter Keypoint R-CNN inference outputs for visualization and metrics."""

from __future__ import annotations

import torch
from torchvision.ops import nms


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


def _keypoints_from_roi_output(output: dict) -> torch.Tensor | None:
    keypoints = output.get("keypoints")
    scores = output.get("scores")
    if keypoints is None or scores is None or keypoints.numel() == 0:
        return None
    best = int(scores.argmax().item())
    return keypoints[best]


@torch.no_grad()
def predict_keypoints_on_proposals(
    model,
    image_tensor: torch.Tensor,
    proposal_boxes: torch.Tensor,
    device: torch.device,
    score_thresh: float = 0.5,
    nms_thresh: float = 0.6,
    proposal_labels: torch.Tensor | None = None,
) -> list[torch.Tensor | None]:
    """
    Paper inference path: fixed boxes (YOLO or GT) → backbone + ROI heads only (no RPN).

    Runs one proposal at a time so each tooth gets exactly one keypoint tensor.
    """
    if proposal_boxes.numel() == 0:
        return []

    model.eval()
    boxes = proposal_boxes.to(device).float()
    if proposal_labels is None:
        labels_all = torch.ones(len(boxes), dtype=torch.int64, device=device)
    else:
        labels_all = proposal_labels.to(device).long()

    images = [image_tensor.to(device)]
    results: list[torch.Tensor | None] = []

    for j in range(len(boxes)):
        box = boxes[j : j + 1]
        labels = labels_all[j : j + 1]
        targets = [{"boxes": box, "labels": labels}]
        images_t, targets_t = model.transform(images, targets)
        features = model.backbone(images_t.tensors)
        proposals = [targets_t[0]["boxes"]]
        detections, _ = model.roi_heads(features, proposals, images_t.image_sizes, None)
        filtered = filter_keypoint_output(detections[0], score_thresh, nms_thresh)
        results.append(_keypoints_from_roi_output(filtered))

    return results

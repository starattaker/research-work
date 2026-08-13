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

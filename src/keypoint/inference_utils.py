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
    prefer_label: int | None = None,
) -> torch.Tensor | None:
    """Pick keypoints from detections that best overlap a reference box."""
    boxes = output.get("boxes")
    keypoints = output.get("keypoints")
    labels = output.get("labels")
    if boxes is None or keypoints is None or boxes.numel() == 0:
        return None

    device = boxes.device
    ious = box_iou(proposal_box.unsqueeze(0).to(device), boxes)[0]
    candidates = ious >= min_iou
    if not bool(candidates.any()):
        return None

    idxs = torch.where(candidates)[0]
    if prefer_label is not None and labels is not None:
        label_match = labels[idxs] == prefer_label
        if bool(label_match.any()):
            idxs = idxs[label_match]

    best_local = int(ious[idxs].argmax().item())
    return keypoints[idxs[best_local]]


def _keypoints_from_roi_output(output: dict, prefer_label: int | None = None) -> torch.Tensor | None:
    keypoints = output.get("keypoints")
    scores = output.get("scores")
    labels = output.get("labels")
    if keypoints is None or scores is None or keypoints.numel() == 0:
        return None
    if prefer_label is not None and labels is not None:
        mask = labels == prefer_label
        if mask.any():
            best = int(scores[mask].argmax().item())
            return keypoints[mask][best]
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
    """Fixed boxes → ROI heads only (paper-style, no RPN)."""
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
        prefer = int(labels_all[j].item())
        results.append(_keypoints_from_roi_output(filtered, prefer_label=prefer))

    return results


@torch.no_grad()
def predict_keypoints_full_forward(
    model,
    image_tensor: torch.Tensor,
    proposal_boxes: torch.Tensor,
    device: torch.device,
    score_thresh: float = 0.5,
    nms_thresh: float = 0.6,
    proposal_labels: torch.Tensor | None = None,
    match_iou: float = 0.3,
) -> list[torch.Tensor | None]:
    """
    Full model forward (same path as OKS test script), then match each detection to a proposal box.
    """
    if proposal_boxes.numel() == 0:
        return []

    model.eval()
    boxes = proposal_boxes.to(device).float()
    if proposal_labels is None:
        labels_all = torch.ones(len(boxes), dtype=torch.int64, device=device)
    else:
        labels_all = proposal_labels.to(device).long()

    output = model([image_tensor.to(device)])[0]
    filtered = filter_keypoint_output(output, score_thresh, nms_thresh)

    results: list[torch.Tensor | None] = []
    for j in range(len(boxes)):
        prefer = int(labels_all[j].item())
        results.append(
            keypoints_for_proposal(filtered, boxes[j], match_iou, prefer_label=prefer)
        )
    return results


def predict_keypoints_for_boxes(
    model,
    image_tensor: torch.Tensor,
    proposal_boxes: torch.Tensor,
    device: torch.device,
    mode: str = "full",
    score_thresh: float = 0.5,
    nms_thresh: float = 0.6,
    proposal_labels: torch.Tensor | None = None,
    match_iou: float = 0.3,
) -> list[torch.Tensor | None]:
    if mode == "roi":
        return predict_keypoints_on_proposals(
            model, image_tensor, proposal_boxes, device,
            score_thresh, nms_thresh, proposal_labels,
        )
    if mode == "full":
        return predict_keypoints_full_forward(
            model, image_tensor, proposal_boxes, device,
            score_thresh, nms_thresh, proposal_labels, match_iou,
        )
    raise ValueError(f"Unknown keypoint inference mode: {mode}")

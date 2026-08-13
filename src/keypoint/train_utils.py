"""Training and evaluation helpers for Keypoint R-CNN."""

from __future__ import annotations

import torch
from torchvision.ops import box_iou

from src.keypoint.debug_log import dbg_log


def _accumulate_losses(losses: dict, totals: dict[str, float]) -> float:
    batch_total = 0.0
    for name, value in losses.items():
        v = float(value.item())
        batch_total += v
        totals[name] = totals.get(name, 0.0) + v
    return batch_total


def train_one_epoch(model, loader, optimizer, device) -> tuple[float, dict[str, float]]:
    model.train()
    total = 0.0
    components: dict[str, float] = {}
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        losses = model(images, targets)
        batch_total = _accumulate_losses(losses, components)
        batch_total_tensor = sum(v for v in losses.values())
        batch_total_tensor.backward()
        optimizer.step()
        total += batch_total
    n = max(len(loader), 1)
    return total / n, {k: v / n for k, v in components.items()}


def validate_one_epoch(model, loader, device) -> tuple[float, dict[str, float]]:
    model.train()  # Keypoint R-CNN returns losses only in train mode
    total = 0.0
    components: dict[str, float] = {}
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses = model(images, targets)
            total += _accumulate_losses(losses, components)
    n = max(len(loader), 1)
    return total / n, {k: v / n for k, v in components.items()}


def keypoint_similarity(gt_kpts, pred_kpts, sigmas, areas):
    eps = torch.finfo(torch.float32).eps
    dist_sq = (gt_kpts[:, None, :, 0] - pred_kpts[..., 0]) ** 2 + (
        gt_kpts[:, None, :, 1] - pred_kpts[..., 1]
    ) ** 2
    vis_mask = gt_kpts[..., 2].int() > 0
    k = 2 * sigmas
    denom = 2 * (k**2) * (areas[:, None, None] + eps)
    exp_term = dist_sq / denom
    oks = (torch.exp(-exp_term) * vis_mask[:, None, :]).sum(-1) / (
        vis_mask[:, None, :].sum(-1) + eps
    )
    return oks


def evaluate_oks(model, loader, device) -> float:
    model.eval()
    scores = []
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            for out, tgt in zip(outputs, targets):
                gt_boxes = tgt["boxes"]
                n = len(gt_boxes)
                if n == 0:
                    continue

                pred_boxes = out["boxes"]
                pred_keypoints = out["keypoints"]
                if len(pred_boxes) == 0:
                    # #region agent log
                    dbg_log(
                        "H2",
                        "train_utils.py:evaluate_oks",
                        "skipped image: zero predictions",
                        {"gt_objects": n},
                    )
                    # #endregion
                    continue

                if len(pred_boxes) < n:
                    # #region agent log
                    dbg_log(
                        "H2",
                        "train_utils.py:evaluate_oks",
                        "pred/gt count mismatch; using IoU matching",
                        {"gt_objects": n, "pred_objects": len(pred_boxes)},
                    )
                    # #endregion

                ious = box_iou(gt_boxes, pred_boxes.to(gt_boxes.device))
                matched = []
                for gt_idx in range(n):
                    matched.append(pred_keypoints[int(ious[gt_idx].argmax().item())])
                pred = torch.stack(matched).reshape(-1, 3)
                gt = tgt["keypoints"].reshape(-1, 3)

                sigmas = torch.ones(len(gt), device=device) / len(gt)
                oks = keypoint_similarity(
                    gt.unsqueeze(0), pred.unsqueeze(0), sigmas, tgt["area"].to(device)
                )
                scores.append(float(oks.mean().item()))
    return sum(scores) / max(len(scores), 1)

"""Training visualizations: TensorBoard, loss curves, sample predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.keypoint.inference_utils import filter_keypoint_output


def _finite(value: float) -> float:
    if value != value or abs(value) == float("inf"):
        return 0.0
    return value


class KeypointTrainingViz:
    """TensorBoard + static plots for Keypoint R-CNN training."""

    def __init__(self, log_dir: Path, enabled: bool = True):
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        self.writer: SummaryWriter | None = None
        if enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        lr: float,
        train_components: dict[str, float] | None = None,
        val_components: dict[str, float] | None = None,
    ) -> None:
        if not self.writer:
            return
        self.writer.add_scalar("loss/train", train_loss, epoch)
        self.writer.add_scalar("loss/val", val_loss, epoch)
        self.writer.add_scalar("lr", lr, epoch)
        if train_components:
            for name, value in train_components.items():
                self.writer.add_scalar(f"loss_components/train/{name}", value, epoch)
        if val_components:
            for name, value in val_components.items():
                self.writer.add_scalar(f"loss_components/val/{name}", value, epoch)

    def log_final_metrics(self, results: dict[str, Any]) -> None:
        if not self.writer:
            return
        for split in ("train", "val", "test"):
            oks = results.get(f"{split}_oks")
            if oks is not None:
                self.writer.add_scalar(f"metrics/{split}_oks", float(oks), 0)
            split_map = results.get(f"{split}_map") or {}
            for key, value in split_map.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(
                        f"metrics/{split}_map/{key}",
                        _finite(float(value)),
                        0,
                    )

    def log_sample_predictions(
        self,
        model: torch.nn.Module,
        loader,
        device: torch.device,
        epoch: int,
        tag: str = "predictions/val",
        max_images: int = 4,
    ) -> None:
        if not self.writer:
            return
        model.eval()
        shown = 0
        with torch.no_grad():
            for images, targets in loader:
                images_dev = [img.to(device) for img in images]
                outputs = model(images_dev)
                for img, tgt, out in zip(images, targets, outputs):
                    if shown >= max_images:
                        break
                    out = filter_keypoint_output(out)
                    canvas = draw_keypoint_predictions(img, tgt, out)
                    self.writer.add_image(tag, canvas, epoch, dataformats="HWC")
                    shown += 1
                if shown >= max_images:
                    break

    def close(self) -> None:
        if self.writer:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    @staticmethod
    def plot_loss_curves(history: list[dict], out_path: Path) -> None:
        if not history:
            return
        epochs = [row["epoch"] for row in history]
        train = [row["train_loss"] for row in history]
        val = [row["val_loss"] for row in history]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, train, label="train_loss", linewidth=2)
        ax.plot(epochs, val, label="val_loss", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Keypoint R-CNN training")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def draw_keypoint_predictions(
    image: torch.Tensor,
    target: dict,
    output: dict,
    max_teeth: int = 16,
    title: str = "",
) -> np.ndarray:
    """Overlay GT (green boxes/keypoints) and predictions (red) on an image tensor."""
    img = _draw_predictions(image, target, output, max_teeth=max_teeth)
    if title:
        import cv2

        cv2.putText(
            img,
            title[:100],
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "Green=GT  Red=Pred",
            (8, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
    return img


def _draw_predictions(
    image: torch.Tensor,
    target: dict,
    output: dict,
    max_teeth: int = 12,
) -> np.ndarray:
    """Overlay GT (green) and predictions (red) on a single image."""
    img = image.detach().cpu().permute(1, 2, 0).numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8).copy()

    gt_boxes = target["boxes"].cpu().numpy()
    gt_kps = target["keypoints"].cpu().numpy()
    pred_boxes = output.get("boxes", torch.empty(0)).detach().cpu().numpy()
    pred_kps = output.get("keypoints", torch.empty(0)).detach().cpu().numpy()

    for idx in range(min(len(gt_boxes), max_teeth)):
        x1, y1, x2, y2 = gt_boxes[idx].astype(int)
        cv2_rect(img, x1, y1, x2, y2, (0, 200, 0))
        for kp in gt_kps[idx]:
            if kp[2] > 0:
                cv2_circle(img, int(kp[0]), int(kp[1]), (0, 255, 0))

    for idx in range(min(len(pred_boxes), max_teeth)):
        x1, y1, x2, y2 = pred_boxes[idx].astype(int)
        cv2_rect(img, x1, y1, x2, y2, (220, 80, 80))
        if idx < len(pred_kps):
            for kp in pred_kps[idx]:
                cv2_circle(img, int(kp[0]), int(kp[1]), (255, 60, 60))

    return img


def cv2_rect(img, x1, y1, x2, y2, color, thickness=2):
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    img[y1 : y1 + thickness, x1:x2] = color
    img[y2 - thickness : y2, x1:x2] = color
    img[y1:y2, x1 : x1 + thickness] = color
    img[y1:y2, x2 - thickness : x2] = color


def cv2_circle(img, x, y, color, radius=4):
    h, w = img.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    img[y0:y1, x0:x1] = color


def plot_yolo_results(results_csv: Path, out_path: Path) -> None:
    """Plot Ultralytics results.csv (detection training)."""
    import pandas as pd

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    epoch_col = "epoch" if "epoch" in df.columns else df.columns[0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ("train/box_loss", "train/cls_loss", "Box / cls loss"),
        ("metrics/precision(B)", "metrics/recall(B)", "Precision / recall"),
        ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "mAP50 / mAP50-95"),
        ("val/box_loss", "val/cls_loss", "Val box / cls loss"),
    ]
    for ax, (col_a, col_b, title) in zip(axes.flat, panels):
        plotted = False
        for col, label in ((col_a, col_a.split("/")[-1]), (col_b, col_b.split("/")[-1])):
            if col in df.columns:
                ax.plot(df[epoch_col], df[col], label=label, linewidth=2)
                plotted = True
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend()
    fig.suptitle(f"YOLO training — {results_csv.parent.name}", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_history(output_dir: Path) -> list[dict]:
    history_file = output_dir / "history.json"
    metrics_file = output_dir / "metrics.json"
    if history_file.exists():
        return json.loads(history_file.read_text(encoding="utf-8"))
    if metrics_file.exists():
        return json.loads(metrics_file.read_text(encoding="utf-8")).get("history", [])
    return []

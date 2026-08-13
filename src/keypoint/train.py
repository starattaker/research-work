"""Keypoint R-CNN training utilities (paper-matched hyperparameters)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import albumentations as A
import torch
import torchvision
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from torchvision.transforms import functional as F

from src.keypoint.dataset import KeypointDataset
from src.keypoint.debug_log import dbg_log
from src.keypoint.model import get_keypoint_model
from src.keypoint.train_utils import evaluate_oks, train_one_epoch, validate_one_epoch


def build_transforms(train: bool):
    transforms = [
        A.CLAHE(clip_limit=40.0, tile_grid_size=(8, 8), p=1.0),
    ]
    return A.Compose(
        transforms,
        keypoint_params=A.KeypointParams(format="xy"),
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["bboxes_labels"]),
    )


def get_loaders(data_root: Path, batch_size: int = 8):
    train_ds = KeypointDataset(data_root / "train", transform=build_transforms(True))
    val_ds = KeypointDataset(data_root / "val", transform=build_transforms(False))
    test_ds = KeypointDataset(data_root / "test", transform=build_transforms(False))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=KeypointDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=KeypointDataset.collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=KeypointDataset.collate_fn,
    )
    return train_loader, val_loader, test_loader


def serialize_detection_metrics(computed: dict) -> dict:
    """Convert torchmetrics mAP output to JSON-safe scalars and lists."""
    # #region agent log
    dbg_log(
        "H3",
        "train.py:serialize_detection_metrics",
        "raw metric keys and types",
        {
            "keys": list(computed.keys()),
            "types": {k: type(v).__name__ for k, v in computed.items()},
            "shapes": {
                k: (list(v.shape) if hasattr(v, "shape") else None)
                for k, v in computed.items()
            },
        },
    )
    # #endregion
    out = {}
    for key, value in computed.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                scalar = float(value.item())
                # #region agent log
                if scalar != scalar or abs(scalar) == float("inf"):
                    dbg_log(
                        "H1",
                        "train.py:serialize_detection_metrics",
                        "non-finite scalar metric",
                        {"key": key, "value": scalar},
                    )
                # #endregion
                out[key] = scalar
            else:
                out[key] = value.detach().cpu().tolist()
        else:
            # #region agent log
            dbg_log(
                "H4",
                "train.py:serialize_detection_metrics",
                "non-tensor metric value",
                {"key": key, "type": type(value).__name__, "repr": repr(value)[:200]},
            )
            # #endregion
            out[key] = value
    return out


def evaluate_map(model, loader, device) -> dict:
    metric = MeanAveragePrecision()
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            outputs = [{k: v.to(device) for k, v in out.items()} for out in outputs]
            metric.update(outputs, targets)
    return serialize_detection_metrics(metric.compute())


def build_results(model, train_loader, val_loader, test_loader, device, keypoint_type, history):
    results = {
        "keypoint_type": keypoint_type,
        "train_map": evaluate_map(model, train_loader, device),
        "val_map": evaluate_map(model, val_loader, device),
        "test_map": evaluate_map(model, test_loader, device),
        "train_oks": evaluate_oks(model, train_loader, device),
        "val_oks": evaluate_oks(model, val_loader, device),
        "test_oks": evaluate_oks(model, test_loader, device),
        "history": history,
    }
    # #region agent log
    try:
        json.dumps(results)
        dbg_log("H1", "train.py:build_results", "json.dumps succeeded", {"keypoint_type": keypoint_type})
    except Exception as exc:
        dbg_log(
            "H1",
            "train.py:build_results",
            "json.dumps failed",
            {"error": str(exc), "keypoint_type": keypoint_type},
        )
        raise
    # #endregion
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--keypoint-type", choices=["cej", "intersection", "apex"], required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/keypoints"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; load best.pt and write metrics.json",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_loader, val_loader, test_loader = get_loaders(args.data_root, args.batch_size)
    model = get_keypoint_model(num_keypoints=2).to(device)

    if args.eval_only:
        weights = args.output_dir / "best.pt"
        if not weights.exists():
            raise FileNotFoundError(f"No checkpoint at {weights}")
        model.load_state_dict(torch.load(weights, map_location=device))
        history_path = args.output_dir / "metrics.json"
        history = []
        history_file = args.output_dir / "history.json"
        if history_file.exists():
            history = json.loads(history_file.read_text(encoding="utf-8"))
        elif history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8")).get("history", [])
        results = build_results(
            model, train_loader, val_loader, test_loader, device, args.keypoint_type, history
        )
        history_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.6)

    best_val = float("inf")
    patience_counter = 0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, device)
        scheduler.step()

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(json.dumps(row))
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), args.output_dir / "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    results = build_results(
        model, train_loader, val_loader, test_loader, device, args.keypoint_type, history
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

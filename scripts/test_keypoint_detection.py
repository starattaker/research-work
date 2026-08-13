"""Visual test for Keypoint R-CNN — GT vs predictions on sample images.

Run in a separate terminal while training continues.

Example:
  export PYTHONPATH=.
  python scripts/test_keypoint_detection.py --keypoint-type cej --n 10 --compare-gt
  python scripts/test_keypoint_detection.py --keypoint-type intersection --n 10
  python scripts/test_keypoint_detection.py --keypoint-type apex --prefer-double
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import torch

from src.keypoint.dataset import KeypointDataset
from src.keypoint.model import get_keypoint_model
from src.keypoint.train import build_transforms
from src.keypoint.train_utils import keypoint_similarity
from src.keypoint.inference_utils import filter_keypoint_output
from src.keypoint.training_viz import draw_keypoint_predictions
from torchvision.ops import box_iou


def resolve_weights(keypoint_type: str, weights: Path | None) -> Path:
    candidates = [
        weights,
        Path("runs/keypoints") / keypoint_type / "best.pt",
        Path("runs/keypoints") / f"v1_{keypoint_type}" / "best.pt",
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No weights for {keypoint_type}. Train first or pass --weights path/to/best.pt"
    )


def image_oks(target: dict, output: dict, device: torch.device) -> float:
    gt_boxes = target["boxes"]
    n = len(gt_boxes)
    if n == 0:
        return 0.0
    pred_boxes = output.get("boxes", torch.empty(0, device=device))
    pred_keypoints = output.get("keypoints", torch.empty(0, device=device))
    if len(pred_boxes) == 0:
        return 0.0

    ious = box_iou(gt_boxes, pred_boxes)
    matched = []
    for gt_idx in range(n):
        matched.append(pred_keypoints[int(ious[gt_idx].argmax().item())])
    pred = torch.stack(matched).reshape(-1, 3)
    gt = target["keypoints"].reshape(-1, 3)
    sigmas = torch.ones(len(gt), device=device) / len(gt)
    oks = keypoint_similarity(
        gt.unsqueeze(0), pred.unsqueeze(0), sigmas, target["area"].to(device)
    )
    return float(oks.mean().item())


def main():
    parser = argparse.ArgumentParser(description="Visual Keypoint R-CNN test on N images")
    parser.add_argument(
        "--keypoint-type",
        choices=["cej", "intersection", "apex"],
        required=True,
    )
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Default: data/processed/keypoints/{type}",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--images",
        nargs="*",
        default=None,
        help="Optional explicit image stems (without .jpg), overrides --n sampling",
    )
    parser.add_argument(
        "--prefer-double",
        action="store_true",
        help="Prefer images with double-root (label=2) teeth",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: research_log/figures/keypoint_inference/{type}",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--nms-thresh", type=float, default=0.6, help="Paper inference NMS IoU")
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Show all raw model outputs (many overlapping boxes)",
    )
    args = parser.parse_args()

    data_root = args.data_root or Path("data/processed/keypoints") / args.keypoint_type
    out_dir = args.out_dir or Path("research_log/figures/keypoint_inference") / args.keypoint_type
    weights = resolve_weights(args.keypoint_type, args.weights)
    device = torch.device(args.device)

    dataset = KeypointDataset(data_root / args.split, transform=build_transforms(False))

    if args.images:
        stems = set(args.images)
        indices = [i for i, p in enumerate(dataset.image_files) if p.stem in stems]
    else:
        all_indices = list(range(len(dataset)))
        if args.prefer_double:
            double_indices = []
            for i in all_indices:
                ann = json.loads(
                    (dataset.ann_dir / f"{dataset.image_files[i].stem}.json").read_text(
                        encoding="utf-8"
                    )
                )
                if any(lbl == 2 for lbl in ann.get("labels", [])):
                    double_indices.append(i)
            pool = double_indices or all_indices
        else:
            pool = all_indices
        rng = random.Random(args.seed)
        indices = rng.sample(pool, min(args.n, len(pool)))

    model = get_keypoint_model(num_keypoints=2, weights_path=str(weights)).to(device)
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    print(f"Model:   {args.keypoint_type}")
    print(f"Weights: {weights}")
    print(f"Split:   {args.split}  ({len(indices)} images)")
    print(f"Output:  {out_dir}\n")

    with torch.no_grad():
        for idx in indices:
            image, target = dataset[idx]
            stem = dataset.image_files[idx].stem
            output = model([image.to(device)])[0]
            if not args.show_raw:
                output = filter_keypoint_output(
                    output, args.score_thresh, args.nms_thresh
                )
            target_dev = {k: v.to(device) for k, v in target.items()}

            oks = image_oks(target_dev, output, device)
            n_teeth = len(target["boxes"])
            n_pred = len(output.get("boxes", []))
            stats = {
                "image": stem,
                "keypoint_type": args.keypoint_type,
                "n_gt_teeth": n_teeth,
                "n_pred": n_pred,
                "image_oks": round(oks, 4),
            }
            summary.append(stats)

            title = f"{args.keypoint_type} | {stem} | OKS={oks:.3f} | GT={n_teeth} pred={n_pred}"
            canvas = draw_keypoint_predictions(image, target, output, title=title)
            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            out_path = out_dir / f"{args.split}_{stem}.jpg"
            cv2.imwrite(str(out_path), canvas_bgr)
            print(f"{stem}  OKS={oks:.3f}  teeth={n_teeth}  preds={n_pred}  -> {out_path.name}")

    if summary:
        mean_oks = sum(s["image_oks"] for s in summary) / len(summary)
        print(f"\nMean image OKS: {mean_oks:.3f} over {len(summary)} images")

    summary_path = out_dir / f"summary_{args.split}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

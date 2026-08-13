"""Plot training curves from saved run artifacts (no TensorBoard required)."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.keypoint.training_viz import KeypointTrainingViz, load_history, plot_yolo_results


def main():
    parser = argparse.ArgumentParser(description="Plot YOLO and/or keypoint training curves")
    parser.add_argument(
        "--keypoint-dir",
        type=Path,
        action="append",
        default=[],
        help="Keypoint run dir (e.g. runs/keypoints/cej). Repeat for multiple models.",
    )
    parser.add_argument(
        "--yolo-results",
        type=Path,
        default=None,
        help="Ultralytics results.csv (e.g. runs/detect/.../results.csv)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("research_log/figures/training"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for run_dir in args.keypoint_dir:
        history = load_history(run_dir)
        out = args.out_dir / f"{run_dir.name}_loss_curve.png"
        KeypointTrainingViz.plot_loss_curves(history, out)
        print(f"Wrote {out}")

    if args.yolo_results and args.yolo_results.exists():
        out = args.out_dir / "yolo_training.png"
        plot_yolo_results(args.yolo_results, out)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()

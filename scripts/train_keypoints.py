"""Launch all three Keypoint R-CNN models sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    python = sys.executable
    for kpt_type in ("cej", "intersection", "apex"):
        data_root = args.processed_root / "keypoints" / kpt_type
        output_dir = Path("runs/keypoints") / kpt_type
        cmd = [
            python,
            "-m",
            "src.keypoint.train",
            "--data-root",
            str(data_root),
            "--keypoint-type",
            kpt_type,
            "--output-dir",
            str(output_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

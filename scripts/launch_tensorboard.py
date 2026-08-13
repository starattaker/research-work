"""Launch TensorBoard for keypoint runs under runs/keypoints/."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        type=Path,
        default=Path("runs/keypoints"),
        help="Root folder containing */tensorboard event files",
    )
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if not args.logdir.exists():
        print(f"Log dir not found: {args.logdir}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(args.logdir),
        "--port",
        str(args.port),
        "--host",
        args.host,
    ]
    print("Starting TensorBoard:", " ".join(cmd))
    print(f"Open http://{args.host}:{args.port}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

"""Run keypoint eval path on a tiny subset to surface post-training failures."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.keypoint.debug_log import dbg_log
from src.keypoint.model import get_keypoint_model
from src.keypoint.train import build_results, get_loaders


def main():
    data_root = Path("data/processed/keypoints/cej")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dbg_log("H0", "debug_keypoint_eval_probe.py", "probe start", {"device": str(device)})

    train_loader, val_loader, test_loader = get_loaders(data_root, batch_size=2)
    model = get_keypoint_model(num_keypoints=2).to(device)
    model.eval()

    # Use only first 2 batches from each split to keep probe fast.
    def take_two(loader):
        for i, batch in enumerate(loader):
            if i >= 2:
                break
            yield batch

    class LimitedLoader:
        def __init__(self, loader):
            self._batches = list(take_two(loader))

        def __iter__(self):
            return iter(self._batches)

        def __len__(self):
            return len(self._batches)

    results = build_results(
        model,
        LimitedLoader(train_loader),
        LimitedLoader(val_loader),
        LimitedLoader(test_loader),
        device,
        "cej",
        [{"epoch": 1, "train_loss": 1.0, "val_loss": 1.0}],
    )
    out = Path("runs/keypoints/_probe_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    dbg_log("H0", "debug_keypoint_eval_probe.py", "probe finished", {"output": str(out)})


if __name__ == "__main__":
    main()

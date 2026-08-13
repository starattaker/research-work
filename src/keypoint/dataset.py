"""Keypoint dataset loader for Keypoint R-CNN."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class KeypointDataset(Dataset):
    def __init__(self, root: Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.images_dir = self.root / "images"
        self.ann_dir = self.root / "annotations"
        self.image_files = sorted(self.images_dir.glob("*.jpg"))

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_path = self.image_files[idx]
        ann_path = self.ann_dir / f"{img_path.stem}.json"

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data = json.loads(ann_path.read_text(encoding="utf-8"))

        bboxes = data["bboxes"]
        labels = data["labels"]
        keypoints = data["keypoints"]
        keypoints = [sorted(kps, key=lambda kp: kp[0]) for kps in keypoints]
        keypoints_original = keypoints

        if self.transform is not None:
            flat = [kp[:2] for obj in keypoints for kp in obj]
            transformed = self.transform(
                image=img,
                bboxes=bboxes,
                bboxes_labels=labels,
                keypoints=flat,
            )
            img = transformed["image"]
            bboxes = transformed["bboxes"]
            flat_t = transformed["keypoints"]
            reshaped = np.reshape(np.array(flat_t), (-1, 2, 2)).tolist()
            keypoints = []
            for o_idx, obj in enumerate(reshaped):
                obj_kps = []
                for k_idx, kp in enumerate(obj):
                    obj_kps.append(kp + [keypoints_original[o_idx][k_idx][2]])
                keypoints.append(obj_kps)

        boxes = torch.as_tensor(bboxes, dtype=torch.float32)
        target = {
            "boxes": boxes,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int32),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
            "keypoints": torch.as_tensor(keypoints, dtype=torch.float32),
        }
        return F.to_tensor(img), target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

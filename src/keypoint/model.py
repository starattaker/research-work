"""Keypoint R-CNN model factory."""

import torchvision


def get_keypoint_model(num_keypoints: int = 2, weights_path: str | None = None):
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
        weights=None,
        weights_backbone="DEFAULT",
        num_keypoints=num_keypoints,
        num_classes=3,
    )
    if weights_path:
        import torch

        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    return model
